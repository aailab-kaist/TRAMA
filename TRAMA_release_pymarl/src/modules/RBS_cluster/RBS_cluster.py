#from pandas.core.arrays.categorical import factorize_from_iterable
import torch
import torch_geometric.utils as utils
from scipy.optimize import linear_sum_assignment
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os 
import pickle
import copy

import numpy as np

class RBSCLUSTER():
    def __init__(self, args, qvectors=None):
        self.args = args
        self.threshold = 0.5
        self.flag_cluster_type = 2 # 1: cdhit, 2: K-means # clustering type
        self.n_cluster    = self.args.n_cluster
        self.flag_reduced = self.args.flag_reduced_seq
        self.sim_type     = self.args.similarity_type
        self.default_node = self.args.n_codes

        self.qvectors = qvectors
        self.qvectors0 = copy.deepcopy(qvectors)
        
    def forward(self, sequences, prev_centroid=None, qvectors=None):
        if self.sim_type <= 3:
            clusters, reduced_seq, cluster_labels, sim_centroids = self.f_cluster(sequences, prev_centroid=prev_centroid, cur_qvectors=qvectors)
            return clusters, reduced_seq, cluster_labels, sim_centroids
        else:
            clusters, reduced_seq, cluster_labels, sim_centroids, meanVq = self.f_cluster(sequences, prev_centroid=prev_centroid, cur_qvectors=qvectors)
            return clusters, reduced_seq, cluster_labels, sim_centroids, meanVq
    
    # Define a function to perform CD-HIT clustering
    def f_cluster(self, sequences, prev_centroid=None, cur_qvectors=None):        
        
        if self.sim_type <= 3:
            # step 1: sequence reduction: reduce the repetitive sequence and dimensionality reduction
            num_sequences = sequences.size(0)
            max_seq_len   = sequences.size(1)
            #output_seq = []
            if self.flag_reduced == True:
                input_sequences = self.sequence_reduction(sequences)
            else:
                input_sequences = sequences
        
            # step2: Calculate pairwise sequence similarities based on binary similarity
            similarity_matrix = self.pairwise_similarity(input_sequences) # based on binary similarity
    
        if self.flag_cluster_type == 1: # cd-hit------------------------------------
        # Apply similarity threshold
            similarity_matrix[similarity_matrix < self.threshold] = 0
    
            # Convert dense similarity matrix to sparse adjacency matrix
            adjacency_matrix = (similarity_matrix > 0).to(torch.float)
            adjacency_matrix = adjacency_matrix.cpu().numpy()  # Convert to numpy array
            sparse_adjacency_matrix = sp.coo_matrix(adjacency_matrix)
    
            # Find connected components using connected components labeling
            components = sp.csgraph.connected_components(sparse_adjacency_matrix, directed=False)[1]
            components = [torch.tensor([i for i, comp in enumerate(components) if comp == j], dtype=torch.long) for j in range(components.max() + 1)]
    
            # Map sequences to their cluster IDs
            cluster_mapping = {}
            for cluster_id, component in enumerate(components):
                for seq_idx in component:
                    cluster_mapping[seq_idx.item()] = cluster_id
    
            # Group sequences by cluster
            clusters = {}
            for seq_idx, cluster_id in cluster_mapping.items():
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                #clusters[cluster_id].append(seq_idx.item())
                clusters[cluster_id].append(seq_idx)
            return clusters
    
        else: # Kmeans----------------------------------------------------------
            if self.sim_type <=3:
                # Convert to numpy array
                similarity_matrix_np = similarity_matrix.cpu().numpy()
    
                # Apply similarity threshold and convert to distance
                distance_matrix = 1 - similarity_matrix_np
            
                #.. sanity check for distance_matrix, set 1.0 distance for NaN elements
                nan_indices = np.isnan(distance_matrix)
                if np.any(nan_indices):
                    distance_matrix[nan_indices] = 1.0

                np.fill_diagonal(distance_matrix, 0)
            
                # Perform KMeans clustering with predetermined distance_matrix
                if self.args.flag_init_centroid and prev_centroid is not None:                
                    if len(prev_centroid) == self.n_cluster:
                        if np.shape(distance_matrix)[1] != np.shape(prev_centroid)[1]: # if size is different -> add zero_pad
                            if np.shape(distance_matrix)[1] > np.shape(prev_centroid)[1]:
                                diff_len = np.shape(distance_matrix)[1] - np.shape(prev_centroid)[1]
                                prev_centroid_input = np.concatenate((prev_centroid, np.zeros((self.n_cluster, diff_len))), axis=1)                    
                            else:
                                prev_centroid_input = prev_centroid[:,:np.shape(distance_matrix)[1]]
                                seq1 = np.concatenate((seq1, np.zeros((self.n_cluster, diff_len))), axis=1)
                        else:
                            prev_centroid_input = prev_centroid
                        kmeans = KMeans(n_clusters=self.n_cluster, init=prev_centroid_input, n_init=1)
                    else:
                        kmeans = KMeans(n_clusters=self.n_cluster, init='k-means++')
                else:
                    kmeans = KMeans(n_clusters=self.n_cluster, init='k-means++')
                    
                cluster_labels = kmeans.fit_predict(distance_matrix) # contains possible errors by outputing NaN elements in cluster_labels 
            else: # self.sim_type == 4
                if self.args.flag_init_centroid and prev_centroid is not None: 
                    kmeans = KMeans(n_clusters=self.n_cluster, init=prev_centroid, n_init=1)
                else:
                    kmeans = KMeans(n_clusters=self.n_cluster, init='k-means++')
                self.qvectors = cur_qvectors
                valid_idx = (sequences != self.default_node).cpu().numpy()
                input_sequences = sequences
                meanVq = []                 
                for k in range(len(input_sequences)):
                    ide = sum(valid_idx[k]).item()
                    if ide ==0: ide=1 # to prevent anomaly
                    if self.args.flag_reduced_seq: # here index order is ignored
                        ndx = np.array( list(set(input_sequences[k,:ide].cpu().numpy().astype(np.int32)) ) )
                    else:
                        ndx = input_sequences[k,:ide].cpu().numpy().astype(np.int32) 
                        
                    meanVq.append(np.mean(self.qvectors[ndx,:], axis=0))
                meanVq = np.array(meanVq)    

                cluster_labels = kmeans.fit_predict(meanVq) 
    
            # Group sequences by cluster
            clusters = {}
            for seq_idx, cluster_id in enumerate(cluster_labels):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(seq_idx)
        
            sim_centroids = kmeans.cluster_centers_
            if self.sim_type <=3:
                return clusters, input_sequences, cluster_labels, sim_centroids
            else:
                return clusters, input_sequences, cluster_labels, sim_centroids, meanVq
    

    # Define a function to calculate pairwise sequence similarities
    def pairwise_similarity(self, sequences):
        if self.sim_type == 1: # Euclidean distance
            similarity_matrix = torch.matmul(sequences, sequences.t())
            return similarity_matrix / (sequences.norm(dim=1)[:, None] * sequences.norm(dim=1)) # element-wise normalization
    
        elif self.sim_type == 2: # considering binary similiarty
            num_sequences = sequences.size(0)
            sequence_length = sequences.size(1)
    
            # Expand dimensions to enable broadcasting
            sequences_expanded = sequences[:, None, :]
    
            # Perform element-wise comparison and sum along the sequence length dimension
            similarity_matrix = (sequences_expanded == sequences).float().sum(dim=2) / sequence_length # seq_len x seq_len
    
            return similarity_matrix
    
        elif self.sim_type == 3: # considering binary similiarty with nonzero elements
            num_sequences = sequences.size(0)
            sequence_length = sequences.size(1)
    
            # Create a mask to identify non-zero elements in the sequences
            non_zero_mask = (sequences != self.default_node)
    
            # Expand dimensions to enable broadcasting
            sequences_expanded = sequences[:, None, :]
    
            # Perform element-wise comparison considering only non-zero elements
            similarity_matrix = (sequences_expanded == sequences) & non_zero_mask[:, None, :]
    
            # Sum along the sequence length dimension considering only non-zero elements
            similarity_sum = similarity_matrix.float().sum(dim=2)
    
            # Count the number of non-zero elements in each sequence
            non_zero_count = non_zero_mask.float().sum(dim=1)
    
            # Normalize by the number of non-zero elements in each sequence
            similarity_matrix = similarity_sum / non_zero_count[:, None]
    
        return similarity_matrix

    def sequence_similarity(self, seq1, seq2 ):
        if self.sim_type == 1:  # Euclidean distance
            similarity = torch.matmul(seq1, seq2.t()) / (seq1.norm(dim=1)[:, None] * seq2.norm(dim=1))

        elif self.sim_type == 2:  # Considering binary similarity
            seq1_expanded = seq1[:, None, :]
            similarity = (seq1_expanded == seq2).float().sum(dim=2) / seq1.size(1)

        elif self.sim_type == 3:  # Considering binary similarity with nonzero elements
            non_zero_mask_seq1 = (seq1 != self.default_node)
            non_zero_mask_seq2 = (seq2 != self.default_node)
            seq1_expanded = seq1[:, None, :]
            similarity = (seq1_expanded == seq2) & non_zero_mask_seq1[:, None, :]
            similarity_sum = similarity.float().sum(dim=2)
            non_zero_count_seq1 = non_zero_mask_seq1.float().sum(dim=1)
            non_zero_count_seq2 = non_zero_mask_seq2.float().sum(dim=1)
            similarity = similarity_sum / (non_zero_count_seq1[:, None] + non_zero_count_seq2)

        return similarity
    
    def sequence_reduction(self, sequences):
        num_sequences = sequences.size(0)
        max_seq_len   = sequences.size(1)
        #output_seq = []
   
        max_len = 0
        #output_seq_th = torch.zeros((num_sequences, max_seq_len))
        output_seq_th = self.default_node*torch.ones((num_sequences, max_seq_len))
        
        for i in range(num_sequences):
            input_seq = sequences[i]    
            idx = (torch.nonzero( input_seq[:-1] - input_seq[1:] ) + 1).squeeze(-1) 
            reduced_seq = torch.cat( (input_seq[[0]], input_seq[idx]) , dim=0)
            #output_seq.append(reduced_seq)
            output_seq_th[i,:len(reduced_seq)] = reduced_seq
            max_len = max(max_len, len(reduced_seq))
    
        # idx = (torch.nonzero( sequences[:,:-1] - sequences[:,1:] ) + 1).squeeze(-1) 
        # output_seq = torch.cat( (input_seq[[0]], input_seq[idx]) , dim=0)
        output_seq_th = output_seq_th[:,:max_len]
        return output_seq_th    

    def match_vectors_brute_force(self, vector1, vector2, similarity_matrix):
        k = len(vector1)
        matching = []

        # Loop through each element in vector1
        for i in range(k):
            max_similarity = -np.inf
            matched_index = -1
        
            # Compare with each element in vector2
            for j in range(k):
                similarity = similarity_matrix[i][j]
            
                # If similarity is higher than previous max, update
                if similarity > max_similarity:
                    max_similarity = similarity
                    matched_index = j
        
            # Add matched index to the matching list
            matching.append((i, matched_index))

        return matching
        
    def match_vectors_Hungarian(self, seq1, seq2, match_type=None):
        
        # compute similairty matrix
        if match_type is None: # seq        
            similarity_matrix = self.sequence_similarity(seq1, seq2).detach().cpu().numpy()
        else: # Euclidean distance
            #similarity_matrix = torch.matmul(seq1, seq2.t()) / (seq1.norm(dim=1)[:, None] * seq2.norm(dim=1)).detach().cpu().numpy()
            if np.shape(seq1)[1] != np.shape(seq2)[1]: # if size is different -> add zero_pad
                if np.shape(seq1)[1] > np.shape(seq2)[1]:
                    diff_len = np.shape(seq1)[1] - np.shape(seq2)[1]
                    seq2 = np.concatenate((seq2, np.zeros((self.n_cluster, diff_len))), axis=1)                    
                else:
                    diff_len = np.shape(seq2)[1] - np.shape(seq1)[1]
                    seq1 = np.concatenate((seq1, np.zeros((self.n_cluster, diff_len))), axis=1)
                
            similarity_matrix = np.matmul(seq1, seq2.T) / (np.linalg.norm(seq1, axis=1)[:, None] * np.linalg.norm(seq2, axis=1))
    
        # check anomaly of NaN
        if np.any(np.isnan(similarity_matrix)): # there is non-lement
            matching = None
        else:
            # Convert similarity matrix to cost matrix
            cost_matrix = np.max(similarity_matrix) - similarity_matrix
    
            # Apply Hungarian algorithm
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
            # Extract matching pairs
            matching = list(zip(row_indices, col_indices))
    
        return matching

    def rearrange_indices(self, seq_cur, seq_prev):
        # compute matching pair         
        matching = self.match_vectors_Hungarian(seq_prev, seq_cur)
        
        # Get the indices from seq1 that correspond to the matched elements in seq2
        rearranged_indices = [matching[i][1] for i in range(len(matching))]
    
        # Rearrange seq1 according to the rearranged indices
        rearranged_seq1 = [seq_cur[i] for i in rearranged_indices]
    
        return torch.stack(rearranged_seq1, dim=0)
    
    def rearrange_label_indices(self, seq_cur, seq_prev, seq_labels):
        # compute matching pair   

        # matching = self.match_vectors_Hungarian(seq_prev, seq_cur, "Euclidean") # 24.08.07 matching error 
        matching = self.match_vectors_Hungarian(seq_cur, seq_prev, "Euclidean")
        
        # Get the indices from seq1 that correspond to the matched elements in seq2
        rearranged_indices = copy.deepcopy(seq_labels) # if non --> return original indices
        if matching is not None:
            for i in range(len(matching)):
                target_indices = np.equal(seq_labels,i)            
                rearranged_indices[target_indices] = matching[i][1]
        
        return rearranged_indices
    
    def compute_centroid_topK(self, clusters, sequences):
        n_cluster = len(clusters)

        centroids = []
        centroid_indices =[]
        valid_indices = []
        existing_cluster_indices = list(clusters.keys())
        
        for k in range(n_cluster):
            #cur_sequence_set = sequences[clusters[k],:]
            cur_sequence_set = sequences[clusters[existing_cluster_indices[k]],:]

            # input sequence reduction
            if self.flag_reduced:
                input_sequences = self.sequence_reduction(cur_sequence_set)
            else:
                input_sequences = cur_sequence_set

            similarity_matrix_np = self.pairwise_similarity(input_sequences).cpu().numpy() # based on binary similarity     
            #mean_similarity = np.mean(similarity_matrix_np, axis=1)
            #max_idx = np.argmax(mean_similarity)
            sum_similarity = np.sum(similarity_matrix_np, axis=1)
            
            # Sort indices in descending order to get top-10 indices
            if self.args.seq_sampling_type == 1: # argmax
                top_indices = (np.argsort(sum_similarity)[::-1][:self.args.num_centroid_sample]).astype(np.int32)
            elif self.args.seq_sampling_type == 2: # random
                top_indices = np.random.randint(0, len(sum_similarity), size=self.args.num_centroid_sample).astype(np.int32)
                
            # Store the indices of centroids
            centroid_indices.append(top_indices)

            # Append corresponding sequences to centroids list
            centroids.extend(cur_sequence_set[top_indices])

        # Convert centroids and indices to torch tensors
        centroids = torch.stack(centroids, dim=0)       # [ n_cluster x n_sample, n_dim ]
        #centroids = centroids.reshape(self.args.n_cluster, self.args.num_centroid_sample, -1)
        centroid_indices = np.array(centroid_indices)   # [ n_cluster x n_sample ]

        return centroids, centroid_indices

    def predict_label_topK(self, input_sequences, input_centroid):    
        # Compute similarity based on the specified similarity type
        
        #.. size check
        len_seq = input_sequences.size(0)
        #.. concatenation
        concatenated_seq = torch.concat((input_sequences, input_centroid), dim=0 )

        #.. conduct sequence reduciton
        if self.flag_reduced == True:
            reduced_concatenated_seq = self.sequence_reduction(concatenated_seq)
            sequences  = reduced_concatenated_seq[:len_seq,:]
            centroid   = reduced_concatenated_seq[len_seq:,:]
        else:
            sequences = input_sequences
            centroid  = input_centroid

        if self.args.prd_sim_type == 1:  # Euclidean distance
            # Compute Euclidean distance (negated since we want to maximize similarity)
            similarity = -torch.norm(sequences - centroid, dim=1)

        elif self.args.prd_sim_type == 2:  # Considering binary similarity
            # Check if sequences are equal to centroid (element-wise comparison)        
            similarity = []
            for i in range(centroid.size(0)):
                centroid_expanded = centroid[i].unsqueeze(0).expand_as(sequences)
                similarity.append( ( (sequences == centroid_expanded) ).float().sum(-1) )

            # Sum of similarity for nonzero elements
            similarity_th = (torch.stack(similarity, dim=0)).reshape((-1, centroid.size(0))) # [n_sample, n_centroid] =  [n_sample, n_sample*n_cluster]
            similarity_th_reshaped = similarity_th.reshape((-1, self.args.num_centroid_sample, self.args.n_cluster)).permute((0,2,1)) # [n_sample, n_centroid] --> [n_sample, n_sample, n_cluster] --> [n_sample, n_cluster, n_sample]
            similarity_th_sum = similarity_th_reshaped.sum(-1)
            
        elif self.args.prd_sim_type == 4:  # Considering the number of common VQ vectors
            
            # Convert sequence2 to a set (since it's fixed)
            n_clst   = self.args.n_cluster
            n_sample = self.args.num_centroid_sample
            #total_common_elements = []
                
            # Initialize a list to store the count of common elements for each row of sequence1
            common_elements_per_row = []
            set2 = {}
            for k in range(n_clst):                
                sequence2 = centroid[n_sample*k:n_sample*(k+1),:]                
                set2[k] = set(np.unique(sequence2.flatten()))

            # Iterate over each row of sequence1
            for row in sequences:
                # Convert the current row of sequence1 to a set
                set1_row = set(np.unique(row))
                # Find the common elements between the current row of sequence1 and sequence2
                
                num_common = np.zeros(n_clst, dtype=np.int32)
                for k in range(n_clst):                
                    # sequence2 = centroid[n_sample*k:n_sample*(k+1),:]                
                    # set2 = set(sequence2.flatten())
                    common_elements_row = set1_row.intersection(set2[k])
                    num_common[k] = len(common_elements_row)
                    
                # Append the count of common elements for the current row to the list
                common_elements_per_row.append(num_common.tolist())
                
            similarity_th_sum = torch.tensor(common_elements_per_row)
        
        elif self.args.prd_sim_type == 3:  # Considering binary similarity with nonzero elements
            # Create masks for nonzero elements
            non_zero_mask_sequences = (sequences != self.default_node)
            #non_zero_mask_centroid  = (centroid != 0)
        
            # Count nonzero elements in sequences and centroid
            non_zero_count_sequences = non_zero_mask_sequences.float().sum(dim=1)
            #non_zero_count_centroid = non_zero_mask_centroid.float().sum()

            # Compute similarity for nonzero elements
            similarity = []
            for i in range(centroid.size(0)):
                centroid_expanded = centroid[i].unsqueeze(0).expand_as(sequences)
                similarity.append(((sequences == centroid_expanded) & non_zero_mask_sequences ).float().sum(-1)/(non_zero_count_sequences))

                #non_zero_mask_centroid = non_zero_mask_centroid[i].unsqueeze(0).expand_as(sequences)
                #similarity.append(((sequences == centroid_expanded) & non_zero_mask_sequences & non_zero_mask_centroid).float())

            # Sum of similarity for nonzero elements
            similarity_th = (torch.stack(similarity, dim=0)).reshape((-1, centroid.size(0))) # [n_sample, n_centroid] =  [n_sample, n_sample*n_cluster]
            similarity_th_reshaped = similarity_th.reshape((-1, self.args.num_centroid_sample, self.args.n_cluster)).permute((0,2,1)) # [n_sample, n_centroid] --> [n_sample, n_sample, n_cluster] --> [n_sample, n_cluster, n_sample]
            similarity_th_sum = similarity_th_reshaped.sum(-1)
            # Compute final similarity
            #similarity = similarity_sum / (non_zero_count_sequences)

        # Get the index of the centroid with the highest similarity for each sequence
        max_similarity_indices = torch.argmax(similarity_th_sum, dim=1) 
    
        return max_similarity_indices

    def compute_centroid(self, clusters, sequences):
        n_cluster = len(clusters)

        centroids = []
        buf_max_idx = np.zeros(n_cluster, dtype=np.int32)
        
        for k in range(n_cluster):
            cur_sequence_set = sequences[clusters[k],:]

            # input sequence reduction
            if self.flag_reduced:
                input_sequences = self.sequence_reduction(cur_sequence_set)
            else:
                input_sequences = cur_sequence_set

            similarity_matrix_np = self.pairwise_similarity(input_sequences).cpu().numpy() # based on binary similarity     
            #mean_similarity = np.mean(similarity_matrix_np, axis=1)
            #max_idx = np.argmax(mean_similarity)
            sum_similarity = np.sum(similarity_matrix_np, axis=1)
            max_idx = np.argmax(sum_similarity)
            buf_max_idx[k] = int(max_idx)
            centroids.append(cur_sequence_set[max_idx]) # before sequence reduction
            
        #.. make array    
        centroids = torch.stack(centroids, dim=0)    
        
        return centroids, buf_max_idx 
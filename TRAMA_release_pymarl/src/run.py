import datetime
import os
import pprint
import time
import threading
import torch as th
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from modules.vae import REGISTRY as vae_REGISTRY
from modules.RBS_cluster import REGISTRY as rbs_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
import pickle
import numpy as np
import copy

def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(abspath(__file__))), "results", "tb_logs", args.env,
                                     args.env_args['map_name'])
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)

def evaluate_sequential(args, runner):
    
    for iter in range(args.test_nepisode):
        runner.run(test_mode=True, t_episode=iter)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.unit_dim = env_info["unit_dim"]

    # Default/Base scheme    
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "goals": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "True_goal": {"vshape": (1,), "group": "agents", "dtype": th.long}, # from the environment
        "flag_win": {"vshape": (1,), "dtype": th.uint8}, # for monitoring
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, args, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
   
    # Give runner the scheme # need to input VQVAE's embeddings
    
    # Setup VQVAE
    state_dim = buffer.scheme["state"]["vshape"]
    if args.use_vqvae:
        vqvae = vae_REGISTRY[args.vae](state_dim, state_dim, args)
        
        if args.use_cuda:
            vqvae.cuda()
    else:
        vqvae = None
    
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac, vqvae=vqvae)
    args.max_seq_length = buffer.max_seq_length
    # Learner   
    learner = le_REGISTRY[args.learner](mac, vqvae, buffer.scheme, logger, args)
    
    if args.flag_seq_cluster and args.use_vqvae:    
        RBS_cluster = rbs_REGISTRY[args.clustering_method](args, qvectors=vqvae.emb.weight.permute(1,0).cpu().detach().numpy())        
        buffer.init_sequence_buffer(args)
        
    if args.use_cuda:
        learner.cuda()

    if args.similarity_type <= 3:
        if args.classifier_input_type==1:
            f_classifier = vae_REGISTRY[args.classifier](buffer.max_seq_length, args.n_cluster)        
            if args.use_cuda:
                f_classifier.cuda()    
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(f_classifier.parameters(), lr=0.001)
        else:        
            f_classifier = None
    else:
        f_classifier = vae_REGISTRY[args.classifier](args.latent_dim, args.n_cluster)        
        if args.use_cuda:
            f_classifier.cuda()    
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(f_classifier.parameters(), lr=0.001)

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = 0 # timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    episode = 0

    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0
    t_last_cluster_update = 0
    t_last_cluster_update_debug = 0
    n_cluster_update_prev = 0
    n_cluster_update      = 0

    start_time = time.time()
    last_time = start_time
    seq_centroid      = None
    seq_centroid_prev = None
    sim_centroid      = None
    sim_centroid_prev = None
    flag_cluster_begin = False

    last_cluster_update_episode = 0
    
    filename_cluster_results = "cls_results" 
    cls_save_folder = args.config_name + '_' + args.env_args['map_name']           
    cls_save_path = os.path.join(args.local_results_path, "models", cls_save_folder, args.unique_token, filename_cluster_results)
    
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    while runner.t_env <= args.t_max:
        add_save_flag = False
        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False, flag_cluster_begin=flag_cluster_begin)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            idx, buffer_seq_labels, episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
                
            if not flag_cluster_begin:
                buffer_seq_labels = None
                
            if args.flag_seq_cluster: 
                visited_seq, _ = learner.train(episode_sample, runner.t_env, episode, seq_centroid=seq_centroid, RBS_cluster=RBS_cluster, buffer_seq_labels=buffer_seq_labels, f_classifier=f_classifier)
                if args.T_cluster_save <= runner.t_env:                
                    buffer.insert_sequence_batch(idx, visited_seq)
            else:
                learner.train(episode_sample, runner.t_env, episode)
     
        # update clustering------------------------------------------------------
        if (args.flag_seq_cluster) and (args.use_vqvae == True) and (runner.t_env >= args.T_cluster_start):
            if flag_cluster_begin == False or ((runner.t_env -  t_last_cluster_update) / (args.cluster_update_interval) >= 1.0) or \
                (episode - last_cluster_update_episode >= args.cluster_update_episode_itv):
                t_last_cluster_update = runner.t_env
                last_cluster_update_episode = episode
            
                #.. step 1. sample sequence data from replay buffer
                if buffer.can_sample_seq(args.batch_size_clustering):
                    training_batch_size = args.batch_size_clustering
                else:
                    training_batch_size = buffer.num_seq_updated()
                
                if training_batch_size >= args.min_batch_size_clustering:
                    idx_seq, sampled_sequences, sampled_seq_label = buffer.sample_seq(training_batch_size)
            
                    #.. step 2. conduct sequence clustering (Kmeans)                   
                    clusters, reduced_seq, cluster_labels, sim_centroid, meanVq = RBS_cluster.forward( sampled_sequences, prev_centroid=sim_centroid_prev, qvectors=vqvae.emb.weight.permute(1,0).cpu().detach().numpy() )
                    
                    if len(clusters) >= args.n_min_cluster:                              
                        n_cluster_update += 1
                        
                        #.. step 3. matching sequence labels idx
                        if args.flag_centroid_matching and sim_centroid_prev is not None:
                            #.. rearrange cluster_labels to be accordance with the previous labeling
                            if len(sim_centroid_prev) == len(sim_centroid):
                                rearranged_cluster_labels = RBS_cluster.rearrange_label_indices(sim_centroid, sim_centroid_prev, cluster_labels)                                
                                cluster_labels = rearranged_cluster_labels
                
                        sim_centroid_prev = sim_centroid

                        #.. step 4. insert clustering results
                        buffer.insert_sequence_label(idx_seq[:,0], th.tensor(cluster_labels))                        

                        #===================================================
                        # classifier learning
                        #===================================================
                        #.. step 5. initialize classifier                            
                        input_seq  = th.tensor(meanVq).to(args.device)
                        input_size = np.shape(input_seq)[0]
                        
                        #.. step 6. training classifier
                        # Define the loss function and the optimizer
                        # Training the neural network
                        batch_size = args.batch_size         
                        num_epochs = int(input_size/batch_size)
                        ts = time.time()
                            
                        f_classifier.train()
                        iter_train = 0 
                            
                        for epoch in range(num_epochs):
                            permutation = th.randperm(input_size)
    
                            for i in range(0, input_size, batch_size):
                                optimizer.zero_grad()
        
                                indices = permutation[i:i+batch_size]
                                batch_x, batch_y = input_seq[indices].to(args.device), th.tensor(cluster_labels[indices], dtype=th.long).to(args.device)
        
                                if len(batch_y.size()) == 0: # anomaly check
                                    batch_y = batch_y.unsqueeze(0)
                                outputs = f_classifier(batch_x)
                                loss = criterion(outputs, batch_y)                                    
        
                                loss.backward()
                                optimizer.step()
                                    
                                
                            if iter_train % 10 == 0:
                                print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')                                  
                            iter_train += 1
                                
                        te = time.time()

                        print(f"Classifier trianing time: {te-ts:.2f} [sec]")
                        
                        flag_cluster_begin = True                            
                              
                    else:
                        clusters= None
                        
        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                test_batch = runner.run(test_mode=True) # not insert batch data

        if (args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0)) :
            model_save_time = runner.t_env
            #save_path = os.path.join(args.local_results_path, "models", args.unique_token, str(runner.t_env))
            save_folder = args.config_name + '_' + args.env_args['map_name']           
            save_path = os.path.join(args.local_results_path, "models", save_folder, args.unique_token, str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path) # save codebook.th & vae.th    

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

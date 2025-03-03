import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from run import run
import warnings

warnings.filterwarnings("ignore",category=FutureWarning)

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

#ex = Experiment("pymarl")
ex = Experiment("pymarl", save_git_info=False) # modified
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] = '5'
    params = deepcopy(sys.argv)    
    config_str = " --config=trama_gc_qplex" # algorithm    
    #config_str = " --config=lagma_gc_qplex" # algorithm    
    env_str    = " --env-config=sc2_gen_protoss_surComb3"  # environment
    #env_str    = " --env-config=sc2_gen_protoss"  # environment
    #env_str    = " --env-config=sc2"  # environment
    
    params = params + config_str.split() + env_str.split()
    
    warnings.filterwarnings("ignore")
    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f: # src/config/default.yaml
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs

    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    #.. config setting for test-------------
    config_dict['test_nepisode']=2
    config_dict['batch_size']   =5
    # # config_dict['buffer_size']  =100
    
    # # #config_dict['n_codes']      =50000
    # # config_dict['latent_dim']    = 8
    # #config_dict['n_codes']       = 256
    # config_dict['n_cluster']       = 6
    # # #config_dict['latent_dim']   =32
    # # #config_dict['n_max_code']   =10
    
    # # #config_dict['goal_sampling_base']=1
    
    # # # config_dict['vqvae_training_batch']=32
    # # # config_dict['vqvae_training_mini_batch']=32
    # # # config_dict['vqvae_update_type']=2

    # # #config_dict['evaluate'] = True
    # # #config_dict['verbose']  = True

    # # config_dict['flag_seq_cluster']        =   True

    config_dict['buffer_update_time']         =  1000
    config_dict['T_cluster_save']             =  500
    config_dict['T_cluster_start']            =  500
    config_dict['cluster_update_interval']    =  500
    config_dict['batch_size_clustering']      =   50
    config_dict['min_batch_size_clustering']  =   10
    config_dict['learner_log_interval']       =   100
    config_dict['t_degub_log_begin']          =  2000
    config_dict['diff_monitor']               =   0.1
        
    #config_dict['flag_centroid_matching'] = True 
    config_dict['flag_seq_cluster']       = True 
    config_dict['flag_init_centroid']     = True 
    config_dict['use_goal_repr']          = True 
    config_dict['goal_repr_dim']          = 32    
    config_dict['goal_reward']            = False
    
    #config_dict['rnn_hidden_dim'] = 128
    # config_dict['evaluate'] = False
    # config_dict['verbose']  = False
    # config_dict['use_vqvae'] = True
    config_dict['goal_reward'] = True
    config_dict['n_min_cluster'] = 1
    config_dict['flag_true_traj_class'] = False
    #---------------------------------------

    # now add all the config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")

    if config_dict['config_name'] == '':
        cur_config_name = config_dict['agent']
    else:
        cur_config_name = config_dict['config_name']
    if config_dict['env_args']['map_name'] == '':
        save_folder = cur_config_name 
    else:
        save_folder = cur_config_name + '_' + config_dict['env_args']['map_name']

    save_folder   = cur_config_name
    file_obs_path = os.path.join(file_obs_path, save_folder )
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline(params)


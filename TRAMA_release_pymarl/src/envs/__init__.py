from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from smacv2.env import StarCraft2Env as StarCraft2Env_v2, StarCraftCapabilityEnvWrapper
from smacv2.env import StarCraft2Env_multigoal as StarCraft2Env_v2_multigoal, StarCraftCapabilityEnvWrapper_multigoal
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)
REGISTRY["sc2wrapped_multi"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper_multigoal)

#if sys.platform == "linux":
#    os.environ.setdefault("SC2PATH",
#                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

REGISTRY = {}

from .rnn_agent import RNNAgent
from .lagma_agent import LAGMAAgent 
from .lagma_gc_agent import LAGMAAgent_GC 
from .lagma_gp_agent import LAGMAAgent_GP

REGISTRY["rnn"] = RNNAgent
REGISTRY["lagma"] = LAGMAAgent
REGISTRY["lagma_gc"] = LAGMAAgent_GC
REGISTRY["lagma_gp"] = LAGMAAgent_GP
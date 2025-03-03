REGISTRY = {}

from .vqvae import VQVAE 
from .classifier import SimpleNN

REGISTRY["vqvae"] = VQVAE
REGISTRY["classifier"] = SimpleNN
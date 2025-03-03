REGISTRY = {}

from .basic_controller import BasicMAC
from .lagma_gc_controller import LAGMAMAC_GC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["lagma_gc_mac"] = LAGMAMAC_GC

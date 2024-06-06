from .flashface_loader import *
from .flashface_sampler import *

NODE_CLASS_MAPPINGS = {
    "FlashFace_Loader_NuA": FlashFaceLoader,
    "FlashFace_Sampler_NuA": FlashFaceSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FlashFace_Loader_NuA": "FlashFace Loader",
    "FlashFace_Sampler_NuA": "FlashFace Sampler",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

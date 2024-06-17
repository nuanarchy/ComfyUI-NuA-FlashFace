import os

import torch
from .flashfacelib.api import FlashFace
from models import sd_v1_ref_unet
from .flashfacelib.flashface.config import cfg
from .flashfacelib.ldm import data, models
from .flashfacelib.ldm.models.vae import sd_v1_vae
from .flashfacelib.ldm.models.retinaface import retinaface

import folder_paths

class FlashFaceLoader:
    def __init__(self):
        self.device = 'cuda'

    @classmethod
    def INPUT_TYPES(s):
        comfyui_dir = folder_paths.base_path
        flashface_dir = os.path.join(comfyui_dir, "models/flashface")
        return {
            "required": {
                "flashface_model": (sorted(os.listdir(flashface_dir)),),
            },
        }

    RETURN_TYPES = ("FLASHFACE_MODELS",)
    RETURN_NAMES = ()

    FUNCTION = "load_models"

    CATEGORY = "NuA/FlashFace"

    def load_models(self, flashface_model):
        retinaface_model = retinaface(pretrained=True, device=self.device).eval().requires_grad_(False)

        clip_tokenizer = data.CLIPTokenizer(padding='eos')

        clip = getattr(models, cfg.clip_model)(pretrained=True).eval().requires_grad_(False).textual.to(self.device)

        autoencoder = sd_v1_vae(pretrained=True).eval().requires_grad_(False).to(self.device)

        comfyui_dir = folder_paths.base_path
        flashface_dir = os.path.join(comfyui_dir, "models/flashface")
        weight_path = os.path.join(flashface_dir, flashface_model)
        model_weight = torch.load(weight_path, map_location="cpu")
        unet = sd_v1_ref_unet(pretrained=True, version='sd-v1-5_nonema', enable_encoder=False).to(self.device)
        unet.replace_input_conv()
        unet = unet.eval().requires_grad_(False).to(self.device)
        unet.share_cache['num_pairs'] = cfg.num_pairs
        unet.load_state_dict(model_weight, strict=True)
        flashface_models = (unet, clip, clip_tokenizer, autoencoder, retinaface_model)

        return (flashface_models, )
import torchvision.transforms.functional as F
from comfy.utils import ProgressBar
from .flashfacelib.api import FlashFace

class FlashFaceSampler:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "flashface_models": ("FLASHFACE_MODELS", {"forceInput": True}),
                "reference_faces": ("IMAGE", {"forceInput": True}),

                "positive": ("STRING", {"multiline": True, "default":"best quality, masterpiece,ultra-detailed, UHD 4K, photographic"}),
                "negative": ("STRING", {"multiline": True, "default":"blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"}),
                "steps": ("INT", {"default":35}),
                "height": ("INT", {"default": 768, "min": 256, "max": 4096}),
                "width": ("INT", {"default": 768, "min": 256, "max": 4096}),
                "face_bbox_x1": ("FLOAT", {"default":0.4, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y1": ("FLOAT", {"default":0.3, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_x2": ("FLOAT", {"default":0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "face_bbox_y2": ("FLOAT", {"default":0.6, "min": 0.0, "max": 1.0, "step": 0.1}),
                "lamda_feat": ("FLOAT", {"default":1.2, "step": 0.1}),
                "lamda_feat_before_ref_guidence": ("FLOAT", {"default":0.85, "step": 0.1}),
                "face_guidence": ("FLOAT", {"default":3.2, "step": 0.1}),
                "step_to_launch_face_guidence": ("INT", {"default":750}),
                "text_control_scale": ("FLOAT", {"default":7.5, "step": 0.1}),
                "seed": ("INT", {"default":0, "min": 0, "max": 0xffffffff}),
                #"need_face_detect": ("BOOLEAN", {"default":True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ()

    FUNCTION = "generate"

    CATEGORY = "NuA/FlashFace"

    def generate(self, flashface_models, reference_faces, positive, negative, steps, height, width, face_bbox_x1, face_bbox_y1, face_bbox_x2, face_bbox_y2, lamda_feat, lamda_feat_before_ref_guidence, face_guidence, step_to_launch_face_guidence, text_control_scale, seed):
        (unet, clip, clip_tokenizer, vae, retinaface_model) = flashface_models

        pbar = ProgressBar(int(steps))
        p = {"prev": 0}

        def prog(i):
            i = i + 1
            if i < p["prev"]:
                p["prev"] = 0
            pbar.update(i - p["prev"])
            p["prev"] = i

        flashface = FlashFace(unet, clip, clip_tokenizer, vae, retinaface_model, on_progress=prog)

        need_face_detect = True
        img = flashface.generate(
            pos_prompt=positive,
            neg_prompt=negative,
            reference_faces=reference_faces,
            steps=steps,
            height=height,
            width=width,
            face_bbox=[face_bbox_x1, face_bbox_y1, face_bbox_x2, face_bbox_y2],
            lamda_feat=lamda_feat,
            lamda_feat_before_ref_guidence=lamda_feat_before_ref_guidence,
            face_guidence=face_guidence,
            num_sample=1,
            text_control_scale=text_control_scale,
            seed=seed,
            step_to_launch_face_guidence=step_to_launch_face_guidence,
            need_detect=need_face_detect)
        return (img)


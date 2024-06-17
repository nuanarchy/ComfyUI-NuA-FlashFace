import sys
import os
package_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, package_dir)
sys.path.insert(0, os.path.join(package_dir, 'flashface'))

import copy
import random
import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F

from .flashface.config import cfg

from .flashface.ops.context_diffusion import ContextGaussianDiffusion

from .ldm import ops
from .ldm.models.retinaface import crop_face

from .flashface.utils import Compose, PadToSquare, get_padding, seed_everything

class FlashFace():
    def __init__(self, model, clip, clip_tokenizer, vae, retinaface, on_progress=None):
        self.device = 'cuda'

        self.padding_to_square = PadToSquare(224)
        self.face_transforms = Compose([T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.retinaface_transforms = T.Compose([PadToSquare(size=640), T.ToTensor()])
        self.retinaface = retinaface
        self.clip_tokenizer = clip_tokenizer
        self.clip = clip
        self.autoencoder = vae
        self.unet = model
        # diffusion
        sigmas = ops.noise_schedule(schedule=cfg.schedule,
                                    n=cfg.num_timesteps,
                                    beta_min=cfg.scale_min,
                                    beta_max=cfg.scale_max)
        self.diffusion = ContextGaussianDiffusion(sigmas=sigmas, prediction_type=cfg.prediction_type)
        self.diffusion.num_pairs = cfg.num_pairs

        self.progress_hook = on_progress if on_progress else None

    def progress_callback(self, i):
        if self.progress_hook:
            self.progress_hook(i)

    def detect_face(self, imgs=None):
        # read images
        pil_imgs = imgs
        b = len(pil_imgs)
        vis_pil_imgs = copy.deepcopy(pil_imgs)

        # detection
        imgs = torch.stack([self.retinaface_transforms(u) for u in pil_imgs]).to(self.device)
        boxes, kpts = self.retinaface.detect(imgs, min_thr=0.6)

        # undo padding and scaling
        face_imgs = []

        for i in range(b):
            # params
            scale = 640 / max(pil_imgs[i].size)
            left, top, _, _ = get_padding(round(scale * pil_imgs[i].width),
                                          round(scale * pil_imgs[i].height), 640)

            # undo padding
            boxes[i][:, [0, 2]] -= left
            boxes[i][:, [1, 3]] -= top
            kpts[i][:, :, 0] -= left
            kpts[i][:, :, 1] -= top

            # undo scaling
            boxes[i][:, :4] /= scale
            kpts[i][:, :, :2] /= scale

            # crop faces
            crops = crop_face(pil_imgs[i], boxes[i], kpts[i])
            if len(crops) != 1:
                raise ValueError(
                    f'Warning: {len(crops)} faces detected in image {i}')

            face_imgs += crops

            # draw boxes on the pil image
            draw = ImageDraw.Draw(vis_pil_imgs[i])
            for box in boxes[i]:
                box = box[:4].tolist()
                box = [int(x) for x in box]
                draw.rectangle(box, outline='red', width=4)

        face_imgs = face_imgs

        return face_imgs

    def encode_text(self, m, x):
        # embeddings
        x = m.token_embedding(x) + m.pos_embedding

        # transformer
        for block in m.transformer:
            x = block(x)

        # output
        x = m.norm(x)

        return x

    def generate(self,
                 pos_prompt,
                 neg_prompt="",
                 steps=35,
                 solver='ddim',
                 height=768,
                 width=768,
                 face_bbox=[0.4, 0.3, 0.6, 0.6],
                 lamda_feat=1.2,
                 face_guidence=3.2,
                 num_sample=1,
                 text_control_scale=7.5,
                 seed=0,
                 step_to_launch_face_guidence=750,
                 lamda_feat_before_ref_guidence=0.85,
                 reference_faces=None,
                 need_detect=True):
        default_neg_prompt = "blurry, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face"
        if (neg_prompt == ""):
            neg_prompt = default_neg_prompt
        seed_everything(seed)

        imgs = []
        for img in reference_faces:
            img = img.squeeze(0)
            img = img.permute(2, 0, 1)
            img = F.to_pil_image(img)
            imgs.append(img)
        reference_faces = imgs

        if need_detect:
            reference_faces = self.detect_face(reference_faces)

            # for i, ref_img in enumerate(reference_faces):
            #     ref_img.save(f'./{i + 1}.png')
            print(f'detected {len(reference_faces)} faces')
            assert len(
                reference_faces) > 0, 'No face detected in the reference images'

        # process the ref_imgs
        H = height
        W = width

        normalized_bbox = face_bbox
        face_bbox = [
            int(normalized_bbox[0] * W),
            int(normalized_bbox[1] * H),
            int(normalized_bbox[2] * W),
            int(normalized_bbox[3] * H)
        ]
        max_size = max(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1])
        empty_mask = torch.zeros((H, W))

        empty_mask[face_bbox[1]:face_bbox[1] + max_size,
                   face_bbox[0]:face_bbox[0] + max_size] = 1

        empty_mask = empty_mask[::8, ::8].cuda()
        empty_mask = empty_mask[None].repeat(num_sample, 1, 1)

        pasted_ref_faces = []
        for ref_img in reference_faces:
            ref_img = ref_img.convert('RGB')
            ref_img = self.padding_to_square(ref_img)
            to_paste = ref_img

            to_paste = self.face_transforms(to_paste)
            pasted_ref_faces.append(to_paste)

        faces = torch.stack(pasted_ref_faces, dim=0).to(self.device)

        c = self.encode_text(self.clip, self.clip_tokenizer([pos_prompt]).to(self.device))
        #c = pos_prompt[0][0][0].to(self.device)
        c = c[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
        c = {'context': c}

        single_null_context = self.encode_text(self.clip, self.clip_tokenizer([neg_prompt ]).cuda()).to(self.device)
        #single_null_context = neg_prompt[0][0][0].to(self.device)
        null_context = single_null_context
        nc = {
            'context': null_context[None].repeat(num_sample, 1, 1, 1).flatten(0, 1)
        }

        ref_z0 = cfg.ae_scale * torch.cat([
            self.autoencoder.sample(u, deterministic=True)
            for u in faces.split(cfg.ae_batch_size)
        ])
        #  ref_z0 = ref_z0[None].repeat(num_sample, 1,1,1,1).flatten(0,1)
        self.unet.share_cache['num_pairs'] = len(faces)
        self.unet.share_cache['ref'] = ref_z0
        self.unet.share_cache['similarity'] = torch.tensor(lamda_feat).cuda()
        self.unet.share_cache['ori_similarity'] = torch.tensor(lamda_feat).cuda()
        self.unet.share_cache['lamda_feat_before_ref_guidence'] = torch.tensor(
            lamda_feat_before_ref_guidence).cuda()
        self.unet.share_cache['ref_context'] = single_null_context.repeat(
            len(ref_z0), 1, 1)
        self.unet.share_cache['masks'] = empty_mask
        self.unet.share_cache['classifier'] = face_guidence
        self.unet.share_cache['step_to_launch_face_guidence'] = step_to_launch_face_guidence

        self.diffusion.classifier = face_guidence
        # sample
        with amp.autocast(dtype=cfg.flash_dtype), torch.no_grad():
            z0 = self.diffusion.sample(solver=solver,
                                  noise=torch.empty(num_sample, 4,
                                                    H // 8,
                                                    W // 8,
                                                    device=self.device).normal_(),
                                  model=self.unet,
                                  model_kwargs=[c, nc],
                                  steps=steps,
                                  guide_scale=text_control_scale,
                                  guide_rescale=0.5,
                                  show_progress=True,
                                  callback=self.progress_callback,
                                  discretization=cfg.discretization)

        imgs = self.autoencoder.decode(z0 / cfg.ae_scale)
        del self.unet.share_cache['ori_similarity']
        # output
        imgs = (imgs.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(
            0, 255).astype(np.uint8)

        # convert to PIL image
        imgs = [Image.fromarray(img) for img in imgs]
        imgs = imgs

        torch_imgs = []
        for img in imgs:
            img = F.to_tensor(img)
            img = img.permute(1, 2, 0).unsqueeze(0)
            torch_imgs.append(img)
        torch_imgs = torch.cat(torch_imgs, dim=0,)

        return (torch_imgs,)

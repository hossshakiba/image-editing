import json
import numpy as np

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.multimodal.clip_score import CLIPScore

from src.config import ConfigManager


class PromptLoader:
    def __init__(self):
        self.prompt_file = ConfigManager().get("paths")["prompts_file_path"]
        self.prompts = self._load_prompts()

    def _load_prompts(self):
        with open(self.prompt_file, "r") as f:
            return json.load(f)

    def get_prompts_for_image(self, image_name):
        if image_name in self.prompts:
            return self.prompts[image_name]
        else:
            return None

def calculate_fid_score(feature, real_images, edited_images):
    # Check if all images in the lists are tensors
    assert all(isinstance(img, torch.Tensor) for img in real_images), "All real images must be tensors."
    assert all(isinstance(img, torch.Tensor) for img in edited_images), "All edited images must be tensors."

    # Check if all tensors have the dtype `torch.uint8`
    assert all(img.dtype == torch.uint8 for img in real_images), "All real images must have dtype torch.uint8."
    assert all(img.dtype == torch.uint8 for img in edited_images), "All edited images must have dtype torch.uint8."

    fid = FrechetInceptionDistance(feature=feature)
    fid.update(real_images, real=True)
    fid.update(edited_images, real=False)

    return fid.compute().item()

def calculate_clip_score(edited_image, prompt):
    assert edited_image.dtype == torch.uint8, "Image must have dtype torch.uint8."
    
    metric = CLIPScore(model_name_or_path=ConfigManager().get("eval")["clip_model_name"])
    clip_score = metric(edited_image, prompt).detach().item()

    return clip_score

def pil_img_to_tensor(image):
    return torch.tensor(np.array(image), dtype=torch.uint8).permute(2, 0, 1)
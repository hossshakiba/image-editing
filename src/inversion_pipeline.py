import torch
from diffusers import LEditsPPPipelineStableDiffusion

from src.config import ConfigManager


class InversionPipeline:
    def __init__(self):
        self.pipe = LEditsPPPipelineStableDiffusion.from_pretrained(
            ConfigManager().get("model")["name"], torch_dtype=torch.float32
        ).to(ConfigManager().get("env")["device"])
        self.generator = torch.manual_seed(ConfigManager().get("env")["seed"])
        self.num_steps = ConfigManager().get("inversion")["num_steps"]
        self.skip = ConfigManager().get("inversion")["skip"]
        if not ConfigManager().get("inversion")["safety_checker"]:
            self._disable_safety_checker()

    def _disable_safety_checker(self):
        """Disable the NSFW safety checker."""
        def dummy_safety_checker(images, clip_input):
            return images, [False] * len(images)
        
        self.pipe.safety_checker = dummy_safety_checker
    
    def invert_image(self, image):
        return self.pipe.invert(
            image=image,
            generator=self.generator,
            num_inversion_steps=self.num_steps,
            skip=self.skip
        )

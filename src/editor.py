from src.config import ConfigManager


class Editor:
    def __init__(self, pipe):
        self.pipe = pipe
        self.edit_guidance_scale = ConfigManager().get("editor")["edit_guidance_scale"]
        self.edit_threshold = ConfigManager().get("editor")["edit_threshold"]

    def apply_edits(self, latent, prompts):
        edited_images = []
        for prompt in prompts:
            edited_image = self.pipe(
                editing_prompt=[prompt],
                edit_guidance_scale=self.edit_guidance_scale,
                edit_threshold=self.edit_threshold,
                latent=latent
            ).images[0]
            edited_images.append(edited_image)
        return edited_images

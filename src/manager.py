import os
import numpy as np

import torch

from src.editor import Editor
from src.dataset import ImageDataset
from src.inversion_pipeline import InversionPipeline
from src.utils import PromptLoader, calculate_fid_score, calculate_clip_score, pil_img_to_tensor
from src.config import ConfigManager


class PipelineManager:
    def __init__(self):
        self.dataset = None
        self.inversion_pipeline = InversionPipeline()
        self.prompt_loader = PromptLoader()
        self.results_dir_path = ConfigManager().get("paths")["results_dir_path"]
        self.fid_feature_size = ConfigManager().get("eval")["fid_feature_size"]
        self.real_images = []
        self.edited_images = []

    def set_dataset(self, image_dir, urls=None):
        self.dataset = ImageDataset(image_dir, urls)

    def process_images(self):
        images = self.dataset.load_images()
        editor = Editor(self.inversion_pipeline.pipe)

        for image_path in images:
            image_name = os.path.basename(image_path)
            print(f"Processing {image_name}...")
            prompts = self.prompt_loader.get_prompts_for_image(image_name)

            if prompts:
                preprocessed_image = self.dataset.preprocess_image(image_path, (512, 512))
                self.real_images.append(preprocessed_image)
                latent = self.inversion_pipeline.invert_image(preprocessed_image)
                edited_images = editor.apply_edits(latent, prompts)
                self.edited_images += edited_images

                clip_scores = self._calc_clip_scores(edited_images, prompts)
                
                # Rank edited images by CLIP scores
                ranked_results = sorted(
                    zip(edited_images, clip_scores, prompts),
                    key=lambda x: x[1],  # Sort by CLIP score
                    reverse=True,
                )

                # Save Edited Images
                results_base_path = os.path.join(self.results_dir_path, image_name)
                os.makedirs(results_base_path, exist_ok=True)
                for img, prompt in zip(edited_images, prompts):
                    p_underscored = prompt.replace(" ", "_")
                    img.save(os.path.join(results_base_path, f"{p_underscored}.jpg"))

                # Save Scores
                with open(os.path.join(results_base_path, f"{image_name}_scores.txt"), "w") as f:
                    for i, (_, score, prompt) in enumerate(ranked_results):
                        f.write(f"Rank {i+1}: CLIP Score = {score:.2f}, Prompt = {prompt}\n")
                    f.write(f"Average CLIP Score: {np.average(ranked_results)}")

        if len(self.real_images) > 1:
            # FID Score: Comparing the real image distribution to the edited image distribution
            print("-"*10)
            print(f"FID Score: {self._calc_fid_score()}")

    def _calc_fid_score(self):
        fid_score = calculate_fid_score(
            self.fid_feature_size,
            torch.stack([pil_img_to_tensor(img) for img in self.real_images]),
            torch.stack([pil_img_to_tensor(img) for img in self.edited_images])
        )
        return fid_score

    def _calc_clip_scores(self, edited_images, prompts):
        clip_scores = []
        for edited_image, prompt in zip(edited_images, prompts):
            score = calculate_clip_score(pil_img_to_tensor(edited_image), prompt)
            clip_scores.append(score)
        return clip_scores

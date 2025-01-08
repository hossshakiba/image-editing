import os
import requests

from PIL import Image


class ImageDataset:
    def __init__(self, image_dir="data/images", urls=None):
        self.image_dir = image_dir
        if urls:
            self.download_images(urls)

    def download_images(self, urls):
        os.makedirs(self.image_dir, exist_ok=True)
        for i, url in enumerate(urls):
            response = requests.get(url)
            with open(os.path.join(self.image_dir, f"image_{i+1}.jpg"), 'wb') as f:
                f.write(response.content)

    def load_images(self):
        return [os.path.join(self.image_dir, img) for img in sorted(os.listdir(self.image_dir)) if img.endswith((".png", ".jpg", "jpeg"))]

    def preprocess_image(self, image_path, img_size=(256, 256)):
        image = Image.open(image_path).convert("RGB")
        image = image.resize(img_size)
        return image




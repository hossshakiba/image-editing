import argparse

from src.manager import PipelineManager
from src.config import ConfigManager


def parse_args():
    parser = argparse.ArgumentParser(description="LEDITS++ Playground")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = ConfigManager(args.config)
    manager = PipelineManager()
    manager.set_dataset(cfg.get("paths")["real_images_dir_path"])
    manager.process_images()

if __name__ == "__main__":
    main()


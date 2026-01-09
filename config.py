# config.py

import torch
import random
import numpy as np

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)

    # Image generation
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9

    # Prompt generation
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

    # Fine-tuning
    batch_size = 2
    learning_rate = 5e-6
    num_epochs = 3

    # Paths
    dataset_path = "./data/text_image_pairs"
    output_dir = "./output"
    logs_dir = "./logs"

    @staticmethod
    def set_seed(seed_value=42):
        random.seed(seed_value)
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_value)

    @staticmethod
    def print_config():
        print("[INFO] Using configuration:")
        print(f"Device: {CFG.device}")
        print(f"Seed: {CFG.seed}")
        print(f"Image Size: {CFG.image_gen_size}")
        print(f"Batch Size: {CFG.batch_size}")
        print(f"Learning Rate: {CFG.learning_rate}")
        print(f"Epochs: {CFG.num_epochs}")

CFG.set_seed(CFG.seed)
CFG.print_config()

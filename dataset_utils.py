# dataset_utils.py

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import CFG

# ----------------------- Custom Dataset -----------------------

class CustomImageTextDataset(Dataset):
    """
    Loads (text, image) pairs for training or fine-tuning.
    """
    def __init__(self, text_image_pairs, transform=None):
        self.text_image_pairs = text_image_pairs
        self.transform = transform

    def __len__(self):
        return len(self.text_image_pairs)

    def __getitem__(self, idx):
        text, image_path = self.text_image_pairs[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"text": text, "image": image}

# ----------------------- Transformation -----------------------

def get_image_transform():
    transform = transforms.Compose([
        transforms.Resize(CFG.image_gen_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return transform

# ----------------------- Data Loader -----------------------

def create_dataloader(text_image_pairs, batch_size=CFG.batch_size, shuffle=True):
    transform = get_image_transform()
    dataset = CustomImageTextDataset(text_image_pairs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader

# ----------------------- Example Data Preparation -----------------------

def load_dummy_data():
    text_image_pairs = [
        ("A cat sleeping on a couch", "./data/images/cat1.jpg"),
        ("A dog running in a park", "./data/images/dog1.jpg"),
        ("A mountain with snow", "./data/images/mountain1.jpg")
    ]
    return text_image_pairs

# ----------------------- Repeated Variant: Extended Dummy Data -----------------------

def load_extended_dummy_data():
    text_image_pairs = [
        ("A sunset at the beach", "./data/images/sunset1.jpg"),
        ("A forest with tall trees", "./data/images/forest1.jpg"),
        ("A city skyline at night", "./data/images/city1.jpg"),
        ("A cute puppy playing", "./data/images/puppy1.jpg"),
        ("A delicious pizza", "./data/images/pizza1.jpg"),
    ]
    return text_image_pairs

# ----------------------- Combined Loader -----------------------

def get_default_dataloader():
    """
    Combines basic and extended dummy data for a bigger dataset.
    """
    pairs = load_dummy_data() + load_extended_dummy_data()
    dataloader = create_dataloader(pairs)
    print(f"[INFO] Loaded {len(pairs)} text-image pairs.")
    return dataloader

# ----------------------- Repeated Variant: Another Combined Loader -----------------------

def get_demo_dataloader():
    """
    Another repeated dataloader for demonstration.
    """
    pairs = load_extended_dummy_data()
    dataloader = create_dataloader(pairs, batch_size=4)
    print(f"[INFO] Demo dataloader with {len(pairs)} items created.")
    return dataloader

# ----------------------- Test Block -----------------------

if __name__ == "__main__":
    print("[INFO] Testing dataset_utils.py...\n")

    pairs = load_dummy_data()
    loader = create_dataloader(pairs)
    print(f"[INFO] Loaded dataloader with {len(loader)} batches.\n")

    extended_loader = get_default_dataloader()
    demo_loader = get_demo_dataloader()

    print("[INFO] dataset_utils.py test run complete.")

import os
import numpy as np
import cv2
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from cldm.model import create_model, load_state_dict

class cocoDataset(Dataset):
    def __init__(self, img_dir, size=256):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, img_name) for img_name in os.listdir(img_dir) if img_name.endswith(".jpg")]
        
        self.size = size
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.size, self.size))
        image = (image.astype(np.float32) / 127.5) - 1.0

        img_name = os.path.basename(img_path)
            
        return image, img_name

if __name__ == "__main__":
    device = "cuda"
    split = 'train2017' # and 'val2017'


    img_size = 256
    batch_size = 16

    img_dir = f"datasets/coco/images/{split}"
    output_dir = f"datasets/coco/{img_size}_latents/{split}"
    os.makedirs(output_dir, exist_ok=True)

    configs = 'configs/controlnet/coco_256.yaml'
    weight = 'models/sd15_ini.ckpt'

    model = create_model(configs)
    model.load_state_dict(load_state_dict(weight, location='cpu'))

    vae = model.first_stage_model
    del model
    vae = vae.to(device)

    # Create dataset and dataloader
    dataset = cocoDataset(img_dir, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Process images using the dataloader
    for batch_images, batch_img_names in tqdm(dataloader):
        batch_images = batch_images.to(device).permute(0, 3, 1, 2)
        
        with torch.no_grad():
            z = vae.encode(batch_images).sample() * 0.18215

        z = z.cpu().numpy()
        for j in range(len(batch_img_names)):
            file_name = batch_img_names[j].split(".")[0]
            np.save(f"{output_dir}/{file_name}.npy", z[j])

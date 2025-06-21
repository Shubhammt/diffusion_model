import os
import multiprocessing
from PIL import Image
import json
from config import *
import numpy as np
import torch

class diffusion_dataset():
    def __init__(self, path):
        self.files = []
        self.path = path
        
        pool = multiprocessing.Pool(processes=8)
        print("Loading Data ...")
        folders = os.listdir(path)[:2]
        outputs = pool.map(self.get_files, folders)
        for Files in outputs:
            self.files.extend(Files)

    def get_files(self, folder):
        Files = os.listdir(os.path.join(self.path, folder))
        Files = [[folder, x[:-5]] for x in Files if x.endswith('.json')]
        return Files

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        json_path = os.path.join(self.path, self.files[idx][0], self.files[idx][1]+'.json')
        
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        conditional_prompt = data['prompt']

        image_path = os.path.join(self.path, self.files[idx][0], self.files[idx][1]+'.jpg')
        input_image = Image.open(image_path)
        input_image = input_image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        input_image_array = (np.array(input_image) - np.array([200,200,200]))/256
        input_image_tensor = torch.tensor(input_image_array, dtype=torch.float32)
        input_image_tensor = input_image_tensor.permute(2, 0, 1)
        return conditional_prompt, input_image_tensor
if __name__ == "__main__":
    path = r"E:\text-to-image-2M\data_512_2M"
    dataset = diffusion_dataset(path)
    print(len(dataset))
    red_mean = 0
    green_mean = 0
    blue_mean = 0
    from tqdm import tqdm
    for i in tqdm(range(1000)):
        red_mean += dataset[3562][1][0].sum()/(512*512)
        green_mean += dataset[3562][1][1].sum()/(512*512)
        blue_mean += dataset[3562][1][2].sum()/(512*512)
    print(red_mean/1000, green_mean/1000, blue_mean/1000)

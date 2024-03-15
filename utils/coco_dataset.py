import cv2
import numpy as np
import os
import random

from torch.utils.data import Dataset
from transformers import AutoImageProcessor



class cocoConditionaltBase(Dataset):
    def __init__(self,
                 txt_file,
                 source_img_path,
                 target_img_path,
                 size=256,
                 flip_p=0.5
                 ):

        self.data_paths = txt_file

        with open(self.data_paths, 'rt') as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.source_img_path = source_img_path
        self.target_img_path = target_img_path

        self.labels = {
            "source_file_path_": [os.path.join(self.source_img_path, l.split(';')[0])
                           for l in self.image_paths],
            "target_file_path_": [os.path.join(self.target_img_path, l.split(';')[0])
                           for l in self.image_paths],
            "prompt": [l.split(';')[1] for l in self.image_paths],
        }
        
        self.size = size
        self.flip = flip_p
        self.image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        example = dict((k, self.labels[k][idx]) for k in self.labels)

        source = cv2.imread(example["source_file_path_"])
        target_path = example["target_file_path_"]
        target_path = target_path.replace('.png', '.jpg')
        target = cv2.imread(target_path)

        prompt = example["prompt"]

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        
        source = cv2.resize(source, (self.size, self.size))
        target = cv2.resize(target, (self.size, self.size))

        if random.random()<self.flip:
            source = cv2.flip(source, 1)
            target = cv2.flip(target, 1)


        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        target = (target.astype(np.float32) / 127.5) - 1.0


        return dict(jpg=target, txt=prompt, hint=source)

class cocoConditionalTrain(cocoConditionaltBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="datasets/coco/annotations/coco_prompts_train2017.txt", source_img_path="datasets/coco/images/80_colors/train2017", target_img_path="datasets/coco/images/train2017", **kwargs)


class cocoConditionalValidation(cocoConditionaltBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="datasets/coco/annotations/coco_prompts_val2017.txt", source_img_path="datasets/coco/images/80_colors/val2017", target_img_path="datasets/coco/images/val2017", **kwargs)
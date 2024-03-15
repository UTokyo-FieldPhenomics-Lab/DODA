#This file generates images according to the labels of Terraref domain
#Using the data generated from this file to train the detectors does not give the best results
#Please use ‘generate_data_for_target_domain.py’ to generate data to train the detector

import os


import cv2
import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from utils.utils import makedir
from transformers import AutoImageProcessor


#Reference images path of target domain
ref_img_path = 'gwhd_2021/Terraref_x9/'

output_path = 'output/wheat/Terraref-testing/'
weight = "models/DODA-wheat-Terraref.ckpt"

seed = 21
batch_size = 8
image_resolution = 512
configs = 'configs/controlnet/DODA_wheat_cldm_kl_4.yaml'

seed_everything(seed)
makedir(output_path)



def process(control, reference_image, ddim_steps=50, guess_mode=False, strength=1.5, scale=1, eta=0.0):
    with torch.no_grad():
        
        B, H, W, C = control.shape

        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        reference = torch.from_numpy(reference_image.copy()).float().cuda()


        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(reference)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning(torch.zeros((B, 3, 224, 224)).cuda())]}
        shape = (3, H // 4, W // 4)


        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, B,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(B)]
    return results


source_Path = ref_img_path + 'source/'
target_Path = ref_img_path + 'target/'
output_img_path = output_path + 'img/'
output_ctr_path = output_path + 'ctr/'


model = create_model(configs).cpu()
model.load_state_dict(load_state_dict(weight, location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")

makedir(output_img_path)
makedir(output_ctr_path)



imgnames_list = os.listdir(source_Path)

# Get the labels used to draw the visualization picture
label_dic = {}
with open(ref_img_path + 'label_x9.txt', 'r') as f:
    label_lines = f.readlines()
    for label_line in label_lines:
        label_line = label_line.rstrip()
        label_line = label_line.split(' ')
        img_name = label_line[0].split('/')[-1]
        if len(label_line) > 1:
            label_dic[img_name] = label_line[1:]
        else:
            label_dic[img_name] = 'no_box'

output_img_path_img = os.listdir(output_img_path)
        
# read image by batch
for i in range(0, len(imgnames_list), batch_size):
    batch_imgnames = imgnames_list[i:i+batch_size]
    if batch_imgnames[-1] in output_img_path_img:
        continue

    reference_images = [image_processor(images=cv2.cvtColor(cv2.imread(target_Path + imgname), cv2.COLOR_BGR2RGB))['pixel_values'][0] for imgname in batch_imgnames]
    
    control_images = [cv2.resize(cv2.imread(source_Path + imgname), (image_resolution, image_resolution)) for imgname in batch_imgnames]
    label_lst = [label_dic[imgname] for imgname in batch_imgnames]

    # Stack images in batch dimension
    reference_images = np.stack(reference_images, axis=0)
    control_images = np.stack(control_images, axis=0)

    out_imgs = process(control_images, reference_images)
    for n, out_img in enumerate(out_imgs):
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_img_path + batch_imgnames[n], out_img)
        bboxes = label_lst[n]
        if bboxes != 'no_box':
            for box in bboxes:
                box = box.split(',')
                ctr_image = cv2.rectangle(out_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (208, 146,0), 2)
        else:
            ctr_image = out_img
        cv2.imwrite(output_ctr_path + batch_imgnames[n], ctr_image)
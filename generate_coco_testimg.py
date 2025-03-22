import cv2
import os

import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from DODA.sampler import eulerSampler


#Path of COCO
dataDir = 'datasets/coco'

output_path = 'output/coco/'
weight = 'models/DODA-L2I-coco-512.ckpt'

strength = 1.5
cfg_scale = 4.5

seed = 21
batch_size = 4
image_resolution = 512
sample_step = 100

configs = 'configs/controlnet/coco_512.yaml'

layout_img_path = f'{dataDir}/images/80_colors/val2017/'
prompt_path = f'{dataDir}/annotations/coco_prompts_val2017.txt'

seed_everything(seed)
os.makedirs(output_path, exist_ok=True)



def process(control, prompt_lst, scale=4.5):
    with torch.no_grad():
        
        B, H, W, C = control.shape

        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        prompt_lst = [model.get_learned_conditioning(prompt) for prompt in prompt_lst]
        prompt_lst = torch.cat(prompt_lst, dim=0)

        cond = {"c_concat": [control], "c_crossattn": [prompt_lst]}
        un_cond = {"c_concat": [torch.zeros_like(control).cuda()], "c_crossattn": [model.get_learned_conditioning([''] * B)]}

        shape = (B, 3, H // 8, W // 8)

        samples = sampler.sample_euler(shape=shape, c=cond, 
                                 unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)

        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(B)]
    return results



model = create_model(configs).cpu()
model.load_state_dict(load_state_dict(weight, location='cuda'))

if strength==0:
    model.control_scales = None
    model.control_model = None
else:
    model.control_scales = ([strength] * 13)
model = model.cuda()
sampler = eulerSampler(model, sample_step)


# read prompts
prompt_dic = {}
with open(prompt_path, 'r') as f:
    prompt_lines = f.readlines()
    for prompt_line in prompt_lines:
        prompt_line = prompt_line.rstrip()
        prompt_line = prompt_line.split(';')
        img_name = prompt_line[0]
        prompt = prompt_line[1]
        prompt_dic[img_name] = prompt

imgnames_list = list(prompt_dic.keys())

        
# read image by batch
for i in range(0, len(imgnames_list), batch_size):
    batch_imgnames = imgnames_list[i:i+batch_size]

    control_images = [cv2.cvtColor(cv2.resize(cv2.imread(layout_img_path + imgname), (image_resolution, image_resolution)), cv2.COLOR_BGR2RGB) for imgname in batch_imgnames]
    prompt_lst = [prompt_dic[imgname] for imgname in batch_imgnames]

    # Stack images in batch dimension
    control_images = np.stack(control_images, axis=0)

    out_imgs = process(control_images, prompt_lst, scale=cfg_scale)
    for n, out_img in enumerate(out_imgs):
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path + batch_imgnames[n], out_img)

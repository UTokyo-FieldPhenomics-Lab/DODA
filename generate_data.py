import os
import cv2
import random
import einops
import numpy as np
import torch

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from DODA.sampler_layout import eulerSampler
from transformers import CLIPImageProcessor


cfg_scale = 1.5
strength = 1

target_domain = 'ARC_1'
n_img_per_domain = 200



configs = 'configs/DODA/DODA_wheat_ldm_kl_4_layout_clip.yaml'
weight = "models/DODA-wheat-resnet.ckpt"


seed = 21
batch_size = 16
img_resolution = 256
sample_step = 50
start_no_img = 0   # from which layout image to start


scale = img_resolution/512

output_path = f'output/{target_domain}/cfg{cfg_scale}_strength{strength}/'
os.makedirs(output_path, exist_ok=True)

data_path = 'datasets/wheat/'

ref_img_path = data_path + 'target/'
layout_path = f'{data_path}/random_layout/'
reference_infor_path = f'{data_path}/reference_infor/'
labels_of_generated_img = open(f'{output_path}/bounding_boxes.txt', 'w')



# Initialize model
seed_everything(seed)

model = create_model(configs).cpu()
model.load_state_dict(load_state_dict(weight, location='cuda'))
model = model.cuda()
sampler = eulerSampler(model, sample_step)
img_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
model.control_scales = [strength] * 13



def process(control, reference_img, scale=1):
    with torch.no_grad():
        
        B, H, W, C = control.shape
        shape = (B, 3, H // 4, W // 4)

        control = torch.from_numpy(control.copy()).float().cuda() / 255.0
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        reference = torch.from_numpy(reference_img.copy()).float().cuda()

        #layout = model.layout_encoder(control)
        layout =control
        c_crossattn = [model.get_learned_conditioning(reference)]
        unconditional_layout = torch.zeros_like(layout)


        c_crossattn = [model.get_learned_conditioning(reference)]
        cond = {"c_crossattn": c_crossattn}
        un_cond = {"c_crossattn": c_crossattn}

        samples = sampler.sample_euler(shape=shape, c=cond, layout=layout, unconditional_layout=unconditional_layout,
                                 unconditional_guidance_scale=scale, unconditional_conditioning=un_cond)


        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(B)]
    return results



for domain in os.listdir(reference_infor_path):
    if domain != target_domain:
        continue
    layout_img_path = layout_path + domain + '/img/'
    label_path = layout_path + domain + '/bounding_boxes.txt'

    names_img_with_wheat = reference_infor_path + domain + "/with_wheat.txt"
    names_img_wo_wheat = reference_infor_path + domain + "/wo_wheat.txt"
    
    output_img_path = output_path + '/img/'
    output_ctr_path = output_path + '/ctr/'
    
    os.makedirs(output_img_path, exist_ok=True)
    os.makedirs(output_ctr_path, exist_ok=True)


    #Get reference img names that there is/isn't object in the img
    with open(names_img_with_wheat) as f:
        imgs_with_wheat = f.readlines()
    with open(names_img_wo_wheat) as f:
        imgs_wo_wheat = f.readlines()

    # Get the labels of layout images
    label_dic = {}
    imgnames_list = []
    with open(label_path) as f:
        label_lines = f.readlines()
        try: 
            label_lines = label_lines[start_no_img:n_img_per_domain]
        except:
            print('The number of images to be generated is greater than the number of layout images')
            print('please use "random_generate_layout_images.py" to generate more layout images')
        for label_line in label_lines:
            label_line = label_line.rstrip()
            label_line = label_line.split(' ')
            img_name = label_line[0].split('/')[-1]
            if len(label_line) > 1:
                label_dic[img_name] = label_line[1:]
            else:
                label_dic[img_name] = 'no_box'
            imgnames_list.append(img_name)

    img_id = 0
    # read img by batch
    for i in range(0, len(imgnames_list), batch_size):
        batch_imgnames = imgnames_list[i:i+batch_size]
        label_lst = [label_dic[imgname] for imgname in batch_imgnames]

        #read and process reference imgs
        ref_imgs = []
        ref_img_names = []
        for bboxes in label_lst:
            if bboxes != 'no_box':
                ref_img_name = random.choice(imgs_with_wheat)
            else:
                ref_img_name = random.choice(imgs_wo_wheat)
            ref_img_name = ref_img_name.strip()
            ref_img = img_processor(cv2.cvtColor(cv2.imread(ref_img_path + ref_img_name), cv2.COLOR_BGR2RGB))['pixel_values'][0]

            ref_img_names.append(ref_img_name)
            ref_imgs.append(ref_img)

        
        control_imgs = [cv2.resize(cv2.imread(layout_img_path + imgname), (img_resolution, img_resolution)) for imgname in batch_imgnames]
        

        # Stack images in batch dimension
        ref_imgs = np.stack(ref_imgs, axis=0)
        control_imgs = np.stack(control_imgs, axis=0)

        out_imgs = process(control_imgs, ref_imgs, cfg_scale)
        for n, out_img in enumerate(out_imgs):
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            generated_img_path = output_img_path + f'{domain}-{img_id}.png'
            generated_img_path = os.path.abspath(generated_img_path).replace('\\', '/')
            labels_of_generated_img.write(f'{generated_img_path}')
            cv2.imwrite(generated_img_path, out_img)

            bboxes = label_lst[n]
            
            if bboxes != 'no_box':
                for box in bboxes:
                    box = np.array(list(map(float,box.split(',')))) * scale
                    box = box.astype(np.int32)
                    ctr_img = cv2.rectangle(out_img, (box[0], box[1]), (box[2], box[3]), (208, 146,0), 2)
                    labels_of_generated_img.write(f" {box[0]},{box[1]},{box[2]},{box[3]},0")
            else:
                ctr_img = out_img
            labels_of_generated_img.write('\n')
            
            cv2.imwrite(generated_img_path.replace('.png', '.jpg').replace('/img/', '/ctr/'), ctr_img)
            img_id += 1
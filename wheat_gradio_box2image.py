from share import *

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from transformers import AutoImageProcessor


model = create_model('configs/controlnet/DODA_wheat_cldm_kl_4.yaml').cpu()
model.load_state_dict(load_state_dict("models/DODA-wheat-cldm.ckpt", location='cuda'))


model = model.cuda()
ddim_sampler = DDIMSampler(model)
image_processor = AutoImageProcessor.from_pretrained("facebook/vit-mae-base")


def process(control_image, reference_image, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta):
    with torch.no_grad():
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)
        
        control = cv2.resize(control_image, (image_resolution, image_resolution))
        H, W, C = control.shape

        control = torch.from_numpy(control.copy()).float().cuda() / 255.0 *1.1
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()
        
        h, w, c = reference_image.shape
        if h > 256 and w > 256:
            y_ref_start = np.random.randint(0, h - 256 + 1)
            x_ref_start = np.random.randint(0, w - 256 + 1)
            reference = reference_image[y_ref_start:y_ref_start + 256, x_ref_start:x_ref_start + 256]
        reference = image_processor(images=reference)['pixel_values'][0]
        reference = torch.from_numpy(reference.copy()).float().cuda().unsqueeze(0)


        cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning(reference)]}
        un_cond = {"c_concat": None if guess_mode else [control]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning(torch.zeros((1, 3, 224, 224)).cuda())]}
        shape = (3, H // 4, W // 4)

        model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)


        x_samples = model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
        out = results[0]
        cv2.imwrite('output/version_0/' + 'test' + '.png', out)
    return results


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Control Latent Diffusion with Bounding Box")
    with gr.Row():
        with gr.Column():
            control_image = gr.Image(source='upload', type="numpy")
            reference_image = gr.Image(source='upload', type="numpy")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                image_resolution = gr.Slider(label="Image Resolution", minimum=256, maximum=1024, value=512, step=64)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.5, step=0.01)
                guess_mode = gr.Checkbox(label='Guess Mode', value=False)
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=45, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=882002212, randomize=False)
                eta = gr.Number(label="eta (DDIM)", value=0.0)
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    ips = [control_image, reference_image, num_samples, image_resolution, ddim_steps, guess_mode, strength, scale, seed, eta]
    run_button.click(fn=process, inputs=ips, outputs=[result_gallery])


block.launch(server_name='0.0.0.0', share=True)

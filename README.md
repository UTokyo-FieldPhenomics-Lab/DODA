# DODA: Adapting Object Detectors to Dynamic Agricultural Environments in Real-Time with Diffusion
[Paper](http://arxiv.org/abs/2403.18334)

DODA is a data synthesizer that can generate high-quality object detection data for new domains in agriculture, and help the detectors adapt to the new domains.

![overview of DODA](figures/DODA_overview)

## Pretrained Models
| Model | Dataset | Resolution | Training Iters | Downlad Link |
|:-:|:-:|:-:|:-:|:-:|
|DODA-L2I|COCO|512x512|30K|[Google drive](https://drive.google.com/file/d/1Xm2gOA5QdtYyGQe6Lik-wXlyJTxFTc-F/view?usp=sharing)|
|DODA-L2I|COCO|256x256|100K|[Google drive](https://drive.google.com/file/d/1l4bJfBRqa0gyLgqpj6Fw1jHsXenEIz15/view?usp=sharing)|
|VAE|GWHD2021|256x256|170K|[Google drive](https://drive.google.com/file/d/1XHmtZR95uSbFcY-y6wCffgV5uUM1x8pC/view?usp=sharing)|
|ldm|GWHD 2+|256x256|150k|[Google drive](https://drive.google.com/file/d/1gD_crs4H4PjbKFiQvoERxX3m1XAVZ1Cb/view?usp=sharing)|
|DODA|GWHD2021|256x256|80K|[Google drive](https://drive.google.com/file/d/1rkSxNTL0mnZiPLmNxnIMf1KRgobF4_dW/view?usp=sharing)|


## Evaluation

### Setup Environment
```
conda create -y -n DODA python=3.8.5
conda activate DODA
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

### Download Datesets
```
bash Download_dataset.sh # download the dataset you need
```

### Prepare Datesets
```
python prepare_coco.py
python prepare_wheat_trainset.py && python prepare_wheat_testset.py  # If you only want to generate data for your domain, and don't need to train the model, please skip this line
```

### Generate images for COCO L2I evaluation
Generate images according to the bounding boxes of the COCO 2017 validation set:
First download the pretrained DODA-L2I to `/models` folder, then run:
```
python generate_coco_testimg.py
```

### Generate images for target domain
#### Generate images for domains in GWHD testset
Modify the `target_domain`, `cfg_scale`, `strength` in `generate_data.py`, then run:
```
python generate_data.py
```
#### Generate images for your custom domain
1. Modify the parameters in `generate_custom_layout.py` based on the shape and number of wheat heads in your data to make the generated layouts match your dataset. And run `generate_custom_layout.py`
2. Modify the `ref_img_path`, `cfg_scale`, `strength` in `generate_custom_data.py`, then run `python generate_custom_data.py`

## Generate images in GUI
You can try our method to generate images for wheat through the GUI: 
```
python wheat_gradio_box2image.py
```

Please upload <u>**BOTH**</u> the <u>**reference image**</u> and <u>**layout image**</u> image respectively as shown:

![web_example](figures/web_example.png)

> PS: The demo <u>**reference image**</u> and <u>**layout image**</u> can be found in the `/figures` folder. More images can be found in `/dataset` folder after run `prepare_wheat_trainset.py`

Or you can simply draw it yourself through drawing software. Each item should have a distinguishable color (with maximized values of the R, G, B channels), for example, `(0, 0, 255)`, `(255, 0, 255)`, etc. Below are some examples of possible layout images:

![layout_example](figures/layout_example.png)

## Train your own DODA
DODA training is divided into three parts, from first to last: VAE, ldm and L2I. This repository reads the data set through a txt file, so first, please write the file names of all the images in your own dataset into a txt file.
### Training of VAE
Modify the `config` in `train.py` :
```
config = 'configs/autoencoder/wheat_autoencoder_kl_64x64x3.yaml'
```
Modify the `txt_file` and `data_root` in the config file to the path of the filenames txt file and the path to your own dataset.
then train the VAE by running:
```
python train.py
```
VAE is very robust, so we recommend skipping VAE training and using the pre-trained weight `kl-f4-wheat.ckpt` we provide.

### Training of ldm
Modify the `config` in `train.py` :
```
config = 'configs/latent-diffusion/wheat_ldm_kl_4.yaml'
```
Modify the `ckpt_path` in the config file `wheat_ldm_kl_4.yaml` to the weight path of your VAE or the VAE provided by us.
Modify the `txt_file` and `data_root` in the config file to the path of the filenames txt file and the path to your own dataset.
then train the ldm by running:
```
python train.py
```

### Training of L2I
Modify the `config` in `train.py` :
```
config = 'configs/controlnet/DODA_wheat_ldm_kl_4_layout_clip.yaml'
```
Modify the `resume_path` in `train.py` to the weight path of your ldm or the ldm provided by us.
Modify the `txt_file` and `data_root` in the config file to the path of the filenames txt file and the path to your own dataset.
then train the cldm by running:
```
python train.py
```

### Hyperparameters for training
![Hyperparameters](figures/Hyperparameters.png)

### Training tips
Diffusion model is data hungry, and using more data always gives better results, so we strongly recommend mixing your data with GWHD for training. Mixing data can be achieved by putting all the images in your own dataset and the GWHD into one folder and writing the filenames of all images to one txt file.

When fine-tuning a detector with generated data, use mosaic data augmentation and train for one epoch with a fixed learning rate.
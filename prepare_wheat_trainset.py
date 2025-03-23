import cv2
import os
import random
from tqdm import tqdm

from utils.utils import read_label, get_layout_image

random.seed(23)

#Path of Global Wheat Head Detection Dataset
gwhd_2021_path = 'datasets/gwhd_2021/'
#Path to save the prepared images for training DODA
save_path = 'datasets/wheat/'



gwhd_2021_img_path = gwhd_2021_path + 'images/'
csvFile = gwhd_2021_path + 'competition_train.csv'

target_path = save_path + 'target/'
os.makedirs(target_path, exist_ok=True)
source_path = save_path + 'source/'
os.makedirs(source_path, exist_ok=True)

ldm_train_ids = open(save_path + 'train_ldm.txt', 'w', encoding='utf-8')
ldm_val_ids = open(save_path + 'val_ldm.txt', 'w', encoding='utf-8')
cldm_train_ids = open(save_path + 'train_cldm.txt', 'w', encoding='utf-8')
cldm_val_ids = open(save_path + 'val_cldm.txt', 'w', encoding='utf-8')

#read labels of trainset 
label_dic = read_label(csvFile)



for imgName in tqdm(os.listdir(gwhd_2021_img_path)):
    if '.png' not in imgName:
         continue

    is_train = random.random()<0.99
    in_official_trainset = imgName in label_dic.keys()


    img = cv2.imread(gwhd_2021_img_path + imgName)
    if in_official_trainset:
        laBel = label_dic[imgName][0]
        laBel = laBel.replace(" ", ",").split(';')
        source_img, n_lay = get_layout_image(img, laBel)

    imgName = imgName[:-4]

    #cut image into 512*512, with step=256(x9)
    i = 0
    for y_i in range(3):
        for x_i in range(3):
            x_start = x_i*256
            y_start = y_i*256
            img_cut = img[y_start:y_start+512, x_start:x_start+512,:]

            cv2.imwrite(target_path + imgName + '_' + str(i) + '.png', img_cut)
            if is_train:
                ldm_train_ids.write('%s.png'%(imgName + '_' + str(i)) + '\n')
            else:
                ldm_val_ids.write('%s.png'%(imgName + '_' + str(i)) + '\n')

            if in_official_trainset and n_lay<=3:   #Filter images with bbox overlap of more than 3 layers
                source_img_cut = source_img[y_start:y_start+512, x_start:x_start+512,:]
                cv2.imwrite(source_path + imgName + '_' + str(i) + '.png', source_img_cut)
                if is_train:
                    cldm_train_ids.write('%s.png'%(imgName + '_' + str(i)) + '\n')
                else:
                    cldm_val_ids.write('%s.png'%(imgName + '_' + str(i)) + '\n')
            i += 1
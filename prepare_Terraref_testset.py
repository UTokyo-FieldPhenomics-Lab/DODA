import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm

from utils.utils import read_label, makedir, get_layout_image

#Path of Global Wheat Head Detection Dataset
gwhd_2021_path = 'gwhd_2021/'

#Path to save the images of Terraref domain
Terraref_path = 'gwhd_2021/Terraref/target/'
makedir(Terraref_path)
Terraref_ori_source_path = 'gwhd_2021/Terraref/source/'
makedir(Terraref_ori_source_path)
Terraref_cut_Path = 'gwhd_2021/Terraref_x9/'
makedir(Terraref_cut_Path)
Terraref_source_Path = Terraref_cut_Path + 'source/'
makedir(Terraref_source_Path)
Terraref_target_Path = Terraref_cut_Path + 'target/'
makedir(Terraref_target_Path)


gwhd_2021_img_path = gwhd_2021_path + 'images/'
csvFiles = gwhd_2021_path + 'competition_test.csv'
Terraref_label = open(Terraref_cut_Path + 'label_x9.txt', 'w', encoding='utf-8')

#read labels of testset 
label_dic = read_label(csvFiles)


#devide images belonging to Terraref
for img_name in os.listdir(gwhd_2021_img_path):
    if img_name in label_dic.keys():
        domain = label_dic[img_name][-1]
        if 'Terraref' in domain:
            shutil.copy2(gwhd_2021_img_path + img_name, Terraref_path)



for imgName in tqdm(os.listdir(Terraref_path)):
    if '.png' not in imgName:
        continue
    img = cv2.imread(Terraref_path + imgName)

    
    if imgName in label_dic.keys() :
        laBel = label_dic[imgName][0]

        imgName = imgName[:-4]
        laBel = laBel.replace(" ", ",").split(';')
        source_img, n_lays = get_layout_image(img, laBel)
        cv2.imwrite(Terraref_ori_source_path + imgName + '.png', source_img)

        if 'no_box' not in laBel:
            orginal_bbox = np.array([np.array(list(map(int,bbox.split(',')))) for bbox in laBel])    

        #cut image into 512*512, with step=256(x9), and get layout images and new labels(txt)
        i = 0
        for y_i in range(3):
            for x_i in range(3):
                x_start = x_i*256
                y_start = y_i*256
                
                img_cut = img[y_start:y_start+512, x_start:x_start+512,:]
                source_img_cut = source_img[y_start:y_start+512, x_start:x_start+512,:]

                cv2.imwrite(Terraref_target_Path + imgName + '_' + str(i) + '.png', img_cut)
                cv2.imwrite(Terraref_source_Path + imgName + '_' + str(i) + '.png', source_img_cut)

                Terraref_label.write('%s.png'%(Terraref_target_Path + imgName + '_' + str(i)))
                
                if 'no_box' not in laBel:
                    bbox=orginal_bbox.copy()
                    bbox[:, 0] = bbox[:, 0] - x_start
                    bbox[:, 2] = bbox[:, 2] - x_start
                    bbox[:, 1] = bbox[:, 1] - y_start
                    bbox[:, 3] = bbox[:, 3] - y_start

                    bbox[:, 0:2][bbox[:, 0:2]<0] = 0
                    bbox[:, 2][bbox[:, 2]>511] = 511
                    bbox[:, 3][bbox[:, 3]>511] = 511

                    bboxes = bbox[np.logical_and((bbox[:, 2] - bbox[:, 0])>10, (bbox[:, 3] - bbox[:, 1])>10)] # discard invalid box 

                    for bbox in bboxes:
                        bbox = ','.join(map(str, bbox))
                        Terraref_label.write(' ' + bbox + ',0')

                Terraref_label.write('\n')

                i += 1

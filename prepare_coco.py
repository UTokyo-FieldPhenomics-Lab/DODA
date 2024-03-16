import os
import cv2
import numpy as np
import colorsys
import inflect
from tqdm import tqdm
from pycocotools.coco import COCO
from utils.utils import makedir

#Path of COCO
dataDir = 'datasets/coco'
dataTypes = ['val2017'] 

p = inflect.engine()

for dataType in dataTypes:
    prompt_txt_name = '{}/annotations/coco_prompts_{}.txt'.format(dataDir, dataType)
    prompt_txt = open(prompt_txt_name, 'w')

    #Load COCO annotation
    annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
    coco = COCO(annFile)

    output_path = '{}/images/80_colors/{}/'.format(dataDir, dataType)
    makedir(output_path)

    # Get category information
    categories = coco.loadCats(coco.getCatIds())

    # Map category_id to hsv color
    category_id_to_color = {}
    num_classes = len(categories)
    for i, category in enumerate(categories):
        hsv = [(i+1) / num_classes, 1., 1.]
        category_id_to_color[category['id']] = hsv
    hsv_tuples = [((x+1) / num_classes, 1., 1.) for x in range(num_classes+1)]
    colors_lst = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors_lst = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors_lst))

    # Map category_id to name
    category_id_to_name = {}
    for category in categories:
        category_id_to_name[category['id']] = category['name']

    # Get image ID
    imgIds = coco.getImgIds()

    # # Sort the labels in reverse order according to area
    for imgId in tqdm(imgIds):
        img_info = coco.loadImgs(imgId)[0]
        file_name = img_info['file_name']
        annIds = coco.getAnnIds(imgIds=img_info['id'])
        anns = coco.loadAnns(annIds)

        areas = [ann['area'] for ann in anns]
        # Sort labels based on area
        sorted_indices = np.argsort(areas,)[::-1]
        anns = [anns[i] for i in sorted_indices]

        mask = np.zeros((img_info['height'], img_info['width'], 3))

        #Get layout image
        category_num = {}
        for ann in anns:
            if ann['iscrowd'] and dataType=='train2017':   #Filter ‘iscrowd’ in training set
                break
            if ann['category_id'] not in category_num.keys():
                category_num[ann['category_id']] = 1
                repeat_time = 0
            else:
                repeat_time = category_num[ann['category_id']]
                category_num[ann['category_id']] = 1 + repeat_time
            hsv = category_id_to_color[ann['category_id']]
            hsv[2] = 1 - repeat_time*0.02
            rgb = colorsys.hsv_to_rgb(*hsv)
            bgr = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            bbox = ann['bbox']
            x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            mask = cv2.rectangle(mask, (x, y), (x + w, y + h), bgr, -1)
        if ann['iscrowd'] and dataType=='train2017':   #Filter ‘iscrowd’ in training set
            continue


        #Get prompt
        areas = [ann['area'] for ann in anns]
        # Sort labels based on area
        sorted_indices = np.argsort(areas,)
        anns = [anns[i] for i in sorted_indices]
        if len(anns)<1 and dataType=='train2017':
            continue
        prompt = 'a photograph with '
        category_dir = {}
        for ann in anns:
            category = category_id_to_name[ann['category_id']]
            if category not in category_dir:
                category_dir[category] = 1
            else:
                category_dir[category] = category_dir[category] + 1
        for category, repeat_time in category_dir.items():
            if repeat_time >1: 
                category = p.plural(category)
            prompt += str(repeat_time) + ' '+ category+', '


        file_name = file_name.replace('.jpg', '.png')

        cv2.imwrite(output_path + file_name, mask)
        prompt_txt.write(file_name + ';')
        prompt_txt.write(prompt[:-2]+'\n')

    
    
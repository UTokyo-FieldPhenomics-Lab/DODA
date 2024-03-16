import os
import csv

import cv2
import random

import numpy as np
import networkx as nx
from tqdm import tqdm
from utils.utils import makedir, is_overlaped



#Number of images to randomly generate
n_img = 5000
image_size = 512
p_background = 0.25
seed = 23

output_path = 'datasets/random_layout/'

# To avoid the size of the bboxes in the layout images being too different from the actual size,
# we use the length and width of the bounding boxes in the label of the target domain. 'use_wh_from_target_domain = True'
#-----------------------------
# If you want to randomly generate the length and width of bboxes, set use_wh_from_target_domain to False,
# and set the base_w, base_h, w_scale, h_scase
use_wh_from_target_domain = True
ann_Path = 'datasets/gwhd_2021/competition_test.csv'
base_w = 55
base_h = 50
w_scale = (0.15, 2)
h_scale = (0.15, 2)

np.random.seed(seed)
random.seed(seed)




def convert_ann2wh(ann_Path, domain_filter=None):
    csvFile = open(ann_Path, "r")
    reader = csv.reader(csvFile)
    laBel_list = []
    for item in reader:
        if reader.line_num == 1:
            continue
        domain = item[2]
        if domain_filter is not None and domain not in domain_filter:
            continue
        laBel_list.append(item[1])

    box_wh_lst = []
    for laBel in laBel_list:
        if 'no_box' not in laBel:
            laBel = laBel.split(';')
            for box in laBel:
                box = np.array(list(map(int,box.split(' ')))).reshape((4))
                w = box[2] - box[0]
                h = box[3] - box[1]
                box_wh = [w, h]

                if box_wh not in box_wh_lst:
                    box_wh_lst.append(box_wh)

    return box_wh_lst

def random_select_boxes(box_wh_lst, max_boxes_number=200, image_size=512, orginal_size=1024):

    selected_box_wh = random.choices(box_wh_lst, k=max_boxes_number)
    selected_box_wh = np.array(selected_box_wh)

    box_left_tops = np.random.rand(max_boxes_number, 2) * image_size
    box_left_tops = box_left_tops.astype(np.int32)
    boxes = np.concatenate((box_left_tops, selected_box_wh), axis=1)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    boxes = boxes[np.logical_and(boxes[:,2]<image_size, boxes[:,3]<image_size)]

    return boxes

def is_overlaped(array_1, array_2):
    x_min, y_min, x_max, y_max = array_2[0]
    x_min -= 16
    y_min -= 16
    x_max += 16
    y_max += 16
    
    x_min_n = array_1[:, 0]
    y_min_n = array_1[:, 1]
    x_max_n = array_1[:, 2]
    y_max_n = array_1[:, 3]
    intersect = ((x_min <= x_max_n) & (x_max >= x_min_n) & (y_min <= y_max_n) & (y_max >= y_min_n))
    
    return intersect

def calculate_iou(box1, box2):
    #Calculate the coordinates of the intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    #Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    #Calculate the area of two boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    iou = intersection_area / min(box1_area, box2_area)
    return iou

def remove_boxes_with_high_iou(boxes, threshold=0.4):
    boxes_to_keep = []
    for i in range(len(boxes)):
        keep_box = True
        for j in range(len(boxes)):
            if i != j:
                iou = calculate_iou(boxes[i], boxes[j])
                if iou > threshold:
                    keep_box = False
                    break
        if keep_box:
            boxes_to_keep.append(boxes[i])
    return np.array(boxes_to_keep)

def draw_control_img(bboxes, min_box_num=2, max_box_num=15, image_size=512):
    isolated_boxes = []
    overlaped_boxes = []

    box_lay_1 = []
    box_lay_2 = []
    box_lay_3 = []

    for bbox in bboxes:
        bbox = bbox.reshape(1, 4)
        overlap = is_overlaped(bboxes, bbox)
        n_box_overlap = np.sum(overlap.astype(int))
        bbox = bbox.reshape(4)
        if np.sum(n_box_overlap)==1:
            isolated_boxes.append(bbox)
        else:
            overlaped_boxes.append(bbox)

    lays = [0]   
    if len(overlaped_boxes) > 0:
        overlaped_boxes = np.array(overlaped_boxes)
        overlaped_matrix = None
        for bbox in overlaped_boxes:
            bbox = bbox.reshape(1, 4)
            overlap = is_overlaped(overlaped_boxes, bbox)
            overlap = overlap.reshape(1, overlap.size)
            if overlaped_matrix is not None:
                overlaped_matrix = np.concatenate((overlaped_matrix, overlap), axis=0) 
            else:
                overlaped_matrix = overlap
        
        diagonal_matrix = abs(np.eye(overlap.size) - 1)
        diagonal_matrix = diagonal_matrix.astype(bool)
        overlaped_matrix *= diagonal_matrix
        overlaped_matrix = overlaped_matrix.astype(bool)

        G = nx.Graph(overlaped_matrix)
        coloring = nx.coloring.greedy_color(G, strategy='largest_first')

        for box_id, lay_id in coloring.items():
            lays.append(lay_id)   
            if lay_id == 0:
                box_lay_1.append(overlaped_boxes[box_id])
            if lay_id == 1:
                box_lay_2.append(overlaped_boxes[box_id])
            if lay_id == 2:
                box_lay_3.append(overlaped_boxes[box_id])

    box_lay_1 = box_lay_1 + isolated_boxes
    box_lay_2 = random.sample(box_lay_2, len(box_lay_2)//2)
    box_lay_3 = random.sample(box_lay_3, len(box_lay_3)//4)
    box_all = box_lay_1 + box_lay_2 + box_lay_3

    max_box_num = min(len(box_all), max_box_num)
    min_box_num = min(len(box_all), min_box_num)

    box_num = random.randint(min_box_num, max_box_num)
    

    #Create a dictionary to record which list the elements come from
    index_dict = {}
    for index, lst in enumerate([box_lay_1, box_lay_2, box_lay_3]):
        for element in lst:
            index_dict[str(element)] = index

    selected_boxes = random.sample(box_all, box_num)

    new_lst = [[] for _ in range(3)] # Create three new empty lists

    for element in selected_boxes:
        index = index_dict[str(element)]  # Get the original list index that the element comes from
        new_lst[index].append(element)  #Put the elements into the corresponding new list


    box_lay_1 = np.array(new_lst[0])
    box_lay_2 = np.array(new_lst[1])
    box_lay_3 = np.array(new_lst[2])

    #draw bbox
    box_img_lay_1 = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    box_img_lay_2 = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    box_img_lay_3 = np.zeros((image_size, image_size, 1), dtype=np.uint8)

    if box_lay_1.shape[0] > 1:
        for bbox in box_lay_1:
            cv2.rectangle(box_img_lay_1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
    if box_lay_2.shape[0] > 1:
        for bbox in box_lay_2:
            cv2.rectangle(box_img_lay_2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
    if box_lay_3.shape[0] > 1:
        for bbox in box_lay_3:
            cv2.rectangle(box_img_lay_3, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)        
            
    source_img = np.concatenate((box_img_lay_1, box_img_lay_2, box_img_lay_3), axis=2)

    return np.array(selected_boxes), source_img

def generate_random_wh(base_w, base_h, w_scale, h_scale, n):
    sizes = []
    for _ in range(n):
        w = int(random.uniform(*w_scale) * base_w)
        h = int(random.uniform(*h_scale) * base_h)
        sizes.append([w, h])
    return sizes




box_img_path = output_path + 'img/'
makedir(box_img_path)
file = open(output_path + 'bounding_boxes.txt', 'w')

if use_wh_from_target_domain:
    box_wh_lst = convert_ann2wh(ann_Path, ['Terraref_1', 'Terraref_2'])
else:
    box_wh_lst = generate_random_wh(base_w, base_h, w_scale, h_scale, n=1000)




for i in tqdm(range(n_img)):
    if np.random.rand() < p_background:
        file_name = box_img_path + str(i) + '.png'
        file.write(file_name)
        file.write("\n")

        image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        cv2.imwrite(box_img_path + f"{i}.png", image)
        continue

    boxes = random_select_boxes(box_wh_lst)
    boxes = remove_boxes_with_high_iou(boxes)

    selected_boxes, source_img = draw_control_img(boxes)
    


    #Write bounding boxes coordinates to txt file
    if len(selected_boxes) > 0:
        file_name = box_img_path + str(i) + '.png'
        file.write(file_name)
        for box in selected_boxes:
            box = np.clip(box, 0, image_size-1)
            file.write(f" {box[0]},{box[1]},{box[2]},{box[3]},0")
        file.write("\n")

        cv2.imwrite(box_img_path + f"{i}.png", source_img)



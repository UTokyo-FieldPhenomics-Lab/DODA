import os
import cv2
import random
import numpy as np
import networkx as nx

from utils.utils import read_label, is_overlaped


def cut_box(domain_infor_path, img_name, original_box):
    names_img_w_wheat = open(domain_infor_path + 'with_wheat.txt', 'w', encoding='utf-8')
    names_img_wo_wheat = open(domain_infor_path + 'wo_wheat.txt', 'w', encoding='utf-8')
    img_name = img_name.split('.')[0]

    i = 0
    total_n_box = 0
    n_img_w_wheat = 0
    n_img_wo_wheat = 0
    for y_i in range(3):
        for x_i in range(3):
            x_start = x_i*256
            y_start = y_i*256
            box = original_box.copy()
            box[:, 0] = box[:, 0] - x_start
            box[:, 2] = box[:, 2] - x_start
            box[:, 1] = box[:, 1] - y_start
            box[:, 3] = box[:, 3] - y_start

            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>511] = 511
            box[:, 3][box[:, 3]>511] = 511

            boxes = box[np.logical_and((box[:, 2] - box[:, 0])>10, (box[:, 3] - box[:, 1])>10)] # discard invalid box 
            n_box = boxes.shape[0]
            total_n_box += n_box
            if n_box > 0:
                names_img_w_wheat.write(f'{img_name}_{i}.png\n')
                n_img_w_wheat += 1
            else:
                names_img_wo_wheat.write(f'{img_name}_{i}.png\n')
                n_img_wo_wheat += 1

            i += 1
    return total_n_box, n_img_w_wheat, n_img_wo_wheat

def random_select_boxes(box_wh_lst, max_boxes_number=100, image_size=512):

    selected_box_wh = random.choices(box_wh_lst, k=max_boxes_number)
    selected_box_wh = np.array(selected_box_wh)

    box_left_tops = np.random.rand(max_boxes_number, 2) * image_size
    box_left_tops = box_left_tops.astype(np.int32)
    boxes = np.concatenate((box_left_tops, selected_box_wh), axis=1)
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

    boxes = boxes[np.logical_and(boxes[:,2]<image_size, boxes[:,3]<image_size)]

    return boxes

def calculate_iou(box1, box2):
    # 计算交集的坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集面积
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # 计算并返回 IoU
    iou = intersection_area / min(box1_area, box2_area)
    return iou

def remove_boxes_with_high_iou(boxes, threshold=0.3):
    boxes_to_keep = []

    for i in range(len(boxes)):
        box_0 = boxes[0, :]
        boxes = np.delete(boxes, 0, axis=0)

        iou = []

        for j in range(len(boxes)):
            iou.append(calculate_iou(box_0, boxes[j]))

        if len(iou) > 0:
            if np.max(np.array(iou)) < threshold:
                boxes_to_keep.append(box_0)

    return np.array(boxes_to_keep)

def draw_layout_img(boxes, min_box_num=3, max_box_num=12, image_size=512):
    isolated_boxes = []
    overlaped_boxes = []

    box_lay_1 = []
    box_lay_2 = []
    box_lay_3 = []

    for box in boxes:
        box = box.reshape(1, 4)
        overlap = is_overlaped(boxes, box)
        n_box_overlap = np.sum(overlap.astype(int))
        box = box.reshape(4)
        if np.sum(n_box_overlap)==1:
            isolated_boxes.append(box)
        else:
            overlaped_boxes.append(box)

    lays = [0]   #放置box的层数，设置为[0]，保证至少有0+1=1层
    if len(overlaped_boxes) > 0:
        overlaped_boxes = np.array(overlaped_boxes)
        overlaped_matrix = None
        for box in overlaped_boxes:
            box = box.reshape(1, 4)
            overlap = is_overlaped(overlaped_boxes, box)
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
    

    # 创建一个字典来记录元素来自于哪个列表
    index_dict = {}
    for index, lst in enumerate([box_lay_1, box_lay_2, box_lay_3]):
        for element in lst:
            index_dict[str(element)] = index

    selected_boxes = random.sample(box_all, box_num)

    new_lst = [[] for _ in range(3)] # 创建三个新的空列表

    for element in selected_boxes:
        index = index_dict[str(element)]  # 获取元素来自的原始列表索引
        new_lst[index].append(element)  # 将元素放入对应的新列表中


    box_lay_1 = np.array(new_lst[0])
    box_lay_2 = np.array(new_lst[1])
    box_lay_3 = np.array(new_lst[2])

    #画出box
    box_img_lay_1 = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    box_img_lay_2 = np.zeros((image_size, image_size, 1), dtype=np.uint8)
    box_img_lay_3 = np.zeros((image_size, image_size, 1), dtype=np.uint8)

    if box_lay_1.shape[0] > 0:
        for box in box_lay_1:
            cv2.rectangle(box_img_lay_1, (box[0], box[1]), (box[2], box[3]), (255), -1)
    if box_lay_2.shape[0] > 0:
        for box in box_lay_2:
            cv2.rectangle(box_img_lay_2, (box[0], box[1]), (box[2], box[3]), (255), -1)
    if box_lay_3.shape[0] > 0:
        for box in box_lay_3:
            cv2.rectangle(box_img_lay_3, (box[0], box[1]), (box[2], box[3]), (255), -1)        
            
    source_img = np.concatenate((box_img_lay_1, box_img_lay_2, box_img_lay_3), axis=2)

    return np.array(selected_boxes), source_img



n_img = 1000
image_size = 512
min_box_num = 3

seed = 23

layout_path = f'datasets/wheat/random_layout/'
reference_infor_path = f'datasets/wheat/reference_infor/'
csvFile = f'datasets/gwhd_2021/competition_test.csv'



np.random.seed(seed)
random.seed(seed)


label_dic = read_label(csvFile)

domain2imgnames_dic = {}
for img_name in label_dic.keys():
    if label_dic[img_name][0] == 'no_box':
        continue
    domain = label_dic[img_name][-1]
    domain2imgnames_dic.setdefault(domain, []).append(img_name)

for domain in domain2imgnames_dic.keys():
    layout_img_path =  f'{layout_path}/{domain}/img/'
    os.makedirs(layout_img_path, exist_ok=True)
    label_txt = open(f'{layout_path}/{domain}/bounding_boxes.txt', 'w')

    domain_infor_path = f'{reference_infor_path}/{domain}/'
    os.makedirs(domain_infor_path, exist_ok=True)

    total = (0, 0, 0)
    box_wh_lst = []
    for img_name in domain2imgnames_dic[domain]:
        laBel = label_dic[img_name][0]
        laBel = laBel.replace(" ", ",").split(';')
        orginal_box = np.array([np.array(list(map(int,box.split(',')))) for box in laBel])    
        for box in orginal_box:
            w = box[2] - box[0]
            h = box[3] - box[1]
            box_wh = [w, h]

            if box_wh not in box_wh_lst:
                box_wh_lst.append(box_wh)

        total = tuple(map(sum, zip(total, cut_box(domain_infor_path, img_name, orginal_box.copy()))))

    total_n_box, n_img_w_wheat, n_img_wo_wheat = total
    max_box_num = round(total_n_box/n_img_w_wheat)
    p_background = n_img_wo_wheat / (n_img_w_wheat+n_img_wo_wheat)

    box_wh_lst = np.array(box_wh_lst)
    w = box_wh_lst[:, 0]
    h = box_wh_lst[:, 1]

    print(domain)
    print('average W', np.average(w))
    print('average H', np.average(h))
    print('p_background', p_background)

    i = 0
    while i < n_img:
        if np.random.rand() < p_background:
            label_txt_name = layout_img_path + str(i) + '.png'
            label_txt.write(label_txt_name)
            label_txt.write("\n")

            image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            cv2.imwrite(layout_img_path + f"{i}.png", image)
            i += 1
            continue

        boxes = random_select_boxes(box_wh_lst)
        boxes = remove_boxes_with_high_iou(boxes)

        selected_boxes, source_img = draw_layout_img(boxes, min_box_num=min_box_num, max_box_num=max_box_num)

        # Write the random generated bounding box to a txt file
        if len(selected_boxes) > min_box_num:
            label_txt_name = layout_img_path + str(i) + '.png'
            label_txt.write(label_txt_name)
            for box in selected_boxes:
                box = np.clip(box, 0, image_size-1)
                label_txt.write(f" {box[0]},{box[1]},{box[2]},{box[3]},0")
            label_txt.write("\n")

            cv2.imwrite(layout_img_path + f"{i}.png", source_img)
            i += 1



import os
import csv
import cv2
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import pytorch_lightning as pl

from typing import Dict
from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config


def DataloaderFromConfig(batch_size, train=None, validation=None,
                 num_workers=None, shuffle_val_dataloader=False):
    if train is not None:
        train_dataset = instantiate_from_config(train)
        train_dataloader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    if validation is not None:
        val_dataset = instantiate_from_config(validation)
        val_dataloader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=shuffle_val_dataloader)
        
    return train_dataloader, val_dataloader

def read_label(csv_label_path, label_dic={}):
    csvFile = open(csv_label_path, "r")
    reader = csv.reader(csvFile)
    for item in reader:
        if reader.line_num == 1:
            continue
        label_dic[item[0]] = [item[-2], item[-1]]
    return label_dic

def makedir(path):
    # Split the path into directories at each level according to '/'
    path_parts = path.split('/')
    current_path = ''
    # Determine whether the path exists step by step, and create it if it does not exist
    for part in path_parts:
        # Splice the current path with the current level directory
        current_path = os.path.join(current_path, part)
        # If the current path does not exist, create it
        if not os.path.exists(current_path):
            os.makedirs(current_path)

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

def get_layout_image(img, laBel):
    h, w, c = img.shape 
    bboxes = []
    isolated_boxes = []
    overlaped_boxes = []

    box_lay_1 = []
    box_lay_2 = []
    box_lay_3 = []
    box_lay_extra = []

    if "no_box" in laBel:
        source_img = np.zeros((h, w, 3), dtype=np.uint8)
        n_lay = 0
        return source_img, n_lay

    else:
        for bbox in laBel:
            bbox = np.array(list(map(int,bbox.split(','))))
            bboxes.append(bbox)
        bboxes = np.array(bboxes)
        for bbox in bboxes:
            bbox = bbox.reshape(1, 4)
            overlap = is_overlaped(bboxes, bbox)
            n_box_overlap = np.sum(overlap.astype(int))
            bbox = bbox.reshape(4)
            if np.sum(n_box_overlap)==1:
                isolated_boxes.append(bbox)
            else:
                overlaped_boxes.append(bbox)

        lays = [0]   #放置box的层数
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
                if lay_id > 2:
                    box_lay_extra.append(overlaped_boxes[box_id])

        n_lay = np.max(np.array(lays)) + 1

        box_lay_1 = np.array(box_lay_1)
        box_lay_2 = np.array(box_lay_2)
        box_lay_3 = np.array(box_lay_3)

        
        isolated_boxes = np.array(isolated_boxes)
        box_lay_extra = np.array(box_lay_extra)


        if isolated_boxes.shape[0]:
            if box_lay_1.size != 0:
                box_lay_1 = np.concatenate((box_lay_1, isolated_boxes), axis=0)
            else:
                box_lay_1 = isolated_boxes

        if box_lay_extra.shape[0]:
            if box_lay_extra.shape[0] > 1:
                indices = np.random.choice(range(len(box_lay_extra)), size=len(box_lay_extra)//2, replace=False)
                arr1 = box_lay_extra[indices]
                arr2 = np.delete(box_lay_extra, indices, axis=0)
                box_lay_2 = np.concatenate((box_lay_2, arr1), axis=0)
                box_lay_3 = np.concatenate((box_lay_3, arr2), axis=0)
            else:
                box_lay_3 = np.concatenate((box_lay_3, box_lay_extra), axis=0)


        #draw bbox
        box_img_lay_1 = np.zeros((h, w, 1), dtype=np.uint8)
        box_img_lay_2 = np.zeros((h, w, 1), dtype=np.uint8)
        box_img_lay_3 = np.zeros((h, w, 1), dtype=np.uint8)

        if box_lay_1.shape[0] > 1:
            for bbox in box_lay_1:
                cv2.rectangle(box_img_lay_1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
        if box_lay_2.shape[0] > 1:
            for bbox in box_lay_2:
                cv2.rectangle(box_img_lay_2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
        if box_lay_3.shape[0] > 1:
            for bbox in box_lay_3:
                if bbox not in box_lay_2:
                    cv2.rectangle(box_img_lay_3, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)        
                
        source_img = np.concatenate((box_img_lay_1, box_img_lay_2, box_img_lay_3), axis=2)

    return source_img, n_lay
        
class SaveLoraOnlyCallback(pl.Callback):
    def __init__(self, save_lora_only, paras_to_save='lora_', save_path='lora_model', filename='lora'):
        self.save_lora_only = save_lora_only
        self.paras_to_save = paras_to_save
        self.save_path = save_path
        self.filename = filename

    def on_epoch_end(self, trainer, pl_module):
        # 创建一个新的字典，只包含要保存的参数
        if self.save_lora_only:
            state_dict_to_save = {key: value for key, value in pl_module.state_dict().items() if self.paras_to_save in key}
        else:
            state_dict_to_save = pl_module.state_dict()
        # 拼接保存的文件路径
        save_filepath = f"{self.save_path}/{self.filename}_epoch{trainer.current_epoch}.pth"

        # 保存权重
        torch.save(state_dict_to_save, save_filepath)
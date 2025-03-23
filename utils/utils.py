import os
import csv
import cv2
from PIL import Image
import numpy as np
import networkx as nx

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from ldm.util import instantiate_from_config

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



def visualize_and_save_features_pca(feats_map, t, save_dir, layer_idx):
    """
    feats_map: [B, N, D]
    """
    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feats_map)
    feature_maps_pca = pca.transform(feats_map)  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(B, -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(feature_maps_pca):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(np.sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = pca_img.resize((512, 512))
        pca_img.save(os.path.join(save_dir, f"{i}_time_{t}_layer_{layer_idx}.png"))

def visualize_and_save_features_kmean(feats_map, save_dir, key_word=''):

    B = len(feats_map)
    feats_map = feats_map.flatten(0, -2)
    feats_map = feats_map.cpu().numpy()

    # K-means 聚类
    n_clusters = 3  # 簇数
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    kmeans.fit(feats_map)

    # 聚类标签
    labels = kmeans.labels_

    h = w = int(np.sqrt(labels.shape[0]))

    # 重塑回空间维度
    clustered_map = labels.reshape(h, w)

    clustered_map += 1
    clustered_map *= 50
    clustered_map += 50
    clustered_map = Image.fromarray(clustered_map.astype(np.uint8))
    clustered_map = clustered_map.resize((512, 512))
    clustered_map.save(os.path.join(save_dir, f"{key_word}_kmean{h}.png"))


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

        used_box = set()

        if box_lay_1.shape[0] > 0:
            for bbox in box_lay_1:
                if tuple(bbox) not in used_box:
                    cv2.rectangle(box_img_lay_1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
                    used_box.add(tuple(bbox))
        if box_lay_2.shape[0] > 0:
            for bbox in box_lay_2:
                if tuple(bbox) not in used_box:
                    cv2.rectangle(box_img_lay_2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)
                    used_box.add(tuple(bbox))
        if box_lay_3.shape[0] > 0:
            for bbox in box_lay_3:
                if tuple(bbox) not in used_box:
                    cv2.rectangle(box_img_lay_3, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255), -1)   
                    used_box.add(tuple(bbox))
                
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
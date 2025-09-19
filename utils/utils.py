#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :utils.py
@Author :CodeCat
@Date   :2025/8/27 11:56
"""

import os
import numpy as np
import torch
from loguru import logger
import time
from scipy.ndimage import binary_fill_holes


def load_pretrained_model(model, pretrained_model, device):
    if pretrained_model is not None:
        logger.info("Loading pretrained model from {} on device {}".format(pretrained_model, device))

        if os.path.exists(pretrained_model):
            checkpoint = torch.load(pretrained_model, map_location=device)
            model_param_state_dict = checkpoint['model']
            from collections import OrderedDict
            param_state_dict = OrderedDict()
            for k, v in model_param_state_dict.items():
                if k[:7] == 'module.':  # remove 'module.'
                    name = k[7:]
                else:
                    name = k
                param_state_dict[name] = v
            model_state_dict = model.state_dict()
            keys = model_state_dict.keys()
            num_param_loaded = 0
            for k in keys:
                if k not in param_state_dict:
                    logger.warning("{} is not in pretrained model".format(k))
                elif list(param_state_dict[k].shape) != list(model_state_dict[k].shape):
                    logger.warning(
                        "[SKIP] Shape of pretrained params {} doesn't match. (Pretrained: {}, Actual: {})".format(
                            k, param_state_dict[k].shape, model_state_dict[k].shape
                        )
                    )
                else:
                    model_state_dict[k] = param_state_dict[k]
                    num_param_loaded += 1
            model.load_state_dict(model_state_dict)
            logger.info("There are {}/{} variables loaded into {}".format(num_param_loaded, len(model_state_dict),
                                                                          model.__class__.__name__))
        else:
            raise ValueError("The pretrained model directory is not found: {}".format(pretrained_model))
    else:
        logger.info("No pretrained model to load, {} will be trained.".format(model.__class__.__name__))


def adjust_window(image, window_level=35, window_width=85):
    # 计算窗宽和窗位的最小和最大值
    min_value = window_level - window_width // 2
    max_value = window_level + window_width // 2

    # 将图像裁剪到指定的窗宽范围内
    windowed_image = np.clip(image, min_value, max_value)

    # 归一化图像到0-255范围
    windowed_image = ((windowed_image - min_value) / (max_value - min_value) * 255).astype(np.uint8)

    return windowed_image


def preprocess(img_path, transforms):
    data = dict()
    data['img'] = img_path
    data = transforms(data)
    data['img'] = data['img'][np.newaxis, ...]
    data['img'] = torch.from_numpy(data['img']).float()
    return data['img']


def fill_inner_holes(segmentation_image):
    # logger.info("Filling inner holes in the segmentation image...")
    filled_segmentation = np.zeros_like(segmentation_image)

    unique_labels = np.unique(segmentation_image)
    for label_value in unique_labels:
        if label_value == 0:
            continue
        class_mask = (segmentation_image == label_value)
        filled_class_mask = binary_fill_holes(class_mask)
        segmentation_image[filled_class_mask] = label_value
        filled_segmentation[filled_class_mask] = label_value

    return filled_segmentation


def fill_between_holes(segmentation_image):
    foreground_mask = (segmentation_image != 0).astype(np.uint8)
    filled_foreground_mask = binary_fill_holes(foreground_mask).astype(np.uint8)
    between_mask = np.where(filled_foreground_mask > foreground_mask, 1, 0)
    indices = np.where(between_mask == 1)
    for x, y in zip(*indices):
        if x is not None and y is not None:
            segmentation_image[x, y] = get_nearest_class((x, y), segmentation_image)
    return segmentation_image


def get_nearest_class(indice, segmentation_image):
    class_ls = []
    i, j = indice
    class_ls.append(segmentation_image[i - 1, j - 1])
    class_ls.append(segmentation_image[i, j - 1])
    class_ls.append(segmentation_image[i + 1, j - 1])
    class_ls.append(segmentation_image[i - 1, j])
    class_ls.append(segmentation_image[i + 1, j])
    class_ls.append(segmentation_image[i - 1, j + 1])
    class_ls.append(segmentation_image[i, j + 1])
    class_ls.append(segmentation_image[i + 1, j + 1])
    return max(set(class_ls), key=class_ls.count)
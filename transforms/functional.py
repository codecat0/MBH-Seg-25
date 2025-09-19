#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :functional.py
@Author :CodeCat
@Date   :2024/6/29 22:18
"""
import cv2
import numpy as np
from PIL import Image, ImageEnhance
from scipy.ndimage import distance_transform_edt


def crop(img, crop_coordinate):
    """
    裁剪图像中的一部分。

    Args:
        img (np.ndarray): 待裁剪的图像，类型为 numpy 数组。
        crop_coordinate (tuple): 包含四个整数值的元组，分别表示裁剪区域的左上角坐标(x1, y1)和右下角坐标(x2, y2)。

    Returns:
        np.ndarray: 裁剪后的图像，类型为 numpy 数组，维度与原图相同（除了裁剪区域的尺寸）。

    """
    x1, y1, x2, y2 = crop_coordinate
    img = img[y1:y2, x1:x2, ...]
    return img


def rescale_size(img_size, target_size):
    """
    将输入图像大小缩放到目标大小范围。

    Args:
        img_size (list[int]): 包含两个整数值的列表，分别代表图像的宽度和高度。
        target_size (list[int]): 包含两个整数值的列表，分别代表目标图像的宽度和高度。

    Returns:
        tuple: 包含两个元素的元组，第一个元素是缩放后的图像大小（list[int]），
        第二个元素是缩放比例（float）。

    """
    scale = min(
        max(target_size) / max(img_size), min(target_size) / min(img_size))
    rescaled_size = [round(i * scale) for i in img_size]
    return rescaled_size, scale


def normalize(im, mean, std):
    """
    对输入图像进行标准化处理。

    Args:
        im (np.ndarray): 输入的图像数据，形状为 (H, W, C)，其中 H 是高度，W 是宽度，C 是通道数。
                         数据类型应为 numpy.ndarray，且数据类型为 uint8。
        mean (list[float] or np.ndarray): 包含三个浮点数的列表或 numpy 数组，分别对应 RGB 三个通道的均值。
        std (list[float] or np.ndarray): 包含三个浮点数的列表或 numpy 数组，分别对应 RGB 三个通道的标准差。

    Returns:
        np.ndarray: 标准化处理后的图像数据，数据类型为 float32，取值范围为 [-1, 1]。

    """
    im = im.astype(np.float32, copy=False) / 255.0
    im -= mean
    im /= std
    return im


def resize(im, target_size=608, interp=cv2.INTER_LINEAR):
    """
    将图像调整到指定大小。

    Args:
        im (numpy.ndarray): 需要调整的图像，应为numpy数组。
        target_size (Union[int, List[int], Tuple[int, int]], optional): 目标图像的大小。
            如果为整数，则图像会等比例缩放到指定宽度和高度；
            如果为列表或元组，则分别指定宽度和高度。默认为608。
        interp (int, optional): 插值方式。默认为cv2.INTER_LINEAR。

    Returns:
        numpy.ndarray: 调整大小后的图像，应为numpy数组。

    """
    if isinstance(target_size, list) or isinstance(target_size, tuple):
        w = target_size[0]
        h = target_size[1]
    else:
        w = target_size
        h = target_size
    im = cv2.resize(im, (w, h), interpolation=interp)
    return im


def resize_long(im, long_size=224, interpolation=cv2.INTER_LINEAR):
    """
    将图像按照最长边缩放至指定长度。

    Args:
        im (np.ndarray): 待缩放的图像，应为numpy数组类型，形状为(H, W, C)，其中H为高度，W为宽度，C为通道数。
        long_size (int, optional): 目标图像最长边的长度，默认为224。
        interpolation (int, optional): 缩放时使用的插值方式，默认为cv2.INTER_LINEAR。

    Returns:
        np.ndarray: 缩放后的图像，同样为numpy数组类型，形状根据缩放比例而变化，但保持原图长宽比不变。

    """
    value = max(im.shape[0], im.shape[1])
    scale = float(long_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def resize_short(im, short_size=224, interpolation=cv2.INTER_LINEAR):
    """
    将图像按照短边缩放至指定长度。

    Args:
        im (np.ndarray): 待缩放的图像，应为numpy数组类型，且至少为二维（灰度图像）或三维（彩色图像）。
        short_size (int, optional): 目标图像短边的长度。默认为224。
        interpolation (int, optional): 缩放时使用的插值方式。默认为cv2.INTER_LINEAR。

    Returns:
        np.ndarray: 缩放后的图像，同样为numpy数组类型，形状根据缩放比例而变化，但保持原图长宽比不变。

    """
    value = min(im.shape[0], im.shape[1])
    scale = float(short_size) / float(value)
    resized_width = int(round(im.shape[1] * scale))
    resized_height = int(round(im.shape[0] * scale))

    im = cv2.resize(
        im, (resized_width, resized_height), interpolation=interpolation)
    return im


def horizontal_flip(im):
    """
    对图像进行水平翻转。

    Args:
        im (np.ndarray): 待翻转的图像，可以是二维或三维的numpy数组。
            二维数组表示灰度图像，三维数组表示彩色图像（通道顺序为BGR）。

    Returns:
        np.ndarray: 翻转后的图像，与输入图像具有相同的形状和数据类型。

    """
    if len(im.shape) == 3:
        im = im[:, ::-1, :]
    elif len(im.shape) == 2:
        im = im[:, ::-1]
    im = im.copy()
    return im


def vertical_flip(im):
    """
    对图像进行垂直翻转。

    Args:
        im (np.ndarray): 待翻转的图像，可以是二维或三维的numpy数组。
            二维数组表示灰度图像，三维数组表示彩色图像（通道顺序为BGR）。

    Returns:
        np.ndarray: 翻转后的图像，与输入图像具有相同的形状和数据类型。

    """
    if len(im.shape) == 3:
        im = im[::-1, :, :]
    elif len(im.shape) == 2:
        im = im[::-1, :]
    im = im.copy()
    return im


def brightness(im, brightness_lower, brightness_upper):
    """
    调整图像亮度。

    Args:
        im (PIL.Image.Image): 需要调整亮度的图像。
        brightness_lower (float): 亮度调整的最小值，范围在 0.0 到 1.0 之间。
        brightness_upper (float): 亮度调整的最大值，范围在 0.0 到 1.0 之间。

    Returns:
        PIL.Image.Image: 亮度调整后的图像。

    """
    brightness_delta = np.random.uniform(brightness_lower, brightness_upper)
    im = ImageEnhance.Brightness(im).enhance(brightness_delta)
    return im


def contrast(im, contrast_lower, contrast_upper):
    """
    调整图像对比度。

    Args:
        im (PIL.Image.Image): 需要调整对比度的图像对象。
        contrast_lower (float): 对比度调整的最小值，范围在 1.0 到 3.0 之间。
        contrast_upper (float): 对比度调整的最大值，范围在 1.0 到 3.0 之间。

    Returns:
        PIL.Image.Image: 对比度调整后的图像对象。

    """
    contrast_delta = np.random.uniform(contrast_lower, contrast_upper)
    im = ImageEnhance.Contrast(im).enhance(contrast_delta)
    return im


def saturation(im, saturation_lower, saturation_upper):
    """
    调整图像饱和度。

    Args:
        im (PIL.Image.Image): 需要调整饱和度的图像对象。
        saturation_lower (float): 饱和度调整的最小值，范围在 0.0（无饱和度）到 1.0（正常饱和度）之间。
        saturation_upper (float): 饱和度调整的最大值，范围在 0.0（无饱和度）到 1.0（正常饱和度）之间。

    Returns:
        PIL.Image.Image: 饱和度调整后的图像对象。

    """
    saturation_delta = np.random.uniform(saturation_lower, saturation_upper)
    im = ImageEnhance.Color(im).enhance(saturation_delta)
    return im


def hue(im, hue_lower, hue_upper):
    """
    调整图像的色调。

    Args:
        im (PIL.Image.Image): 需要调整色调的图像对象。
        hue_lower (float): 色调调整的最小值，范围在 -180.0 到 180.0 之间（对应于HSV颜色空间中的色调值）。
        hue_upper (float): 色调调整的最大值，范围在 -180.0 到 180.0 之间（对应于HSV颜色空间中的色调值）。

    Returns:
        PIL.Image.Image: 色调调整后的图像对象。

    """
    hue_delta = np.random.uniform(hue_lower, hue_upper)
    im = np.array(im.convert('HSV'))
    im[:, :, 0] = im[:, :, 0] + hue_delta
    im = Image.fromarray(im, mode='HSV').convert('RGB')
    return im


def sharpness(im, sharpness_lower, sharpness_upper):
    """
    调整图像的锐度。

    Args:
        im (PIL.Image.Image): 需要调整锐度的图像对象。
        sharpness_lower (float): 锐度调整的最小值，范围在 0.0（无锐度）到 1.0（正常锐度）之间。
        sharpness_upper (float): 锐度调整的最大值，范围在 0.0（无锐度）到 1.0（正常锐度）之间。

    Returns:
        PIL.Image.Image: 锐度调整后的图像对象。

    """
    sharpness_delta = np.random.uniform(sharpness_lower, sharpness_upper)
    im = ImageEnhance.Sharpness(im).enhance(sharpness_delta)
    return im


def rotate(im, rotate_lower, rotate_upper):
    """
    对图像进行随机旋转。

    Args:
        im (PIL.Image.Image): 需要旋转的图像对象，类型为PIL Image。
        rotate_lower (float): 旋转角度的最小值，单位为度。
        rotate_upper (float): 旋转角度的最大值，单位为度。

    Returns:
        PIL.Image.Image: 旋转后的图像对象，类型为PIL Image。

    """
    rotate_delta = np.random.uniform(rotate_lower, rotate_upper)
    im = im.rotate(int(rotate_delta))
    return im


def mask_to_onehot(mask, num_classes):
    """
    Convert a mask (H, W) to onehot (K, H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Onehot mask with shape(K, H, W).
    """
    _mask = [mask == i for i in range(num_classes)]
    _mask = np.array(_mask).astype(np.uint8)
    return _mask


def onehot_to_binary_edge(mask, radius):
    """
    Convert a onehot mask (K, H, W) to a edge mask.

    Args:
        mask (np.ndarray): Onehot mask with shape (K, H, W)
        radius (int|float): Radius of edge.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    if radius < 1:
        raise ValueError('`radius` should be greater than or equal to 1')
    num_classes = mask.shape[0]

    edge = np.zeros(mask.shape[1:])
    # pad borders
    mask = np.pad(mask, ((0, 0), (1, 1), (1, 1)),
                  mode='constant',
                  constant_values=0)
    for i in range(num_classes):
        dist = distance_transform_edt(mask[i, :]) + distance_transform_edt(
            1.0 - mask[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edge += dist

    edge = np.expand_dims(edge, axis=0)
    edge = (edge > 0).astype(np.uint8)
    return edge


def mask_to_binary_edge(mask, radius, num_classes):
    """
    Convert a segmentic segmentation mask (H, W) to a binary edge mask(H, W).

    Args:
        mask (np.ndarray): Label mask with shape (H, W)
        radius (int|float): Radius of edge.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: Edge mask with shape(H, W).
    """
    mask = mask.squeeze()
    onehot = mask_to_onehot(mask, num_classes)
    edge = onehot_to_binary_edge(onehot, radius)
    return edge
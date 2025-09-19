#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :infer.py
@Author :CodeCat
@Date   :2025/8/27 14:25
"""

from itertools import combinations
import numpy as np
import cv2
import torch
import torch.nn.functional as F


def slide_inference(model, im, crop_size, stride):
    h_im, w_im = im.shape[-2:]
    w_crop, h_crop = crop_size
    w_stride, h_stride = stride
    rows = int(np.ceil((h_im - h_crop) / h_stride)) + 1
    cols = int(np.ceil((w_im - w_crop) / w_stride)) + 1
    rows = 1 if h_im <= h_crop else rows
    cols = 1 if w_im <= w_crop else cols

    final_logit = None
    count = np.zeros([1, 1, h_im, w_im])
    for r in range(rows):
        for c in range(cols):
            h1 = r * h_stride
            w1 = c * w_stride
            h2 = min(h1 + h_crop, h_im)
            w2 = min(w1 + w_crop, w_im)
            h1 = max(h2 - h_crop, 0)
            w1 = max(w2 - w_crop, 0)
            im_crop = im[:, :, h1:h2, w1:w2]
            logit = model(im_crop)
            if isinstance(logit, list):
                logit = logit[0]
            logit = logit.detach().cpu().numpy()
            if final_logit is None:
                final_logit = np.zeros_like([1, logit.shape[1], h_im, w_im])
            final_logit[:, :, h1:h2, w1:w2] += logit[:, :, :h2 - h1, :w2 - w1]
            count[:, :, h1:h2, w1:w2] += 1
    if np.sum(count == 0) != 0:
        raise RunTimeError(
            'There are pixel not predicted. It is possible that stride is greater than crop_size'
        )
    final_logit = final_logit / count
    final_logit = torch.from_numpy(final_logit)
    return final_logit


def inference(model,
              im,
              is_slide=False,
              stride=None,
              crop_size=None,
              use_multilabel=False,
              use_multiclass=False,
              img_size=None,
              mode='bilinear',
              thresh=0.5):
    if not is_slide:
        logit = model(im)
        if isinstance(logit, list):
            logit = logit[0]
    else:
        logit = slide_inference(model, im, crop_size=crop_size, stride=stride)

    if img_size is not None and logit.shape[2:] != img_size:
        logit = F.interpolate(logit, size=img_size, mode=mode)

    if use_multilabel:
        pred = (F.sigmoid(logit) > 0.5).long()
    elif use_multiclass:
        pred = logit.argmax(dim=1, keepdim=True)
    else:
        probs = torch.softmax(logit, dim=1)
        pred = probs[:, 1, :, :]
        pred[pred < thresh] = 0
        pred[pred >= thresh] = 1
    return pred, logit

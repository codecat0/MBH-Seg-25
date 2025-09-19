#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File   :stroke_predict.py
@Author :CodeCat
@Date   :2025/8/27 11:43
"""
import os
import sys
import argparse
import cv2
import cc3d
import numpy as np
from loguru import logger
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from copy import deepcopy
from config.constants import Config
from models.msvm_unet import MSVMUNet_C
from utils.utils import load_pretrained_model, adjust_window, preprocess, fill_inner_holes, fill_between_holes
from transforms import transforms as T
from utils import infer


class StrokePredictor:
    def __init__(self, config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        self.ich_model_seg, self.ich_model_cls = self.load_model()

        self.ich_seg_transform = T.Compose(
            [
                T.Resize(self.config.ICH_SEG_IMG_SIZE),
                T.Normalize()
            ], img_channels=self.config.ICH_SEG_IMG_CHANNELS, foreground=True
        )
        self.ich_cls_transform = T.Compose(
            [
                T.Resize(self.config.ICH_CLS_IMG_SIZE),
                T.Normalize()
            ], img_channels=self.config.ICH_CLS_IMG_CHANNELS, foreground=False
        )

    def load_model(self):
        ich_model_seg = MSVMUNet_C(
            in_channels=self.config.ICH_SEG_IN_CHANNELS,
            num_classes=self.config.ICH_SEG_NUM_CLASSES,
        )
        load_pretrained_model(ich_model_seg, self.config.ICH_SEG_MODEL_PATH, self.device)
        ich_model_seg.eval()

        ich_model_cls = MSVMUNet_C(
            in_channels=self.config.ICH_CLS_IN_CHANNELS,
            num_classes=self.config.ICH_CLS_NUM_CLASSES,
        )
        load_pretrained_model(ich_model_cls, self.config.ICH_CLS_MODEL_PATH, self.device)
        ich_model_cls.eval()

        return ich_model_seg, ich_model_cls

    def predict(self, vol):
        ich_model_seg = self.ich_model_seg.to(self.device)
        ich_model_cls = self.ich_model_cls.to(self.device)
        origin_img_size = vol.shape[1:]
        vol = adjust_window(vol, self.config.WINDOW_LEVEL, self.config.WINDOW_WIDTH)
        depth = vol.shape[0]
        pred_result = np.zeros_like(vol)

        for ith in range(depth):
            img_slice = vol[ith, :, :]
            if self.config.ICH_SEG_IMG_CHANNELS == 3:
                img_slice = cv2.cvtColor(img_slice, cv2.COLOR_GRAY2RGB)
            img = preprocess(img_slice, self.ich_seg_transform)

            ith_seg_pred, _ = infer.inference(
                ich_model_seg,
                img.to(self.device),
                thresh=self.config.ICH_SEG_THRESH,
                img_size=origin_img_size,
                mode=self.config.INTERPOLATION_MODE
            )

            ith_seg_pred_in = ith_seg_pred.unsqueeze(1)
            if ith_seg_pred.shape[2:] != self.config.ICH_CLS_IMG_SIZE:
                ith_seg_pred_in = F.interpolate(ith_seg_pred_in.float(), size=self.config.ICH_CLS_IMG_SIZE,
                                                mode='nearest')

            img_slice = vol[ith, :, :]
            img = preprocess(img_slice, self.ich_cls_transform)

            ith_cls_pred, _ = infer.inference(
                ich_model_cls,
                torch.cat([img.to(self.device), ith_seg_pred_in], dim=1),
                use_multiclass=True,
                img_size=origin_img_size,
                mode=self.config.INTERPOLATION_MODE
            )

            ith_cls_pred = torch.squeeze(ith_cls_pred)
            ith_cls_pred = ith_cls_pred.detach().cpu().numpy().astype(np.uint8)

            pred_result[ith, :, :] = ith_cls_pred

        labels_out = cc3d.connected_components(pred_result)
        for i in range(1, labels_out.max() + 1):
            if np.sum(labels_out == i) < self.config.REMOVE_VOLUME:
                pred_result[labels_out == i] = 0

        depth = pred_result.shape[0]
        for i in range(depth):
            pred_result[i, :, :] = fill_inner_holes(pred_result[i, :, :])
            pred_result[i, :, :] = fill_between_holes(pred_result[i, :, :])

        return pred_result


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Competition Model Prediction Pipeline")
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Path to Image (MBH-Seg 2025)",
    )
    parser.add_argument(
        "--output_dir",
        default="competition_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    conf = Config()
    predictor = StrokePredictor(conf)

    for sample_id in os.listdir(args.input_dir):
        logger.info(f"Processing sample {sample_id}")
        output_path = os.path.join(args.output_dir, sample_id + '.nii.gz')
        sample_path = os.path.join(args.input_dir, sample_id)
        for file in os.listdir(sample_path):
            if 'image' in file:
                image_path = os.path.join(sample_path, file)
                vol_sitk = sitk.ReadImage(image_path)
                vol = sitk.GetArrayFromImage(vol_sitk)
                pred_result = predictor.predict(vol)
                pred_sitk = sitk.GetImageFromArray(pred_result)
                pred_sitk.CopyInformation(vol_sitk)
                sitk.WriteImage(pred_sitk, output_path)
        logger.info(f"Saved prediction to {output_path}")


if __name__ == '__main__':
    main()

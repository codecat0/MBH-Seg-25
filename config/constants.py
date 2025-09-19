# -*- coding:utf-8 -*-
import os
from loguru import logger


class Config:

    MODEL_PATH = 'weights'

    # 模型配置
    ICH_SEG_IN_CHANNELS = 3
    ICH_SEG_NUM_CLASSES = 2
    ICH_CLS_IN_CHANNELS = 2
    ICH_CLS_NUM_CLASSES = 6
    ICH_SEG_MODEL_PATH = os.path.join(MODEL_PATH, 'ich_seg_model.pth')
    ICH_CLS_MODEL_PATH = os.path.join(MODEL_PATH, 'ich_cls_model.pth')

    # 数据预处理配置
    WINDOW_LEVEL = 35
    WINDOW_WIDTH = 85
    ICH_SEG_IMG_SIZE = (512, 512)
    ICH_SEG_IMG_CHANNELS = 3
    ICH_CLS_IMG_SIZE = (512, 512)
    ICH_CLS_IMG_CHANNELS = 1

    # 后处理配置
    ICH_SEG_THRESH = 0.6
    INTERPOLATION_MODE = 'bilinear'
    REMOVE_VOLUME = 50


if __name__ == "__main__":
    conf = Config()
    logger.info(f"conf: {conf}")

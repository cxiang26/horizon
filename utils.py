# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import numpy as np
import colorsys
import cv2
from easydict import EasyDict

from coco_metric import MSCOCODetMetric
import sys
sys.path.append("../../../01_common/python/data/")
from dataloader import *


def get_classes(class_file_name='coco_classes.names'):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

## 新增画实例分割函数
def draw_bboxs_mask(image, bboxes, masks, mask_thr=0.4, gt_classes_index=None, classes=get_classes()):
    """draw the bboxes and masks in the original image
    """
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        mask_color = colors[np.random.randint(0, num_classes)]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = classes[class_index]
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        mask = cv2.resize(masks[i], (image_w, image_h))
        mask = mask >= mask_thr
        image[mask] = ([0.6*bbox_color[0], 0.6*bbox_color[1], 0.6*bbox_color[2]] + 0.4 * image[mask])
        mask = mask.astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, mask_color, 2, cv2.LINE_8, hierarchy, 100)
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(
            image,
            bbox_mess, (c1[0], c1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, (0, 0, 0),
            bbox_thick // 2,
            lineType=cv2.LINE_AA)
        print("{} is in the picture with confidence:{:.4f}".format(
            classes_name, score))
        cv2.imwrite("demo.jpg", image)
    return image


def draw_bboxs(image, bboxes, gt_classes_index=None, classes=get_classes()):
    """draw the bboxes in the original image
    """
    num_classes = len(classes)
    image_h, image_w, channel = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))

    fontScale = 0.5
    bbox_thick = int(0.6 * (image_h + image_w) / 600)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)

        if gt_classes_index == None:
            class_index = int(bbox[5])
            score = bbox[4]
        else:
            class_index = gt_classes_index[i]
            score = 1

        bbox_color = colors[class_index]
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        classes_name = classes[class_index]
        bbox_mess = '%s: %.2f' % (classes_name, score)
        t_size = cv2.getTextSize(
            bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3),
                      bbox_color, -1)
        cv2.putText(
            image,
            bbox_mess, (c1[0], c1[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale, (0, 0, 0),
            bbox_thick // 2,
            lineType=cv2.LINE_AA)
        print("{} is in the picture with confidence:{:.4f}".format(
            classes_name, score))
        cv2.imwrite("demo.jpg", image)
    return image


def eval_update(pred_result, image_name, results, category_id):
    assert isinstance(pred_result, list)
    for pred in pred_result:
        assert isinstance(pred, dict)
        assert "bbox" in pred, "missing bbox for prediction"
    if image_name in results:
        warnings.warn("warning: you are overwriting {}".format(image_name))

    parsed_name = image_name.strip()
    parsed_name = parsed_name.split(".")[0]
    image_id = int(parsed_name[-12:])
    inst_list = []
    for pred in pred_result:
        coco_inst = {}
        bbox = pred["bbox"].reshape((-1, ))
        assert bbox.shape == (6, ), (
            "bbox should with shape (6,), get %s" % bbox.shape)
        coco_inst.update({
            "image_id":
                image_id,
            "category_id":
                category_id[int(bbox[5])],
            "score":
                float(bbox[4]),
            "bbox": [
                float(bbox[0]),
                float(bbox[1]),
                float(bbox[2] - bbox[0]),
                float(bbox[3] - bbox[1]),
            ],
        })
        inst_list.append(coco_inst)
    results[image_name] = inst_list
    return results


def get_data_loader(image_path, label_path, transformers, batch_size=100):
    """Load the validation data."""
    data_loader = COCODataLoader(
        transformers, image_path, label_path, imread_mode='opencv')
    return data_loader

# Copyright (c) 2020 Horizon Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of Horizon Robotics Inc. This is proprietary information owned by
# Horizon Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of Horizon Robotics Inc.

import cv2
import os
import numpy as np

from x3_tc_ui import HB_QuantiONNXRuntime
import argparse
import json
from horizon_nn import horizon_onnx
from data_transformer import data_transformer
import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *
from dataloader import *
import utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from x3_tc_ui import HB_ONNXRuntime

def sigmoid(x):
    return 1/(1+np.exp(-x))

##crop函数：通过pred_bbox来提取mask区域，区域内mask得分保留，区域外至0。
def crop(masks, boxes, padding=1):
    n, h, w = masks.shape
    boxes = boxes / 4
    x1, x2, y1, y2 = boxes[...,0], boxes[...,2], boxes[...,1], boxes[...,3]
    rows = np.arange(w, dtype=masks.dtype).reshape(1, 1, -1).repeat(n, axis=0).repeat(h, axis=1)
    cols = np.arange(h, dtype=masks.dtype).reshape(1, -1, 1).repeat(n, axis=0).repeat(w, axis=2)
    masks_left = rows >= x1.reshape(-1, 1, 1)
    masks_right = rows <= x2.reshape(-1, 1, 1)
    masks_up = cols >= y1.reshape(-1, 1, 1)
    masks_down = cols < y2.reshape(-1, 1, 1)
    crop_mask = masks_left * masks_right * masks_up * masks_down
    return masks * crop_mask.astype(np.float)

## bbox_decoder: 整合anchors与bbox_pred，生成真实预测锚框，这里的结果是基于输入图像尺寸而非原始图像尺寸
def bbox_decoder(anchors, bbox_pred, max_shape):
    means = np.array([0.0, 0.0, 0.0, 0.0])
    stds = np.array([0.1, 0.1, 0.2, 0.2])
    wh_ratio_clip = 0.016
    denorm_deltas = bbox_pred * stds + means
    dx = denorm_deltas[..., 0::4]
    dy = denorm_deltas[..., 1::4]
    dw = denorm_deltas[..., 2::4]
    dh = denorm_deltas[..., 3::4]
    max_ratio = np.abs(np.log(wh_ratio_clip))
    dw = np.clip(dw, -max_ratio, max_ratio)
    dh = np.clip(dh, -max_ratio, max_ratio)
    x1, y1 = anchors[..., 0], anchors[..., 1]
    x2, y2 = anchors[..., 2], anchors[..., 3]
    # Compute center of each roi
    px = ((x1 + x2) * 0.5)[..., None]
    py = ((y1 + y2) * 0.5)[..., None]
    # Compute width/height of each roi
    pw = (x2 - x1)[..., None]
    ph = (y2 - y1)[..., None]
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    # Use network energy to shift the center of each roi
    gx = px + pw * dx
    gy = py + ph * dy
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5
    
    max_xy = np.concatenate([max_shape] * 2)
    bboxes = np.concatenate([x1, y1, x2, y2], axis=-1)
    bboxes = np.where(bboxes < 0., 0., bboxes)
    bboxes = np.where(bboxes > max_xy, max_xy, bboxes)
    return bboxes

## 快速非极大抑制函数，采用了YOLACT提供的快速抑制算法，工具链中使用了mxnet实现的算法，但是由于yolact输出中包含了mask_coeffs参数，无法直接用mxnet中算法
## 在c++的实现过程中可直接使用yolo5_nms
def fast_nms(multi_bboxes, multi_scores, multi_coeffs, iou_thr, score_thr, top_k, max_num=-1):
    scores = np.transpose(multi_scores[:, :-1])
    idx = np.argsort(scores, axis=-1)[:, ::-1][:, :top_k]
    num_classes, num_dets = idx.shape
    column_idx = np.arange(num_classes)[:,None]
    scores = scores[column_idx, idx]
    boxes = multi_bboxes[idx.flatten(), :].reshape(num_classes, num_dets, 4)
    coeffs = multi_coeffs[idx.flatten(), :].reshape(num_classes, num_dets, -1)

    # Compute ious
    area = (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    lt = np.maximum(boxes[..., :, None, :2], boxes[..., None, :, :2])
    rb = np.minimum(boxes[..., :, None, 2:], boxes[..., None, :, 2:])
    wh = np.clip((rb - lt), a_min=0, a_max=1000)
    overlap = wh[..., 0] * wh[..., 1]
    union = area[..., None] + area[..., None, :] - overlap
    ious = overlap / np.maximum(union, 1e-6)
    ious = np.triu(ious, 1)
    
    iou_max = np.max(ious, axis=1)
    keep = iou_max <= iou_thr
    keep *= scores > score_thr
    
    classes = np.arange(num_classes)[:, None]
    classes = np.repeat(classes, num_dets, axis=-1)
    
    classes = classes[keep]
    boxes = boxes[keep]
    coeffs = coeffs[keep]
    scores = scores[keep]
    
    if scores.shape[0] > max_num:
        idx = np.argsort(scores)
        scores = scores[idx[-max_num:]] 
        classes = classes[idx[-max_num:]]
        boxes = boxes[idx[-max_num:]]
        coeffs = coeffs[idx[-max_num:]]
    return boxes, scores[:, None], classes[:, None], coeffs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        type=str,
        help='input onnx model(.onnx) file',
        required=True)
    parser.add_argument(
        '--image',
        type=str,
        dest='image',
        help='input image file',
        required=False)
    parser.add_argument(
        '--type',
        type=str,
        default="inference",
        help='work type, can be inference or evaluation',
        required=False)
    parser.add_argument(
        '--anno_path', type=str, help='annotation path', required=False)
    parser.add_argument(
        '--image_path', type=str, help='evaluation image path', required=False)
    parser.add_argument(
        '--input_layout', type=str, help="model input layout", required=False)
    args = parser.parse_args()
    return args


def dump_image(image):
    print(image.shape)
    print("dump_image shape: ", image.shape)
    cv2.imwrite("./dump_image_t2.png", image[0])
    exit(0)


def load_image(image_file, input_layout):
    transformers = data_transformer()
    transformers.append(RGB2YUV444Transformer("CHW"))
    origin_image, process_image = SingleImageDataLoaderWithOrigin(
        transformers, image_file, imread_mode='opencv')
    if input_layout == "NHWC":
        process_image = process_image.transpose([0, 2, 3, 1])
    return origin_image, process_image


def model_infer(onnx_model, input_shape, process_image):
    sess = HB_ONNXRuntime(onnx_model=onnx_model)
    input_name = sess.get_inputs()[0].name
    output_name = [output.name for output in sess.get_outputs()]
    outputs = sess.run(output_name, {input_name: process_image},
                       {input_name: 'yuv444'})
    return outputs

## 后处理流程 outputs是一个包含16个成员的list，0-4号为score，5-10成员为box和anchor成员（box和anchor被concat在一起）
## 11-15为mask_coeffs得分； 16号成员为mask_proto。之所以为5层是因为YOLACT中FPN结构有5层输出。
## 后续改进后按FPN层进行了整合（c++中比较好处理），这里没有体现出来。
def postprocess(outputs, origin_shape):
    input_height = 550
    input_width = 550
    scores_list = outputs[0:5]  #
    box_anchors_list = outputs[5:10] #bbox与anchors被concat在outputs[1]中 0-4为bbox，4-8为anchor
    coeffs_list = outputs[10:15]
    protos = outputs[15][0]

    mlvl_bboxes_anchors = []
    mlvl_scores = []
    mlvl_coeffs = []
    for scores, box_anchors, coeffs in zip(scores_list, box_anchors_list, coeffs_list):
        scores = scores[0]
        box_anchors = box_anchors[0]
        coeffs = coeffs[0]
        if scores.shape[0] > 1000:  #利用scores每层帅选出1000个样本， 1000可调节，约小速度越快，但精度会受影响
            max_scores = scores[:, :-1].max(axis=-1)
#            print('max_scores.max: {}, min: {}'.format(max_scores.max(), max_scores.min()))
            topk_inds = max_scores.argsort()[-1000:]  # 从小到大排序，最后1000为最大
            # topk_inds = np.argpartition(-max_scores, 1000) # 对max_scores取负，获得-max_scores的1000个最小值索引
            box_anchors = box_anchors[topk_inds, :]
            scores = scores[topk_inds, :]
            coeffs = coeffs[topk_inds, :]
        mlvl_bboxes_anchors.append(box_anchors)
        mlvl_scores.append(scores)
        mlvl_coeffs.append(coeffs)
    mlvl_scores = np.concatenate(mlvl_scores)
    mlvl_coeffs = np.concatenate(mlvl_coeffs)
    mlvl_bboxes_anchors = np.concatenate(mlvl_bboxes_anchors)
    bboxes = bbox_decoder(mlvl_bboxes_anchors[:,4:], mlvl_bboxes_anchors[:, :4], (input_width, input_height))
    return bboxes, mlvl_scores, mlvl_coeffs, protos

## 没有修改，可以只评测bbox的精度
def evaluation(anno_path, image_path, onnx_model, input_shape):
    results = {}
    category_id = {}
    anno_file = os.path.abspath(os.path.expanduser(anno_path))
    coco_anno = COCO(anno_file)
    class_cat = coco_anno.dataset["categories"]
    for (i, cat) in enumerate(class_cat):
        category_id[i] = cat["id"]
    num_samples = 5000
    sess = HB_QuantiONNXRuntime(onnx_model=onnx_model)
    input_name = sess.get_inputs()[0].name
    output_name = [output.name for output in sess.get_outputs()]
    transformers = data_transformer()
    transformers.append(RGB2YUV444Transformer("CHW"))
    data_loader = utils.get_data_loader(image_path, anno_path, transformers)
    for i in range(num_samples):
        if i % 10 == 0:
            print('process: {} / {}'.format(i, num_samples))
        image, entry_dict = next(data_loader)
        anno_info = entry_dict[0]
        origin_shape = anno_info['origin_shape']
        image = image.transpose([0, 2, 3, 1]).astype(np.int8)
        bboxes_pr, _ = inference(origin_shape, image, onnx_model, input_shape)
        pred_result = []
        for one_bbox in bboxes_pr:
            one_result = {'bbox': one_bbox, 'mask': False}
            pred_result.append(one_result)
        results = utils.eval_update(pred_result, anno_info['image_name'],
                                    results, category_id)

    filename = os.path.abspath(os.path.expanduser("eval_result.json"))
    ret = []
    for (_, v) in results.items():
        ret.extend(v)
    with open(filename, "w") as f:
        json.dump(ret, f)
    pred = coco_anno.loadRes(filename)
    coco_eval = COCOeval(coco_anno, pred, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # IoU=0.5
    precision_0_5 = coco_eval.eval["precision"][0, :, :, 0, 2]
    ap_0_5 = np.mean(precision_0_5)
    # IoU=0.75
    precision_0_75 = coco_eval.eval["precision"][5, :, :, 0, 2]
    ap_0_75 = np.mean(precision_0_75)
    # IoU=0.5:0.95
    precision_0_5_0_95 = coco_eval.eval["precision"][0:10, :, :, 0, 2]
    ap_0_5_0_95 = np.mean(precision_0_5_0_95)

    # Model validation results.
    result_dict = {}
    result_dict['DataSet'] = 'COCO_Val2017'
    result_dict['Samples'] = num_samples
    result_dict["quanti model"] = {
        'IoU=0.50:0.95': ap_0_5_0_95,
        'IoU=0.50': ap_0_5,
        'IoU=0.75': ap_0_75
    }
    print(result_dict)

## 推理过程
def inference(origin_shape,
              process_image,
              onnx_model,
              input_shape,
              score_thr=0.2,
              iou_thr=0.5,
              top_k=200,
              max_per_img=100):
    model_output = model_infer(onnx_model, input_shape, process_image) ##获取模型推理结果
    box, score, coeffs, proto = postprocess(model_output, origin_shape) ##利用后处理函数得到box, score, coeffs, proto
    boxes, scores, classes, coeffs = fast_nms(box, score, coeffs, iou_thr, score_thr, top_k, max_per_img) ## 极大抑制，滤除重叠框   
    d, w, h = proto.shape
    n, _ = coeffs.shape 
    mask = coeffs @ proto.reshape(d, -1) ## 计算最后保留样本的实例mask
    mask = mask.reshape(n, w, h)
    mask = sigmoid(mask) 
    mask = crop(mask, boxes) ## 利用crop函数排除框外区域带来的影响，提高精度
	
	## 下面三行代码将预测box映射到原始图像
    scale_h, scale_w = origin_shape[0] / input_shape[0], origin_shape[1] / input_shape[1]
    boxes[:, 0], boxes[:, 2] = boxes[:, 0] * scale_w, boxes[:, 2] * scale_w
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] * scale_h, boxes[:, 3] * scale_h
    cls_dets = np.concatenate([boxes, scores, classes], axis=-1) ## 打包检测结果，用于后面的evaluation
    return cls_dets, mask


def main():
    args = get_args()
    input_shape = (550, 550)

    onnx_model = horizon_onnx.load_model(args.model)
    if args.type == "inference":
        origin_image, process_image = load_image(args.image, args.input_layout)
        origin_shape = origin_image.shape[1:]
        bboxes_pr, mask = inference(
            origin_shape,
            process_image,
            onnx_model,
            input_shape,
            )
		## 在utils中增加了绘画mask函数
        utils.draw_bboxs_mask(origin_image[0], bboxes_pr, mask, mask_thr=0.45)
    else:
        evaluation(args.anno_path, args.image_path, onnx_model, input_shape)


if __name__ == '__main__':
    main()

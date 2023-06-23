# COLLINS AEROSPACE
# THIS FILE CONTAINS NO EU OR US EXPORT-CONTROLLED TECHNICAL DATA
#
# This script uses and modifies the code from https://github.com/ultralytics/yolov5/blob/master/ 
# created by Ultralytics and released under the AGPL.3.0 license.
#
# Creation date: 26 May, 2023
#
# This script is distributed under the AGPL-3.0 license.
#
# Run as ' python3 generate_property.py <seed value> [--test_mode True] '

import csv
import os
import random
import cv2
import argparse
import torch
import torchvision
import numpy
import onnx
import onnxruntime
import shutil
import ast
from pathlib import Path


# costants
DELTAS = [0.1/100, 0.5/100, 1/100, 2/100, 5/100, 10/100]
EPSILON = 10/100
IMG_SIZE = [640, 640]
MODEL_NAME = 'yolov5nano_LRelu_640.onnx'


def run(seed, test_mode):
    print('Seed: ' + str(seed))
    random.seed(seed)
    im_fold_path = 'data'
    prop_fold_path = 'vnnlib'
    instances_fname = 'instances.csv'
    if os.path.exists(prop_fold_path) and os.path.isdir(prop_fold_path):
        shutil.rmtree(prop_fold_path)
    if os.path.exists(instances_fname):
        os.remove(instances_fname)
    os.mkdir(prop_fold_path)
    model_path = f'onnx/{MODEL_NAME}'

    # Serialize properties, one per delta
    for delta in DELTAS:
        # Randomly choose and resize image, if necessary with padding
        im_name = random.choice(os.listdir(im_fold_path))
        im_path = str(Path(im_fold_path).joinpath(Path(im_name)))
        print('Image: ' + im_path)
        im = cv2.imread(im_path)
        im_scaled = rescale_image(im, IMG_SIZE)

        # Perform prediction
        print('Model: ' + model_path)
        check_model(model_path)
        raw_pred = predict(im_scaled, model_path)
        c_names = get_model_classes(model_path)
        pred, out_to_raw_out = non_max_suppression_simple(raw_pred)
        rescale_bboxes(im_scaled, pred, im)
    
        print('Delta: ' + str(delta))
        # Randomly choose bounding box 
        bbox_ind = random.randint(0,len(pred)-1)
        bbox = pred[bbox_ind]
        print('Bounding box index: ' + str(bbox_ind) + ' out of: ' + str(len(pred)))
        # Apply perturbation and rescale images
        im_minus, im_plus = add_delta_noise_to_bbox(im, bbox, delta)
        im_minus_scaled = rescale_image(im_minus, IMG_SIZE)
        im_plus_scaled = rescale_image(im_plus, IMG_SIZE)
        # Serialize property: the probability of existence of the object in bbox does not change by more than EPSILON
        # prop_fn = str(Path(im_path).stem) + '_'  + str(bbox_ind) + '_' + str(delta) + '.vnnlib'
        prop_fn = f'img_{Path(im_path).stem}_perturbed_bbox_{bbox_ind}_delta_{delta}.vnnlib'
        prop_path = str(Path(prop_fold_path).joinpath(Path(prop_fn)))
        print('Property: ' + prop_path)
        raw_bbox_ind = out_to_raw_out[bbox_ind]
        serialize_property(prop_path, model_path, im_minus_scaled, im_plus_scaled, raw_pred, raw_bbox_ind, EPSILON, IMG_SIZE)
        
        with open(instances_fname, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([MODEL_NAME, prop_fn, '3600'])

    # (optional) Annotate original image and perturbed image copies with related predictions 
    if test_mode:
        im_c = im.copy()
        draw_bboxes(im_c, pred, (0, 255, 0), c_names)
        im_c_path = 'test1.jpg'
        if os.path.exists(im_c_path):
            os.remove(im_c_path)
        cv2.imwrite(im_c_path, im_c)

        raw_pred = predict(im_minus_scaled, model_path)
        pred, out_to_raw_out = non_max_suppression_simple(raw_pred)
        rescale_bboxes(im_minus_scaled, pred, im_minus)
        im_c = im_minus.copy()
        draw_bboxes(im_c, pred, (0, 255, 0), c_names)
        im_c_path = 'test2.jpg'
        if os.path.exists(im_c_path):
            os.remove(im_c_path)
        cv2.imwrite(im_c_path, im_c)

        raw_pred = predict(im_plus_scaled, model_path)
        pred, out_to_raw_out = non_max_suppression_simple(raw_pred)
        rescale_bboxes(im_plus_scaled, pred, im_plus)
        im_c = im_plus.copy()
        draw_bboxes(im_c, pred, (0, 255, 0), c_names)
        im_c_path = 'test3.jpg'
        if os.path.exists(im_c_path):
            os.remove(im_c_path)
        cv2.imwrite(im_c_path, im_c)


def get_model_interface(model_path):
    session = onnxruntime.InferenceSession(model_path)
    input = session.get_inputs()[0]
    output = session.get_outputs()[0]
    h = input.shape[2]
    w = input.shape[3]
    n_bboxes = output.shape[1]
    n_data = output.shape[2]
    return h, w, n_bboxes, n_data


def get_model_classes(model_path):
    model = onnx.load(model_path)
    d = ast.literal_eval(model.metadata_props[1].value)
    classes = [c for c in d.values()]
    return classes


def serialize_property(prop_path, model_path, im_minus, im_plus, raw_pred, raw_bbox_ind, eps, img_size):
    h, w, n_bboxes, n_data = get_model_interface(model_path)
    n_channels = 3
    with open(prop_path, 'a') as f:
        # Input variables declaration
        n_inp = 0
        f.write(';Input variables:\n')
        for y in range(h):
            for x in range(w):
                for k in range(n_channels):
                    f.write('(declare-const X_' + str(n_inp) + ' Real)' + '\n')
                    n_inp += 1
        # Output variables declaration
        n_out = 0
        f.write('\n;Output variables:\n')
        for b in range(n_bboxes):
            for d in range(n_data):
                f.write('(declare-const Y_' + str(n_out) + ' Real)' + '\n')
                n_out += 1
        # Input constraints definition
        upper = im_plus.numpy()[0]
        upper = upper.transpose((1, 2, 0))
        lower = im_minus.numpy()[0]
        lower = lower.transpose((1, 2, 0))
        n_inp = 0
        f.write('\n;Input constraints:\n')
        for y in range(h):
            for x in range(w):
                for k in range(n_channels):
                    ub = upper[y][x][k]
                    lb = lower[y][x][k]
                    f.write('(assert (<= X_' + str(n_inp) + ' ' + str(ub) + '))' + '\n')
                    f.write('(assert (>= X_' + str(n_inp) + ' ' + str(lb) + '))' + '\n')
                    n_inp += 1
        # Output constraints definition
        pred = raw_pred.numpy()
        n_out = 0
        f.write('\n;Output constraints:\n')
        f.write('(assert (or \n')
        for b in range(n_bboxes):
            bbox = pred[b]
            for d in range(n_data): # Constrain upper and lower bounds of each bounding box element
                data = bbox[d]
                ub = data
                lb = data
                if b == raw_bbox_ind:
                    #if d in [0,2]: # Bounding box does note move more than 1/100 of the image resolution on either axis (x)
                    #   w = img_size[0]
                    #   assert(data >= 0 and data <= w)
                    #   ub = min(data + w/100, w)
                    #   lb = max(data - w/100, 0)
                    #elif d in [1,3]: # (y)
                    #   y = img_size[1]
                    #   assert(data >= 0 and data <= y)
                    #   ub = min(data + y/100, y)
                    #   lb = max(data - y/100, 0)
                    # f.write(f'HERE: Y_{n_out}')
                    if d == 4: # Object existence probability does not change more than eps
                        assert(data >= 0 and data <= 1)
                        ub = min(data*(1+eps), 1.0)
                        lb = max(data*(1-eps), 0.0)
                        constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                        f.write(constr)
                    elif d in [5,6,7,8,9,10]:  # Class conditional probability allowed to fluctuate
                        assert(data >= 0 and data <= 1)
                        ub = 1.0
                        lb = 0.0
                        constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                        f.write(constr)
                    # Negated property
                    # constr = '\t(and (>= Y_' + str(n_out) + ' ' + str(ub) + ')) (and (<= Y_' + str(n_out) + ' ' + str(lb) + '))' + '\n'
                    # f.write(constr)
                n_out += 1
            if b == raw_bbox_ind: # Highest class conditional probability remains the highest despite of perturbation (negated property)
                max_class_prob_ind = numpy.argmax(bbox[5:11])
                n_out_max_class_prob = n_out - 6 + max_class_prob_ind
                n_out_class_probs = [n for n in range(n_out - 6, n_out) if n != n_out_max_class_prob]
                # f.write(f'HERE: Y_{n_out}')
                for n in n_out_class_probs:
                    constr = '\t(and (>= Y_' + str(n) + ' Y_' + str(n_out_max_class_prob) + '))' + '\n'
                    f.write(constr)
        f.write('))')


def add_delta_noise_to_bbox(im, bbox, d):
    im_plus = im.copy()
    im_minus = im.copy()
    for y in range(int(bbox[1])+1, int(bbox[3])):
        for x in range(int(bbox[0])+1, int(bbox[2])):
            color = im[y][x]
            color_plus = [min(int(color[0]*(1+d)),255), min(int(color[1]*(1+d)),255), min(int(color[2]*(1+d)),255)]
            color_minus = [max(int(color[0]*(1-d)),0), max(int(color[1]*(1-d)),0), max(int(color[2]*(1-d)),0)]
            im_plus[y][x] = color_plus
            im_minus[y][x] = color_minus
    return im_minus, im_plus


def draw_bboxes(im, bboxes, color, c_names):
    for bbox in bboxes:
        top_left, bottom_right = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(im, top_left, bottom_right, color, thickness=1, lineType=cv2.LINE_AA)
        label = f'{c_names[int(bbox[5])]} {bbox[4]:.2f}'
        cv2.putText(im, text=label, org=top_left, fontFace=0, fontScale=0.5, color = (255,255,255), thickness=1, lineType=cv2.LINE_AA)


def rescale_bboxes(im_scaled, pred, im):
    pred[:, :4] = scale_boxes(im_scaled.shape[2:], pred[:, :4], im.shape).round()


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def check_model(model_path):
    model = onnx.load(model_path)
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("The model is valid")


def predict(im_scaled, model_path):
    session = onnxruntime.InferenceSession(model_path)
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    #for i, info in enumerate(inputs):
    #    print(f"Input {i}: name = {info.name}, shape = {info.shape}")
    #for i, info in enumerate(outputs):
    #    print(f"Output {i}: name = {info.name}, shape = {info.shape}")
    input_name = inputs[0].name
    output_name = outputs[0].name
    # Perform prediction
    raw_pred = session.run([output_name], {input_name: im_scaled.numpy()})
    if isinstance(raw_pred, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        raw_pred = raw_pred[0]
    raw_pred = raw_pred[0]

    return torch.from_numpy(raw_pred)


def rescale_image(im, img_size):
    im_scaled = letterbox(im, img_size, stride=32, auto=True)[0]  # padded resize
    im_scaled = im_scaled.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im_scaled = numpy.ascontiguousarray(im_scaled)  # contiguous
    im_scaled = torch.from_numpy(im_scaled).to(torch.device('cpu'))
    im_scaled = im_scaled.float()
    im_scaled /= 255
    if len(im_scaled.shape) == 3:
        im_scaled = im_scaled[None]  # expand for batch dim
    return im_scaled


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        # NOTE: not clear the purpose of the original code
        dw, dh = dw, dh #numpy.mod(dw, stride), numpy.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh // 2, dh - (dh // 2)
    left, right = dw // 2, dw - (dw // 2)
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    assert(im.shape[0] == new_shape[1] and im.shape[1] == new_shape[0])
    return im, ratio, (dw, dh)


def non_max_suppression_simple(pred):
    conf_thres=0.25
    iou_thres=0.45
    max_det = 300
    nc = pred.shape[1] - 5  # number of classes

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05  # seconds to quit after

    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6), device='cpu')]
    x = pred
    # Map for tracing final output to raw output
    out_to_raw_out1 = dict()

    # first filter: object existence confidence (on col 4) > threshold
    f1 = x[..., 4] > conf_thres  # candidates, Boolean vector
    f1_idxs = [i for i, y in enumerate(f1) if y]
    x = x[f1]   
    for i in range(0,len(x)):
        out_to_raw_out1[i] = f1_idxs[i]

    # If none remain return
    if not x.shape[0]:
        return 

    # Compute prediction confidence
    # coords on col 0..3
    # obj_conf on col 4
    # cls_conf from col 5 to last 
    x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

    # Box/Mask
    box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)

    # Detections matrix nx6 (xyxy, conf, cls)
    # best class only
    # conf: vector of max confidences; j: corresponding vector of classes indexed from 0
    conf, j = x[:, 5:mi].max(1, keepdim=True)
    # second filter: max class confidence > threshold
    x = torch.cat((box, conf, j.float()), 1)
    f2 = conf.view(-1) > conf_thres # Boolean vector
    f2_idxs = [i for i, y in enumerate(f2) if y]
    x = x[f2]
    out_to_raw_out2 = dict()
    for i in range(0,len(x)):
        out_to_raw_out2[i] = out_to_raw_out1[f2_idxs[i]]

    # Check shape
    n = x.shape[0]  # number of boxes
    if not n:  # no boxes
        return 
    # third filter: sort by confidence and remove excess boxes
    f3_idxs = x[:, 4].argsort(descending=True)[:max_nms] # indexes vector
    x = x[f3_idxs]
    out_to_raw_out3 = dict()
    for i in range(0,len(x)):
        out_to_raw_out3[i] = out_to_raw_out2[f3_idxs[i].item()]

    # Batched NMS
    c = x[:, 5:6] * max_wh  # classes
    boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
    # fourth filter
    f4_idxs = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS, indexes vector
    f4_idxs = f4_idxs[:max_det]  # limit detections
    output = x[f4_idxs]
    out_to_raw_out4 = dict()
    for i in range(0,len(output)):
        out_to_raw_out4[i] = out_to_raw_out3[f4_idxs[i].item()]

    # Mapping check w.r.t confidence values: 
    # pred[out_to_raw_out4[idx]][4]*pred[out_to_raw_out4[idx]][5+int(output[idx][5])] 
    # against output[idx][4]
    for i in range(0,len(output)):
        conf = pred[out_to_raw_out4[i]][4]*pred[out_to_raw_out4[i]][5+int(output[i][5])]
        conf_ = output[i][4]
        assert(conf == conf_) # NB: equality between float 

    return output, out_to_raw_out4


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=str, default=10, help='random seed (int)')
    parser.add_argument('--test_mode', type=bool, default=False, help='annotate images with predictions (Bool)')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
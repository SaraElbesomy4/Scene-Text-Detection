#-*- coding:utf-8 -*-
#'''
# Created on 18-12-11 上午10:03
#
# @Author: Greg Gao(laygin)
#'''
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import scipy.io as sio
import torch
import torch.nn.functional as F
from ctpn_model import CTPN_Model
from ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from ctpn_utils import resize
import config
from shapely.geometry import Polygon

prob_thresh = 0.5
height = 720
gpu = True
if not torch.cuda.is_available():
    gpu = False
device = torch.device('cuda:0' if gpu else 'cpu')
weights = os.path.join(config.checkpoints_dir, 'CTPN.pth')
model = CTPN_Model()
model.load_state_dict(torch.load(weights, map_location=device)['model_state_dict'])
model.to(device)
model.eval()



def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_2_points(text):
    '''
     Objective: to drop the unused elements from the input text and return only the 2 points to compare them with the opposite 2 points in the label.
     Input: a numpy array that retrieved from the model and represent the coordinates of the points that used to draw the detection box.
     output: a numpy array that have only 2 points.
    '''

    text = np.delete(text, np.s_[2:6], axis=1)
    text = text[:, 0:4]

    return text


def get_4_points(label):
    '''
    Objective: to return the four points coordinates from the two points in the label array
    Input: the label with shape (1,4) has 2 points: (xmin, ymin) and (xmax, ymax)
    output: 4 points (xmin, ymin) (x1, y1) (x2, y2) (xmax, ymax)
    '''
    p1 = (label[:, 0], label[:, 1])
    p2 = (label[:, 2], label[:, 1])
    p3 = (label[:, 2], label[:, 3])
    p4 = (label[:, 0], label[:, 3])

    return p1, p2, p3, p4

def intersect(p1, p2, p3, p4, m1, m2, m3, m4):
    '''
     Objective: to calculate the area of the intersection between 2 rectangles.
     Inputs: 8 tuples: 4 tuples for each rectangle represent the 4 points to draw it.
     Output: the area of the intersection (float)
    '''
    box1 = Polygon([p1, p2, p3, p4])
    box2 = Polygon([m1, m2, m3, m4])
    intersection = box1.intersection(box2)
    return intersection.area


def get_iou(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    a1, a2, a3, a4 = get_4_points(box_a)
    b1, b2, b3, b4 = get_4_points(box_b)
    inter = intersect(a1, a2, a3, a4, b1, b2, b3, b4)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def get_accuracy(label, target):
    '''
    Objective: get an image that contains multiple detection boxes and loop over them to get the iou accuracy of each box and then get the average accuracy
               of all boxes to represent the accuracy for this image.
    Input:
           label: tensor of the groundtruth labels of one image
           target: tensor of the label that returned by the model for one image
    Output: a float number represents the iou accuracy of one image
    '''
    target_rows = target.shape[0]
    label_rows = label.shape[0]
    iou_list = []
    for t in range(target_rows):
        for L in range(label_rows):
            iou_box = get_iou(label[L, :].unsqueeze(0), target[t, :].unsqueeze(0))
            iou_list.append(iou_box)
    iou_list = sorted(iou_list, reverse=True)
    iou_list = np.asarray(iou_list)
    final_list = iou_list[0:target_rows]
    acc_image = sum(final_list)/len(final_list)

    return acc_image


def get_det_boxes(image,display = True, expand = True):
    #image = resize(image, height=height)
    #image = cv2.resize(image, (224,224), interpolation=cv2.INTER_AREA)

    image_r = image.copy()
    image_c = image.copy()
    h, w = image.shape[:2]
    image = image.astype(np.float32) - config.IMAGE_MEAN
    image = torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).float()

    with torch.no_grad():
        image = image.to(device)
        cls, regr = model(image)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])
        # print(bbox.shape)

        fg = np.where(cls_prob[0, :, 1] > prob_thresh)[0]
        # print(np.max(cls_prob[0, :, 1]))
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        # print(select_anchor.shape)
        keep_index = filter_bbox(select_anchor, 16)

        # nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        # print(keep)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])

        # expand text

        if expand:
            for idx in range(len(text)):
                text[idx][0] = max(text[idx][0] - 10, 0)
                text[idx][2] = min(text[idx][2] + 10, w - 1)
                text[idx][4] = max(text[idx][4] - 10, 0)
                text[idx][6] = min(text[idx][6] + 10, w - 1)



        # print(text)
        if display:

            blank = np.zeros(image_c.shape,dtype=np.uint8)
            for box in select_anchor:
                pt1 = (box[0], box[1])
                pt2 = (box[2], box[3])
                blank = cv2.rectangle(blank, pt1, pt2, (50, 0, 0), -1)
            image_c = image_c+blank
            image_c[image_c>255] = 255


            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(image_c, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(image_c, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(image_c, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                cv2.putText(image_c, s, (i[0]+13, i[1]+13),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255,0,0),
                            2,
                            cv2.LINE_AA)
            # dis(image_c)


        return text, image_c, image_r                   #text[:,1:5]

if __name__ == '__main__':
    img_path = 'E:/CIE/CIE 3/Spring 2020/Computer Vision/Final Project/Dataset/Images/All/img5.jpg'
    image = cv2.imread(img_path)
    print(image.shape)
    #cv2.imshow('Sample', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    label = sio.loadmat('E:/CIE/CIE 3/Spring 2020/Computer Vision/Final Project/Dataset/Groundtruth/Rectangular/All/rect_gt_img5.mat')
    label = list(label.items())
    label = np.array(label)
    label = label[3, 1]
    label = label[:, 0:4]
    '''
    label[:, 0] = label[:, 0] * (224 / 672)
    label[:, 1] = label[:, 1] * (224 / 450)
    label[:, 2] = label[:, 2] * (224 / 672)
    label[:, 3] = label[:, 3] * (224 / 450)
    '''
    label = label.astype(float)
    label = torch.from_numpy(label)

    text, image_c, image_r = get_det_boxes(image)
    text = get_2_points(text)
    text = torch.from_numpy(text)

    dis(image_c)
    #dis(image_r)
    acc = get_accuracy(text, label)
    print(acc)
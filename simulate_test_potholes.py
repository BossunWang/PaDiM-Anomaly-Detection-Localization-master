import random
from random import sample
import argparse
import numpy as np
import os
import pickle
from collections import OrderedDict
from scipy.ndimage import gaussian_filter, maximum_filter
import matplotlib.pyplot as plt
import time
import cv2
from threading import Thread
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from resnet import CustomResnet
from ultralytics import YOLO


class AnomalyDetectionThread(Thread):
    # constructor
    def __init__(self,
                device,
                idx,
                model, train_outputs,
                split_format,
                resize_image, resize_gt_image, resize_roi_image,
                img_size,
                block_size,
                bbox_area_mean, bbox_area_std,
                bbox_wh_ratio_mean, bbox_wh_ratio_std):

        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.device = device
        self.idx = idx
        self.model = model
        self.train_outputs = train_outputs
        self.split_format = split_format
        self.resize_image = resize_image
        self.resize_gt_image = resize_gt_image
        self.resize_roi_image = resize_roi_image
        self.img_size = img_size
        self.block_size = block_size
        self.bbox_area_mean = bbox_area_mean
        self.bbox_area_std = bbox_area_std
        self.bbox_wh_ratio_mean = bbox_wh_ratio_mean
        self.bbox_area_mean = bbox_area_mean
        self.bbox_wh_ratio_std = bbox_wh_ratio_std
        self.mask_img = None

    # function executed in a new thread
    def run(self):
        # store data in an instance variable
        self.mask_img = runAnomalyDetection(self.device,
                                        self.idx,
                                        self.model, self.train_outputs,
                                        self.split_format,
                                        self.resize_image, self.resize_gt_image, self.resize_roi_image,
                                        self.img_size,
                                        self.block_size,
                                        self.bbox_area_mean, self.bbox_area_std,
                                        self.bbox_wh_ratio_mean, self.bbox_wh_ratio_std,
                                        show=False)


class YoloDetectionThread(Thread):
    # constructor
    def __init__(self, model, image):
        # execute the base constructor
        Thread.__init__(self)
        # set a default value
        self.model = model
        self.image = image
        self.mask_img = None

    def run(self):
        # store data in an instance variable
        self.mask_img = runYoloDetection(self.model, self.image, show=False)


def to_Tensor(image, img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.resize(image, (img_size, img_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = ((image / 255.0) - mean) / std
    image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image.astype(np.float32))
    return image.unsqueeze(0)


def embedding_concat(x, y):
    B, C1, H1, W1 = x.size()
    _, C2, H2, W2 = y.size()
    s = int(H1 / H2)
    x = F.unfold(x, kernel_size=s, dilation=1, stride=s)
    x = x.view(B, C1, -1, H2, W2)
    z = torch.zeros(B, C1 + C2, x.size(2), H2, W2).to(x.device)
    for i in range(x.size(2)):
        z[:, :, i, :, :] = torch.cat((x[:, :, i, :, :], y), 1)
    z = z.view(B, -1, H2 * W2)
    z = F.fold(z, kernel_size=s, output_size=(H1, W1), stride=s)

    return z


def mahalanobis_torch(u, v, cov_inv):
    delta = u - v
    m = torch.dot(delta, torch.matmul(cov_inv, delta))
    return torch.sqrt(m)


def runAnomalyDetection(device,
                        idx,
                        model, train_outputs,
                        split_format,
                        resize_image, resize_gt_image, resize_roi_image,
                        img_size,
                        block_size,
                        bbox_area_mean, bbox_area_std,
                        bbox_wh_ratio_mean, bbox_wh_ratio_std, show=True):
    grid_w = split_format[0] // split_format[2]
    grid_h = split_format[1] // split_format[3]
    grid_ws_list = [grid_w * i for i in range(split_format[2])]
    grid_hs_list = [grid_h * j for j in range(split_format[3])]

    start_total_time = time.time()
    grid_imgs = torch.Tensor([]).to(device)
    grid_img_list = []
    grid_gt_list = []
    grid_roi_list = []
    for gh in grid_hs_list:
        for gw in grid_ws_list:
            grid_img = resize_image[gh: gh + grid_h, gw: gw + grid_w].copy()
            grid_gt_img = resize_gt_image[gh: gh + grid_h, gw: gw + grid_w].copy()
            grid_roi_img = resize_roi_image[gh: gh + grid_h, gw: gw + grid_w].copy()

            torch_img = to_Tensor(grid_img, img_size).to(device)
            grid_imgs = torch.cat([grid_imgs, torch_img])
            grid_img_list.append(grid_img)
            grid_gt_list.append(grid_gt_img)
            grid_roi_list.append(grid_roi_img)

    # feature extraction
    start_inf_time = time.time()
    with torch.no_grad():
        outputs = model(grid_imgs)

    test_outputs = OrderedDict([('layer1', []), ('layer2', []), ('layer3', [])])

    # get intermediate layer outputs
    for k, v in zip(test_outputs.keys(), outputs):
        test_outputs[k].append(v)
    end_inf_time = time.time()
    print(f'inference time:{end_inf_time - start_inf_time}')

    start_fill_time = time.time()
    for k, v in test_outputs.items():
        test_outputs[k] = torch.cat(v, 0)

    # Embedding concat
    embedding_vectors = test_outputs['layer1']
    for layer_name in ['layer2', 'layer3']:
        embedding_vectors = embedding_concat(embedding_vectors, test_outputs[layer_name])

    # randomly select d dimension
    embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
    end_fill_time = time.time()
    print(f'fill vector time:{end_fill_time - start_fill_time}')

    start_dist_time = time.time()
    # calculate distance matrix
    B, C, H, W = embedding_vectors.size()
    # print(f'embedding_vectors:{embedding_vectors.size()}')
    embedding_vectors = embedding_vectors.view(B, C, H * W)
    dist_list = []

    # print(f"train_outputs[0]:{train_outputs[0].shape}")
    # print(f"train_outputs[1]:{train_outputs[1].shape}")
    # print(f"embedding_vectors:{embedding_vectors.shape}")
    mean_matrix = torch.from_numpy(train_outputs[0]).to(device)
    embedding_vectors_zero_mean = (embedding_vectors - mean_matrix).permute(0, 2, 1)
    cov_inv_matrix = torch.from_numpy(train_outputs[1]).permute(2, 0, 1).to(device)

    # print(f"embedding_vectors_zero_mean:{embedding_vectors_zero_mean.shape}")
    # print(f"cov_inv_matrix:{cov_inv_matrix.shape}")
    mul_mat = torch.einsum('bcd,abd->abc', cov_inv_matrix, embedding_vectors_zero_mean)
    dist_list = torch.einsum('abc,abc->ab', embedding_vectors_zero_mean, mul_mat)
    dist_list = dist_list.reshape(B, H, W).cpu().detach().numpy()
    end_inf_time = time.time()
    print(f'calculate distance time:{end_inf_time - start_dist_time}')

    # upsample
    dist_list = torch.tensor(dist_list)
    score_map = F.interpolate(dist_list.unsqueeze(1), size=224, mode='bilinear',
                              align_corners=False).squeeze().numpy()

    heatmap_image = np.zeros((img_size * len(grid_hs_list), img_size * len(grid_ws_list)))
    grid_count = 0
    for j in range(len(grid_hs_list)):
        for i in range(len(grid_ws_list)):
            heatmap_image[j * img_size: (j + 1) * img_size,
            i * img_size: (i + 1) * img_size] = score_map[grid_count]
            grid_count += 1

    # Normalization
    eta = 1e-10
    max_score = heatmap_image.max()
    min_score = heatmap_image.min()
    heatmap_image = (heatmap_image - min_score) / (max_score - min_score + eta)

    # heatmap_image = cv2.GaussianBlur(heatmap_image, (5,5), 0)
    heatmap_image = (heatmap_image * 255).astype('uint8')
    heatmap_image = cv2.resize(heatmap_image, (split_format[0], split_format[1]))
    heatmap_image = cv2.bitwise_and(heatmap_image, resize_roi_image)

    # Normalization again
    eta = 1e-10
    max_score = heatmap_image.max()
    min_score = heatmap_image.min()
    heatmap_image = (heatmap_image - min_score) / (max_score - min_score + eta)
    heatmap_image = (heatmap_image * 255).astype('uint8')
    heatmap_resize_image = cv2.resize(heatmap_image, (split_format[0] // 1, split_format[1] // 1))
    # image_max is the dilation of im with a 20*20 structuring element
    # It is used within peak_local_max function
    image_max = maximum_filter(heatmap_resize_image, size=block_size, mode='constant')

    threshold = np.mean(image_max[np.nonzero(image_max)]) \
        if np.count_nonzero(image_max) > 0 else 0
    print(f'threshold:{threshold}')
    _, heatmap_thres = cv2.threshold(heatmap_image, int(threshold), 255, cv2.THRESH_BINARY)

    ekernel = np.ones((7, 7), np.uint8)
    dkernel = np.ones((7, 7), np.uint8)

    img_erosion = cv2.erode(heatmap_thres, ekernel, iterations=1)
    img_dilation = cv2.dilate(img_erosion, dkernel, iterations=1)

    heatmap_thres_cond = img_dilation.copy()
    connected_outputs = cv2.connectedComponentsWithStats(heatmap_thres_cond, 8, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = connected_outputs
    for i in range(0, numLabels):
        if i == 0:
            continue

        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if np.log(w * h) < bbox_area_mean - 0 * bbox_area_std or \
                bbox_wh_ratio_mean - 1 * bbox_wh_ratio_std > w / h:
            # if np.log(w * h) < bbox_area_mean:
            heatmap_thres_cond[y: y + h, x: x + w] = 0

    mask_total_image = cv2.cvtColor(heatmap_thres_cond, cv2.COLOR_GRAY2BGR)
    blend_image = cv2.addWeighted(resize_image, 0.5, mask_total_image, 0.5, 0)

    end_total_time = time.time()
    print(f'total time:{end_total_time - start_total_time}')

    if show:
        cv2.imshow('image', resize_image)
        cv2.imshow('gt image', resize_gt_image)
        cv2.imshow('ROI image', resize_roi_image)
        cv2.imshow('heatmap', heatmap_image)
        cv2.imshow('heatmap maximum', image_max)
        cv2.imshow('heatmap threshold', heatmap_thres)
        cv2.imshow('heatmap threshold er_dl', img_dilation)
        cv2.imshow('blend_image', blend_image)
        cv2.waitKey(0)

    return mask_total_image


def runYoloDetection(model, image, show=True):
    start_det_time = time.time()
    results = list(model.predict(image,
                                 save=False,
                                 imgsz=1024,
                                 conf=0.329,
                                 iou=0.5,
                                 show_labels=False,
                                 name='yolov8n_test_one_image'))
    end_det_time = time.time()
    print(f'det time:{end_det_time - start_det_time}')

    result = results[0]
    mask_total_image = np.zeros_like(image)
    if hasattr(result.masks, "data"):
        mask_total_image = np.zeros(result.masks.data.shape[1:]).astype('uint8')
        # print(mask_total_image.shape)
        for i in range(result.masks.data.shape[0]):
            mask_image = result.masks.data[i].cpu().detach().numpy() * 255
            mask_image = mask_image.astype('uint8')
            # print(mask_image.shape)

            mask_total_image = cv2.bitwise_or(mask_total_image, mask_image)

            # nonzero_index = (mask_image == 1).nonzero()
            # print(nonzero_index)

        mask_total_image = cv2.cvtColor(mask_total_image, cv2.COLOR_GRAY2BGR)
        blend_image = cv2.addWeighted(image, 0.5, mask_total_image, 0.5, 0)

        if show:
            cv2.imshow("image", image)
            cv2.imshow("blend_image", blend_image)
            cv2.imshow("mask", mask_total_image)
            cv2.waitKey(0)

    return mask_total_image


def parse_args():
    parser = argparse.ArgumentParser('PaDiM')
    parser.add_argument('--data_path', type=str, default='D:/dataset/mvtec_anomaly_detection')
    parser.add_argument('--train_feature_filepath', type=str, default='train_potholes.pkl')
    parser.add_argument('--yolo_weight_path', type=str, default='train_potholes.pkl')
    parser.add_argument('--save_path', type=str, default='./sim_potholes_result')
    parser.add_argument('--arch', type=str, choices=['resnet18', 'wide_resnet50_2'], default='wide_resnet50_2')
    parser.add_argument("--ensemble", default=False, help="ensemble all datection",
                        action="store_true")
    return parser.parse_args()


def main():
    # device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parse_args()

    # load model
    if args.arch == 'resnet18':
        t_d = 448
        d = 100
    elif args.arch == 'wide_resnet50_2':
        t_d = 1792
        d = 550
    else:
        print('arch not support')
        return 0

    if not os.path.isfile(args.yolo_weight_path):
        print('can not load yolo weight from: %s' % args.yolo_weight_path)
        return 0

    yolo_model = YOLO(args.yolo_weight_path)

    if not os.path.isfile(args.train_feature_filepath):
        print('can not load training features from: %s' % args.train_feature_filepath)
        return 0

    print('load train set feature from: %s' % args.train_feature_filepath)
    with open(args.train_feature_filepath, 'rb') as f:
        train_outputs = pickle.load(f)

    model = CustomResnet(args.arch, pretrained=True)
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    if use_cuda:
        torch.cuda.manual_seed_all(1024)

    idx = torch.tensor(sample(range(0, t_d), d)).to(device)

    os.makedirs(args.save_path, exist_ok=True)

    # grid method
    split_format = [1024, 640, 4, 2]
    img_size = 224
    block_size = 28
    bbox_area_mean = 7.4454131512885615
    bbox_area_std = 1.6787692863341874
    bbox_wh_ratio_mean = 1.8448105276099263
    bbox_wh_ratio_std = 0.8495229148189919

    # create video writer
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    image_size = (split_format[0], split_format[1])
    video_writer = cv2.VideoWriter(os.path.join(args.save_path, 'output.mp4'),
                                  fourcc,
                                  20.0,
                                  image_size)

    gt_mask_list = []
    pred_mask_list = []

    for dirPath, dirNames, fileNames in os.walk(args.data_path):
        for f in fileNames:
            img_path = os.path.join(dirPath, f)
            image = cv2.imread(img_path)
            gt_image = cv2.imread(img_path.replace('/images', '/gt').replace(".jpg", ".png"), 0)
            roi_image = cv2.imread(img_path.replace('/images', '/ROI').replace(".jpg", ".png"), 0)

            resize_image = cv2.resize(image, (split_format[0], split_format[1]))
            resize_gt_image = cv2.resize(gt_image, (split_format[0], split_format[1]))
            resize_roi_image = cv2.resize(roi_image, (split_format[0], split_format[1]))

            _, resize_gt_image = cv2.threshold(resize_gt_image, 127, 255, cv2.THRESH_BINARY)

            # multi-thread
            anomaly_thread = AnomalyDetectionThread(device,
                                    idx,
                                    model, train_outputs,
                                    split_format,
                                    resize_image, resize_gt_image, resize_roi_image,
                                    img_size,
                                    block_size,
                                    bbox_area_mean, bbox_area_std,
                                    bbox_wh_ratio_mean, bbox_wh_ratio_std)
            yolo_thread = YoloDetectionThread(yolo_model, resize_image)

            run_start_time = time.time()
            anomaly_thread.start()
            yolo_thread.start()
            anomaly_thread.join()
            yolo_thread.join()

            if args.ensemble:
                mask_total_image = cv2.bitwise_or(anomaly_thread.mask_img, yolo_thread.mask_img)
            else:
                mask_total_image = yolo_thread.mask_img
            blend_image = cv2.addWeighted(resize_image, 0.5, mask_total_image, 0.5, 0)

            run_end_time = time.time()
            print(f'run thread time:{run_end_time - run_start_time}')

            mask_total_image = cv2.cvtColor(mask_total_image, cv2.COLOR_BGR2GRAY)
            _, mask_total_image = cv2.threshold(mask_total_image, 127, 255, cv2.THRESH_BINARY)
            gt_mask_list.append(resize_gt_image)
            pred_mask_list.append(mask_total_image)

            cv2.imshow("image", resize_image)
            cv2.imshow("blend_image", blend_image)
            cv2.imshow("mask", mask_total_image)
            cv2.imshow("gt", resize_gt_image)
            cv2.waitKey(1)

            # write to video
            run_fps = 1. / (run_end_time - run_start_time + 1e-10)
            text = "fps:{:.4f}".format(run_fps)
            cv2.putText(blend_image, text, (10, 10),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            video_writer.write(blend_image)

            # single thread
            # runAnomalyDetection(device,
            #                     idx,
            #                     model, train_outputs,
            #                     split_format,
            #                     resize_image, resize_gt_image, resize_roi_image,
            #                     img_size,
            #                     block_size,
            #                     bbox_area_mean, bbox_area_std,
            #                     bbox_wh_ratio_mean, bbox_wh_ratio_std)

            # runYoloDetection(yolo_model, resize_image)

    video_writer.release()

    gt_mask = np.asarray(gt_mask_list) / 255.0
    pred_mask = np.asarray(pred_mask_list) / 255.0

    # calculate per-pixel level PR
    per_pixel_cm = confusion_matrix(gt_mask.flatten(), pred_mask.flatten())
    per_pixel_precision = precision_score(gt_mask.flatten(), pred_mask.flatten(), zero_division=0)
    per_pixel_recall = recall_score(gt_mask.flatten(), pred_mask.flatten())
    per_pixel_f1 = f1_score(gt_mask.flatten(), pred_mask.flatten())
    print('confusion matrix: \n', per_pixel_cm)
    print('precision: %.3f' % per_pixel_precision)
    print('recall: %.3f' % per_pixel_recall)
    print('f1: %.3f' % per_pixel_f1)

    # calculate per-pixel level ROCAUC
    fpr, tpr, _ = roc_curve(gt_mask.flatten(), pred_mask.flatten())
    per_pixel_auc = roc_auc_score(gt_mask.flatten(), pred_mask.flatten())
    print('pixel ROCAUC: %.3f' % per_pixel_auc)
    print('pixel fpr:', fpr)
    print('pixel tpr:', tpr)

    plt.figure('ROC Curve')
    plt.plot(fpr, tpr, label='ROC_AUC: %.3f' % per_pixel_auc)
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend()
    plt.savefig(os.path.join(args.save_path, 'roc_curve.png'), dpi=100)


if __name__ == '__main__':
    main()

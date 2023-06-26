import glob
import cv2
import os
import pickle
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm


# Function to convert bounding boxes in YOLO format to xmin, ymin, xmax, ymax.
def yolo2bbox(bboxes):
    xmin, ymin = bboxes[0]-bboxes[2]/2, bboxes[1]-bboxes[3]/2
    xmax, ymax = bboxes[0]+bboxes[2]/2, bboxes[1]+bboxes[3]/2
    return xmin, ymin, xmax, ymax


def plot_box(image, bboxes, labels):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    for box_num, box in enumerate(bboxes):
        x1, y1, x2, y2 = yolo2bbox(box)
        # Denormalize the coordinates.
        xmin = int(x1 * w)
        ymin = int(y1 * h)
        xmax = int(x2 * w)
        ymax = int(y2 * h)

        thickness = max(2, int(w / 275))

        cv2.rectangle(
            image,
            (xmin, ymin), (xmax, ymax),
            color=(0, 0, 255),
            thickness=thickness
        )
    return image


def save_ROI(image, bbox, save_path):
    # Need the image height and width to denormalize
    # the bounding box coordinates
    h, w, _ = image.shape
    x1, y1, x2, y2 = yolo2bbox(bbox)
    # Denormalize the coordinates.
    xmin = int(x1 * w)
    ymin = int(y1 * h)
    xmax = int(x2 * w)
    ymax = int(y2 * h)

    xmin = xmin if xmin > 0 else 0
    xmin = xmin if xmin < w else w - 1
    ymin = ymin if ymin > 0 else 0
    ymin = ymin if ymin < h else h - 1
    xmax = xmax if xmax > 0 else 0
    xmax = xmax if xmax < w else w - 1
    ymax = ymax if ymax > 0 else 0
    ymax = ymax if ymax < h else h - 1

    if xmin < xmax and ymin < ymax:
        img_crop = image[ymin: ymax, xmin: xmax]
        cv2.imwrite(save_path, img_crop)


def getData(image_paths, label_paths):
    all_images = []
    all_images.extend(glob.glob(image_paths + '/*.jpg'))
    all_images.extend(glob.glob(image_paths + '/*.JPG'))
    all_images.extend(glob.glob(image_paths + '/*.png'))

    all_images.sort()

    num_images = len(all_images)

    box_data_dict = {'area': [], 'wh_ratio': []}

    for i in range(num_images):
        image_name = all_images[i]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[i])
        img_h, img_w, _ = image.shape

        if not os.path.isfile(os.path.join(label_paths, image_name + '.txt')):
            continue

        with open(os.path.join(label_paths, image_name + '.txt'), 'r') as f:
            bboxes = []
            labels = []
            label_lines = f.readlines()
            for label_line in label_lines:
                label = label_line[0]
                bbox_string = label_line[2:]
                x_c, y_c, w, h = bbox_string.split(' ')
                x_c = float(x_c)
                y_c = float(y_c)
                w = float(w)
                h = float(h)

                if w * h <= 0:
                    continue

                box_data_dict['area'].append(w * img_w * h * img_h)
                box_data_dict['wh_ratio'].append((w * img_w) / (h * img_h))
                bboxes.append([x_c, y_c, w, h])
                labels.append(label)
        result_image = plot_box(image, bboxes, labels)
        cv2.imshow("gt image", result_image)
        if cv2.waitKey(1) == ord('q'):
            break

    return box_data_dict


def saveImageROI(image_paths,
                 label_paths,
                 gt_paths,
                 save_normal_folder1, save_normal_folder2,
                 save_anomaly_folder,
                 save_gt_folder,
                 data_dict,
                 bbox_cond=False,
                 split_format=None,
                 bbox_scale=1):
    os.makedirs(save_normal_folder1, exist_ok=True)
    os.makedirs(save_normal_folder2, exist_ok=True)
    os.makedirs(save_anomaly_folder, exist_ok=True)

    all_images = []
    all_images.extend(glob.glob(image_paths + '/*.jpg'))
    all_images.extend(glob.glob(image_paths + '/*.JPG'))
    all_images.extend(glob.glob(image_paths + '/*.png'))

    all_images.sort()

    num_images = len(all_images)

    area_mean = data_dict['area_mean']
    area_std = data_dict['area_std']
    wh_ratio_mean = data_dict['wh_ratio_mean']
    wh_ratio_std = data_dict['wh_ratio_std']

    total_labels = 0
    valid_labels = 0

    for i in tqdm(range(num_images)):
        image_name = all_images[i]
        image_name = '.'.join(image_name.split(os.path.sep)[-1].split('.')[:-1])
        image = cv2.imread(all_images[i])
        img_h, img_w, _ = image.shape

        if not os.path.isfile(os.path.join(label_paths, image_name + '.txt')):
            continue

        if split_format is not None:
            gt_image = cv2.imread(all_images[i].replace(image_paths, gt_paths).replace(".jpg", ".png"))
            resize_image = cv2.resize(image, (split_format[0], split_format[1]))
            resize_gt_image = cv2.resize(gt_image, (split_format[0], split_format[1]))

            grid_w = split_format[0] // split_format[2]
            grid_h = split_format[1] // split_format[3]
            grid_ws_list= [grid_w * i for i in range(split_format[2])]
            grid_hs_list = [grid_h * j for j in range(split_format[3])]

            save_path = all_images[i].replace(".jpg", "")
            save_path = save_path.replace(".JPG", "")
            save_path = save_path.replace(".png", "")

            g_count = 0
            for gh in grid_hs_list:
                for gw in grid_ws_list:
                    grid_img = resize_image[gh: gh + grid_h, gw: gw + grid_w, :].copy()
                    grid_gt_img = resize_gt_image[gh: gh + grid_h, gw: gw + grid_w, :].copy()

                    # save to test/good
                    if np.count_nonzero(grid_gt_img) == 0:
                        save_normal_path2 = save_path.replace(image_paths, save_normal_folder2)
                        save_normal_path2 += f'_{g_count}.png'
                        cv2.imwrite(save_normal_path2, grid_img)
                    # save to test/abnormal
                    else:
                        save_anomaly_path = save_path.replace(image_paths, save_anomaly_folder)
                        save_anomaly_path += f'_{g_count}.png'
                        cv2.imwrite(save_anomaly_path, grid_img)
                        if save_gt_folder is not None:
                            os.makedirs(save_gt_folder, exist_ok=True)
                            save_gt_path = save_path.replace(image_paths, save_gt_folder)
                            save_gt_path += f'_{g_count}.png'
                            cv2.imwrite(save_gt_path, grid_gt_img)

                    g_count += 1

        else:
            with open(os.path.join(label_paths, image_name + '.txt'), 'r') as f:
                bboxes = []
                labels = []
                label_lines = f.readlines()
                for li, label_line in enumerate(label_lines):
                    label = label_line[0]
                    bbox_string = label_line[2:]
                    x_c, y_c, w, h = bbox_string.split(' ')
                    x_c = float(x_c)
                    y_c = float(y_c)
                    w = float(w)
                    h = float(h)

                    total_labels += 1

                    if w * h <= 0:
                        continue
                    # elif np.log(w * img_w * h * img_h) >= area_mean + 0 * area_std \
                    #     and wh_ratio_mean - 1 * wh_ratio_std < (w * img_w) / (h * img_h) < wh_ratio_mean + 1 * wh_ratio_std:
                    elif np.log(w * img_w * h * img_h) >= area_mean + 0 * area_std \
                         and 0.5 < (w * img_w) / (h * img_h) < 2.0\
                         or not bbox_cond:
                        # print(w * img_w / h * img_h)
                        # print(np.log(w * img_w * h * img_h))
                        save_path = all_images[i].replace(".jpg", "")
                        save_path = save_path.replace(".JPG", "")
                        save_path = save_path.replace(".png", "")

                        if bbox_scale != 1:
                            new_w = max(w, h)
                            new_h = max(w, h)
                        else:
                            new_w = w
                            new_h = h

                        save_anomaly_path = save_path.replace(image_paths, save_anomaly_folder)
                        save_anomaly_path += f'_{li}.png'
                        save_ROI(image, [x_c, y_c, new_w * bbox_scale, new_h * bbox_scale], save_anomaly_path)
                        # saved neighbor as good example
                        save_normal_path1 = save_path.replace(image_paths, save_normal_folder1)
                        save_normal_path1 += f'_{li}.png'
                        save_ROI(image, [x_c - w, y_c, w, h], save_normal_path1)

                        save_normal_path2 = save_path.replace(image_paths, save_normal_folder2)
                        save_normal_path2 += f'_{li}.png'
                        save_ROI(image, [x_c + w, y_c, w, h], save_normal_path2)

                        if save_gt_folder is not None:
                            os.makedirs(save_gt_folder, exist_ok=True)
                            gt_image = cv2.imread(all_images[i].replace(image_paths, gt_paths).replace(".jpg", ".png"))
                            save_gt_path = save_path.replace(image_paths, save_gt_folder)
                            save_gt_path += f'_{li}.png'
                            save_ROI(gt_image, [x_c , y_c, new_w * bbox_scale, new_h * bbox_scale], save_gt_path)

                        valid_labels += 1

    print(f'valid_labels/total_labels: {valid_labels}/{total_labels}')


def analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  bbox_cond,
                  split_format=None,
                  bbox_scale=1):
    data_dict = getData(
        image_paths=os.path.join(dataset_folder, img_folder),
        label_paths=os.path.join(dataset_folder, label_folder)
    )
    with open(os.path.join(analysis_folder, 'box_data_dict.pickle'), 'wb') as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(analysis_folder, 'box_data_dict.pickle'), 'rb') as handle:
        data_dict = pickle.load(handle)

    data_new_dict = data_dict.copy()
    for k in data_dict.keys():
        data = np.array(data_dict[k])

        if k == 'area':
            data = np.log(data)

        mean = np.mean(data)
        std = np.std(data)
        print(f'{k} mean: {mean}')
        print(f'{k} std: {std}')

        data_new_dict[f'{k}_mean'] = mean
        data_new_dict[f'{k}_std'] = std

        plt.figure(f'box {k}')
        sns.histplot(data=data)
        plt.savefig(os.path.join(analysis_folder, f'box_{k}.png'))

    saveImageROI(
        os.path.join(dataset_folder, img_folder),
        os.path.join(dataset_folder, label_folder),
        os.path.join(dataset_folder, gt_folder),
        save_normal_folder1,
        save_normal_folder2,
        save_anomaly_folder,
        save_gt_folder,
        data_new_dict,
        bbox_cond,
        split_format,
        bbox_scale
    )


def main():
    analysis_folder = "pothole_dataset_v8_analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    img_folder = 'pothole_dataset_v8/train/images/'
    label_folder = 'pothole_dataset_v8/train/labels/'
    gt_folder = ''

    save_anomaly_folder = os.path.join(dataset_folder, 'pothole_dataset_v8/potholes_roi/test/potholes/')
    save_normal_folder1 = os.path.join(dataset_folder, 'pothole_dataset_v8/potholes_roi/train/potholes_neighbor/')
    save_normal_folder2 = os.path.join(dataset_folder, 'pothole_dataset_v8/potholes_roi/test/potholes_neighbor/')
    save_gt_folder = None

    analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  bbox_cond=True
                  )

    analysis_folder = "pothole600_analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    img_folder = 'pothole600/training/images/'
    label_folder = 'pothole600/training/labels/'
    gt_folder = 'pothole600/training/gt/'

    # saved bbox of potholes' neighbor as good samples
    save_anomaly_folder = os.path.join(dataset_folder, 'pothole600/training/potholes_roi/test/potholes/')
    save_normal_folder1 = os.path.join(dataset_folder, 'pothole600/training/potholes_roi/train/good/')
    save_normal_folder2 = os.path.join(dataset_folder, 'pothole600/training/potholes_roi/test/good/')
    save_gt_folder = os.path.join(dataset_folder, 'pothole600/training/potholes_roi/ground_truth/potholes/')

    analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  bbox_cond=False
                  )

    analysis_folder = "Pothole.v1-raw.yolov8_analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    img_folder = 'Pothole.v1-raw.yolov8/train/images/'
    label_folder = 'Pothole.v1-raw.yolov8/train/labels/'
    gt_folder = ''

    # saved bbox of potholes' neighbor as good samples
    save_anomaly_folder = os.path.join(dataset_folder, 'Pothole.v1-raw.yolov8/train/potholes_roi/test/potholes/')
    save_normal_folder1 = os.path.join(dataset_folder, 'Pothole.v1-raw.yolov8/train/potholes_roi/train/potholes_neighbor/')
    save_normal_folder2 = os.path.join(dataset_folder, 'Pothole.v1-raw.yolov8/train/potholes_roi/test/potholes_neighbor/')
    save_gt_folder = None

    analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  bbox_cond=True
                  )

    # Cracks-and-Potholes-in-Road-Images-Dataset ROI mask
    analysis_folder = "Cracks_and_Potholes_in_Road_dataset_analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    img_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/ROI_mask/'
    label_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/labels/'
    gt_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/gt/'

    save_anomaly_folder = os.path.join(dataset_folder, 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi_mask/test/potholes/')
    save_normal_folder1 = os.path.join(dataset_folder, 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi_mask/train/')
    save_normal_folder2 = os.path.join(dataset_folder, 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi_mask/test/good/')
    save_gt_folder = os.path.join(dataset_folder, 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi_mask/ground_truth/potholes/')
    split_format = [1024, 640, 4, 2]

    analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  False,
                  split_format
                  )

    # Cracks-and-Potholes-in-Road-Images-Dataset ROI
    analysis_folder = "Cracks_and_Potholes_in_Road_dataset_analysis"
    os.makedirs(analysis_folder, exist_ok=True)

    dataset_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets"
    img_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/images/'
    label_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/labels/'
    gt_folder = 'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/gt/'

    save_anomaly_folder = os.path.join(dataset_folder,
                                       'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi/test/potholes/')
    save_normal_folder1 = os.path.join(dataset_folder,
                                       'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi/train/potholes_neighbor/')
    save_normal_folder2 = os.path.join(dataset_folder,
                                       'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi/test/potholes_neighbor/')
    save_gt_folder = os.path.join(dataset_folder,
                                  'biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_roi/ground_truth/potholes/')

    analysis_bbox(analysis_folder,
                  dataset_folder, img_folder, label_folder, gt_folder,
                  save_anomaly_folder,
                  save_normal_folder1, save_normal_folder2,
                  save_gt_folder,
                  bbox_cond=True,
                  bbox_scale=2
                  )


if __name__ == '__main__':
    main()
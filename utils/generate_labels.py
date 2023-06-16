import os
import cv2


def generate_labels(data_folder, labels_folder):
    for dirPath, dirNames, fileNames in os.walk(data_folder):
        for f in fileNames:
            image_path = os.path.join(dirPath, f)
            label_path = image_path.replace(".jpg", "")
            label_path = label_path.replace(".JPG", "")
            label_path = label_path.replace(".png", "")
            label_path = label_path.replace(data_folder, labels_folder)
            image = cv2.imread(image_path)
            imgh, imgw, _ = image.shape
            show_img = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            connected_outputs = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)
            (numLabels, labels, stats, centroids) = connected_outputs

            with open(label_path + '.txt', 'w') as f:
                for i in range(0, numLabels):
                    if i == 0:
                        continue
                    # if this is the first component then we examine the
                    # *background* (typically we would just ignore this
                    # component in our loop)
                    # if i == 0:
                    #     text = "examining component {}/{} (background)".format(
                    #         i + 1, numLabels)
                    # # otherwise, we are examining an actual connected component
                    # else:
                    #     text = "examining component {}/{}".format(i + 1, numLabels)
                    # print a status message update for the current connected
                    # component
                    # print("[INFO] {}".format(text))
                    # extract the connected component statistics and centroid for
                    # the current label
                    x = stats[i, cv2.CC_STAT_LEFT]
                    y = stats[i, cv2.CC_STAT_TOP]
                    w = stats[i, cv2.CC_STAT_WIDTH]
                    h = stats[i, cv2.CC_STAT_HEIGHT]
                    area = stats[i, cv2.CC_STAT_AREA]
                    (cX, cY) = centroids[i]
                    cv2.rectangle(show_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    f.write(f'0 {cX/imgw} {cY/imgh} {w/imgw} {h/imgh}\n')

                cv2.imshow("bbox", show_img)
                if cv2.waitKey(1) == ord('q'):
                    break


def main():
    data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/training/gt"
    labels_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/training/labels"
    os.makedirs(labels_folder, exist_ok=True)
    generate_labels(data_folder, labels_folder)

    data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/testing/gt"
    labels_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/testing/labels"
    os.makedirs(labels_folder, exist_ok=True)
    generate_labels(data_folder, labels_folder)

    data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/validation/gt"
    labels_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/validation/labels"
    os.makedirs(labels_folder, exist_ok=True)
    generate_labels(data_folder, labels_folder)


if __name__ == '__main__':
    main()
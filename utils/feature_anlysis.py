import torch
import torch.nn.functional as F
import cv2
import numpy as np
import os
import random
from random import sample
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import KernelPCA, PCA
from matplotlib import pyplot as plt
import umap

from resnet import CustomResnet


def to_Tensor(image_path, img_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    image = cv2.imread(image_path)
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


def dimension_reduction(data_folder, device, model, idx, save_folder, figure_name):
    embedding_list = []
    for dirPath, dirNames, fileNames in os.walk(data_folder):
        for f in fileNames:
            image_path = os.path.join(dirPath, f)
            img_tensor = to_Tensor(image_path, img_size=224).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
            embedding_vectors = outputs[0]
            for li in range(1, 3):
                embedding_vectors = embedding_concat(embedding_vectors, outputs[li])
            embedding_vectors = torch.index_select(embedding_vectors, 1, idx)
            embedding_list.append(embedding_vectors.reshape(-1).cpu().detach().numpy())

    embedding_list = np.array(embedding_list)
    np.save(os.path.join(save_folder, f'{figure_name}_embedding_features.npy'), embedding_list)
    embedding_list = np.load(os.path.join(save_folder, f'{figure_name}_embedding_features.npy'))
    print(f'embedding_list size: {embedding_list.shape}')

    # TSNE
    embedded_tsne = TSNE(n_components=2).fit_transform(embedding_list)
    np.save(os.path.join(save_folder, f'{figure_name}_tsne_features.npy'), embedded_tsne)

    # UMAP
    reducer = umap.UMAP()
    embedded_umap = reducer.fit_transform(embedding_list)
    np.save(os.path.join(save_folder, f'{figure_name}_umap_features.npy'), embedded_umap)

    # KernelPCA
    transformer = KernelPCA(n_components=2, kernel='poly')
    embedded_kpca = transformer.fit_transform(embedding_list)
    np.save(os.path.join(save_folder, f'{figure_name}_kpca_features.npy'), embedded_kpca)

    embedded_tsne = np.load(os.path.join(save_folder, f'{figure_name}_tsne_features.npy'))
    plt.figure(figure_name + '_tsne')
    plt.scatter(embedded_tsne[:, 0], embedded_tsne[:, 1])
    plt.savefig(os.path.join(save_folder, f'{figure_name}_tsne.jpg'))

    embedded_umap = np.load(os.path.join(save_folder, f'{figure_name}_umap_features.npy'))
    plt.figure(figure_name + '_umap')
    plt.scatter(embedded_umap[:, 0], embedded_umap[:, 1])
    plt.savefig(os.path.join(save_folder, f'{figure_name}_umap.jpg'))

    embedded_kpca = np.load(os.path.join(save_folder, f'{figure_name}_umap_features.npy'))
    plt.figure(figure_name + '_kpca')
    plt.scatter(embedded_kpca[:, 0], embedded_kpca[:, 1])
    plt.savefig(os.path.join(save_folder, f'{figure_name}_kpca.jpg'))


def dimension_reduction_mix_data(save_folder1, save_folder2,
                                 figure_name1, figure_name2,
                                 save_folder_target, figure_name_target):
    os.makedirs(save_folder_target, exist_ok=True)
    embedding_list1 = np.load(os.path.join(save_folder1, f'{figure_name1}_embedding_features.npy'))
    embedding_list2 = np.load(os.path.join(save_folder2, f'{figure_name2}_embedding_features.npy'))
    embedding_list = np.concatenate([embedding_list1, embedding_list2])

    list1_size = len(embedding_list1)

    # TSNE
    embedded_tsne = TSNE(n_components=2).fit_transform(embedding_list)
    np.save(os.path.join(save_folder_target, f'{figure_name_target}_tsne_features.npy'), embedded_tsne)

    # UMAP
    reducer = umap.UMAP()
    embedded_umap = reducer.fit_transform(embedding_list)
    np.save(os.path.join(save_folder_target, f'{figure_name_target}_umap_features.npy'), embedded_umap)

    # KernelPCA
    transformer = KernelPCA(n_components=2, kernel='poly')
    embedded_kpca = transformer.fit_transform(embedding_list)
    np.save(os.path.join(save_folder_target, f'{figure_name_target}_kpca_features.npy'), embedded_kpca)

    # PCA
    transformer = PCA(n_components=2)
    embedded_pca = transformer.fit_transform(embedding_list)
    np.save(os.path.join(save_folder_target, f'{figure_name_target}_pca_features.npy'), embedded_pca)

    plt.figure(figure_name_target + '_tsne')
    plt.scatter(embedded_tsne[:list1_size, 0], embedded_tsne[:list1_size, 1], color='red')
    plt.scatter(embedded_tsne[list1_size:, 0], embedded_tsne[list1_size:, 1], color='green')
    plt.savefig(os.path.join(save_folder_target, f'{figure_name_target}_tsne.jpg'))

    plt.figure(figure_name_target + '_umap')
    plt.scatter(embedded_umap[:list1_size, 0], embedded_umap[:list1_size, 1], color='red')
    plt.scatter(embedded_umap[list1_size:, 0], embedded_umap[list1_size:, 1], color='green')
    plt.savefig(os.path.join(save_folder_target, f'{figure_name_target}_umap.jpg'))

    plt.figure(figure_name_target + '_kpca')
    plt.scatter(embedded_kpca[:list1_size, 0], embedded_kpca[:list1_size, 1], color='red')
    plt.scatter(embedded_kpca[list1_size:, 0], embedded_kpca[list1_size:, 1], color='green')
    plt.savefig(os.path.join(save_folder_target, f'{figure_name_target}_kpca.jpg'))

    plt.figure(figure_name_target + '_pca')
    plt.scatter(embedded_pca[:list1_size, 0], embedded_pca[:list1_size, 1], color='red')
    plt.scatter(embedded_pca[list1_size:, 0], embedded_pca[list1_size:, 1], color='green')
    plt.savefig(os.path.join(save_folder_target, f'{figure_name_target}_pca.jpg'))


def cluster(embedded_data, figure_name):
    cluster_num = 1
    clustering = KMeans(n_clusters=cluster_num, random_state=0, n_init="auto").fit(embedded_data)
    print(f'cluster centers:{clustering.cluster_centers_}')
    color_list = ['red']
    plt.figure(figure_name)
    # filter rows of original data
    for label, c in zip(range(cluster_num), color_list):
        filtered_embedded_data = embedded_data[clustering.labels_ == label]
        plt.scatter(filtered_embedded_data[:, 0], filtered_embedded_data[:, 1], color=c)
    plt.scatter(clustering.cluster_centers_[:, 0], clustering.cluster_centers_[:, 1], color='green')
    plt.savefig(f'{figure_name}.jpg')


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CustomResnet('resnet18', pretrained=True)
    model.to(device)
    model.eval()
    random.seed(1024)
    torch.manual_seed(1024)
    torch.cuda.manual_seed_all(1024)
    t_d = 448
    d = 100
    idx = torch.tensor(sample(range(0, t_d), d)).to(device)

    # analysis_folder = "mvtec_anomaly_detection_tile_analysis"
    # os.makedirs(analysis_folder, exist_ok=True)
    # tile_train_data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/mvtec_anomaly_detection/tile/train/good"
    # dimension_reduction(tile_train_data_folder,
    #                     device,
    #                     model,
    #                     idx,
    #                     analysis_folder,
    #                     'mvtec_anomaly_detection_tile')
    #
    # analysis_folder = "pothole_dataset_v8_analysis"
    # pothole_train_data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole_dataset_v8/potholes_roi/train/potholes_neighbor"
    # dimension_reduction(pothole_train_data_folder,
    #                     device,
    #                     model,
    #                     idx,
    #                     analysis_folder,
    #                     'pothole_dataset_v8_train_data')

    # analysis_folder = "pothole600_analysis"
    # pothole600_train_data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/training/potholes_roi/train/good"
    # dimension_reduction(pothole600_train_data_folder,
    #                     device,
    #                     model,
    #                     idx,
    #                     analysis_folder,
    #                     'pothole600_train_data')
    #
    # analysis_folder = "pothole600_analysis"
    # pothole600_test_data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/training/potholes_roi/test/potholes"
    # dimension_reduction(pothole600_test_data_folder,
    #                     device,
    #                     model,
    #                     idx,
    #                     analysis_folder,
    #                     'pothole600_test_data')

    # analysis_folder = "Pothole.v1-raw.yolov8_analysis"
    # pothole_v1_train_data_folder = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/Pothole.v1-raw.yolov8/train/potholes_roi/train/potholes_neighbor"
    # dimension_reduction(pothole_v1_train_data_folder,
    #                     device,
    #                     model,
    #                     idx,
    #                     analysis_folder,
    #                     'pothole_v1_train_data')

    # analysis_folder = "pothole600_analysis"
    # dimension_reduction_mix_data(analysis_folder, analysis_folder,
    #                              'pothole600_train_data', 'pothole600_test_data',
    #                              analysis_folder, 'pothole600_data')

    analysis_folder1 = "pothole600_analysis"
    analysis_folder2 = "Pothole.v1-raw.yolov8_analysis"
    analysis_tar_folder = "pothole600_vs_pothole_v1_analysis"
    dimension_reduction_mix_data(analysis_folder1, analysis_folder2,
                                 'pothole600_train_data', 'pothole_v1_train_data',
                                 analysis_tar_folder, 'pothole600_vs_pothole_v1_data')

    # cluster(embedded_feats, 'train_data_clusters')



if __name__ == '__main__':
    main()
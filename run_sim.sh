#python simulate_test_potholes.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/train/images \
#  --train_feature_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/temp_resnet18/train_potholes_roi.pkl \
#  --yolo_weight_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_develop/instance_segment/runs/segment/potholes_yolo_segment2/weights/best.pt \
#  --save_path sim_potholes_train_data_result \
#  --arch resnet18 \
#  --ensemble

#python simulate_test_potholes.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/val/images \
#  --train_feature_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/temp_resnet18/train_potholes_roi.pkl \
#  --train_idx_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/idx.npy \
#  --yolo_weight_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_develop/instance_segment/runs/segment/potholes_yolo_segment2/weights/best.pt \
#  --save_path sim_potholes_val_data_result \
#  --arch resnet18 \
#  --ensemble
#
#python simulate_test_potholes.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/test/images \
#  --train_feature_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/temp_resnet18/train_potholes_roi.pkl \
#  --train_idx_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/idx.npy \
#  --yolo_weight_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_develop/instance_segment/runs/segment/potholes_yolo_segment2/weights/best.pt \
#  --save_path sim_potholes_test_data_result \
#  --arch resnet18 \
#  --ensemble

python simulate_test_potholes.py \
  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/val/images \
  --train_feature_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/temp_resnet18/train_potholes_roi.pkl \
  --train_idx_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/idx.npy \
  --yolo_weight_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_develop/instance_segment/runs/segment/potholes_yolo_segment2/weights/best.pt \
  --save_path sim_potholes_val_data_yolo_only_result \
  --arch resnet18

python simulate_test_potholes.py \
  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054/potholes_yolo_detection/test/images \
  --train_feature_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/temp_resnet18/train_potholes_roi.pkl \
  --train_idx_filepath Cracks-and-Potholes-in-Road-Images-Dataset_result/idx.npy \
  --yolo_weight_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_develop/instance_segment/runs/segment/potholes_yolo_segment2/weights/best.pt \
  --save_path sim_potholes_test_data_yolo_only_result \
  --arch resnet18
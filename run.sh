#python main.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/mvtec_anomaly_detection \
#  --arch resnet18

#python run_potholes.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/pothole600/training \
#  --save_path potholes_result \
#  --arch resnet18
##
#python run_potholes.py \
#  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054 \
#  --save_path Cracks-and-Potholes-in-Road-Images-Dataset_result \
#  --arch resnet18

python run_potholes.py \
  --data_path /media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/ultralytics_datasets/biankatpas-Cracks-and-Potholes-in-Road-Images-Dataset-1f20054 \
  --save_path Cracks-and-Potholes-in-Road-Images-Dataset_result \
  --arch resnet18 \
  --d 125
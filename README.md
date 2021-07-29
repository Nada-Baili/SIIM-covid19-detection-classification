# SIIM-covid19-detection-classification
Problem description
## YOLOv5
The first task is to detect the opacities. To do so, we retrain YOLOv5 on the provided data, as follows:
* Modify the value of the column "_dcm_path_" in the csv files  **input/train.csv** and **input/ss.csv** to match the directory where you store the competition data.
* `cd yolov5`
* Run extract_data.py (`python Extract_data.py`). This script generates JPEG images out of the dcm files and generates labels of the images as text files that fit the required format of YOLOv5. The output will be saved in **yolov5/covid19/images** and **yolov5/covid19/labels**.
* Run split_train_val.py (`python split_train_val.py`) to split the data into training and validation. The output will be saved in **yolov5/covid19/train** and **yolov5/covid19/val**.
* Train YOLOv5 by running the following line: `python train.py --img-size 256 --epochs 35--batch-size 128 --data data/covid19_single.yaml --name "single_class" --single-cls --cfg models/yolov5x.yaml --weights yolov5x.pt --exist-ok`
The best results have been achieved using YOLOvx which is a large model. Other yolov5 versions can also be experienced with.

The training results as well as the trained model's weights will be saved in **yolov5/runs/train**.

## CNN
The second task is to identify the detected opacities as _'atypical', 'indeterminate', 'negative', 'typical'_. To do so, we use a CNN, specifically, we use an efficientnet. We use a 5-fold cross validation training design. To train the CNN, follow these steps:
* Modify the file CNN/params.json with your data directories.
* Run `python Train.py` after specifying the network: efficientnet-b0 or efficientnet-b7, and the loss function: CE (cross-entropy) or FOCAL (focal loss).
The results of the trained CNN will be saved under **CNN/results**. Plots of the evolution of the training and validation loss and accuracy will be saved, as well as the confusion matrix.

### LightGBM
We explore an approach that is based on machine learning to improve the classification results. For each image, we extract a 128-D feature vector from each trained CNN model (of the 5 folds). These feature vectors are extracted from the last layer of the network, right before the classification layer. We also consider the box area and the box aspect ratio (if there are no detections in an image then the box area and the aspect ratio are null). Thus, each image can be represented by a 130-D feature vector that we pass to a lightgbm model for classification.
To get classification results using this approach, run `python ML_model.py` and specify the directory where you stored the 5 checkpoints of the CNN model.

## Inference

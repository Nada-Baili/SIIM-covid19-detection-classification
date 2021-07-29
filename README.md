# SIIM-covid19-detection-classification
## YOLOv5
The first task is to detect the opacities. To do so, we retrain YOLOv5 on the provided data, as follows:
* Modify the value of the column "_dcm_path_" in the csv files  **input/train.csv** and **input/ss.csv** to match the directory where you store the competition data.
* `cd yolov5`
* Run extract_data.py (`python Extract_data.py`). This script generates JPEG images out of the dcm files and generates labels of the images as text files that fit the required format of YOLOv5. The output will be saved in **yolov5/covid19/images** and **yolov5/covid19/labels**.
* Run split_train_val.py (`python split_train_val.py`) to split the data into training and validation. The output will be saved in **yolov5/covid19/train** and **yolov5/covid19/val**.
* Train YOLOv5 by running the following line: `python train.py --img-size 256 --epochs 35--batch-size 128 --data data/covid19_single.yaml --name "single_class" --single-cls --cfg models/yolov5x.yaml --weights yolov5x.pt --exist-ok`
The best results have been achieved using YOLOvx which is a large model. Other yolov5 versions can also be experienced with.

The training results as well as the trained model's weights will be saved in **yolov5/runs/train**.

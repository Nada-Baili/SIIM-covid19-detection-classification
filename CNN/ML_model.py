import os
import argparse
import json
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import prepare_data
from model import *
from sklearn.decomposition import PCA
import lightgbm as lgb

parser = argparse.ArgumentParser()
parser.add_argument('--Models_DIR', default='./results/efficientnet-b0')

def train(models_paths):
    device = "cuda"
    if not os.path.exists(os.path.join(models_paths, "ML_models")):
        os.mkdir(os.path.join(models_paths, "ML_models"))

    models = [p for p in os.listdir(models_paths) if p.endswith("pth")]

    LABEL_LIST = ['atypical', 'indeterminate', 'negative', 'typical']
    num_classes = len(LABEL_LIST)

    with open('params.json', 'r') as f:
        params = json.load(f)

    train_df = pd.read_csv(params["train_df"])

    labels = []
    for id, df in train_df.groupby("id"):
        labels.append(list(df.integer_label)[0])

    X = np.array([os.path.split(e)[1][:-4] for e in train_df.dcm_path.unique()])
    y = np.array(labels)

    for i in range(len(LABEL_LIST)):
        print("{}: {}".format(LABEL_LIST[i], (y==i).sum()))

    skf = StratifiedKFold(n_splits=5, random_state=43, shuffle=True)
    fold_nb = 0
    all_preds = []
    pca = PCA(n_components=128)
    for train_index, val_index in skf.split(X, y):
        fold_nb += 1
        print()
        print("FOLD # "+str(fold_nb))

        train_split = train_df.loc[train_index].reset_index(drop=True)
        val_split = train_df.loc[val_index].reset_index(drop=True)

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_dataset = prepare_data.Data(params,
                                          "",
                                          X_train,
                                          y_train,
                                          transform=prepare_data.get_transforms(data='valid'))
        valid_dataset = prepare_data.Data(params,
                                        "",
                                          X_val,
                                          y_val,
                                          transform=prepare_data.get_transforms(data='valid'))
        train_loader = DataLoader(train_dataset, batch_size=params["BATCH_SIZE"], shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=params["BATCH_SIZE"], shuffle=False)

        weights_path = os.path.join(models_paths, models[fold_nb-1])

        model = EfficientNet.from_pretrained("efficientnet-b0", weights_path=weights_path, advprop=False,
                                             include_top=True, num_classes=len(LABEL_LIST)).to(device)

        model.eval()
        preds = []
        all_features = []
        tk = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, labels) in tk:
            images, labels = images.to(device).float(), labels.to(device).long()

            with torch.no_grad():
                y_preds = model(images)
                features = model.extract_features(images)
                all_features.append(torch.reshape(features, (features.shape[0], -1)).cpu().numpy())

            preds.append(F.softmax(y_preds).cpu().numpy())
        all_features = np.concatenate(all_features)
        reduced_features = pca.fit_transform(all_features)
        df_meta_train = pd.DataFrame(columns = ["f_{}".format(i+1) for i in range(128)], data=reduced_features)
        df_meta_train["BBoxArea"] = (train_split.frac_xmax-train_split.frac_xmin) * (train_split.frac_ymax-train_split.frac_ymin)
        df_meta_train["AspectRatio"] = (train_split.xmax-train_split.xmin) / (train_split.ymax-train_split.ymin)

        model.eval()
        all_features = []
        tk = tqdm(enumerate(valid_loader), total=len(valid_loader))
        for i, (images, labels) in tk:
            images, labels = images.to(device).float(), labels.to(device).long()
            with torch.no_grad():
                y_preds = model(images)
                features = model.extract_features(images)
                all_features.append(torch.reshape(features, (features.shape[0], -1)).cpu().numpy())
        all_features = np.concatenate(all_features)
        reduced_features = pca.fit_transform(all_features)
        df_meta_val = pd.DataFrame(columns = ["f_{}".format(i+1) for i in range(128)], data=reduced_features)
        df_meta_val["BBoxArea"] = (val_split.frac_xmax-val_split.frac_xmin) * (val_split.frac_ymax-val_split.frac_ymin)
        df_meta_val["AspectRatio"] = (val_split.xmax-val_split.xmin) / (val_split.ymax-val_split.ymin)


        train_data = lgb.Dataset(df_meta_train, label=y_train)
        val_data = lgb.Dataset(df_meta_val, label=y_val)

        param = {}
        param['learning_rate'] = 0.03
        param['boosting_type'] = 'gbdt'  # GradientBoostingDecisionTree
        param['objective'] = 'multiclass'  # Multi-class target feature
        param['metric'] = 'multi_logloss'  # metric for multi-class
        param['max_depth'] = 30
        param['num_class'] = 4  # no.of unique values in the target class not inclusive of the end value

        num_round = 100
        clf = lgb.train(param, train_data, num_round, valid_sets=[val_data])
        y_pred = clf.predict(df_meta_val)
        val_acc = (np.argmax(y_pred, axis=1)==y_val).sum()/len(y_val)
        clf.save_model(os.path.join(models_paths, "ML_models/Fold{}_{:.2f}.txt".format(fold_nb,val_acc*100)))

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.Models_DIR)
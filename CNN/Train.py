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
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import Adam
from efficientnet_pytorch import EfficientNet
import prepare_data
import matplotlib.pyplot as plt
from focal_loss.focal_loss import FocalLoss
from sklearn.metrics import confusion_matrix
import seaborn as sn
import collections
from model import *
import torchvision
import torch.nn as nn
from vit_pytorch import ViT
from vit_pytorch.deepvit import DeepViT
from sklearn.metrics import roc_auc_score

if torch.cuda.is_available():
    device = "cuda"
    torch.backends.cudnn.benchmark = True
else:
    device = "cpu"

parser = argparse.ArgumentParser()
parser.add_argument('--Project', default="efficientnet-b0")
parser.add_argument('--Model', default="efficientnet0")
parser.add_argument('--Loss', default="FOCAL")

def train(output_dir, model_name, loss):
    if not os.path.exists("./results"):
        os.mkdir("./results")
    output_path = "./results/{}".format(output_dir)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path+"/plots"):
        os.mkdir(output_path+"/plots")

    LABEL_LIST = ['atypical', 'indeterminate', 'negative', 'typical']
    num_classes = len(LABEL_LIST)
    with open('params.json', 'r') as f:
        params = json.load(f)

    train_df = pd.read_csv(params["train_df"])
    #train_df = train_df[:100]
    labels = []
    for id, df in train_df.groupby("id"):
        labels.append(list(df.integer_label)[0])

    X = np.array([os.path.split(e)[1][:-4] for e in train_df.dcm_path.unique()])
    y = np.array(labels)

    for i in range(len(LABEL_LIST)):
        print("{}: {}".format(LABEL_LIST[i], (y==i).sum()))


    skf = StratifiedKFold(n_splits=5, random_state=43, shuffle=True)
    fold_nb = 0
    for train_index, val_index in skf.split(X, y):
        fold_nb += 1
        print()
        print("FOLD # "+str(fold_nb))

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        train_dataset = prepare_data.Data(params,
                                          loss,
                                          X_train,
                                          y_train,
                                          transform=prepare_data.get_transforms(data='train'))
        valid_dataset = prepare_data.Data(params,
                                          loss,
                                          X_val,
                                          y_val,
                                          transform=prepare_data.get_transforms(data='valid'))
        train_loader = DataLoader(train_dataset, batch_size=params["BATCH_SIZE"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=params["BATCH_SIZE"], shuffle=False)

        if model_name == "efficientnet7":
            model = EfficientNet.from_pretrained("efficientnet-b7", advprop=False, include_top=True, num_classes=num_classes).to(device)
        if model_name == "efficientnet0":
            model = EfficientNet.from_pretrained("efficientnet-b0", advprop=False,
                                                 include_top=True, num_classes=num_classes).to(device)

        optimizer = Adam(model.parameters(), lr=params["LR"])
        scheduler_cos = CosineAnnealingLR(optimizer, params["EPOCHS"], eta_min=0, verbose=False)

        criterion = FocalLoss(alpha=0.25, gamma=2, reduction="mean")

        train_acc_epochs, val_acc_epochs, train_loss_epochs, val_loss_epochs, val_auc_epochs = [], [], [], [], []
        best_acc = 0.
        for epoch in range(params["EPOCHS"]):
            model.train()
            tk0 = tqdm(enumerate(train_loader), total=len(train_loader))
            avg_train_loss, avg_train_acc= 0., 0.
            for i, (train_input, target) in tk0:
                train_input, target = train_input.to(device).float(), target.to(device).long()
                outputs = model(train_input)
                if loss == "CE":
                    train_loss = F.cross_entropy(outputs, target, reduction="mean", ignore_index=-1)
                    avg_train_acc += (outputs.max(1)[1] == target).sum()/len(train_dataset)
                elif loss == "FOCAL":
                    y_preds = torch.sigmoid(outputs)
                    train_loss = criterion(y_preds, target)
                    avg_train_acc += (outputs.max(1)[1] == target.max(1)[1]).sum()/len(train_dataset)

                avg_train_loss += train_loss.item() / len(train_loader)
                
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            train_loss_epochs.append(avg_train_loss)
            train_acc_epochs.append(avg_train_acc.item())

            model.eval()
            avg_val_loss, avg_val_acc = 0., 0.
            y_true, y_pred = [], []
            y_pred_norm= []

            tk1 = tqdm(enumerate(valid_loader), total=len(valid_loader))
            for i, (images, labels) in tk1:
                images = images.to(device).float()
                labels = labels.to(device).long()
                
                with torch.no_grad():
                    outputs = model(images)
                if loss == "CE":
                    val_loss = F.cross_entropy(outputs, labels, reduction="mean", ignore_index=-1)
                    avg_val_acc += (outputs.max(1)[1] == labels).sum() / len(valid_dataset)
                    y_true.append(labels.cpu().numpy())
                elif loss =="FOCAL":
                    y_preds = torch.sigmoid(outputs)
                    val_loss = criterion(y_preds, labels)
                    avg_val_acc += (y_preds.max(1)[1] == labels.max(1)[1]).sum() / len(valid_dataset)
                    y_true.append(labels.max(1)[1].cpu().numpy())
                y_pred.append(outputs.max(1)[1].cpu().numpy())
                y_pred_norm.append(F.softmax(outputs, dim=1).cpu().numpy())

                avg_val_loss += val_loss.item() / len(valid_loader)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            y_pred_norm = np.concatenate(y_pred_norm)

            scheduler_cos.step()

            val_loss_epochs.append(avg_val_loss)
            val_acc_epochs.append(avg_val_acc.item())

            auc = roc_auc_score(y_true, y_pred_norm, multi_class="ovr")
            val_auc_epochs.append(auc)

            metric = auc

            if best_acc < metric:
                best_acc = metric
                best_model = model.state_dict()

            cm = confusion_matrix(y_true, y_pred)
            cm = 100*cm / cm.sum(axis=1).reshape((-1,1))
            df_cm = pd.DataFrame(cm, index=[i for i in LABEL_LIST],
                                 columns=[i for i in LABEL_LIST])
            plt.figure(figsize=(10, 7))
            sn.heatmap(df_cm, annot=True)
            plt.ylim(0,len(LABEL_LIST))
            plt.savefig(output_path + '/plots/Fold#{}_ConfMat.png'.format(fold_nb))

            plt.figure()
            _ = plt.plot(train_loss_epochs, label="Train")
            _ = plt.plot(val_loss_epochs, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.legend()
            plt.savefig(output_path + '/plots/Fold#{}_Loss.png'.format(fold_nb))

            plt.figure()
            _ = plt.plot(val_auc_epochs, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("AUC")
            plt.legend()
            plt.savefig(output_path + '/plots/Fold#{}_AUC.png'.format(fold_nb))

            plt.figure()
            _ = plt.plot(train_acc_epochs, label="Train")
            _ = plt.plot(val_acc_epochs, label="Val")
            plt.xlabel("Epochs")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.savefig(output_path + '/plots/Fold#{}_Acc.png'.format(fold_nb))
        torch.save(best_model,
                   os.path.join(output_path, 'FOLD{}_{:.2f}.pth'.format(fold_nb, best_acc*100)))

if __name__ == '__main__':
    args = parser.parse_args()
    train(args.Project, args.Model, args.Loss)
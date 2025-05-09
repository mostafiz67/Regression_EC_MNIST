"""
Author: Md Mostafziur Rahman
File: Traning a CNN architecture using the MNIST dataset
Some code is inherited from https://github.com/christianversloot/machine-learning-articles/blob/main/how-to-use-k-fold-cross-validation-with-pytorch.md
"""

import os
import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data import DataLoader,SubsetRandomSampler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# module packages
from src.constants import NB_EPOCS, CHECKPOINT, LR, K_Fold, BATCH_SIZE
from src import data_preprocess, my_model


def train_epoch(model, train_dataloaders, optimizer, criterion):
    train_loss = 0.0
    model.train()

    for images, labels in train_dataloaders:
        b_x = images   
        b_y = labels  

        optimizer.zero_grad() # clear gradients for this training step

        output = model(b_x)[0]          
        loss = criterion(output.squeeze(-1), b_y.float())    
        loss.backward() # backpropagation, compute gradients  
        optimizer.step()   # apply gradients               
        train_loss +=loss.item() * images.size(0)

    return train_loss

def valid_epoch(model, valid_dataloaders, criterion):
    valid_loss = 0.0
    model.eval()
    for images, labels in valid_dataloaders:
        b_x = images  
        b_y = labels   
        output = model(b_x)[0]          
        loss = criterion(output.squeeze(-1), b_y.float())                
        valid_loss +=loss.item() * images.size(0)
    return valid_loss

def model_test(test_dataloader, model):
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images, labels
            test_output, _ = model(images)
            actual_labels = labels.detach().numpy()
            pred_y = test_output.squeeze().detach().numpy()
            # residuals =  actual_labels - pred_y
            residuals =  pred_y - actual_labels
            rep_gof = pd.DataFrame()
            rep_gof["Test_MSqE"] = [mean_squared_error(actual_labels, pred_y)]
            rep_gof["Test_MAE"] = [mean_absolute_error(actual_labels, pred_y)]
            rep_gof["Test_R2"] = [r2_score(actual_labels, pred_y)]
            rep_gof["Test_MAPE"] = [np.mean(np.abs(pred_y - actual_labels) / ((pred_y) - 1e-5))]
    return residuals, rep_gof, actual_labels, pred_y
        
def model_train(nb_rep):
    fold_residuals, fold_gofs, fold_actual_labels, fold_predict_y = [], [], [], []
    train_data, test_data = data_preprocess.data_loaders() # Loading train Data
    splits=KFold(n_splits=K_Fold,shuffle=True,random_state=42)
    foldperf={}
    for fold, (train_idx,val_idx) in enumerate(splits.split(np.arange(len(train_data)))):
        print('Fold {}'.format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=train_sampler)
        valid_loader = DataLoader(train_data, batch_size=BATCH_SIZE, sampler=valid_sampler)

        model = my_model.get_model() # Loading Model
        optimizer = optim.SGD(params=model.parameters(), lr=LR)
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'valid_loss': []}

        for epoch in range(NB_EPOCS):
            train_loss=train_epoch(model, train_loader, optimizer, criterion)
            valid_loss=valid_epoch(model,valid_loader, criterion)

            train_loss = train_loss / len(train_loader.sampler)
            valid_loss = valid_loss / len(valid_loader.sampler)

            print("Epoch:{}/{} AVG Training Loss:{:.3f} AVG valid Loss:{:.3f} %".format(epoch + 1, NB_EPOCS, train_loss, valid_loss))
            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)
        foldperf['fold{}'.format(fold+1)] = history 
    
        # Save Model 
        model_name = "baseline_" + str(nb_rep) + "_" + str(fold) + ".h5"
        model_checkpoint_dir = os.path.join(CHECKPOINT, model_name)
        torch.save(model.state_dict(), model_checkpoint_dir)

        test_dataloader = DataLoader(test_data, batch_size=10000)
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT, model_name))) #Loading Model

        residuals, gofs, actual_lab, predic_y = model_test(test_dataloader, model)
        fold_residuals.append(residuals)
        fold_actual_labels.append(actual_lab)
        fold_predict_y.append(predic_y)
        fold_gofs.append(gofs)
        
        # print("-------------fold shape-------", np.shape(fold_residuals))

    all_fold_gofs = pd.concat(fold_gofs, axis=0, ignore_index=True)
    return all_fold_gofs, fold_residuals, fold_actual_labels, fold_predict_y



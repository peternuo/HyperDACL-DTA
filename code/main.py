# -*-coding:utf-8-*-
# -*-coding:utf-8-*-
import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from dataloader import create_DTA_dataset
from model import Graph_GAT
from utils import *
import time
from nt_xent import NT_Xent
# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, Cross_bool):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    LOG_INTERVAL = 20
    loss_train = 0
    num_sample = 0
    a_all=[]
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data_mol = data[0].to(device)
        data_pro = data[1].to(device)
        if Cross_bool:
            output,a = model(data_mol, data_pro, Cross_bool)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))
            loss.backward()
        else:
            output,a, xt_svd, xt_random, drug_diff1, drug_diff2 = model(data_mol, data_pro, Cross_bool)
            loss = loss_fn(output, data_mol.y.view(-1, 1).float().to(device))

            criterion1 = NT_Xent(xt_svd.shape[0], 0.1, 1)
            cl_loss1 = criterion1(xt_svd, xt_random)
            criterion2 = NT_Xent(drug_diff1.shape[0], 0.1, 1)
            cl_loss2 = criterion2(drug_diff1, drug_diff2)
            loss_ = loss + 0.025 * cl_loss1 + 0.025 * cl_loss2

            loss_.backward()
        loss_train = loss_train + data_mol.y.shape[0] * loss.item()
        num_sample = num_sample + data_mol.y.shape[0]
        optimizer.step()
        # running_loss.update(loss.item(), data[1].y.size(0))
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [({:.0f}%)]\tLoss: {:.6f}'.format(epoch,100. * batch_idx / len(train_loader),
                                                                           loss.item()))
        a = a.cpu().detach().numpy().sum(axis=0)/data_mol.y.shape[0]
        a_all.append(list(a.flatten()))
    print('Train epoch: {}\tLoss: {:.6f}'.format(epoch, loss_train / num_sample))
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    print("注意力分数 drug, pro:", np.array(a_all).sum(axis=0)/len(a_all))
    return loss_train / num_sample, np.array(a_all).sum(axis=0)/len(a_all)

def predicting(model, device, dataloader, Cross_bool):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            if Cross_bool:
                output,_ = model(data_mol, data_pro, Cross_bool)
            else:
                output,_,_,_,_,_ = model(data_mol, data_pro, Cross_bool)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = [['davis', 'kiba'][int(sys.argv[1])]]
cuda_name = ['cuda:0', 'cuda:1'][int(sys.argv[2])]
modeling = [Graph_GAT][0]
model_st = modeling.__name__
BATCH_SIZE = 256
LR = 0.0005

# # 定义学习率调整函数
# def adjust_learning_rate(epoch):
#     if epoch < 200:
#         return LR
#     elif epoch < 700:
#         return 0.0005
#     else:
#         return 0.0001
    
# Main program: iterate over different datasets
for dataset in datasets:
    loss_train_list = []
    loss_test_list = []
    a_list = []
    print('\nrunning on ', model_st + '_' + dataset )
    train_data, test_data= create_DTA_dataset(dataset)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False,  collate_fn=collate)
    # training the model
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model = modeling().to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epochs = 2000
    best_mse = 1000
    best_epoch = -1
    model_file_name = 'model_' + dataset + '.model'
    for epoch in range(epochs):
        # 在前500轮，不冻结任何层
        if epoch < 1000:
            for param in model.Mol_encoder.parameters():
                param.requires_grad = True
            for param in model.Pro_encoder.parameters():
                param.requires_grad = True
        # 在后100轮，冻结Mol_encoder和Pro_encoder
            Cross_bool = False
        else:
            for param in model.Mol_encoder.parameters():
                param.requires_grad = False
            for param in model.Pro_encoder.parameters():
                param.requires_grad = False
            Cross_bool = True
        start = time.time()
        loss_train,a = train(model, device, train_loader, optimizer, epoch+1, Cross_bool)
        loss_train_list.append(loss_train)
        a_list.append(list(a))
        G,P = predicting(model, device, test_loader, Cross_bool)
        test_loss =  mse(G,P)
        loss_test_list.append(test_loss)
        # logger.info(msg)
        if test_loss<best_mse:
            # save_model_dict(model, logger.get_model_dir(), msg)
            torch.save(model.state_dict(), model_file_name)
            best_epoch = epoch+1
            best_mse = test_loss
            best_rm2 = get_rm2(G, P)
            print('rmse improved at epoch ', best_epoch, '; best_mse:', best_mse,get_rm2(G, P),dataset)
        else:
            print(test_loss, 'No improvement since epoch ', best_epoch, '; best_mse:', best_mse,best_rm2,dataset)
        end = time.time()
        print(end-start,"s")
        d = pd.DataFrame(loss_train_list, columns=['train_loss'])
        d['test_loss'] = loss_test_list
        d.to_csv("训练损失.csv", index=0)
        d = pd.DataFrame(a_list, columns=['drug', 'pro'])
        d.to_csv("注意力分数.csv", index=0)
        # print("learning_LR:", lr)
    print('train success!')
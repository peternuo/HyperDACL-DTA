import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from dataloader import create_DTA_dataset
# from dataloader_test import create_DTA_dataset
from model import Graph_GAT
from torch_geometric.data import DataLoader
from utils import *

def predicting(model, device, dataloader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(dataloader.dataset)))
    with torch.no_grad():
        for data in dataloader:
            data_mol = data[0].to(device)
            data_pro = data[1].to(device)
            output,_ = model(data_mol, data_pro, Cross_bool=True)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels,data_mol.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()

datasets = 'kiba'
modeling = Graph_GAT
model_st = modeling.__name__
print('\npredicting for test dataset using ', model_st)
TEST_BATCH_SIZE = 512

result = []
train_data, test_data= create_DTA_dataset(datasets)
test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False,  collate_fn=collate)

# training the model
device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model = modeling().to(device)
model_file_name = 'model_kiba.model'
if os.path.isfile(model_file_name):
    model.load_state_dict(torch.load(model_file_name,map_location=torch.device('cpu')),strict=False)
    G,P = predicting(model, device, test_loader)
    ret = [mse(G, P), rmse(G, P), ci(G, P), r2s(G, P), pearson(G, P), spearman(G, P), get_rm2(G, P)]
    ret = [datasets,model_st]+[round(e,3) for e in ret]
    result += [ret]
    print('dataset,model,mse,rmse,ci,r2s,pearson,spearman,rm2')
    print(ret)
else:
    print('model is not available!')

with open('result_'+ datasets +'.csv','a') as f:
    f.write('dataset,model,mse,rmse,ci,r2s,pearson,spearman\n')
    for ret in result:
        f.write(','.join(map(str,ret)) + '\n')



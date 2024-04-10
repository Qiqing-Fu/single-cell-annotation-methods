import TOSICA
import scanpy as sc
import numpy as np
from numpy import genfromtxt as gft
import pandas as pd
import os
import csv
import time as tm

import warnings
warnings.filterwarnings("ignore")

import torch
print(torch.__version__)
print(torch.cuda.get_device_capability(device=None), torch.cuda.get_device_name(device=None))

import rpy2.robjects as robjects 

def run_TOSICA(DataPath, LabelsPath, CV_RDataPath, OutputDir):
    
    # read the Rdata file
    robjects.r['load'](CV_RDataPath)
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int')
    col = col - 1 
    test_ind = np.array(robjects.r['Test_Idx'])
    train_ind = np.array(robjects.r['Train_Idx'])
    

    # read the data and labels
    data_old = sc.read(DataPath) 
    labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols=col)
    cells = data_old.obs.index.values
    data_old.obs = pd.DataFrame(index=cells, data=labels.x.values, columns=["cell_type"])
    GeneSymbol = data_old.var.index.values
    data_old.var = pd.DataFrame(index=GeneSymbol, data=GeneSymbol, columns=["Gene Symbol"])
    data = data_old
    labels = gft(LabelsPath, dtype = "str", skip_header = 1, delimiter = ",", usecols = col) 

    truelab = []
    pred = []
    tr_time = []
    ts_time = []
    os.chdir(OutputDir) 
    
    for i in range(np.squeeze(nfolds)):

        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1

        train=data[train_ind_i,:]
        test=data[test_ind_i,:]
        y_train = labels[train_ind_i]
        y_test = labels[test_ind_i]

        train.obs['cell_type'] = y_train
        
        # train model
        os.chdir(OutputDir)     
        start = tm.time()            
        TOSICA.train(train, gmt_path='human_gobp', label_name='cell_type', epochs=3, project='hGOBP_demo') 
        tr_time.append(tm.time()-start)
        
        # predict labels
        start = tm.time()
        model_weight_path = './hGOBP_demo/model-0.pth'
        test_pred = TOSICA.pre(test, model_weight_path = model_weight_path, project='hGOBP_demo')
        ts_time.append(tm.time()-start)

        truelab.extend(y_test)
        pred.extend(test_pred.obs.Prediction.values)

  
    truelab = pd.DataFrame(truelab)
    truelab.iloc[:,0]=truelab.iloc[:,0].str.replace('"','') 
    pred = pd.DataFrame(pred)
    pred.iloc[:,0]=pred.iloc[:,0].str.replace('"','')
            
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)
    
      
    truelab.to_csv("TOSICA_True_Labels.csv", index = False, quoting=1) 
    pred.to_csv("TOSICA_Pred_Labels.csv", index = False, quoting=1) 
    tr_time.to_csv("TOSICA_Training_Time.csv", index = False)
    ts_time.to_csv("TOSICA_Testing_Time.csv", index = False)
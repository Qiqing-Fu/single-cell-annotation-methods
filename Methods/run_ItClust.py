import os
import time as tm
import ItClust as ic
import scanpy.api as sc
import rpy2.robjects as robjects
from numpy.random import seed
from numpy import genfromtxt as gft
import numpy as np
import pandas as pd
import csv
import warnings
from tensorflow import set_random_seed
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"]="1"
warnings.filterwarnings("ignore")

seed(20180806)
np.random.seed(10)
set_random_seed(20180806) 

def run_ItClust(DataPath, LabelsPath, CV_RDataPath, OutputDir):
    robjects.r['load'](CV_RDataPath)
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int') 
    col = col - 1 
    test_ind = np.array(robjects.r['Test_Idx'])
    train_ind = np.array(robjects.r['Train_Idx'])

    data = sc.read(DataPath, header=0, index_col=0, sep=',') 
    labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',', usecols = col)
    labels = labels[tokeep]
    data = data[tokeep]
    labels = gft(LabelsPath, dtype = "str", skip_header = 1, delimiter = ",", usecols = col)  #convert  

    os.chdir(OutputDir)
    tr_time = []
    ts_time = []
    truelab1 = []
    pred1 = []
    
    for i in range(np.squeeze(nfolds)):
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        train=data[train_ind_i,:]
        test=data[test_ind_i,:]
        y_train = labels[train_ind_i]
        y_test = labels[test_ind_i]
        train.obs['celltype'] = y_train
                
        # train model 
        start = tm.time()
        clf=ic.transfer_learning_clf()
        clf.fit(train, test)
        tr_time.append(tm.time()-start)

        # predict labels
        start = tm.time()
        pred, prob, cell_type_pred=clf.predict()
        ts_time.append(tm.time()-start)

        # change the cluster to the cell type
        cluster_mapping = cell_type_pred 
        pred["cell_type"] = 'NA'
        nrow = pred.shape[0]
        for i in range(nrow):
            cluster_num = pred['cluster'].iloc[i]  
            if cluster_num in cluster_mapping:
                cell_type = cluster_mapping[cluster_num][0]  
                pred["cell_type"].iloc[i] = cell_type  

        test_pred = pred['cell_type'] 
        test_pred = test_pred.tolist()
        pred1.extend(test_pred)
        
        y_test = y_test.tolist()
        truelab1.extend(y_test)

    pred1 = pd.DataFrame(pred1) 
    pred1 = pred1.rename(columns={0:"x"})
    pred1.to_csv("ItClust_"  + "Pred_Labels.csv", index = False, quoting=csv.QUOTE_NONE, quotechar = None)
    
    truelab1 = pd.DataFrame(truelab1) 
    truelab1 = truelab1.rename(columns={0:"x"})
    truelab1.to_csv("ItClust_" + "True_Labels.csv", index = False, quoting=csv.QUOTE_NONE, quotechar = None)
    
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)
    tr_time.to_csv("ItClust_Training_Time.csv", index = False)
    ts_time.to_csv("ItClust_Testing_Time.csv", index = False)
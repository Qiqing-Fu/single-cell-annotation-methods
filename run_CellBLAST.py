import numpy as np
import Cell_BLAST as cb
from numpy import genfromtxt as gft
import rpy2.robjects as robjects
import os
import time as tm
import pandas as pd

def run_cellblast(DataPath, LabelsPath, CV_RDataPath, OutputDir):
    robjects.r['load'](CV_RDataPath) 
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int')
    col = col - 1
    test_ind = np.array(robjects.r['Test_Idx'])
    train_ind = np.array(robjects.r['Train_Idx'])
    
    # read the data and labels
    data_old = cb.data.read_table(DataPath, orientation="cg", sep=",", index_col = 0, header = 0, sparsify = True)
    cb.data.normalize(data_old, target=10000.0)
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',', usecols = col)
    data = data_old[tokeep] 
    labels = gft(LabelsPath, dtype = "str", skip_header = 1, delimiter = ",", usecols = col)
    labels = labels[tokeep]
    
    os.chdir(OutputDir)  
    
    truelab = []
    pred = []
    tr_time = []
    ts_time = []
    
    for i in range(np.squeeze(nfolds)):
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        train=data[train_ind_i,:] 
        test=data[test_ind_i,:] 
        y_train = labels[train_ind_i]
        y_test = labels[test_ind_i]
        train.obs['cell_type'] = y_train
        
        models = []
        axes = cb.data.find_variable_genes(train)
        for j in range(4):
            models.append(cb.directi.fit_DIRECTi(train, genes=train.var.query("variable_genes").index,
            latent_dim=10, cat_dim=20, random_seed=i, path="%d" % j))
        
        # train model
        start = tm.time()
        blast = cb.blast.BLAST(models, train)
        tr_time.append(tm.time()-start)
        
        # predict labels
        start = tm.time()
        test_pred = blast.query(test).annotate('cell_type')
        ts_time.append(tm.time()-start)
        
        truelab.extend(y_test)
        pred.extend(test_pred.values)
    
    truelab = pd.DataFrame(truelab)
    truelab.iloc[:,0]=truelab.iloc[:,0].str.replace('"','')
    pred = pd.DataFrame(pred)
    pred.iloc[:,0]=pred.iloc[:,0].str.replace('"','')
    
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)

    truelab.to_csv("Cell_BLAST_True_Labels.csv", index = False)
    pred.to_csv("Cell_BLAST_Pred_Labels.csv", index = False)
    tr_time.to_csv("Cell_BLAST_Training_Time.csv", index = False)
    ts_time.to_csv("Cell_BLAST_Testing_Time.csv", index = False)
import os
import rpy2.robjects as robjects
import time as tm
import scanpy as sc
import numpy as np
import pandas as pd
import anndata as ad
import scipy
from scipy import sparse



def run_scBERT(DataPath, LabelsPath, CV_RDataPath, OutputDir):
            
    #read the data
    data = pd.read_csv(DataPath,index_col=0,sep=',') 
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',')

    #read the R file 
    robjects.r['load'](CV_RDataPath)
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int') 
    col = np.array(robjects.r['col_Index'], dtype = 'int') 
    col = col - 1

    data = data[tokeep] 
    labels =  labels[tokeep]

    #read the train and test data
    test_ind = np.array(robjects.r['Test_Idx'], dtype = 'object')
    train_ind = np.array(robjects.r['Train_Idx'], dtype = 'object') 
    truelab = [] 
    pred = [] 
    tr_time = []
    ts_time = []

    for i in range(np.squeeze(nfolds)):  
    
        os.chdir("./scBERT-master/")
           
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train=data.iloc[train_ind_i]
        test=data.iloc[test_ind_i]
        y_train=labels.iloc[train_ind_i]
        y_test=labels.iloc[test_ind_i] 

        adata_train = ad.AnnData(train)
        adata_train.X = scipy.sparse.csr_matrix(adata_train.X)
        adata_train.write('./train.h5ad')

        adata_test = ad.AnnData(test)
        adata_test.X = scipy.sparse.csr_matrix(adata_test.X)
        adata_test.write('./test.h5ad')

        # preprocess the data
        panglao = sc.read_h5ad('./data/panglao_10000.h5ad')
        data_test = sc.read_h5ad('./test.h5ad') 
        counts = sparse.lil_matrix((data_test.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
        ref = panglao.var_names.tolist()
        obj = data_test.var_names.tolist()
        for i in range(len(ref)):
            if ref[i] in obj:
                loc = obj.index(ref[i])
                counts[:,i] = data_test.X[:,loc]
        counts = counts.tocsr()
        new_test = ad.AnnData(X=counts)
        new_test.var_names = ref
        new_test.obs_names = data_test.obs_names
        new_test.obs = data_test.obs
        new_test.uns = panglao.uns
        sc.pp.normalize_total(new_test, target_sum=1e4)
        sc.pp.log1p(new_test, base=2)
        new_test.write('./preprocessed_test.h5ad')


        data_train = sc.read_h5ad('./train.h5ad') 
        counts = sparse.lil_matrix((data_train.X.shape[0],panglao.X.shape[1]),dtype=np.float32)
        ref = panglao.var_names.tolist()
        obj = data_train.var_names.tolist()
        for i in range(len(ref)):
            if ref[i] in obj:
                loc = obj.index(ref[i])
                counts[:,i] = data_train.X[:,loc]
        counts = counts.tocsr()
        new_train = ad.AnnData(X=counts)
        new_train.var_names = ref
        new_train.obs_names = data_train.obs_names
        new_train.obs = data_train.obs
        new_train.uns = panglao.uns
        new_train.obs["celltype"] = y_train.values # add the celltype imformation of train data 
        #sc.pp.filter_cells(new_train, min_genes=200)  # don't filter the  cells
        sc.pp.normalize_total(new_train, target_sum=1e4)
        sc.pp.log1p(new_train, base=2)
        new_train.write('./preprocessed_train.h5ad')

        # finetune the model by the trainning datasets
        start = tm.time()
        os.system("python -m torch.distributed.launch finetune.py --data_path preprocessed_train.h5ad --model_path ./data/panglao_pretrain.pth")
        tr_time.append(tm.time()-start)
        
        
        # predict the result by the finetuned model
        start = tm.time()
        os.system("python predict.py --data_path preprocessed_test.h5ad --model_path ./ckpts/finetune_best.pth")
        ts_time.append(tm.time()-start)
        
        # obtain the predicting result
        predlabels = pd.read_csv("predict_result.csv", header=0, index_col=0, sep=',')
        pred.extend(predlabels.loc[:,'0'].tolist())
        y_test = y_test.loc[:,'x'].tolist()
        truelab.extend(y_test)

 
    os.chdir(OutputDir)
    
    truelab1 = pd.DataFrame(truelab)
    truelab1 = truelab1.rename(columns={0:"x"})
    truelab1.to_csv("scBERT_"  + "True_Labels.csv", index = False)
    
    pred1 = pd.DataFrame(pred)
    pred1 = pred1.rename(columns={0:"x"})
    pred1.to_csv("scBERT_"  + "Pred_Labels.csv", index = False)
    
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)
    tr_time.to_csv("scBERT_Training_Time.csv", index = False)
    ts_time.to_csv("scBERT_Testing_Time.csv", index = False)

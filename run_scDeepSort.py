import os
import pandas as pd
import numpy as np
import time as tm
import rpy2.robjects as robjects

def run_scDeepSort(DataPath, LabelsPath, CV_RDataPath, OutputDir, WorkPath):
    

    data = pd.read_csv(DataPath, index_col=0, sep=',') 
    labels = pd.read_csv(LabelsPath, header=0, index_col=None, sep=',')
    robjects.r['load'](CV_RDataPath)
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int') 
    col = np.array(robjects.r['col_Index'], dtype = 'int') 
    col = col - 1

    test_ind = np.array(robjects.r['Test_Idx'])
    train_ind = np.array(robjects.r['Train_Idx']) 
    
    os.chdir(WorkPath)  
      
    tr_time = []
    ts_time = []
    truelab = []
    pred =[]
        
    for i in range(np.squeeze(nfolds)):    
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train=data.iloc[train_ind_i]
        test=data.iloc[test_ind_i]
        y_train=labels.iloc[train_ind_i]
        y_test=labels.iloc[test_ind_i]
            
        # modify the file format of the train labels
        y_train = y_train.rename(columns={'x': 'Cell_type'}) 
        cell = []
        char = "C"
        numbers = range(1, y_train.shape[0]+1)
        for num in (numbers):
            result = char + "_" + str(num) 
            cell.append(result)                
        y_train.insert(0,"Cell",cell) # Add the data named "Cell" in column 1
        y_train.to_csv("./train/human/human_PBMC_celltype.csv") 
        
        train.index = cell
        train = train.T  #Transposite the train data , column is cell, row is gene
        train.to_csv("./train/human/human_PBMC_data.csv") 
        
        #modify the file format of the t
        test = test.T #Transposite the test data 
        test.to_csv("./test/human/human_PBMC13720_data.csv") 
        
        #run the training data  
        start = tm.time()
        os.system("python /XXX/XXX/XXX/XXX/scDeepSort-master/train.py --species human --tissue PBMC --gpu -1 --filetype csv")
        tr_time.append(tm.time() - start)
        
        #predict the test data
        start = tm.time()
        os.system("python /XXX/XXX/XXX/XXX/scDeepSort-master/predict.py --species human --tissue PBMC --test_dataset 13720 --gpu -1 --test --filetype csv --unsure_rate 2")
        ts_time.append(tm.time() - start)
        
        predlabels = pd.read_csv('/XXX/XXX/XXX/XXX/scDeepSort-master/result/human_PBMC_13720.csv',header=0,index_col=None, sep=',') 
        pred.extend(predlabels.loc[:,'cell_type'].tolist())
        
        y_test = y_test.loc[:,'x'].tolist() 
        truelab.extend(y_test) 

       
    truelab = pd.DataFrame(truelab)
    pred = pd.DataFrame(pred)
    tr_time = pd.DataFrame(tr_time)
    ts_time = pd.DataFrame(ts_time)

    os.chdir(OutputDir)
    truelab.to_csv("scDeepSort_" + "True_Labels.csv", index = False)
    pred.to_csv("scDeepSort_"  + "Pred_Labels.csv", index = False)
    tr_time.to_csv("scDeepSort_Training_Time.csv", index = False)
    ts_time.to_csv("scDeepSort_Testing_Time.csv", index = False)

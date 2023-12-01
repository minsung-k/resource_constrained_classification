import numpy as np
np.set_printoptions(suppress=True)
import math
import pandas as pd

import matplotlib.pyplot as plt
import os

import random
from random import SystemRandom  

from sklearn.metrics import confusion_matrix, plot_confusion_matrix, roc_auc_score, roc_curve, f1_score, precision_score, recall_score, matthews_corrcoef
import sklearn as sk
from sklearn.preprocessing import StandardScaler
import copy
import warnings 
warnings.filterwarnings('ignore')
import math

import keras
from keras import backend as K
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split




# ---------------------------------------------------------------------------
def reset_random_seeds():
    os.environ['PYTHONHASHSEED']=str(1)
    tf.random.set_seed(1)
    np.random.seed(1)
    random.seed(1)

# ---------------------------------------------------------------------------
def algorithm_cs(c_01,dataset, N1_ratio,N2_ratio):
    list_p = []
    list_cost = []
    list_num1 = []

    list_conmat = []

    list_precision = []
    list_recall = []
    list_f1 = []
    list_N1N2 = []

    list_auc = []
    list_mcc = []
    list_gmean = []
    for p in [0.5,1,2,3,4,5]:

        if dataset == 'synthetic': 
            
            X_train, y_train, X_test, y_test, X_valid, y_valid= mid_data()

            '''
            X_train = X_train[:int(len(X_train)/10) ]
            y_train = y_train[:int(len(y_train)/10) ]

            X_test = X_test[:int(len(X_test)/10)]
            y_test = y_test[:int(len(y_test)/10)]

            X_valid = X_valid[:int(len(X_valid)/10)]
            y_valid = y_valid[:int(len(y_valid)/10)]
            '''

            y_train_pred, y_test_pred = Model_synthetic(p,X_train, y_train, X_test, X_valid, y_valid)

        elif dataset == 'churn': 
            X_train, y_train, X_test, y_test, X_valid, y_valid= churn()

            '''
            X_train = X_train[:int(len(X_train)/10) ]
            y_train = y_train[:int(len(y_train)/10) ]

            X_test = X_test[:int(len(X_test)/10)]
            y_test = y_test[:int(len(y_test)/10)]

            X_valid = X_valid[:int(len(X_valid)/10)]
            y_valid = y_valid[:int(len(y_valid)/10)]
            '''

            y_train_pred, y_test_pred = Model_churn(p,X_train, y_train, X_test, X_valid, y_valid)

        elif dataset == 'credit': 
            X_train, y_train, X_test, y_test, X_valid, y_valid= credit()
            
            '''
            X_train = X_train[int(len(X_train)*0.1) :int(len(X_train)*0.2) ]
            y_train = y_train[int(len(y_train)*0.1): int(len(y_train)*0.2)]

            X_test = X_test[int(len(X_test)*0.1): int(len(X_test)*0.2)]
            y_test = y_test[int(len(y_test)*0.1): int(len(y_test)*0.2)]

            X_valid = X_valid[int(len(X_valid)*0.1): int(len(X_valid)*0.2)]
            y_valid = y_valid[int(len(y_valid)*0.1): int(len(y_valid)*0.2)]
            '''

            y_train_pred, y_test_pred = Model_credit(p,X_train, y_train, X_test, X_valid, y_valid)

        D_n_train = np.unique(y_train, return_counts = True)[1][0]
        D_p_train = np.unique(y_train, return_counts = True)[1][1]

        D_n_test = int(len(y_test) * D_n_train/len(y_train))
        D_p_test = int(len(y_test) * D_p_train/len(y_train))


        N1 = int(N1_ratio * len(y_test) / len(y_train) * D_p_train)
        N2 = int(N2_ratio * len(y_test) / len(y_train) * D_p_train)



        fpr, fnr, thresholds = fpr_fnr(y_train_pred, y_train, y_test)

        interfpr_con, interfnr_con, threshold_con = find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2)

        y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)

        precision,recall,f1, auc,mcc, gmean, con_mat = metrics(y_test, y_test_classification)

        TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()

        C= int(c_01 * FN +  FP)

        list_p.append(p)
        list_cost.append(C)
        list_num1.append(number_of_1)

        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1)
        list_auc.append(auc)
        list_mcc.append(mcc)
        list_gmean.append(gmean)
        list_N1N2.append([N1, N2])

        list_conmat.append(con_mat)

    dict = {'p': list_p, 'cost': list_cost, 'num1': list_num1, 'list_N1N2': list_N1N2, 'list_conmat': list_conmat, 
            'list_precision':list_precision, 'list_recall':list_recall, 'list_f1':list_f1, 'list_auc': list_auc, 'list_mcc': list_mcc, 'list_gmean': list_gmean}

    df = pd.DataFrame(dict)
    if dataset == 'synthetic': location = 'test'
    elif dataset == 'churn': location = 'test_case1'
    elif dataset == 'credit': location = 'test_case2'
    
    
    method = 'cs'

    # saving the dataframe 
    df.to_csv('result/{}/cs/{}_{}_{}.csv'.format(location,method,N1_ratio,N2_ratio))
# ---------------------------------------------------------------------------

def algorithm_nn(c_01,dataset, N1_ratio,N2_ratio):
    p = 1
    
    if dataset == 'synthetic': 
        X_train, y_train, X_test, y_test, X_valid, y_valid= mid_data()
        
        '''
        X_train = X_train[:int(len(X_train)/10) ]
        y_train = y_train[:int(len(y_train)/10) ]

        X_test = X_test[:int(len(X_test)/10)]
        y_test = y_test[:int(len(y_test)/10)]

        X_valid = X_valid[:int(len(X_valid)/10)]
        y_valid = y_valid[:int(len(y_valid)/10)]
        '''

        y_train_pred, y_test_pred = Model_synthetic(p,X_train, y_train, X_test, X_valid, y_valid)
        
    elif dataset == 'churn': 
        X_train, y_train, X_test, y_test, X_valid, y_valid= churn()
        
        '''
        X_train = X_train[:int(len(X_train)/10) ]
        y_train = y_train[:int(len(y_train)/10) ]

        X_test = X_test[:int(len(X_test)/10)]
        y_test = y_test[:int(len(y_test)/10)]

        X_valid = X_valid[:int(len(X_valid)/10)]
        y_valid = y_valid[:int(len(y_valid)/10)]
        '''
        
        y_train_pred, y_test_pred = Model_churn(p,X_train, y_train, X_test, X_valid, y_valid)
        
    elif dataset == 'credit': 
        X_train, y_train, X_test, y_test, X_valid, y_valid= credit()
        
        '''
        X_train = X_train[int(len(X_train)*0.1) :int(len(X_train)*0.2) ]
        y_train = y_train[int(len(y_train)*0.1): int(len(y_train)*0.2)]

        X_test = X_test[int(len(X_test)*0.1): int(len(X_test)*0.2)]
        y_test = y_test[int(len(y_test)*0.1): int(len(y_test)*0.2)]

        X_valid = X_valid[int(len(X_valid)*0.1): int(len(X_valid)*0.2)]
        y_valid = y_valid[int(len(y_valid)*0.1): int(len(y_valid)*0.2)]
        '''

        y_train_pred, y_test_pred = Model_credit(p,X_train, y_train, X_test, X_valid, y_valid)
        
        
    
    D_n_train = np.unique(y_train, return_counts = True)[1][0]
    D_p_train = np.unique(y_train, return_counts = True)[1][1]
    
    D_n_test = int(len(y_test) * D_n_train/len(y_train))
    D_p_test = int(len(y_test) * D_p_train/len(y_train))


    N1 = int(N1_ratio * len(y_test) / len(y_train) * D_p_train)
    N2 = int(N2_ratio * len(y_test) / len(y_train) * D_p_train)
    
    list_p = []
    list_cost = []
    list_num1 = []
    
    list_conmat = []
    
    list_precision = []
    list_recall = []
    list_f1 = []
    list_N1N2 = []
    
    list_auc = []
    list_mcc = []
    list_gmean = []

    fpr, fnr, thresholds = fpr_fnr(y_train_pred, y_train, y_test)
       
    interfpr_con, interfnr_con, threshold_con = find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2)
    
    y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
        
    precision,recall,f1, auc,mcc, gmean, con_mat = metrics(y_test, y_test_classification)
    
    TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
    
    C= int(c_01 *FN + FP) 
    
    
    list_p.append(p)
    list_cost.append(C)
    list_num1.append(number_of_1)

    list_precision.append(precision)
    list_recall.append(recall)
    list_f1.append(f1)
    list_auc.append(auc)
    list_mcc.append(mcc)
    list_gmean.append(gmean)
    list_N1N2.append([N1, N2])

    list_conmat.append(con_mat)

    dict = {'p': list_p, 'cost': list_cost, 'num1': list_num1, 'list_N1N2': list_N1N2, 'list_conmat': list_conmat, 
            'list_precision':list_precision, 'list_recall':list_recall, 'list_f1':list_f1, 'list_auc': list_auc, 'list_mcc': list_mcc, 'list_gmean': list_gmean}

    df = pd.DataFrame(dict)
    if dataset == 'synthetic': location = 'test'
    elif dataset == 'churn': location = 'test_case1'
    elif dataset == 'credit': location = 'test_case2'
    
    
    method = 'nn'

    # saving the dataframe 
    df.to_csv('result/{}/nn/{}_{}_{}_{}.csv'.format(location,method,N1_ratio,N2_ratio,p))


    

# ---------------------------------------------------------------------------

def algorithm(c_01,initial_p, epochs, total_run,N1_ratio,N2_ratio):

    X_train, y_train, X_test, y_test, X_valid, y_valid= mid_data()
    
    '''
    X_train = X_train[:int(len(X_train)/10) ]
    y_train = y_train[:int(len(y_train)/10) ]

    X_test = X_test[:int(len(X_test)/10)]
    y_test = y_test[:int(len(y_test)/10)]

    X_valid = X_valid[:int(len(X_valid)/10)]
    y_valid = y_valid[:int(len(y_valid)/10)]
    '''

    # initial setting

    D_n_train = np.unique(y_train, return_counts = True)[1][0]
    D_p_train = np.unique(y_train, return_counts = True)[1][1]
    
    D_n_test = int(len(y_test) * D_n_train/len(y_train))
    D_p_test = int(len(y_test) * D_p_train/len(y_train))


    N1 = int(N1_ratio * len(y_test) / len(y_train) * D_p_train)
    N2 = int(N2_ratio * len(y_test) / len(y_train) * D_p_train)
   
    

    list_minp = []
    list_mincost = []
    list_num1 = []
    
    list_conmat = []
    
    list_precision = []
    list_recall = []
    list_f1 = []
    list_N1N2 = []
    
    list_auc = []
    list_mcc = []
    list_gmean = []
 
    y_train_pred, y_test_pred = Model_synthetic(initial_p,X_train, y_train, X_test, X_valid, y_valid)
    fpr, fnr, thresholds = fpr_fnr(y_train_pred, y_train, y_test)
       
    interfpr_con, interfnr_con, threshold_con = find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2)
      
    y_test_classification, number_of_1 = threshold_change(c_01, initial_p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
    TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
    initial_cost = int(c_01 *FN + FP) 

    rho = 0.07
    b = 1.6
    
    alpha = 0.9
    p_max = 5
    p_min = 0

    N =total_run

    for e in range(epochs):

        list_cost = []
        list_p = []
        
        p_new = initial_p
        p = initial_p
    
        t= 100

        C_right = initial_cost * 3
        C_left = initial_cost * 3
        C_feasible = initial_cost * 3

        
        delta = 0
        C_last = 0
        delta_rho = 0
        
        for i in range(total_run):
            
            y_train_pred, y_test_pred = Model_synthetic(p,X_train, y_train, X_test, X_valid, y_valid)
            fpr, fnr, thresholds = fpr_fnr(y_train_pred, y_train, y_test)

            interfpr_con, interfnr_con, threshold_con = find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2)
            
            (interfpr_opt, interfnr_opt, threshold_opt) = find_cost_intersection(c_01,y_train, y_test, fpr, fnr, thresholds)
            (interfpr1,interfnr1,threshold1), (interfpr2,interfnr2,threshold2)= find_limit_point(fpr, fnr, thresholds,N1,N2)
                
            if interfpr2 < interfpr_opt:
                y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
                TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
                C= int(c_01 *FN + FP) 
                C_right = min(C, C_right)

            elif interfpr_opt < interfpr1:
                y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
                TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
                C= int(c_01 *FN + FP) 
                C_left = min(C, C_left)

            else:
                y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
                TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
                C= int(c_01 *FN + FP) 
                C_feasible = min(C_feasible, C)
          
            # 3
            t = alpha * t
            
            # 4
            C_opt = min(C_left, C_right, C_feasible)
            
            # 5
            delta_C = C_opt - C_last 
            
            # 6
            probability = (C_right + 0.5 * C_feasible) / (C_right + C_left + C_feasible) 
            
            # 9
            probability2 = np.exp(-delta_C / t)
            if delta_C <= 0 or ( delta_C > 0 and  probability2 > SystemRandom().random() ):
            
                # 7
            
                delta_rho = (1 - rho  ** ( (1 - (i/N) ) ** b ) )
                
                # if opt_point in right side, class weight p should be increased to predict less 1.
                if probability <= SystemRandom().random():
                    delta = (p_max - p) * delta_rho   # 8
                    p_new = p + delta
                else:
                    delta = (p - p_min) *  delta_rho  # 8
                    p_new = p - delta

            list_p.append(p)
            list_cost.append(C_opt)
            
            p = p_new
            C_last = C
        
        min_cost = min(list_cost)
        min_p = list_p[list_cost.index(min_cost)]

        y_train_pred, y_test_pred = Model_synthetic(min_p,X_train, y_train, X_test, X_valid, y_valid)
        y_test_classification, number_of_1 = threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2)
        
        precision,recall,f1, auc,mcc, gmean, con_mat = metrics(y_test, y_test_classification)
        
        TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
    
        min_cost= int(c_01 *FN + FP) 
        
        min_p = np.round(min_p,5)
        list_minp.append(min_p)
        list_mincost.append(min_cost)
        list_num1.append(number_of_1)
        
        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1)
        list_auc.append(auc)
        list_mcc.append(mcc)
        list_gmean.append(gmean)
        list_N1N2.append([N1, N2])
        
        list_conmat.append(con_mat)
        
        dict = {'p': list_minp, 'cost': list_mincost, 'num1': list_num1, 'list_N1N2': list_N1N2, 'list_conmat': list_conmat, 
                'list_precision':list_precision, 'list_recall':list_recall, 'list_f1':list_f1, 'list_auc': list_auc, 'list_mcc': list_mcc, 'list_gmean': list_gmean}

        df = pd.DataFrame(dict) 
        method = 'on'

    # saving the dataframe 
    df.to_csv('result/test/on/{}_{}_{}_{}.csv'.format(method,N1_ratio,N2_ratio,initial_p))
        
    
# ---------------------------------------------------------------------------
def threshold_change(c_01, p, X_train,y_train, X_test, y_test, X_valid, y_valid, y_train_pred, y_test_pred, N1,N2):  

    D_n_train = np.unique(y_train, return_counts = True)[1][0]
    D_p_train = np.unique(y_train, return_counts = True)[1][1]
    
    D_n_test = int(len(y_test) * D_n_train/len(y_train))
    D_p_test = int(len(y_test) * D_p_train/len(y_train))

    fpr, fnr, thresholds = fpr_fnr(y_train_pred, y_train, y_test)
     
    interfpr_con, interfnr_con, threshold_con = find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2)

    number_of_1 = D_n_test * interfpr_con + D_p_test * (1-interfnr_con)
    number_of_1 = int(round(number_of_1))
    
    y_test_pred_copy = copy.copy(y_test_pred)
    y_test_classification = np.where(y_test_pred_copy >= threshold_con,1,0)

    if number_of_1 < N1:
        y_test_pred_copy = copy.copy(y_test_pred)
        num = 0

        threshold_next = thresholds[(thresholds.index(threshold_con)) +1]
        current_threshold_sum = sum(np.where(y_test_pred >= threshold_con,1,0))

        for i in range(len(y_test_pred)):
            if y_test_pred[i] ==  threshold_next:
                y_test_pred_copy[i] = 1
                num = num + 1

                if current_threshold_sum + num == N1: break

        y_test_classification = np.where(y_test_pred_copy > threshold_con,1,0)
        TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
        number_of_1 = FP + TP
        
    
    if N2 < number_of_1:
        y_test_pred_copy = copy.copy(y_test_pred)
        num = 0
        
        current_threshold_sum = sum(np.where(y_test_pred >= threshold_con,1,0))

        for i in range(len(y_test_pred)):
            if y_test_pred[i] == threshold_con:
                y_test_pred_copy[i] = 0
                num = num + 1

                if current_threshold_sum - num == N2: 
                    break
                
        y_test_classification = np.where(y_test_pred_copy >= threshold_con,1,0)
        TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
        number_of_1 = FP + TP

    return y_test_classification, number_of_1


# ---------------------------------------------------------------------------
def metrics(y_test, y_test_classification):

    TN, FN, FP, TP = confusion_matrix(y_test, y_test_classification).T.flatten()
        
    p = round(precision_score(y_test, y_test_classification),3)
    r = round(recall_score(y_test, y_test_classification),3)
    f = round(f1_score(y_test, y_test_classification),3)
    auc = 1- round(roc_auc_score(y_test, y_test_classification),3)
    mcc = round(matthews_corrcoef(y_test, y_test_classification),3)
    
    con_mat = confusion_matrix(y_test, y_test_classification).T.flatten().tolist()
    tpr = TP / (TP+FN)
    tnr = TN / (TN+FP)
    gmean = np.sqrt(tpr * tnr)
    
    return p,r,f, auc,mcc, gmean, con_mat

# ---------------------------------------------------------------------------------------------------------------


def find_limit_point(fpr, fnr, thresholds,N1,N2):
    (interfpr1,interfnr1,threshold1) = (fpr[N1-1], fnr[N1-1], thresholds[N1-1])
    (interfpr2,interfnr2,threshold2) = (fpr[N2-1], fnr[N2-1], thresholds[N2-1])
    
    return (interfpr1,interfnr1,threshold1), (interfpr2,interfnr2,threshold2)
    
# ---------------------------------------------------------------------------------------------------------------
def fpr_fnr(y_train_pred, y_train, y_test):
    
    list_fpr = []
    list_fnr = []
    list_th = []
    y_train_pred_sort =sorted(y_train_pred, reverse=True)
    y_train_pred_sort[0] = np.array([1])
    y_train_pred_sort[-1] = np.array([0])

    for i in range(1,len(y_train_pred)+1):
        if i % round(len(y_train) /len(y_test)) ==0:
        
            th = y_train_pred_sort[i-1][0]
            y_result = np.where(y_train_pred >= th,1,0)
            tn, fn, fp, tp = confusion_matrix(y_train, y_result).T.ravel()
            fpr = fp / (tn+fp)
            fnr = fn / (tp+fn)
            
            list_fpr.append(fpr)
            list_fnr.append(fnr)
            list_th.append(th)
            
    return [list_fpr, list_fnr, list_th]
    
# find optimal point in whole area
def find_cost_intersection(c_01,y_train, y_test, fpr, fnr, thresholds):
    list_y_intercept = []
    
    D_n_train = np.unique(y_train, return_counts = True)[1][0]
    D_p_train = np.unique(y_train, return_counts = True)[1][1]

    D_n_test = int(len(y_test) * D_n_train/len(y_train))
    D_p_test = int(len(y_test) * D_p_train/len(y_train))
    
    for i in range(len(fpr)): 
        cost = 1 * fpr[i] * D_n_test + fnr[i] * D_p_test * c_01
        y_intercept = cost / (c_01* D_p_test)
        list_y_intercept.append(y_intercept)
        
    index = list_y_intercept.index(min(list_y_intercept))

    return fpr[index], fnr[index], thresholds[index]
# ---------------------------------------------------------------------------------------------------------------
# find constrained optimal point within feasible area
def find_cost_intersection_within(c_01,y_train, y_test, fpr, fnr, thresholds,N1,N2):

    list_cost = []
    
    D_n_train = np.unique(y_train, return_counts = True)[1][0]
    D_p_train = np.unique(y_train, return_counts = True)[1][1]

    D_n_test = int(len(y_test) * D_n_train/len(y_train))
    D_p_test = int(len(y_test) * D_p_train/len(y_train))
    
    (interfpr1,interfnr1,threshold1) = (fpr[N1-1], fnr[N1-1], thresholds[N1-1])
    (interfpr2,interfnr2,threshold2) = (fpr[N2-1], fnr[N2-1], thresholds[N2-1])

    for i in range(N1-1, N2): list_cost.append(1 * fpr[i] * D_n_test + fnr[i] * D_p_test * c_01)

    index = list_cost.index(min(list_cost))
    
    return fpr[N1-1 + index], fnr[N1-1 + index], thresholds[N1-1 + index]
# ---------------------------------------------------------------------------------------------------------------


    


def mid_data():
    np.random.seed(2)
    X =  np.random.rand(50000,12) * 20

    y = []
    for p in range(len(X)):


        a = X[p][0] 
        b = X[p][1] 
        c = X[p][2] 
        d = X[p][3] 
        e = X[p][4] 
        f = X[p][5] 
        
        g = math.cos(X[p][6]) * 10
        h = math.cos(X[p][7]) * 10
        i = math.cos(X[p][8]) * 10
        j = math.sin(X[p][9]) * 10
        k = math.sin(X[p][10]) * 10
        l = math.sin(X[p][11]) * 10

        sum_ = a-b+c-d+e-f+g-h+i-j+k-l
        result = 1 if sum_ > 0 else 0

        y.append(result)

    df =  pd.DataFrame(X)
    df['y'] = y

    df_0 = df[df['y'] == 0][:10000]
    df_1 = df[df['y'] == 1][:10000]

    df__0 = df_0[:3750].append(df_1[:3000]).sample(frac = 1).reset_index(drop = True)
    df__1 = df_0[3750:5000].append(df_1[3000:4000]).sample(frac = 1).reset_index(drop = True)
    df__2 = df_0[5000:6250].append(df_1[4000:5000]).sample(frac = 1).reset_index(drop = True)

    X_train = df__0.iloc[:, [0,1,2,3,4,5,6,7,8,9]].to_numpy()
    y_train = df__0.iloc[:, [-1]].to_numpy()
    X_test = df__1.iloc[:, [0,1,2,3,4,5,6,7,8,9]].to_numpy()
    y_test = df__1.iloc[:, [-1]].to_numpy()
    X_valid = df__2.iloc[:, [0,1,2,3,4,5,6,7,8,9]].to_numpy()
    y_valid = df__2.iloc[:, [-1]].to_numpy()


    return X_train, y_train, X_test, y_test, X_valid, y_valid


# ---------------------------------------------------------------------------------------------------------------

def churn():
    # data preprocessing

    data = pd.read_csv('data/TelcoChurn.csv')

    # data type change
    l1 = [len(i.split()) for i in data['TotalCharges']]
    l2 = [i for i in range(len(l1)) if l1[i] != 1]

    for i in l2:
        data.loc[i,'TotalCharges'] = data.loc[(i-1),'TotalCharges']

    data['TotalCharges'] = data['TotalCharges'].astype(float)
    data.drop(columns = ['customerID'], inplace = True)

    # Label Encoder Transformation
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    df1 = data.copy(deep = True)
    text_data_features = [i for i in list(data.columns) if i not in list(data.describe().columns)]

    for i in text_data_features :
        df1[i] = le.fit_transform(df1[i])

    col = list(df1.columns)
    categorical_features = []
    numerical_features = []
    for i in col:
        if len(data[i].unique()) > 6:
            numerical_features.append(i)
        else:
            categorical_features.append(i)

    categorical_features.remove('Churn')

    l1 = ['gender','SeniorCitizen','Partner','Dependents'] # Customer Information
    l2 = ['PhoneService','MultipleLines','InternetService','StreamingTV','StreamingMovies',
          'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport'] # Services Signed Up for!
    l3 = ['Contract','PaperlessBilling','PaymentMethod'] # Payment Information

    df1['MonthlyCharges_Group'] = [int(i / 5) for i in df1['MonthlyCharges']]
    df1['TotalCharges_Group'] = [int(i / 500) for i in df1['TotalCharges']]

    from sklearn.preprocessing import MinMaxScaler,StandardScaler
    mms = MinMaxScaler() # Normalization
    ss = StandardScaler() # Standardization

    df1.drop(columns = ['MonthlyCharges_Group','TotalCharges_Group'], inplace = True)

    df1['tenure'] = mms.fit_transform(df1[['tenure']])
    df1['MonthlyCharges'] = mms.fit_transform(df1[['MonthlyCharges']])
    df1['TotalCharges'] = mms.fit_transform(df1[['TotalCharges']])


    df1.drop(columns = ['PhoneService', 'gender','StreamingTV','StreamingMovies','MultipleLines','InternetService'],inplace = True)

    f1 = df1.iloc[:,:13].values
    t1 = df1.iloc[:,13].values

    X_train, X_test, y_train, y_test = train_test_split(f1, t1, test_size=0.2, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    '''
    # ----------small size-----------------------------------------------
    X_train = X_train[: int(len(X_train)*0.1)]
    y_train = y_train[: int(len(y_train)*0.1)]

    X_test = X_test[: int(len(X_test)*0.1)]
    y_test = y_test[: int(len(y_test)*0.1)]

    X_valid = X_valid[: int(len(X_valid)*0.1)]
    y_valid = y_valid[: int(len(y_valid)*0.1)]
    # ----------small size-----------------------------------------------
    '''
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid

# ---------------------------------------------------------------------------------------------------------------

def credit():
    
    df = pd.read_csv('data/creditcard.csv')

    df['IsDefaulter'] =df ['default.payment.next.month']
    df = df.drop('default.payment.next.month',axis = 1)

    fil = (df['EDUCATION'] == 5) | (df['EDUCATION'] == 6) | (df['EDUCATION'] == 0)
    df.loc[fil, 'EDUCATION'] = 4

    fil = df['MARRIAGE'] == 0
    df.loc[fil, 'MARRIAGE'] = 3

    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    df_cat = df[categorical_features]
    df_cat['Defaulter'] = df['IsDefaulter']

    df_cat.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)

    df.rename(columns={'PAY_0':'PAY_SEPT','PAY_2':'PAY_AUG','PAY_3':'PAY_JUL','PAY_4':'PAY_JUN','PAY_5':'PAY_MAY','PAY_6':'PAY_APR'},inplace=True)
    df.rename(columns={'BILL_AMT1':'BILL_AMT_SEPT','BILL_AMT2':'BILL_AMT_AUG','BILL_AMT3':'BILL_AMT_JUL','BILL_AMT4':'BILL_AMT_JUN','BILL_AMT5':'BILL_AMT_MAY','BILL_AMT6':'BILL_AMT_APR'}, inplace = True)
    df.rename(columns={'PAY_AMT1':'PAY_AMT_SEPT','PAY_AMT2':'PAY_AMT_AUG','PAY_AMT3':'PAY_AMT_JUL','PAY_AMT4':'PAY_AMT_JUN','PAY_AMT5':'PAY_AMT_MAY','PAY_AMT6':'PAY_AMT_APR'},inplace=True)

    df['AGE']=df['AGE'].astype('int')
    df.groupby('IsDefaulter')['AGE'].mean()
    df = df.astype('int')

    bill_amnt_df = df[['BILL_AMT_SEPT',	'BILL_AMT_AUG',	'BILL_AMT_JUL',	'BILL_AMT_JUN',	'BILL_AMT_MAY',	'BILL_AMT_APR']]
    pay_amnt_df = df[['PAY_AMT_SEPT',	'PAY_AMT_AUG',	'PAY_AMT_JUL',	'PAY_AMT_JUN',	'PAY_AMT_MAY',	'PAY_AMT_APR', 'IsDefaulter']]

    columns = list(df.columns)
    columns.pop()

    X, y = df.iloc[:,0:-1], df['IsDefaulter']

    df_fr = X.copy()
    df_fr['IsDefaulter'] = y
    df_fr['Payement_Value'] = df_fr['PAY_SEPT'] + df_fr['PAY_AUG'] + df_fr['PAY_JUL'] + df_fr['PAY_JUN'] + df_fr['PAY_MAY'] + df_fr['PAY_APR']

    df_fr['Dues'] = (df_fr['BILL_AMT_APR']+df_fr['BILL_AMT_MAY']+df_fr['BILL_AMT_JUN']+df_fr['BILL_AMT_JUL']+df_fr['BILL_AMT_SEPT'])-(df_fr['PAY_AMT_APR']+df_fr['PAY_AMT_MAY']+df_fr['PAY_AMT_JUN']+df_fr['PAY_AMT_JUL']+df_fr['PAY_AMT_AUG']+df_fr['PAY_AMT_SEPT'])

    df_fr['EDUCATION']=np.where(df_fr['EDUCATION'] == 6, 4, df_fr['EDUCATION'])
    df_fr['EDUCATION']=np.where(df_fr['EDUCATION'] == 0, 4, df_fr['EDUCATION'])

    df_fr['MARRIAGE']=np.where(df_fr['MARRIAGE'] == 0, 3, df_fr['MARRIAGE'])

    df_fr.replace({'SEX': {1 : 'MALE', 2 : 'FEMALE'}, 'EDUCATION' : {1 : 'graduate school', 2 : 'university', 3 : 'high school', 4 : 'others'}, 'MARRIAGE' : {1 : 'married', 2 : 'single', 3 : 'others'}}, inplace = True)

    df_fr = pd.get_dummies(df_fr,columns=['EDUCATION','MARRIAGE'])

    df_fr.drop(['EDUCATION_others','MARRIAGE_others'],axis = 1, inplace = True)

    df_fr = pd.get_dummies(df_fr, columns = ['PAY_SEPT',	'PAY_AUG',	'PAY_JUL',	'PAY_JUN',	'PAY_MAY',	'PAY_APR'], drop_first = True )

    encoders_nums = {
                     "SEX":{"FEMALE": 0, "MALE": 1}
    }
    df_fr = df_fr.replace(encoders_nums)


    df_fr.drop('ID',axis = 1, inplace = True)

    X = df_fr.drop(['IsDefaulter','Payement_Value','Dues'],axis=1)
    y = df_fr['IsDefaulter']

    columns = X.columns

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    '''
    #--------------samll dataset----------------------------------------------------
    X_train = X_train[: int(len(X_train)*0.1)]
    y_train = y_train[: int(len(y_train)*0.1)]

    X_test = X_test[: int(len(X_test)*0.1)]
    y_test = y_test[: int(len(y_test)*0.1)]

    X_valid = X_valid[: int(len(X_valid)*0.1)]
    y_valid = y_valid[: int(len(y_valid)*0.1)]
    #--------------samll dataset----------------------------------------------------
    '''


    return X_train, y_train, X_test, y_test, X_valid, y_valid

# ---------------------------------------------------------------------------

def Model_synthetic(p,X_train, y_train, X_test, X_valid, y_valid):
    reset_random_seeds()

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape = (np.array(X_train).shape[-1],)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=keras.losses.BinaryCrossentropy()
    )
    
    model.fit(X_train,y_train, epochs=30, validation_data = (X_valid,y_valid) ,shuffle = True, batch_size=16 , class_weight =  {0: 1, 1: p}, verbose =0)

    y_train_pred = model.predict(X_train, batch_size=16, verbose =0)
    y_test_pred = model.predict(X_test, batch_size=16, verbose =0)
    
    return y_train_pred, y_test_pred
# ---------------------------------------------------------------------------------------------------------------

def Model_credit(p,X_train, y_train, X_test, X_valid, y_valid):
    reset_random_seeds()

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(78, activation='relu', input_shape = (np.array(X_train).shape[-1],)),
    tf.keras.layers.Dense(34, activation='relu'),
    tf.keras.layers.Dense(17, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=keras.losses.BinaryCrossentropy()
    )
    
    model.fit(X_train,y_train, epochs=30, validation_data = (X_valid,y_valid) ,shuffle = True, batch_size=16 , class_weight =  {0: 1, 1: p}, verbose =0)

    y_train_pred = model.predict(X_train, batch_size=16, verbose =0)
    y_test_pred = model.predict(X_test, batch_size=16, verbose =0)
    
    return y_train_pred, y_test_pred

# ---------------------------------------------------------------------------------------------------------------


def Model_churn(p,X_train, y_train, X_test, X_valid, y_valid):
    reset_random_seeds()

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(13, activation='relu', input_shape = (np.array(X_train).shape[-1],)),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(3, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
      optimizer=tf.keras.optimizers.Adam(),
      loss=keras.losses.BinaryCrossentropy()
    )
    
    model.fit(X_train,y_train, epochs=30, validation_data = (X_valid,y_valid) ,shuffle = True, batch_size=16 , class_weight =  {0: 1, 1: p}, verbose =0)

    y_train_pred = model.predict(X_train, batch_size=16, verbose =0)
    y_test_pred = model.predict(X_test, batch_size=16, verbose =0)
    
    return y_train_pred, y_test_pred
"""
Date: 2021-08-09 16:44:14
LastEditors: GodK
"""

import time
import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import size
from numpy.linalg.linalg import pinv
from mylorenz import mylorenz
from mylorenz_dynamic import mylorenz_dynamic
from NN_F2 import NN_F2
import matplotlib.pyplot as plt
import scipy.io as sio


Y = mylorenz(30)  # coupled lorenz system
# Y = mylorenz_dynamic(30, 1000) 
# Y = sio.loadmat("Y.mat")["Y"]

noisestrength = 0

X = Y + noisestrength * np.random.rand(*Y.shape)

Accurate_predictions = 0
ii = 0

while ii < 2000:  # run each case sequentially with different initials
    ii = ii + 2
    print(f"Case number: {ii/2}")
    INPUT_trainlength = 11  # length of training data (observed data), m > 2L
    selected_variables_idx = list(range(90))  # selected the most correlated variables, can be changed
    xx = X[2999 + ii : X.shape[0], selected_variables_idx].T    # after transient dynamics
    noisestrength=0     # strength of noise
    xx_noise = xx+noisestrength*np.random.rand(*xx.shape)
    
    predict_len=5   # L
    
    # use the most recent short term high-dimensional time-series to predict
    traindata = xx_noise[:, max(0, INPUT_trainlength-3*predict_len):INPUT_trainlength]
    trainlength = traindata.shape[1]
    k = 60  # randomly selected variables of matrix B
    
    jd = 0  # the index of target variable
    
    D = xx_noise.shape[0]   # number of variables in the system
    origin_real_y = xx[jd, :]
    real_y = xx[jd, max(0,INPUT_trainlength-3*predict_len):]
    real_y_noise = real_y+noisestrength*np.random.rand(*real_y.shape)
    traindata_y = real_y_noise[:trainlength]
    
    """
    ****************************** ARNN start **************************
    """
    
    # Given a set of fixed weights for F for each time points: A*F(X^t)=Y^t, F(X^t)=B*(Y^t)
    traindata_x_NN = NN_F2(traindata)
    # traindata_x_NN=sio.loadmat("traindata_x_NN.mat")["traindata_x_NN"]
    
    w_flag = np.zeros((traindata_x_NN.shape[0], traindata_x_NN.shape[0]))
    A = np.zeros((predict_len, traindata_x_NN.shape[0]))   # matrix A
    B = np.zeros((traindata_x_NN.shape[0], predict_len))   # matrix B
    
    predict_pred = np.zeros(predict_len-1)
    
    # End of ITERATION 1:  sufficient iterations
    for iter in range(1000):    # cal coeffcient B
        
        temp_set = list(set(range(traindata_x_NN.shape[0]))-set([jd]))
        np.random.shuffle(temp_set)
        
        random_idx = sorted([jd,*temp_set[:k-1]])
        traindata_x = traindata_x_NN[random_idx,:trainlength] # random chose k variables from F(D)
        
        for i in range(traindata_x.shape[0]):
            b = traindata_x[i,:trainlength-predict_len+1].T # 1*(m-L+1)

            B_w = np.zeros((trainlength-predict_len+1, predict_len))
            for j in range(trainlength-predict_len+1):
                B_w[j, :] = traindata_y[j:j+predict_len]
            B_para = np.transpose(np.linalg.pinv(B_w).dot(b))
            B[random_idx[i], :] = (B[random_idx[i], :] + B_para + B_para*(1-w_flag[random_idx[i], 0]))/2
            w_flag[random_idx[i],0] = 1
        
        ####################### tmp predict based on B ############################
        super_bb = np.zeros((predict_len-1)*traindata_x_NN.shape[0])
        super_AA = np.zeros(((predict_len-1)*traindata_x_NN.shape[0], predict_len-1))
        for i in range(traindata_x_NN.shape[0]):
            kt = -1
            bb = np.zeros(predict_len-1)
            AA = np.zeros((predict_len-1,predict_len-1))
            for j in range(trainlength-(predict_len-1), trainlength):
                kt = kt + 1
                bb[kt] = traindata_x_NN[i,j]
                col_known_y_num = trainlength-j
                for r in range(col_known_y_num):
                    bb[kt] = bb[kt]-B[i,r]*traindata_y[trainlength-col_known_y_num+r]
                AA[kt, :predict_len-col_known_y_num] = B[i, col_known_y_num:predict_len]
            
            super_bb[(predict_len-1)*i:(predict_len-1)*i+predict_len-1] = bb[:]
            super_AA[(predict_len-1)*i:(predict_len-1)*i+predict_len-1, :] = AA
        
        pred_y_tmp = (np.linalg.pinv(super_AA).dot(super_bb.T)).T
        
        ####################### update the values of matrix A and Y ############################
        tmp_y = np.concatenate([real_y[:trainlength], pred_y_tmp])
        Ym = np.zeros((predict_len, trainlength))
        for j in range(predict_len):
            Ym[j, :] = tmp_y[j: j+trainlength]
        
        BX = np.concatenate([B, traindata_x_NN], axis=1)
        IY = np.concatenate([np.eye(predict_len), Ym], axis=1)
        A = IY@np.linalg.pinv(BX)
        union_predict_y_ARNN = np.zeros(predict_len-1)
        for j1 in range(predict_len -1):
            tmp_y = np.zeros((predict_len - j1-1, 1))
            kt = -1
            for j2 in range(j1, predict_len-1):
                kt = kt + 1
                row = j2+1
                col = trainlength - j2 + j1-1
                tmp_y[kt] = A[row, :]@traindata_x_NN[:, col]
            union_predict_y_ARNN[j1] = np.mean(tmp_y)
        
        # End of ITERATION 2: the predicting result converges.

        eof_error = np.sqrt(np.square(union_predict_y_ARNN-predict_pred).mean())
        if eof_error<0.0001:
            break
        
        predict_pred=union_predict_y_ARNN
        
    ######################### result display #################################
    my_real = real_y[trainlength:trainlength+predict_len-1]
    RMSE = np.sqrt(np.square(union_predict_y_ARNN-my_real).mean())
    RMSE = RMSE/(np.std(real_y[trainlength-2*predict_len:trainlength+predict_len-1])+0.001) # normalize RMSE
    if RMSE<0.5:
        Accurate_predictions = Accurate_predictions + 1
    Accurate_prediction_rate = Accurate_predictions/(ii/2)
    print(f'Accurate_prediction_rate: {Accurate_prediction_rate}')
    
    refx = X[2999+ii-100:X.shape[0], :].T
    fig1 = plt.figure(1)
    plt.subplot(2,1,1)
    plt.plot(refx[jd, :150], 'c-*',linewidth=2, markersize=4)
    plt.plot(list(range(100,100+INPUT_trainlength)), origin_real_y[:INPUT_trainlength], 'b-*', linewidth=2, markersize=4)
    plt.title(f'original attractor. Init: {ii}, Noise strength: {noisestrength}', fontsize=18)
    
    plt.subplot(2,1,2)
    plt.plot(list(range(INPUT_trainlength)), origin_real_y[:INPUT_trainlength], 'b-*', linewidth=2, markersize=4)
    plt.plot(list(range(INPUT_trainlength, INPUT_trainlength+predict_len-1)), origin_real_y[INPUT_trainlength:INPUT_trainlength+predict_len-1], 'c-p', linewidth=2, markersize=4)
    plt.plot(list(range(INPUT_trainlength, INPUT_trainlength+predict_len-1)), union_predict_y_ARNN, 'ro', linewidth=2, markersize=5)
    plt.title(f'ARNN Union Pred: KnownLen={trainlength}, PredLen={predict_len-1}, RMSE={RMSE}', fontsize=18)
    
    plt.draw()
    plt.pause(0.1)
    plt.clf()
    
    
    
    
    
    
    

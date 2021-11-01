import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot3 as uproot

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from pyquickhelper.helpgen.graphviz_helper import plot_graphviz

from sklearn.metrics import log_loss
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from imblearn.under_sampling import RandomUnderSampler
import optuna
import gc

import time

from pathlib import Path
from scipy.misc import derivative
import dill
import pickle
def getTrainingModel():
    return os.environ['ML_MODULE']

def getParameters():

    params = []

    params.append(matchTree["MFT_X"].array())
    params.append(matchTree["MFT_Y"].array())
    params.append(matchTree["MFT_Phi"].array())
    params.append(matchTree["MFT_Tanl"].array())
    params.append(matchTree["MFT_InvQPt"].array())
    params.append(matchTree["MFT_Cov00"].array())
    params.append(matchTree["MFT_Cov01"].array())
    params.append(matchTree["MFT_Cov11"].array())
    params.append(matchTree["MFT_Cov02"].array())
    params.append(matchTree["MFT_Cov12"].array())
    params.append(matchTree["MFT_Cov22"].array())
    params.append(matchTree["MFT_Cov03"].array())
    params.append(matchTree["MFT_Cov13"].array())
    params.append(matchTree["MFT_Cov23"].array())
    params.append(matchTree["MFT_Cov33"].array())
    params.append(matchTree["MFT_Cov04"].array())
    params.append(matchTree["MFT_Cov14"].array())
    params.append(matchTree["MFT_Cov24"].array())
    params.append(matchTree["MFT_Cov34"].array())
    params.append(matchTree["MFT_Cov44"].array())

    params.append(matchTree["MCH_X"].array())
    params.append(matchTree["MCH_Y"].array())
    params.append(matchTree["MCH_Phi"].array())
    params.append(matchTree["MCH_Tanl"].array())
    params.append(matchTree["MCH_InvQPt"].array())
    params.append(matchTree["MCH_Cov00"].array())
    params.append(matchTree["MCH_Cov01"].array())
    params.append(matchTree["MCH_Cov11"].array())
    params.append(matchTree["MCH_Cov02"].array())
    params.append(matchTree["MCH_Cov12"].array())
    params.append(matchTree["MCH_Cov22"].array())
    params.append(matchTree["MCH_Cov03"].array())
    params.append(matchTree["MCH_Cov13"].array())
    params.append(matchTree["MCH_Cov23"].array())
    params.append(matchTree["MCH_Cov33"].array())
    params.append(matchTree["MCH_Cov04"].array())
    params.append(matchTree["MCH_Cov14"].array())
    params.append(matchTree["MCH_Cov24"].array())
    params.append(matchTree["MCH_Cov34"].array())
    params.append(matchTree["MCH_Cov44"].array())

    params.append(matchTree["MFT_TrackChi2"].array())
    params.append(matchTree["MFT_NClust"].array())

    params.append(matchTree["MatchingScore"].array())

    params.append(addTree["Delta_Z"].array())

    return np.array(params)

def calcFeatures():

    params = getParameters()

    features = []

    MFT_X = params[0]
    MFT_Y = params[1]
    MFT_Phi = params[2]
    MFT_Tanl = params[3]
    MFT_InvQPt = params[4]
    MFT_QPt = 1./MFT_InvQPt

    MCH_X = params[20]
    MCH_Y = params[21]
    MCH_Phi = params[22]
    MCH_Tanl = params[23]
    MCH_InvQPt = params[24]
    MCH_QPt = 1./MCH_InvQPt

    MFT_Cov00 = params[5]
    MFT_Cov01 = params[6]
    MFT_Cov11 = params[7]
    MFT_Cov02 = params[8]
    MFT_Cov12 = params[9]
    MFT_Cov22 = params[10]
    MFT_Cov03 = params[11]
    MFT_Cov13 = params[12]
    MFT_Cov23 = params[13]
    MFT_Cov33 = params[14]
    MFT_Cov04 = params[15]
    MFT_Cov14 = params[16]
    MFT_Cov24 = params[17]
    MFT_Cov34 = params[18]
    MFT_Cov44 = params[19]

    MCH_Cov00 = params[25]
    MCH_Cov01 = params[26]
    MCH_Cov11 = params[27]
    MCH_Cov02 = params[28]
    MCH_Cov12 = params[29]
    MCH_Cov22 = params[30]
    MCH_Cov03 = params[31]
    MCH_Cov13 = params[32]
    MCH_Cov23 = params[33]
    MCH_Cov33 = params[34]
    MCH_Cov04 = params[35]
    MCH_Cov14 = params[36]
    MCH_Cov24 = params[37]
    MCH_Cov34 = params[38]
    MCH_Cov44 = params[39]

    MFT_TrackChi2 = params[40]
    MFT_NClust = params[41]
    MFT_TrackReducedChi2 = MFT_TrackChi2/MFT_NClust

    MatchingScore = params[42]

    Delta_Z = params[43]

    MFT_Ch = np.where( MFT_InvQPt < 0, -1, 1)

    MFT_Pt = 1./np.abs(MFT_InvQPt)
    MFT_Px = np.cos(MFT_Phi) * MFT_Pt
    MFT_Py = np.sin(MFT_Phi) * MFT_Pt
    MFT_Pz = MFT_Tanl * MFT_Pt
    MFT_P = MFT_Pt * np.sqrt(1. + MFT_Tanl*MFT_Tanl)
    #MFT_Eta = -np.log(np.tan((np.pi/2. - np.arctan(MFT_Tanl)) / 2))

    MCH_Ch = np.where( MCH_InvQPt < 0, -1, 1)

    MCH_Pt = 1./np.abs(MCH_InvQPt)
    MCH_Px = np.cos(MCH_Phi) * MCH_Pt
    MCH_Py = np.sin(MCH_Phi) * MCH_Pt
    MCH_Pz = MCH_Tanl * MCH_Pt
    MCH_P = MCH_Pt * np.sqrt(1. + MCH_Tanl*MCH_Tanl)

    #MCH_Eta = -np.log(np.tan((np.pi/2. - np.arctan(MCH_Tanl)) / 2))

    Delta_X = MCH_X - MFT_X
    Delta_Y = MCH_Y - MFT_Y
    Delta_XY = np.sqrt((MCH_X - MFT_X)**2 + (MCH_Y - MFT_Y)**2)
    Delta_Phi = MCH_Phi - MFT_Phi
    #Delta_Eta = MCH_Eta - MFT_Eta
    Delta_Tanl = MCH_Tanl - MFT_Tanl
    Delta_InvQPt = MCH_InvQPt - MFT_InvQPt
    Delta_Pt = MCH_Pt - MFT_Pt
    Delta_Px = MCH_Px - MFT_Px
    Delta_Py = MCH_Py - MFT_Py
    Delta_Pz = MCH_Pz - MFT_Pz
    Delta_P = MCH_P - MFT_P
    Delta_Ch = MCH_Ch - MFT_Ch

    Ratio_X = MCH_X / MFT_X
    Ratio_Y = MCH_Y / MFT_Y
    Ratio_Phi = MCH_Phi / MFT_Phi
    Ratio_Tanl = MCH_Tanl / MFT_Tanl
    Ratio_InvQPt = MCH_InvQPt / MFT_InvQPt
    Ratio_Pt = MCH_Pt / MFT_Pt
    Ratio_Px = MCH_Px / MFT_Px
    Ratio_Py = MCH_Py / MFT_Py
    Ratio_Pz = MCH_Pz / MFT_Pz
    Ratio_P = MCH_P / MFT_P
    Ratio_Ch = MCH_Ch / MFT_Ch

    features.append(MFT_InvQPt)
    features.append(MFT_X)
    features.append(MFT_Y)
    features.append(MFT_Phi)
    features.append(MFT_Tanl)

    features.append(MCH_InvQPt)
    features.append(MCH_X)
    features.append(MCH_Y)
    features.append(MCH_Phi)
    features.append(MCH_Tanl)

    features.append(MFT_TrackChi2)
    features.append(MFT_NClust)
    features.append(MFT_TrackReducedChi2)
    features.append(MatchingScore)

    features.append(Delta_InvQPt)
    features.append(Delta_X)
    features.append(Delta_Y)
    features.append(Delta_XY)
    features.append(Delta_Phi)
    features.append(Delta_Tanl)

    features.append(Delta_Ch)
    '''
    features.append(MFT_Pt)
    features.append(MCH_Pt)
    features.append(Delta_Pt)
    features.append(Ratio_Pt)
    '''
    features.append(Ratio_InvQPt)
    features.append(Ratio_X)
    features.append(Ratio_Y)
    features.append(Ratio_Phi)
    features.append(Ratio_Tanl)

    features.append(Ratio_Ch)

    features.append(Delta_Z)

    print(Ratio_Ch)

    return features

def getExpVar():
    features = calcFeatures()
    training_list = []
    for index in range(len(features)):
        training_list.append(features[index])
    X = np.stack(training_list,1)

    return X

def getInputDim(X):
    rowExpVarDim,colExpVarDim = X.shape
    return rowExpVarDim,colExpVarDim

def getObjVar():
    return matchTree.array("Truth")

def getData(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0,stratify=y)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,test_size=0.2,random_state=0,stratify=y_train)
    del X
    del y
    gc.collect()
    return X_train,y_train,X_test,y_test,X_eval,y_eval

def getSampledTrainData(X_train,y_train):
    print('down sampling now ...')
    sampler = RandomUnderSampler(sampling_strategy={0: y_train.sum(), 1: y_train.sum()}, random_state=42)
    X_train_sampled, y_train_sampled = sampler.fit_resample(X_train, y_train)
    del X_train
    del y_train
    gc.collect()
    return X_train_sampled,y_train_sampled


########################################################################
########################################################################
####                                                                ####
####                                                                ####
####                        MAIN FUNCTION                           ####
####                                                                ####
####                                                                ####
########################################################################
########################################################################


#Global variables
file = uproot.open(os.environ['ML_TRAINING_FILE'])
matchTree = file["matchTree"]

X = getExpVar()
y = getObjVar()

print("y")
print(y)

rowExpVarDim,colExpVarDim = getInputDim(X)
#X_train,y_train,X_test,y_test,X_eval,y_eval = getData(X,y)
#X_train,y_train = getSampledTrainData(X_train,y_train) #get balanced training data

def prauc(data,preds):
    #sampling_rate = 4446./(4446.+713951.)
    #calibPreds = calibration(preds, sampling_rate)
    precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(data, preds)
    area_lgb = auc(recall_lgb, precision_lgb)
    metric = area_lgb
    return 'PR-AUC', metric, True

def normalizedEntropy(data,preds):
    p =4446./(4446.+713951.)
    normalizedlogloss = log_loss(data,preds)/(-p*math.log(p)-(1.-p)*math.log(1.-p))
    metric = normalizedlogloss
    return 'NE', metric, False
def get_focal_loss(x, y):
    return focal_loss_lgb_sk(x, y, 0.25, 2.)
def get_focal_error(x, y):
    return focal_loss_lgb_eval_error_sk(x, y, 0.25, 2.)

def main():
    with open('./deltaZmodel.dill', 'rb') as pklmodel:
        model = dill.load(pklmodel)
    #model = pickle.load(pklmodel, open('./focalmodel.pkl','rb'))
    pred_model_proba = model.predict_proba(X)
    #pred_model_proba = 1./(1.+np.exp(-model.predict_proba(X)))
    #print('prediction')
    #print(pred_model_proba)
    atest = 0.99975
    gtest = 3.
    #focal_test = -(atest*y*(1.-pred_model_proba)**gtest)*np.log(pred_model_proba)-((1.-atest)*(1.-y)*pred_model_proba**gtest)*np.log(1-pred_model_proba)
    #focal_test = np.mean(focal_test)
    #print ("test Focal Loss: %0.8f" % focal_test)
    #pred_array = np.insert(getParameters(), 0, pred_model_proba, axis=1)
    predTree = uproot.newtree({"EventID": int,
                               "MFTtrackID": int,
                               "MCHtrackID": int,
                               "prediction": float,
                               "truth": int
                           })
    expoFile = uproot.recreate("prediction.root")
    expoFile["predTree"] = predTree
    expoFile["predTree"].extend({"EventID": matchTree.array("EventID"),
                                 "MFTtrackID": matchTree.array("MFTtrackID"),
                                 "MCHtrackID": matchTree.array("MCHtrackID"),
                                 "prediction": pred_model_proba[:,1],
                                 "truth": matchTree.array("Truth"),
                                 })

if __name__ == "__main__":
    main()

import os
import sys
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot3 as uproot

import onnxmltools
import onnxruntime as rt
from onnxmltools.convert.common.data_types import FloatTensorType, BooleanTensorType, Int32TensorType, DoubleTensorType, Int64TensorType

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from pyquickhelper.helpgen.graphviz_helper import plot_graphviz
from mlprodict.onnxrt import OnnxInference

from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.shape_calculator import calculate_linear_classifier_output_shapes  # noqa

from onnxmltools.convert.lightgbm.operator_converters.LightGbm import convert_lightgbm  # noqa
from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost  # noqa
from skl2onnx.common.data_types import FloatTensorType

import lightgbm as lgb
import xgboost as xgb

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers.experimental import preprocessing

import keras2onnx
import tf2onnx
'''

from sklearn.metrics import log_loss
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from imblearn.under_sampling import RandomUnderSampler
import optuna
import gc

import time

from pathlib import Path
from sklearn.metrics import f1_score
from scipy.misc import derivative
import dill
import pickle
def getTrainingModel():
    return os.environ['ML_MODULE']

def getParameters():

    params = []

    params.append(matchTree.array("MFT_X"))
    params.append(matchTree.array("MFT_Y"))
    params.append(matchTree.array("MFT_Phi"))
    params.append(matchTree.array("MFT_Tanl"))
    params.append(matchTree.array("MFT_InvQPt"))
    params.append(matchTree.array("MFT_Cov00"))
    params.append(matchTree.array("MFT_Cov01"))
    params.append(matchTree.array("MFT_Cov11"))
    params.append(matchTree.array("MFT_Cov02"))
    params.append(matchTree.array("MFT_Cov12"))
    params.append(matchTree.array("MFT_Cov22"))
    params.append(matchTree.array("MFT_Cov03"))
    params.append(matchTree.array("MFT_Cov13"))
    params.append(matchTree.array("MFT_Cov23"))
    params.append(matchTree.array("MFT_Cov33"))
    params.append(matchTree.array("MFT_Cov04"))
    params.append(matchTree.array("MFT_Cov14"))
    params.append(matchTree.array("MFT_Cov24"))
    params.append(matchTree.array("MFT_Cov34"))
    params.append(matchTree.array("MFT_Cov44"))

    params.append(matchTree.array("MCH_X"))
    params.append(matchTree.array("MCH_Y"))
    params.append(matchTree.array("MCH_Phi"))
    params.append(matchTree.array("MCH_Tanl"))
    params.append(matchTree.array("MCH_InvQPt"))
    params.append(matchTree.array("MCH_Cov00"))
    params.append(matchTree.array("MCH_Cov01"))
    params.append(matchTree.array("MCH_Cov11"))
    params.append(matchTree.array("MCH_Cov02"))
    params.append(matchTree.array("MCH_Cov12"))
    params.append(matchTree.array("MCH_Cov22"))
    params.append(matchTree.array("MCH_Cov03"))
    params.append(matchTree.array("MCH_Cov13"))
    params.append(matchTree.array("MCH_Cov23"))
    params.append(matchTree.array("MCH_Cov33"))
    params.append(matchTree.array("MCH_Cov04"))
    params.append(matchTree.array("MCH_Cov14"))
    params.append(matchTree.array("MCH_Cov24"))
    params.append(matchTree.array("MCH_Cov34"))
    params.append(matchTree.array("MCH_Cov44"))

    params.append(matchTree.array("MFT_TrackChi2"))
    params.append(matchTree.array("MFT_NClust"))

    params.append(matchTree.array("MatchingScore"))

    params.append(addTree.array("Delta_Z"))
    print("np.array(params)")
    print(addTree.array("Delta_Z").shape)
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
    #print(Ratio_Ch)
    print("MFT_X")
    print(MFT_X.shape)
    print("Delta_Z")
    print(Delta_Z.shape)
    print("params[43]")
    print(params[43].shape)
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

def buildModel_lightGBM():
    model = LGBMClassifier(boosting_type='gbdt',objective='binary',learning_rate=0.01,max_depth=10,n_estimators=150000,metric="custom")
    '''
    model = LGBMClassifier(boosting_type='gbdt',
                           objective='binary',
                           learning_rate=0.586339908940463,
                           max_depth=3,
                           n_estimators=150000,
                           metric="custom",
                           num_leaves=83,
                           min_child_samples=68,
                           min_child_weight=72.20006600361943,
                           reg_alpha=28.94771523379064,
                           reg_lambda=18.327304287091614,
                           subsample=0.02108743864130606,
                           subsample_freq=1,
                           colsample_bytree=0.0056907807163545186)
    '''
    '''
    model = LGBMClassifier(boosting_type='gbdt',
                           objective='binary',
                           learning_rate=0.58,
                           max_depth=3,
                           n_estimators=150000,
                           metric="custom",
                           num_leaves=83,
                           min_child_samples=68,
                           min_child_weight=72.2,
                           reg_alpha=28.9,
                           reg_lambda=18.3,)
    '''
    return model

def registerConvONNX_lightGBM():
    update_registered_converter(
        LGBMClassifier, 'LightGbmLGBMClassifier',
        calculate_linear_classifier_output_shapes, convert_lightgbm,
        options={'nocl': [True, False]}
    )

def buildONNXModel_lightGBM(model):
    registerConvONNX_lightGBM()
    return convert_sklearn(model, 'lightgbm',[('input', FloatTensorType([None, colExpVarDim]))],target_opset=12)

def getPredict_lightGBM(model,onnx_model_name,X_test,y_test):

    session_model = rt.InferenceSession(onnx_model_name)

    input_name = session_model.get_inputs()[0].name
    output_name1 = session_model.get_outputs()[0].name #labels
    output_name2= session_model.get_outputs()[1].name #probabilitoes
    pred_onnx_model = session_model.run([output_name1], {"input": X_test.astype(np.float32)})
    pred_onnx_model_proba = session_model.run([output_name2], {"input": X_test.astype(np.float32)})
    pred_model = model.predict(X_test)
    pred_model_proba = model.predict_proba(X_test)

    pred_onnx_model = np.array(pred_onnx_model[0])
    pred_onnx_model_proba = np.array(pred_onnx_model_proba[0])

    pred_onnx_model_proba = pred_onnx_model_proba[:,1]
    pred_model_proba = pred_model_proba[:,1]

    print("accuracy:  ",accuracy_score(y_test,pred_onnx_model))
    print("precision: ",precision_score(y_test,pred_onnx_model))

    print('Pure-LightGBM predicted proba')
    print(pred_model_proba)
    #print(X_test[:,2])
    #print(X_test[:,6])

    print('ONNX-LightGBM predicted proba')
    print(pred_onnx_model_proba)

    return pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba

def buildModel_XGBoost():
    model = XGBClassifier(n_estimators=10000,use_label_encoder=False)
    return model

def registerConvONNX_XGBoost():
    update_registered_converter(
        XGBClassifier, 'XGBoostXGBClassifier',
        calculate_linear_classifier_output_shapes, convert_xgboost,
        options={'nocl': [True, False]}
    )

def buildONNXModel_XGBoost(model):
    registerConvONNX_XGBoost()
    return convert_sklearn(model, 'xgboost',[('input', FloatTensorType([None, colExpVarDim]))],target_opset=12)

def getPredict_XGBoost(model,onnx_model_name,X_test,y_test):

    session_model = rt.InferenceSession(onnx_model_name)

    input_name = session_model.get_inputs()[0].name
    output_name1 = session_model.get_outputs()[0].name #labels
    output_name2= session_model.get_outputs()[1].name #probabilitoes

    pred_onnx_model = session_model.run([output_name1], {"input": X_test.astype(np.float32)})
    pred_onnx_model_proba = session_model.run([output_name2], {"input": X_test.astype(np.float32)})
    pred_model = model.predict(X_test)
    pred_model_proba = model.predict_proba(X_test)

    pred_onnx_model = np.array(pred_onnx_model[0])
    pred_onnx_model_proba = np.array(pred_onnx_model_proba[0])

    pred_onnx_model_proba = pred_onnx_model_proba[:,1]
    pred_model_proba = pred_model_proba[:,1]

    print("accuracy:  ",accuracy_score(y_test,pred_onnx_model))
    print("precision: ",precision_score(y_test,pred_onnx_model))

    print(type(pred_model_proba))
    print('Pure-XGBoost predicted proba')
    print(pred_model_proba)

    print('ONNX-XGBoost predicted proba')
    print(pred_onnx_model_proba)

    return pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba

'''
def buildModel_tensorflowNN(normalize, colExpVarDim):

    model = tf.keras.Sequential([
        #normalize,
        layers.Dense(colExpVarDim, use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(colExpVarDim, activation='sigmoid',use_bias=False),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer=tf.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])

    model.summary

    return model

def showHistory_tensorFlowNN(history):
    #{'loss': [0.14186885952949524], 'categorical_accuracy': [1.0], 'val_loss': [0.11649563908576965], 'val_categorical_accuracy': [1.0]}
    # Plot training & validation accuracy values
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def buildONNXModel_tensorflowNN(model):
    model_onnx, external_tensor_storage = tf2onnx.convert.from_keras(model,input_signature=None, opset=None, custom_ops=None,
                                                                      custom_op_handlers=None, custom_rewriter=None,
                                                                      inputs_as_nchw=None, extra_opset=None, shape_override=None,
                                                                      target=None, large_model=False, output_path=None)
    return  model_onnx

def getPredict_tensorflowNN(model,onnx_model_name,X_test,y_test):

    pred_model_proba = model.predict(X_test)
    pred_model = np.where(pred_model_proba>0.5, True, False)

    model_onnx = rt.InferenceSession(onnx_model_name)

    pred_onnx_model_proba = model_onnx.run([model_onnx.get_outputs()[0].name], { model_onnx.get_inputs()[0].name: X_test.astype(np.float32)})
    pred_onnx_model_proba = pred_onnx_model_proba[0]

    pred_onnx_model = np.where(pred_onnx_model_proba>0.5, True, False)

    return pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba

def showInfo_tensorflowNN(training_history):
    default_theme()

    # summarize history for accuracy
    plt.plot(training_history.history['accuracy'])
    plt.plot(training_history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # summarize history for loss
    plt.plot(training_history.history['loss'])
    plt.plot(training_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
'''
def default_theme():
    plt.rc('axes', labelsize=16, labelweight='bold')
    plt.rc('xtick', labelsize=14)
    plt.rc('figure', figsize=(10, 6))
    plt.rc('ytick', labelsize=14)
    plt.rc('svg', fonttype='none')

def saveONNXModel(model_onnx,model_name):
    with open(model_name, "wb") as f:
        f.write(model_onnx.SerializeToString())

def showONNXInfo(model_onnx):
    oinf = OnnxInference(model_onnx)
    ax = plot_graphviz(oinf.to_dot())
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

def calibration(y_proba, beta):
    return y_proba / (y_proba + (1 - y_proba) / beta)

def focal_loss_lgb_sk(y_true, y_pred, alpha, gamma):
    """
    Focal Loss for lightgbm

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a,g = alpha, gamma
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        #p = x
        #return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
        return -(a*t*(1.-p)**g)*np.log(p)*2./p-((1.-a)*(1.-t)*p**g)*np.log(1-p)*2./(1-p)
    def get_partial_fl(x):
        return fl(x, y_true)
    #partial_fl = lambda x: fl(x, y_true)
    #partial_fl = get_partial_fl(x)
    grad = derivative(get_partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(get_partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

def focal_loss_lgb_eval_error_sk(y_true, y_pred, alpha, gamma):
    """
    Adapation of the Focal Loss for lightgbm to be used as evaluation loss

    Parameters:
    -----------
    y_pred: numpy.ndarray
        array with the predictions
    dtrain: lightgbm.Dataset
    alpha, gamma: float
        See original paper https://arxiv.org/pdf/1708.02002.pdf
    """
    a,g = alpha, gamma
    p = 1/(1+np.exp(-y_pred))
    #p = y_pred
    #loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    loss = -(a*y_true*(1.-p)**g)*np.log(p)*2./p-((1.-a)*(1.-y_true)*p**g)*np.log(1-p)*2./(1-p)
    return 'focal_loss', np.mean(loss), False

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
addTree = file["addTree"]

X = getExpVar()
y = getObjVar()

print("y")
print(y)

rowExpVarDim,colExpVarDim = getInputDim(X)
X_train,y_train,X_test,y_test,X_eval,y_eval = getData(X,y)
#X_train,y_train = getSampledTrainData(X_train,y_train) #get balanced training data

print("training data")
print("positive")
print(np.sum(y_train == 1))
print("negative")
print(np.sum(y_train == 0))
print("validation data")
print("positive")
print(np.sum(y_eval == 1))
print("negative")
print(np.sum(y_eval == 0))
print("test data")
print("positive")
print(np.sum(y_test == 1))
print("negative")
print(np.sum(y_test == 0))

model_type=getTrainingModel()

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
    return focal_loss_lgb_sk(x, y, 0.99 , 5.)
def get_focal_error(x, y):
    return focal_loss_lgb_eval_error_sk(x, y, 0.99 , 5.)
#def classScaledLogloss(data,preds):

'''    
def objectives(trial):

    # optunaでのハイパーパラメータサーチ範囲の設定
    params = {
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 2, 100),
            'max_depth': trial.suggest_int('max_depth', 2, 10),
            'min_child_weight': trial.suggest_uniform('min_child_weight', 2., 100.),
            'reg_alpha': trial.suggest_uniform('reg_alpha', 0.01, 100.),
            'reg_lambda': trial.suggest_uniform('reg_lambda', 0.01, 100.),
            'learning_rate': trial.suggest_uniform('learning_rate',0.001,0.999),
            'n_estimators': 10000,
            'subsample': trial.suggest_uniform('subsample',0.001,1.0),
            'subsample_freq': trial.suggest_int('subsample_freq',0.,1.0),
            'colsample_bytree':trial.suggest_uniform('colsample_bytree',0.001,1.0),
            }

    # LightGBMで学習+予測
    model = lgb.LGBMClassifier(**params)# 追加部分
    model.fit(X_train, y_train,eval_set=(X_eval,y_eval),eval_metric='logloss',early_stopping_rounds=2000,verbose=False)

    y_pred = model.predict(X_test)
    y_pred_proba_eval = model.predict_proba(X_eval)

    # 検証データを用いた評価
    precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(y_eval, y_pred_proba_eval[:, 1])
    area_lgb = auc(recall_lgb, precision_lgb)
    score = area_lgb

    return score
'''
def main():

    if model_type == 'lightGBM':

        #focal_loss = lambda x,y: focal_loss_lgb_sk(x, y, 0.25, 2.)
        #eval_error = lambda x,y: focal_loss_lgb_eval_error_sk(x, y, 0.25, 2.)
        '''
        def get_focal_loss(x, y):
            return focal_loss_lgb_sk(x, y, 0.25, 2.)
        def get_focal_error(x, y):
            return focal_loss_lgb_eval_error_sk(x, y, 0.25, 2.)
        #focal_loss = get_focal_loss(x, y)
        #eval_error = get_focal_error(x, y)
        '''
        #model = LGBMClassifier(boosting_type='gbdt',objective=get_focal_loss,n_estimators=150,metric="custom")
        model=LGBMClassifier(boosting_type='gbdt',objective='binary',learning_rate=0.01,max_depth=10,n_estimators=150000,metric="custom")
        print("training is starting")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric=prauc,
            early_stopping_rounds=1000,)

        '''
        model = buildModel_lightGBM()
        training_start = time.time()
        training_history = model.fit(X_train, y_train,
            eval_metric=prauc,
            eval_set=[
                (X_train, y_train),
                (X_eval, y_eval),
            ],
            eval_names=['train', 'validation'],
            early_stopping_rounds=1000,
        )
        training_elapsed_time = time.time() - training_start
        print ("training_elapsed_time:{0}".format(training_elapsed_time) + "[sec]")
        '''
        '''
        # optunaによる最適化呼び出し
        opt = optuna.create_study(direction='maximize',sampler=optuna.samplers.RandomSampler(seed=0))
        opt.optimize(objectives, n_trials=500)

        # 最適パラメータ取得
        trial = opt.best_trial
        params_best = dict(trial.params.items())
        params_best['random_seed'] = 0

        # 最適パラメータで学習/予測
        model_o = lgb.LGBMClassifier(**params_best)# 追加部分
        model_o.fit(X_train, y_train,eval_set=(X_eval,y_eval),eval_metric='logloss',early_stopping_rounds=2000,verbose=False)
        y_test_pred = model_o.predict(X_test)
        y_test_proba = model_o.predict_proba(X_test)

        precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(y_test, y_test_proba[:, 1])
        area_lgb = auc(recall_lgb, precision_lgb)
        print ("AUPR score: %0.2f" % area_lgb)
        '''
        pickle.dump(model, open('./deltaZmodel.dill','wb'))
        #model_onnx = buildONNXModel_lightGBM(model)
        
        #saveONNXModel(model_onnx,'lightGBM.onnx')

        #pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba = getPredict_lightGBM(model,'lightGBM.onnx',X_test,y_test)
        #pred_model_proba = 1./(1.+np.exp(-model.predict_proba(X_test)))
        pred_model_proba = model.predict_proba(X_test)
        #pred_model_proba = pred_model_proba[:,1]
        #print('prediction')
        np.set_printoptions(threshold=np.inf)
        #print(pred_model_proba)
        precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(y_test, pred_model_proba[:,1])
        area_lgb = auc(recall_lgb, precision_lgb)
        atest = 0.99975
        gtest = 4.
        #focal_test = -(atest*y_test*(1.-pred_model_proba)**gtest)*np.log(pred_model_proba)-((1.-atest)*(1.-y_test)*pred_model_proba**gtest)*np.log(1-pred_model_proba)
        #focal_test = np.mean(focal_test)
        #print ("test PR-AUC score: %0.8f" % area_lgb)
        #print ("test Focal Loss: %0.8f" % focal_test)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(1, 1, 1)
        lgb.plot_metric(model)
        ax.plot(recall_lgb, precision_lgb, label='LightGBM(AUC = %0.2f)' % area_lgb)
        ax.set_xlabel('Recall(=Efficiency)')
        ax.set_ylabel('Precision(=Purity)')
        ax.set_ylim([0.0, 2.0])
        ax.set_xlim([0.0, 1.2])
        ax.set_title('Precision(Purity)-Recall(Efficiency) curve')
        ax.legend(loc="upper right")
        fig.savefig('./MLResultPRCurveLGBM.png', format="png")
        plt.savefig("./MLResultPRAUCLGBM.png", format="png")
        plt.show()
        
        print('LightGBM feature importance')
        print(model.booster_.feature_importance(importance_type='gain'))

    elif model_type == 'XGBoost':
        model = buildModel_XGBoost()

        #training_history = model.fit(X_train, y_train)

        training_history = model.fit(X_train, y_train,
            eval_metric='aucpr',
            eval_set=[
                (X_train, y_train),
                (X_eval, y_eval),
            ],
            early_stopping_rounds=1000,
        )

        model_onnx = buildONNXModel_XGBoost(model)

        saveONNXModel(model_onnx,'XGBoost.onnx')

        pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba = getPredict_XGBoost(model,'XGBoost.onnx',X_test,y_test)
        precision_xgb, recall_xgb, thresholds_xgb = precision_recall_curve(y_test, pred_model_proba)
        area_xgb = auc(recall_xgb, precision_xgb)

        print ("AUPR score: %0.2f" % area_xgb)

        results = model.evals_result()
        epochs = len(results['validation_0']['aucpr'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['aucpr'], label='Train')
        ax.plot(x_axis, results['validation_1']['aucpr'], label='Test')
        ax.legend()
        plt.ylabel('PR-AUC')
        plt.title('XGBoost PR-AUC')
        plt.savefig("./MLResultPRAUCXGBoost.png", format="png")

        ax.plot(recall_xgb, precision_xgb, label='XGBoost(AUC = %0.2f)' % area_xgb)
        ax.set_xlabel('Recall(=Efficiency)')
        ax.set_ylabel('Precision(=Purity)')
        ax.set_ylim([0.0, 2.0])
        ax.set_xlim([0.0, 1.2])
        ax.set_title('Precision(Purity)-Recall(Efficiency) curve')
        ax.legend(loc="upper right")
        fig.savefig('MLResultPRCurveXBoost.png', format="png")
        '''
    elif model_type == 'tfNN':

        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

        normalize = preprocessing.Normalization()

        model = buildModel_tensorflowNN(normalize, colExpVarDim) #mode = model instance, training_history = test results with eval data

        training_history = model.fit(X_train, y_train,
                                     epochs=150,
                                     batch_size=1000,
                                     verbose=1,
                                     validation_data=(X_eval, y_eval))


        model_onnx = buildONNXModel_tensorflowNN(model)

        saveONNXModel(model_onnx,'tfNN.onnx')

        pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba = getPredict_tensorflowNN(model,'tfNN.onnx',X_test,y_test)

        print("accuracy:  ",accuracy_score(y_test,pred_onnx_model))
        print("precision: ",precision_score(y_test,pred_onnx_model))

        showInfo_tensorflowNN(training_history)
        '''
    else:
        print("You didn't select corect ML module option\n")
        print("You must chose one module from lightGBM, TensorFlowNN, XGBoost\n")

if __name__ == "__main__":
    main()

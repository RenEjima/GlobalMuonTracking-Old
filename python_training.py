import os
import sys
import math
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import uproot

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

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.layers.experimental import preprocessing

import keras2onnx
import tf2onnx

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

from imblearn.under_sampling import RandomUnderSampler

import gc

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

    MFT_Ch = np.where( MFT_InvQPt < 0, -1, 1)

    MFT_Pt = 1./np.abs(MFT_InvQPt)
    MFT_Px = np.cos(MFT_Phi) * MFT_Pt
    MFT_Py = np.sin(MFT_Phi) * MFT_Pt
    MFT_Pz = MFT_Tanl * MFT_Pt
    MFT_P = MFT_Pt * np.sqrt(1. + MFT_Tanl*MFT_Tanl)
    MFT_Eta = -np.log(np.tan((np.pi/2. - np.arctan(MFT_Tanl)) / 2))

    MCH_Ch = np.where( MCH_InvQPt < 0, -1, 1)

    MCH_Pt = 1./np.abs(MCH_InvQPt)
    MCH_Px = np.cos(MCH_Phi) * MCH_Pt
    MCH_Py = np.sin(MCH_Phi) * MCH_Pt
    MCH_Pz = MCH_Tanl * MCH_Pt
    MCH_P = MCH_Pt * np.sqrt(1. + MCH_Tanl*MCH_Tanl)
    MCH_Eta = -np.log(np.tan((np.pi/2. - np.arctan(MCH_Tanl)) / 2))

    Delta_X = MCH_X - MFT_X
    Delta_Y = MCH_Y - MFT_Y
    Delta_XY = np.sqrt((MCH_X - MFT_X)**2 + (MCH_Y - MFT_Y)**2)
    Delta_Phi = MCH_Phi - MFT_Phi
    Delta_Eta = MCH_Eta - MFT_Eta
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

    #features.append(MFT_X)
    #features.append(MFT_Y)
    features.append(MFT_Phi)
    #features.append(MFT_Tanl)
    features.append(MFT_Eta)
    features.append(MFT_Pt)
    #features.append(MFT_Px)
    #features.append(MFT_Py)
    #features.append(MFT_Pz)
    features.append(MFT_P)
    features.append(MFT_Ch)

    #features.append(MCH_X)
    #features.append(MCH_Y)
    features.append(MCH_Phi)
    #features.append(MCH_Tanl)
    features.append(MCH_Eta)
    features.append(MCH_Pt)
    #features.append(MCH_Px)
    #features.append(MCH_Py)
    #features.append(MCH_Pz)
    features.append(MCH_P)
    features.append(MCH_Ch)

    features.append(MFT_TrackChi2)
    features.append(MFT_NClust)
    features.append(MFT_TrackReducedChi2)
    features.append(MatchingScore)
    '''
    features.append(MFT_Cov00)
    features.append(MFT_Cov01)
    features.append(MFT_Cov11)
    features.append(MFT_Cov02)
    features.append(MFT_Cov12)
    features.append(MFT_Cov22)
    features.append(MFT_Cov03)
    features.append(MFT_Cov13)
    features.append(MFT_Cov23)
    features.append(MFT_Cov33)
    features.append(MFT_Cov04)
    features.append(MFT_Cov14)
    features.append(MFT_Cov24)
    features.append(MFT_Cov34)
    features.append(MFT_Cov44)

    features.append(MCH_Cov00)
    features.append(MCH_Cov01)
    features.append(MCH_Cov11)
    features.append(MCH_Cov02)
    features.append(MCH_Cov12)
    features.append(MCH_Cov22)
    features.append(MCH_Cov03)
    features.append(MCH_Cov13)
    features.append(MCH_Cov23)
    features.append(MCH_Cov33)
    features.append(MCH_Cov04)
    features.append(MCH_Cov14)
    features.append(MCH_Cov24)
    features.append(MCH_Cov34)
    features.append(MCH_Cov44)
    '''
    features.append(Delta_X)
    features.append(Delta_Y)
    features.append(Delta_XY)
    features.append(Delta_Phi)
    #features.append(Delta_Tanl)
    features.append(Delta_Eta)
    features.append(Delta_Pt)
    features.append(Delta_Px)
    features.append(Delta_Py)
    features.append(Delta_Pz)
    features.append(Delta_P)
    features.append(Delta_Ch)
    '''
    features.append(Ratio_X)
    features.append(Ratio_Y)
    features.append(Ratio_Phi)
    features.append(Ratio_Tanl)
    features.append(Ratio_Pt)
    features.append(Ratio_Px)
    features.append(Ratio_Py)
    features.append(Ratio_Pz)
    features.append(Ratio_P)
    features.append(Ratio_Ch)
    '''
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
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled,test_size=0.2,random_state=0,stratify=y_sampled)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,test_size=0.2,random_state=0,stratify=y_train)
    return X_train,y_train,X_test,y_test,X_eval,y_eval
    
def getSampledData(X,y):
    print('down sampling now ...')
    sampler = RandomUnderSampler(sampling_strategy={0: y.sum(), 1: y.sum()}, random_state=42)
    X_sampled, y_sampled = sampler.fit_resample(X, y)
    del X
    del y
    gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled,test_size=0.2,random_state=0,stratify=y_sampled)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train,test_size=0.2,random_state=0,stratify=y_train)
    return X_train,y_train,X_test,y_test,X_eval,y_eval

def buildModel_lightGBM():
    model = LGBMClassifier(boosting_type='gbdt',objective='binary',learning_rate=0.01,max_depth=20,n_estimators=10000,metric="custom")
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

    print(type(pred_model_proba))
    print('Pure-LightGBM predicted proba')
    print(pred_model_proba)

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

rowExpVarDim,colExpVarDim = getInputDim(X)
#X_train,y_train,X_test,y_test,X_eval,y_eval = getData(X,y)
X_train,y_train,X_test,y_test,X_eval,y_eval = getSampledData(X,y) #get balanced data

model_type=getTrainingModel()

def prauc(data,preds):
    precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(data, preds)
    area_lgb = auc(recall_lgb, precision_lgb)
    metric = area_lgb
    return 'PR-AUC', metric, True

def main():

    if model_type == 'lightGBM':
        model = buildModel_lightGBM()

        #training_history = model.fit(X_train, y_train)

        training_history = model.fit(X_train, y_train,
            eval_metric=prauc,
            eval_set=[
                (X_train, y_train),
                (X_eval, y_eval),
            ],
            eval_names=['train', 'validation'],
            early_stopping_rounds=1000,
        )

        model_onnx = buildONNXModel_lightGBM(model)

        saveONNXModel(model_onnx,'lightGBM.onnx')

        pred_model, pred_onnx_model, pred_model_proba, pred_onnx_model_proba = getPredict_lightGBM(model,'lightGBM.onnx',X_test,y_test)
        precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(y_test, pred_model_proba)
        area_lgb = auc(recall_lgb, precision_lgb)
        print ("AUPR score: %0.2f" % area_lgb)
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

    else:
        print("You didn't select corect ML module option\n")
        print("You must chose one module from lightGBM, TensorFlowNN, XGBoost\n")

if __name__ == "__main__":
    main()

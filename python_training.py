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

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.pretraining import TabNetPretrainer
import torch
import torchvision

import gc

def LoadData():
    print('Loading Branch as Numpy array...')
    MFT_X      = matchTree["MFT_X"].array()
    MFT_X      = np.array(MFT_X)
    MFT_Y      = matchTree["MFT_Y"].array()
    MFT_Y      = np.array(MFT_Y)
    MFT_Phi    = matchTree["MFT_Phi"].array()
    MFT_Phi    = np.array(MFT_Phi)
    MFT_Tanl   = matchTree["MFT_Tanl"].array()
    MFT_Tanl   = np.array(MFT_Tanl)
    MFT_InvQPt = matchTree["MFT_InvQPt"].array()
    MFT_InvQPt = np.array(MFT_InvQPt)
    MFT_Cov00  = matchTree["MFT_Cov00"].array()
    MFT_Cov00  = np.array(MFT_Cov00)
    MFT_Cov01  = matchTree["MFT_Cov01"].array()
    MFT_Cov01  = np.array(MFT_Cov01)
    MFT_Cov11  = matchTree["MFT_Cov11"].array()
    MFT_Cov11  = np.array(MFT_Cov11)
    MFT_Cov02  = matchTree["MFT_Cov02"].array()
    MFT_Cov02  = np.array(MFT_Cov02)
    MFT_Cov12  = matchTree["MFT_Cov12"].array()
    MFT_Cov12  = np.array(MFT_Cov12)
    MFT_Cov22  = matchTree["MFT_Cov22"].array()
    MFT_Cov22  = np.array(MFT_Cov22)
    MFT_Cov03  = matchTree["MFT_Cov03"].array()
    MFT_Cov03  = np.array(MFT_Cov03)
    MFT_Cov13  = matchTree["MFT_Cov13"].array()
    MFT_Cov13  = np.array(MFT_Cov13)
    MFT_Cov23  = matchTree["MFT_Cov23"].array()
    MFT_Cov23  = np.array(MFT_Cov23)
    MFT_Cov33  = matchTree["MFT_Cov33"].array()
    MFT_Cov33  = np.array(MFT_Cov33)
    MFT_Cov04  = matchTree["MFT_Cov04"].array()
    MFT_Cov04  = np.array(MFT_Cov04)
    MFT_Cov14  = matchTree["MFT_Cov14"].array()
    MFT_Cov14  = np.array(MFT_Cov14)
    MFT_Cov24  = matchTree["MFT_Cov24"].array()
    MFT_Cov24  = np.array(MFT_Cov24)
    MFT_Cov34  = matchTree["MFT_Cov34"].array()
    MFT_Cov34  = np.array(MFT_Cov34)
    MFT_Cov44  = matchTree["MFT_Cov44"].array()
    MFT_Cov44  = np.array(MFT_Cov44)
    MCH_X      = matchTree["MCH_X"].array()
    MCH_X      = np.array(MCH_X)
    MCH_Y      = matchTree["MCH_Y"].array()
    MCH_Y      = np.array(MCH_Y)
    MCH_Phi    = matchTree["MCH_Phi"].array()
    MCH_Phi    = np.array(MCH_Phi)
    MCH_Tanl   = matchTree["MCH_Tanl"].array()
    MCH_Tanl   = np.array(MCH_Tanl)
    MCH_InvQPt = matchTree["MCH_InvQPt"].array()
    MCH_InvQPt = np.array(MCH_InvQPt)
    MCH_Cov00  = matchTree["MCH_Cov00"].array()
    MCH_Cov00  = np.array(MCH_Cov00)
    MCH_Cov01  = matchTree["MCH_Cov01"].array()
    MCH_Cov01  = np.array(MCH_Cov01)
    MCH_Cov11  = matchTree["MCH_Cov11"].array()
    MCH_Cov11  = np.array(MCH_Cov11)
    MCH_Cov02  = matchTree["MCH_Cov02"].array()
    MCH_Cov02  = np.array(MCH_Cov02)
    MCH_Cov12  = matchTree["MCH_Cov12"].array()
    MCH_Cov12  = np.array(MCH_Cov12)
    MCH_Cov22  = matchTree["MCH_Cov22"].array()
    MCH_Cov22  = np.array(MCH_Cov22)
    MCH_Cov03  = matchTree["MCH_Cov03"].array()
    MCH_Cov03  = np.array(MCH_Cov03)
    MCH_Cov13  = matchTree["MCH_Cov13"].array()
    MCH_Cov13  = np.array(MCH_Cov13)
    MCH_Cov23  = matchTree["MCH_Cov23"].array()
    MCH_Cov23  = np.array(MCH_Cov23)
    MCH_Cov33  = matchTree["MCH_Cov33"].array()
    MCH_Cov33  = np.array(MCH_Cov33)
    MCH_Cov04  = matchTree["MCH_Cov04"].array()
    MCH_Cov04  = np.array(MCH_Cov04)
    MCH_Cov14  = matchTree["MCH_Cov14"].array()
    MCH_Cov14  = np.array(MCH_Cov14)
    MCH_Cov24  = matchTree["MCH_Cov24"].array()
    MCH_Cov24  = np.array(MCH_Cov24)
    MCH_Cov34  = matchTree["MCH_Cov34"].array()
    MCH_Cov34  = np.array(MCH_Cov34)
    MCH_Cov44  = matchTree["MCH_Cov44"].array()
    MCH_Cov44  = np.array(MCH_Cov44)

    MFT_TrackChi2   = matchTree["MFT_TrackChi2"].array()
    MFT_TrackChi2   = np.array(MFT_TrackChi2)
    MFT_NClust      = matchTree["MFT_NClust"].array()
    MFT_NClust      = np.array(MFT_NClust)
    #HIROSHIMA_MATCHING_SCORE = matchTree["HIROSHIMA_MATCHING_SCORE"].array()
    #HIROSHIMA_MATCHING_SCORE = np.array(HIROSHIMA_MATCHING_SCORE)

    CorrectMatching = matchTree["Truth"].array()
    CorrectMatching = np.array(CorrectMatching)

    Delta_X      = MCH_X      - MFT_X;
    Delta_Y      = MCH_Y      - MFT_Y;
    Delta_XY     = np.sqrt((MCH_X-MFT_X)**2 + (MCH_Y-MFT_Y)**2)
    Delta_Phi    = MCH_Phi    - MFT_Phi;
    Delta_Tanl   = MCH_Tanl   - MFT_Tanl;
    Delta_InvQPt = MCH_InvQPt - MFT_InvQPt;
    Delta_Cov00  = MCH_Cov00 - MFT_Cov00;
    Delta_Cov01  = MCH_Cov01 - MFT_Cov01;
    Delta_Cov11  = MCH_Cov11 - MFT_Cov11;
    Delta_Cov02  = MCH_Cov02 - MFT_Cov02;
    Delta_Cov12  = MCH_Cov12 - MFT_Cov12;
    Delta_Cov22  = MCH_Cov22 - MFT_Cov22;
    Delta_Cov03  = MCH_Cov03 - MFT_Cov03;
    Delta_Cov13  = MCH_Cov13 - MFT_Cov13;
    Delta_Cov23  = MCH_Cov23 - MFT_Cov23;
    Delta_Cov33  = MCH_Cov33 - MFT_Cov33;
    Delta_Cov04  = MCH_Cov04 - MFT_Cov04;
    Delta_Cov14  = MCH_Cov14 - MFT_Cov14;
    Delta_Cov24  = MCH_Cov24 - MFT_Cov24;
    Delta_Cov34  = MCH_Cov34 - MFT_Cov34;
    Delta_Cov44  = MCH_Cov44 - MFT_Cov44;

    Ratio_X      = MCH_X      - MFT_X;
    Ratio_Y      = MCH_Y      - MFT_Y;
    Ratio_XY     = np.sqrt((MCH_X-MFT_X)**2 + (MCH_Y-MFT_Y)**2)
    Ratio_Phi    = MCH_Phi    - MFT_Phi;
    Ratio_Tanl   = MCH_Tanl   - MFT_Tanl;
    Ratio_InvQPt = MCH_InvQPt - MFT_InvQPt;
    Ratio_Cov00  = MCH_Cov00 - MFT_Cov00;
    Ratio_Cov01  = MCH_Cov01 - MFT_Cov01;
    Ratio_Cov11  = MCH_Cov11 - MFT_Cov11;
    Ratio_Cov02  = MCH_Cov02 - MFT_Cov02;
    Ratio_Cov12  = MCH_Cov12 - MFT_Cov12;
    Ratio_Cov22  = MCH_Cov22 - MFT_Cov22;
    Ratio_Cov03  = MCH_Cov03 - MFT_Cov03;
    Ratio_Cov13  = MCH_Cov13 - MFT_Cov13;
    Ratio_Cov23  = MCH_Cov23 - MFT_Cov23;
    Ratio_Cov33  = MCH_Cov33 - MFT_Cov33;
    Ratio_Cov04  = MCH_Cov04 - MFT_Cov04;
    Ratio_Cov14  = MCH_Cov14 - MFT_Cov14;
    Ratio_Cov24  = MCH_Cov24 - MFT_Cov24;
    Ratio_Cov34  = MCH_Cov34 - MFT_Cov34;
    Ratio_Cov44  = MCH_Cov44 - MFT_Cov44;


    MFT_TrackReducedChi2 = MFT_TrackChi2/MFT_NClust;

    print('Stacking arrays ...')
    training_list=np.stack([MFT_X,MFT_Y,MFT_Phi,MFT_Tanl,MFT_InvQPt,MFT_Cov00,MFT_Cov01,MFT_Cov11,MFT_Cov02,MFT_Cov12,MFT_Cov22,MFT_Cov03,MFT_Cov13,MFT_Cov23,MFT_Cov33,MFT_Cov04,MFT_Cov14,MFT_Cov24,MFT_Cov34,MFT_Cov44,
                            MCH_X,MCH_Y,MCH_Phi,MCH_Tanl,MCH_InvQPt,MCH_Cov00,MCH_Cov01,MCH_Cov11,MCH_Cov02,MCH_Cov12,MCH_Cov22,MCH_Cov03,MCH_Cov13,MCH_Cov23,MCH_Cov33,MCH_Cov04,MCH_Cov14,MCH_Cov24,MCH_Cov34,MCH_Cov44,
                            MFT_TrackChi2,MFT_NClust,MFT_TrackReducedChi2,
                            Delta_X,Delta_Y,Delta_XY,Delta_Phi,Delta_Tanl,Delta_InvQPt,Delta_Cov00,Delta_Cov01,Delta_Cov11,Delta_Cov02,Delta_Cov12,Delta_Cov22,Delta_Cov03,Delta_Cov13,Delta_Cov23,Delta_Cov33,Delta_Cov04,Delta_Cov14,Delta_Cov24,Delta_Cov34,Delta_Cov44,
                            Ratio_X,Ratio_Y,Ratio_XY,Ratio_Phi,Ratio_Tanl,Ratio_InvQPt,Ratio_Cov00,Ratio_Cov01,Ratio_Cov11,Ratio_Cov02,Ratio_Cov12,Ratio_Cov22,Ratio_Cov03,Ratio_Cov13,Ratio_Cov23,Ratio_Cov33,Ratio_Cov04,Ratio_Cov14,Ratio_Cov24,Ratio_Cov34,Ratio_Cov44,
                            #HIROSHIMA_MATCHING_SCORE
                            ], axis=1);

    X = training_list
    y = CorrectMatching
    return X,y

def getTrainingModel():
    return os.environ['ML_MODULE']

def getInputDim(X):
    rowExpVarDim,colExpVarDim = X.shape
    return rowExpVarDim,colExpVarDim

def getData(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0,stratify=y)
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

def prauc(data,preds):
    precision_lgb, recall_lgb, thresholds_lgb = precision_recall_curve(data, preds)
    area_lgb = auc(recall_lgb, precision_lgb)
    metric = area_lgb
    return 'PR-AUC', metric, True

def main():
    #X = getExpVar()
    #y = getObjVar()
    X,y=LoadData()
    rowExpVarDim,colExpVarDim = getInputDim(X)
    X_train,y_train,X_test,y_test,X_eval,y_eval = getData(X,y)

    model_type=getTrainingModel()

    if model_type == 'lightGBM':
        model = buildModel_lightGBM()
        print(np.isfinite(X_train))
        print(np.isfinite(y_train))
        print(np.isfinite(X_test))
        print(np.isfinite(y_test))
        print(X_train)
        print(y_train)
        print(X_test)
        print(y_test)

        #training_history = model.fit(X_train, y_train)

        training_history = model.fit(X_train, y_train,
            eval_metric=prauc,
            eval_set=[
                (X_train, y_train),
                (X_test, y_test),
            ],
            eval_names=['train', 'test'],
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
        fig.savefig('/home/ejima/disk1/GlobalMuonTracking-ONNXRuntime/hijingTrain2/MLResultPRCurveLGBM.png', format="png")
        plt.savefig("/home/ejima/disk1/GlobalMuonTracking-ONNXRuntime/hijingTrain2/MLResultPRAUCLGBM.png", format="png")

    elif model_type == 'XGBoost':
        model = buildModel_XGBoost()

        #training_history = model.fit(X_train, y_train)

        training_history = model.fit(X_train, y_train,
            eval_metric='aucpr',
            eval_set=[
                (X_train, y_train),
                (X_test, y_test),
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
        plt.savefig("/home/ejima/disk1/GlobalMuonTracking-ONNXRuntime/hijingTrain2/MLResultPRAUCXGBoost.png", format="png")

        ax.plot(recall_xgb, precision_xgb, label='XGBoost(AUC = %0.2f)' % area_xgb)
        ax.set_xlabel('Recall(=Efficiency)')
        ax.set_ylabel('Precision(=Purity)')
        ax.set_ylim([0.0, 2.0])
        ax.set_xlim([0.0, 1.2])
        ax.set_title('Precision(Purity)-Recall(Efficiency) curve')
        ax.legend(loc="upper right")
        fig.savefig('/home/ejima/disk1/GlobalMuonTracking-ONNXRuntime/hijingTrain2/MLResultPRCurveXBoost.png', format="png")

    elif model_type == 'TabNet':
        del X_test
        del y_test
        del X_train
        del y_train
        gc.collect()
        X_testvalid, X_train, y_testvalid, y_train = train_test_split(X, y,test_size=0.8,random_state=0,stratify=y)
        X_test, X_valid, y_test, y_valid = train_test_split(X_testvalid, y_testvalid,test_size=0.5,random_state=0,stratify=y_testvalid)
        del X
        del y
        del X_testvalid
        del y_testvalid
        gc.collect()
        # TabNetPretrainer
        unsupervised_model = TabNetPretrainer(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            mask_type='entmax' # "sparsemax"
        )

        unsupervised_model.fit(X_train=X_train,eval_set=[X_valid],pretraining_ratio=0.8)

        tab_clf = TabNetClassifier(
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size":10, # how to use learning rate scheduler
                              "gamma":0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type='sparsemax' # This will be overwritten if using pretrain model
        )

        tab_clf.fit(
            X_train=X_train, y_train=y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_name=['train', 'valid'],
            eval_metric=['auc'],
            from_unsupervised=unsupervised_model
        )
        #ROC, PR and AUC
        y_proba_tab = tab_clf.predict_proba(X_test)

        precision_tab, recall_tab, thresholds_tab = precision_recall_curve(y_test, y_proba_tab[:, 1])
        area_tab = auc(recall_tab, precision_tab)
        print ("AUPR score: %0.2f" % area_tab)

        torch.onnx.export(tab_clf,'TabNet.onnx', verbose=True)

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

# -*- coding: utf-8 -*-
# @Author  : Peida Wu

import json

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from keras.models import Sequential, Model
from keras.layers import Convolution2D, MaxPooling2D, Conv3D, MaxPooling3D, ZeroPadding3D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.utils.np_utils import to_categorical
from scipy.io import savemat
from sklearn.decomposition import PCA
from keras.optimizers import Adam, SGD, Adadelta, RMSprop, Nadam
import keras.callbacks as kcallbacks
from keras.regularizers import l2
import time
import collections
from sklearn import metrics, preprocessing
from tensorflow import Tensor
from tensorflow.python.framework import ops

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ResneXt_IN_Dual_Network
import os
from keras.utils import plot_model
from pylab import *
from keras import backend as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
        # counter
        assign_0 = value // Col + pad_length
        assign_1 = value % Col + pad_length
        new_assign[counter] = [assign_0, assign_1]
    return new_assign


def assignmentToIndex(assign_0, assign_1, Row, Col):
    new_index = assign_0 * Col + assign_1
    return new_index


def selectNeighboringPatch(matrix, pos_row, pos_col, ex_len):
    selected_rows = matrix[range(pos_row - ex_len, pos_row + ex_len + 1), :]
    selected_patch = selected_rows[:, range(pos_col - ex_len, pos_col + ex_len + 1)]
    return selected_patch


# divide dataset into train and test datasets
def sampling(proptionVal, groundTruth):
    # train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    print(m)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    train_indices = []
    test_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    print(len(test_indices))
    # 8194
    print(len(train_indices))
    # 2055
    return train_indices, test_indices


def model():
    model = ResneXt_IN_Dual_Network.ResneXt_IN((1, img_rows, img_cols, img_channels), classes=9)

    RMS = RMSprop(lr=0.0003)

    # Training

    # TODO modified loss
    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)

        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)

        return (1 - e) * loss1 + e * loss2

    model.compile(loss=mycrossentropy, optimizer=RMS, metrics=['accuracy'])  # categorical_crossentropy

    model.summary()
    plot_model(model, show_shapes=True, to_file='./model_3D_ResNeXt_UP.png')

    return model


# load data
mat_data = sio.loadmat('./Datasets/UP/PaviaU.mat')
data_IN = mat_data['paviaU']
# load label
mat_gt = sio.loadmat('./Datasets/UP/PaviaU_gt.mat')
gt_IN = mat_gt['paviaU_gt']

print('data_IN shape:', data_IN.shape)
# (145,145,200)
print('gt_IN shape:', gt_IN.shape)
# (145,145)

new_gt_IN = gt_IN

batch_size = 16
nb_classes = 9
nb_epoch = 60
# 7 9 11 13 15
img_rows, img_cols = 11, 11

patience = 100

INPUT_DIMENSION_CONV = 103  # 200
INPUT_DIMENSION = 103  # 200

TOTAL_SIZE = 42776
VAL_SIZE = 4281
# 8558 12838 17113 21391
# Train:Val:Test = 2:1:7 3:1:6 4:1:5 5:1:4
# Validation_split = 0.8 0.7 0.6 0.5
TRAIN_SIZE = 17113
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.6

img_channels = 103  # 200
# TODO 和可变参数1一起改变，7--15，3--7
PATCH_LENGTH = 5  # Patch_size

print('data_IN.shape[:2]:', data_IN.shape[:2])
# (145,145)
print('np.prod(data_IN.shape[:2]:', np.prod(data_IN.shape[:2]))
# 21025 = 145 * 145
print('data_IN.shape[2:]:', data_IN.shape[2:])
# (200,)
print('np.prod(data_IN.shape[2:]:', np.prod(data_IN.shape[2:]))
# 200
print('np.prod(new_gt_IN.shape[:2]:', np.prod(new_gt_IN.shape[:2]))
# 21025

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

data = preprocessing.scale(data)
print('data.shape:', data.shape)
# (21025, 200)


data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
# (145, 145, 200)
whole_data = data_

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)
print('padded_data.shape:', padded_data.shape)

ITER = 1
CATEGORY = 9  # 16

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print('train_data.shape:', train_data.shape)
# (2055, 11, 11, 200)
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
print('test_data.shape:', test_data.shape)
# (8194, 11, 11, 200)


KAPPA_3D_ResNeXt = []
OA_3D_ResNeXt = []
AA_3D_ResNeXt = []
TRAINING_TIME_3D_ResNeXt = []
TESTING_TIME_3D_ResNeXt = []
ELEMENT_ACC_3D_ResNeXt = np.zeros((ITER, CATEGORY))

seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))
    # # 1 Iteration

    # save the best validated model, using easystopping
    best_weights_ResNeXt_path = './models/UP_best_3D_ResNeXt_4_1_5_60_' + str(index_iter + 1) + '.hdf5'

    # sampling
    np.random.seed(seeds[index_iter])
    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt)

    y_train = gt[train_indices] - 1
    y_train = to_categorical(np.asarray(y_train))

    y_test = gt[test_indices] - 1
    y_test = to_categorical(np.asarray(y_test))

    train_assign = indexToAssignment(train_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(train_assign)):
        train_data[i] = selectNeighboringPatch(padded_data, train_assign[i][0], train_assign[i][1], PATCH_LENGTH)

    test_assign = indexToAssignment(test_indices, whole_data.shape[0], whole_data.shape[1], PATCH_LENGTH)
    for i in range(len(test_assign)):
        test_data[i] = selectNeighboringPatch(padded_data, test_assign[i][0], test_assign[i][1], PATCH_LENGTH)

    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION_CONV)
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION_CONV)

    # x_train y_train
    # x_test y_test
    # x_val y_val
    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    ############################################################################################################
    model_ResNeXt = model()

    earlyStopping6 = kcallbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='auto')

    saveBestModel6 = kcallbacks.ModelCheckpoint(best_weights_ResNeXt_path, monitor='val_loss', verbose=1,
                                                save_best_only=True,
                                                mode='auto')

    tic6 = time.clock()
    print(x_train.shape, x_test.shape)
    history_3d_ResNeXt = model_ResNeXt.fit(
        x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3], 1), y_train,
        validation_data=(x_val.reshape(x_val.shape[0], x_val.shape[1], x_val.shape[2], x_val.shape[3], 1), y_val),
        batch_size=batch_size,
        epochs=nb_epoch, shuffle=True, callbacks=[earlyStopping6, saveBestModel6])
    toc6 = time.clock()

    # save loss_acc curve
    with open('./Loss_Acc/UP_label.hdf5', 'w') as f:
        json.dump(history_3d_ResNeXt.history, f)

    # plot loss and acc curve
    plt.plot(history_3d_ResNeXt.history['acc'])
    plt.plot(history_3d_ResNeXt.history['val_acc'])
    # my_y_ticks = np.arange(0, 1.1, 0.1)
    # my_x_ticks = np.arange(0, 65, 5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_Acc', 'Val_Acc'], loc='lower right')
    plt.show()

    plt.plot(history_3d_ResNeXt.history['loss'])
    plt.plot(history_3d_ResNeXt.history['val_loss'])
    # my_y_ticks = np.arange(0, 16, 1)
    # my_x_ticks = np.arange(0, 65, 5)
    # plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_Loss', 'Val_Loss'], loc='upper right')
    plt.show()

    tic7 = time.clock()
    loss_and_metrics = model_ResNeXt.evaluate(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1), y_test,
        batch_size=batch_size)
    toc7 = time.clock()

    print('3D ResNeXt Time: ', toc6 - tic6)
    print('3D ResNeXt Test time:', toc7 - tic7)

    print('3D ResNeXt Test score:', loss_and_metrics[0])
    print('3D ResNeXt Test accuracy:', loss_and_metrics[1])

    pred_test = model_ResNeXt.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)

    collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1

    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_ResNeXt.append(kappa)
    OA_3D_ResNeXt.append(overall_acc)
    AA_3D_ResNeXt.append(average_acc)
    TRAINING_TIME_3D_ResNeXt.append(toc6 - tic6)
    TESTING_TIME_3D_ResNeXt.append(toc7 - tic7)
    ELEMENT_ACC_3D_ResNeXt[index_iter, :] = each_acc

    print("3D ResNeXt finished.")
    print("# %d Iteration" % (index_iter + 1))

# save records
modelStatsRecord.outputStats(KAPPA_3D_ResNeXt, OA_3D_ResNeXt, AA_3D_ResNeXt, ELEMENT_ACC_3D_ResNeXt,
                             TRAINING_TIME_3D_ResNeXt, TESTING_TIME_3D_ResNeXt,
                             history_3d_ResNeXt, loss_and_metrics, CATEGORY,
                             './records/UP_train_3D_ResNeXt_4_1_5_60_1.txt',
                             './records/UP_train_element_3D_ResNeXt_4_1_5_60_1.txt')

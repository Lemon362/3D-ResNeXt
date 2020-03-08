# -*- coding: utf-8 -*-
# @Author  : Peida Wu
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
from keras import backend as K

from Utils import zeroPadding, normalization, doPCA, modelStatsRecord, averageAccuracy, ResneXt_IN_Dual_Network


def indexToAssignment(index_, Row, Col, pad_length):
    new_assign = {}
    for counter, value in enumerate(index_):
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


def sampling(proptionVal, groundTruth):  # divide dataset into train and test datasets
    labels_loc = {}
    train = {}
    test = {}
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        nb_val = int(proptionVal * len(indices))
        train[i] = indices[:-nb_val]
        test[i] = indices[-nb_val:]
    # whole_indices = []
    train_indices = []
    test_indices = []
    for i in range(m):
        #        whole_indices += labels_loc[i]
        train_indices += train[i]
        test_indices += test[i]
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    return train_indices, test_indices


def model():
    model = ResneXt_IN_Dual_Network.ResneXt_IN((1, img_rows, img_cols, img_channels), cardinality=8, classes=16)

    RMS = RMSprop(lr=0.0003)

    def mycrossentropy(y_true, y_pred, e=0.1):
        loss1 = K.categorical_crossentropy(y_true, y_pred)

        loss2 = K.categorical_crossentropy(K.ones_like(y_pred) / nb_classes, y_pred)  # K.ones_like(y_pred) / nb_classes

        return (1 - e) * loss1 + e * loss2

    model.compile(loss=mycrossentropy, optimizer=RMS, metrics=['accuracy'])

    return model


mat_data = sio.loadmat('D:/3D-ResNeXt-master/Datasets/IN/Indian_pines_corrected.mat')
data_IN = mat_data['indian_pines_corrected']
mat_gt = sio.loadmat('D:/3D-ResNeXt-master/Datasets/IN/Indian_pines_gt.mat')
gt_IN = mat_gt['indian_pines_gt']

print(data_IN.shape)

new_gt_IN = gt_IN

batch_size = 16
nb_classes = 16
nb_epoch = 100
img_rows, img_cols = 11, 11
patience = 100

INPUT_DIMENSION_CONV = 200  # 200
INPUT_DIMENSION = 200  # 200

TOTAL_SIZE = 10249  # 10249
VAL_SIZE = 1025  # 1025

TRAIN_SIZE = 5128  # 1031 2055 3081 4106 5128 6153

TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE
VALIDATION_SPLIT = 0.5

img_channels = 200  # 200
PATCH_LENGTH = 5  # Patch_size

data = data_IN.reshape(np.prod(data_IN.shape[:2]), np.prod(data_IN.shape[2:]))
gt = new_gt_IN.reshape(np.prod(new_gt_IN.shape[:2]), )

data = preprocessing.scale(data)

data_ = data.reshape(data_IN.shape[0], data_IN.shape[1], data_IN.shape[2])
whole_data = data_

padded_data = zeroPadding.zeroPadding_3D(whole_data, PATCH_LENGTH)

ITER = 1
CATEGORY = 16  # 16

train_data = np.zeros((TRAIN_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))
test_data = np.zeros((TEST_SIZE, 2 * PATCH_LENGTH + 1, 2 * PATCH_LENGTH + 1, INPUT_DIMENSION_CONV))

KAPPA_3D_ResNeXt = []
OA_3D_ResNeXt = []
AA_3D_ResNeXt = []
TRAINING_TIME_3D_ResNeXt = []
TESTING_TIME_3D_ResNeXt = []
ELEMENT_ACC_3D_ResNeXt = np.zeros((ITER, CATEGORY))

seeds = [1334]

for index_iter in range(ITER):
    print("# %d Iteration" % (index_iter + 1))

    best_weights_ResNeXt_path = 'D:/3D-ResNeXt-master/models/Indian_best_3D_ResNeXt_5_1_4_60_' + str(
        index_iter + 1) + '.hdf5'

    np.random.seed(seeds[index_iter])
    #    train_indices, test_indices = sampleFixNum.samplingFixedNum(TRAIN_NUM, gt)
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

    x_val = x_test_all[-VAL_SIZE:]
    y_val = y_test[-VAL_SIZE:]

    x_test = x_test_all[:-VAL_SIZE]
    y_test = y_test[:-VAL_SIZE]

    model_ResNeXt = model()

    model_ResNeXt.load_weights(best_weights_ResNeXt_path)

    pred_test = model_ResNeXt.predict(
        x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], x_test.shape[3], 1)).argmax(axis=1)

    counter = collections.Counter(pred_test)

    gt_test = gt[test_indices] - 1
    overall_acc = metrics.accuracy_score(pred_test, gt_test[:-VAL_SIZE])
    confusion_matrix = metrics.confusion_matrix(pred_test, gt_test[:-VAL_SIZE])
    each_acc, average_acc = averageAccuracy.AA_andEachClassAccuracy(confusion_matrix)
    kappa = metrics.cohen_kappa_score(pred_test, gt_test[:-VAL_SIZE])
    KAPPA_3D_ResNeXt.append(kappa)
    OA_3D_ResNeXt.append(overall_acc)
    AA_3D_ResNeXt.append(average_acc)
    # TRAINING_TIME_3D_ResNeXt.append(toc6 - tic6)
    # TESTING_TIME_3D_ResNeXt.append(toc7 - tic7)
    ELEMENT_ACC_3D_ResNeXt[index_iter, :] = each_acc

    print("3D ResNeXt  finished.")
    print("# %d Iteration" % (index_iter + 1))

    # print(counter)
modelStatsRecord.outputStats_assess(KAPPA_3D_ResNeXt, OA_3D_ResNeXt, AA_3D_ResNeXt, ELEMENT_ACC_3D_ResNeXt,
                                    CATEGORY,
                                    'D:/3D-ResNeXt-master/records/IN_test_3D_ResNeXt_5_1_4_60_1.txt',
                                    'D:/3D-ResNeXt-master/records/IN_test_element_3D_ResNeXt_5_1_4_60_1.txt')

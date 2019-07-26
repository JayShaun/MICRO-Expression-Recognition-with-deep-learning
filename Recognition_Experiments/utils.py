'''
load_data:resize and iterator

reconstruct data : channel connect

by zxl
'''
import random
import numpy as np
import cv2
import os
from tensorflow.python.keras.models import Model, load_model
def center_crop(x, center_crop_size):
    centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
    halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
    cropped = x[centerw - halfw : centerw + halfw,
                 centerh - halfh : centerh + halfh, :]

    return cropped

def scale_byRatio(img_path, return_width=224, crop_method=center_crop, rgb2gray=False):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img = crop_method(img, (shorter, shorter))
    img = cv2.resize(img, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    if rgb2gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img
def reconstruct_data(x):
    shape = x.shape
    return x.reshape(shape[0]*shape[1],shape[2], shape[3], shape[4])
def generator_batch_5c(raw_path, flowimg_path, strainimg_path, data_list, nbr_classes=5, length = 10,
                       batch_size=32, return_label=True, crop_method=center_crop, reconstruct = True,
                       img_width=224, img_height=224, shuffle=True):
    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, length, img_width, img_height, 5))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line

            img_path = line[0]
            raw_imgs = os.listdir(raw_path + img_path)
            raw_imgs.sort(key=lambda x: int(x[:-4]))
            flow_imgs = os.listdir(flowimg_path + img_path)
            flow_imgs.sort(key=lambda x: int(x[:-4]))
            strain_imgs = os.listdir(strainimg_path + img_path)
            strain_imgs.sort(key=lambda x: int(x[:-4]))
            for j in range(length):
                raw_img_name = raw_imgs[j]
                gray_img = scale_byRatio(raw_path + img_path + '/'+raw_img_name, return_width=img_width,
                                crop_method=crop_method, rgb2gray=True)

                flow_img_name = flow_imgs[j]
                flow_img = scale_byRatio(flowimg_path + img_path + '/'+flow_img_name, return_width=img_width,
                                crop_method=crop_method)

                strain_img_name = flow_imgs[j]
                strain_img = scale_byRatio(strainimg_path + img_path + '/'+strain_img_name, return_width=img_width,
                                crop_method=crop_method, rgb2gray=True)

                X_batch[i - current_index, j, :, :, 0] = gray_img
                X_batch[i - current_index, j, :, :, 1:4] = flow_img
                X_batch[i - current_index, j, :, :, 4] = strain_img

            if return_label:
                label = int(line[-1])
                Y_batch[i - current_index, label] = 1
        X_batch = X_batch.astype(np.float64)

        if reconstruct:
            X_batch = reconstruct_data(X_batch)
            Y_batch = np.repeat(Y_batch, length, axis=0)

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch

def generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, data_list, nbr_classes=5, length = 10,
                       batch_size=32, return_label=True, crop_method=center_crop, reconstruct = True,
                       img_width=224, img_height=224, shuffle=True):
    N = len(data_list)

    if shuffle:
        random.shuffle(data_list)

    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                random.shuffle(data_list)

        X_batch = np.zeros((current_batch_size, length, img_width, img_height, 5))
        Y_batch = np.zeros((current_batch_size, nbr_classes))

        for i in range(current_index, current_index + current_batch_size):
            line = data_list[i].strip().split(' ')
            #print line

            img_path = line[0]
            raw_imgs = os.listdir(raw_path + img_path)
            raw_imgs.sort(key=lambda x: int(x[:-4]))
            flow_imgs = os.listdir(flowimg_path + img_path)
            flow_imgs.sort(key=lambda x: int(x[:-4]))
            strain_imgs = os.listdir(strainimg_path + img_path)
            strain_imgs.sort(key=lambda x: int(x[:-4]))
            for j in range(length):
                raw_img_name = raw_imgs[j]
                gray_img = scale_byRatio(raw_path + img_path + '/'+raw_img_name, return_width=img_width,
                                crop_method=crop_method, rgb2gray=True)

                flow_img_name = flow_imgs[j]
                flow_img = scale_byRatio(flowimg_path + img_path + '/'+flow_img_name, return_width=img_width,
                                crop_method=crop_method)

                strain_img_name = flow_imgs[j]
                strain_img = scale_byRatio(strainimg_path + img_path + '/'+strain_img_name, return_width=img_width,
                                crop_method=crop_method, rgb2gray=True)

                X_batch[i - current_index, j, :, :, 0] = gray_img
                X_batch[i - current_index, j, :, :, 1:4] = flow_img
                X_batch[i - current_index, j, :, :, 4] = strain_img

            if return_label:
                label = int(line[-1])
                Y_batch[i - current_index, label] = 1
        X_batch = X_batch.astype(np.float64)

        if reconstruct:
            X_batch = reconstruct_data(X_batch)
            #Y_batch = np.repeat(Y_batch, length, axis=0)

        X_batch_output = model.predict_on_batch(X_batch)
        X_batch_feature = X_batch_output.reshape(-1, length, X_batch_output.shape[1])
        if return_label:
            yield (X_batch_feature, Y_batch)
        else:
            yield X_batch_feature

if __name__ == '__main__':
    raw_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/'
    flowimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/'
    strainimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticalstrain_image/'
    data_list_path = './train_list.txt'
    f = open(data_list_path, 'r')
    data_list = f.readlines()
    vgg_model = load_model('./models/VGG_16_5_channels.h5')
    model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
    model.summary()
    X = generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, data_list, batch_size=6)
    for x in X:
        print(x[0].shape, type(x[1]))
    f.close()
    # # img = cv2.imread('/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/disgust/1/1.jpg')
    # # print(img.shape)
    # # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('s', img)
    # # cv2.waitKey(0)
from models import  temporal_module, VGG_16_5_channels, temporal_module_with_attention
from utils import generator_batch_5c, generator_batch_feature
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from math import ceil
from tensorflow.python.keras.models import Model, load_model
import numpy as np
def train_spatial(batch_size = 32, epoch=1000):

    raw_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/'
    flowimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/'
    strainimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticalstrain_image/'
    best_model_file = "./models/VGG_16_5_channels.h5"

    train_data_list_path = './train_list.txt'
    f1 = open(train_data_list_path, 'r')
    train_data_list = f1.readlines()
    steps_per_epoch = int(ceil(len(train_data_list)*1.0 / batch_size))

    test_data_list_path = './test_list.txt'
    f2 = open(test_data_list_path, 'r')
    test_data_list = f2.readlines()
    validation_steps = int(ceil(len(test_data_list)*1.0 / batch_size))
    # X = generator_batch_5c(raw_path, flowimg_path, strainimg_path, data_list)
    vgg_model = VGG_16_5_channels()
    vgg_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, decay=0.000001), metrics=["accuracy"])
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc',
                                verbose = 1, save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)
    vgg_model.fit_generator(generator_batch_5c(raw_path, flowimg_path, strainimg_path, train_data_list, batch_size=batch_size),
                            steps_per_epoch = steps_per_epoch, epochs = epoch, verbose = 1,
                        validation_data = generator_batch_5c(raw_path, flowimg_path, strainimg_path, test_data_list, batch_size=batch_size),
                            validation_steps=validation_steps,
                        class_weight = None, callbacks = [best_model, reduce_lr],
                        max_queue_size = 80, workers = 8, use_multiprocessing=False)
    f1.close()
    f2.close()

def train_temporal(batch_size = 6, epoch=1000):
    raw_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/'
    flowimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/'
    strainimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticalstrain_image/'
    best_model_file = "./models/VGG_16_5_channels_temporal.h5"

    train_data_list_path = './train_list.txt'
    f1 = open(train_data_list_path, 'r')
    train_data_list = f1.readlines()
    steps_per_epoch = int(ceil(len(train_data_list)*1.0 / batch_size))

    test_data_list_path = './test_list.txt'
    f2 = open(test_data_list_path, 'r')
    test_data_list = f2.readlines()
    validation_steps = int(ceil(len(test_data_list)*1.0 / batch_size))

    vgg_model = load_model('./models/VGG_16_5_channels.h5')
    model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
    model.predict_on_batch(np.zeros((10, 224, 224, 5))) #https://www.jianshu.com/p/c84ae0527a3f
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc',
                                verbose = 1, save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)
    temporal_model = temporal_module(data_dim=4096)
    temporal_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, decay=0.000001), metrics=["accuracy"])
    temporal_model.fit_generator(generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, train_data_list, batch_size=batch_size),
                            steps_per_epoch = steps_per_epoch, epochs = epoch, verbose = 1,
                        validation_data = generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, test_data_list, batch_size=batch_size),
                            validation_steps=validation_steps,
                        class_weight = None, callbacks = [best_model, reduce_lr])
    f1.close()
    f2.close()

def train_temporal_with_attention(batch_size = 6, epoch=1000):
    raw_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM/'
    flowimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticflow_image/'
    strainimg_path = '/media/zxl/other/pjh/datasetsss/CASME_II_TIM_opticalstrain_image/'
    best_model_file = "./models/VGG_16_5_channels_temporal_with_attention.h5"

    train_data_list_path = './train_list.txt'
    f1 = open(train_data_list_path, 'r')
    train_data_list = f1.readlines()
    steps_per_epoch = int(ceil(len(train_data_list)*1.0 / batch_size))

    test_data_list_path = './test_list.txt'
    f2 = open(test_data_list_path, 'r')
    test_data_list = f2.readlines()
    validation_steps = int(ceil(len(test_data_list)*1.0 / batch_size))

    vgg_model = load_model('./models/VGG_16_5_channels.h5')
    model = Model(inputs=vgg_model.input, outputs=vgg_model.layers[35].output)
    model.predict_on_batch(np.zeros((10, 224, 224, 5))) #https://www.jianshu.com/p/c84ae0527a3f
    best_model = ModelCheckpoint(best_model_file, monitor='val_acc',
                                verbose = 1, save_best_only = True)
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5,
                  patience=3, verbose=1, min_lr=0.00001)
    temporal_model = temporal_module_with_attention(data_dim=4096)
    temporal_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.00001, decay=0.000001), metrics=["accuracy"])
    temporal_model.fit_generator(generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, train_data_list, batch_size=batch_size),
                            steps_per_epoch = steps_per_epoch, epochs = epoch, verbose = 1,
                        validation_data = generator_batch_feature(model, raw_path, flowimg_path, strainimg_path, test_data_list, batch_size=batch_size),
                            validation_steps=validation_steps,
                        class_weight = None, callbacks = [best_model, reduce_lr])
    f1.close()
    f2.close()
if '__name__' == '__main__':
	# train_spatial(batch_size = 6, epoch=1000)
	# train_temporal(batch_size = 6, epoch=1000)
	train_temporal_with_attention(batch_size=6, epoch=1000)
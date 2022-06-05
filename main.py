import os
import time
import cv2
import glob
import tensorflow as tf
import pandas as pd
import numpy as np
import shutil
import sklearn.model_selection as sk

# local
# DATA_PATH = os.path.join(os.getcwd(), "r", "r_dataset")

# google colab
# https://www.kaggle.com/datasets/volkandl/car-brand-logos
# 위 Kaggle 페이지에서 dataset을 다운로드 받은 뒤 r_dataset 폴더 안에 Test 폴더와 Train 폴더를 넣었습니다.
DATASET_PATH = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'r_dataset')
TRAIN_DATASET_PATH = os.path.join(DATASET_PATH, 'Train')
TEST_DATASET_PATH = os.path.join(DATASET_PATH, 'Test')

LEN_CATEGORY = 8


def get_img_size(path, size_limit):
    exts = ['.jpg', '.png', '.jpeg', '.bmp']
    x_list = []
    y_list = []
    folder_list = os.listdir(path)
    for folder in folder_list:
        folder_path = os.path.join(path, folder)
        if(os.path.isdir(folder_path)):
            data_list = []
            for ext in exts:
                data_list += glob.glob(folder_path+'/*'+ext)
            img_norm = list()
            img_std = list()
            for data in data_list:
                img = cv2.imread(data, cv2.IMREAD_COLOR).astype(np.float32)
                if len(img.shape) < 2:
                    continue
                h, w, c = img.shape
                x_list.append(w)
                y_list.append(h)
    x_list.sort()
    y_list.sort()
    return y_list[int(len(x_list) / 10 * size_limit)], x_list[int(len(x_list) / 10 * size_limit)]


def get_nomal_machine_learning_model(data_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=data_shape),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(LEN_CATEGORY, activation="softmax")
    ])
    return model


def get_base_model(data_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                               activation="relu", input_shape=data_shape),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(LEN_CATEGORY, activation="softmax")
    ])
    return model


def get_img_trans_model(data_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.RandomFlip(
            'horizontal', seed=8282, input_shape=data_shape),
        tf.keras.layers.RandomRotation(
            factor=0.2,
            fill_mode='reflect',
            interpolation='bilinear',),
        tf.keras.layers.RandomZoom((0.2, 0.3),
                                   width_factor=None,
                                   fill_mode='reflect',
                                   interpolation='bilinear',),
        tf.keras.layers.Conv2D(32, (3, 3), padding="same",
                               activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
        tf.keras.layers.MaxPooling2D((3, 3)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(LEN_CATEGORY, activation="softmax")
    ])
    return model


def get_callback(mode):
    save_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(os.getcwd(), 'drive',
                              'MyDrive', 'r_weight', mode),
        monitor='val_acc',
        mode='max',
        save_weights_only=True,
        save_freq='epoch'
    )
    return save_callback


def simulation(mode, epochs, model):
    model_path = os.path.join(os.getcwd(), 'drive', 'MyDrive', 'r_model', mode)
    logs_path = os.path.join(
        os.getcwd(), 'drive', 'MyDrive', 'r_logs', mode)

    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="adam", metrics=['accuracy'])
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=logs_path)
    save_callback = get_callback(mode)
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=epochs,
              verbose=2, callbacks=[save_callback, tensorboard])
    model.save(model_path)
    score = model.evaluate(test_ds)
    return score


train_img_size = get_img_size(TRAIN_DATASET_PATH, 9)
test_img_size = get_img_size(TEST_DATASET_PATH, 9)
train_data_shape = (train_img_size[0], train_img_size[1], 3)
test_data_shape = (test_img_size[0], test_img_size[1], 3)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DATASET_PATH,
    image_size=train_img_size,
    batch_size=32,
    subset='training',
    validation_split=0.2,
    seed=8282
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DATASET_PATH,
    image_size=train_img_size,
    batch_size=32,
    subset='validation',
    validation_split=0.2,
    seed=8282
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DATASET_PATH,
    image_size=test_img_size,
    batch_size=32,
    shuffle=True,
    seed=8282
)


nomal_machine_learning_model = get_nomal_machine_learning_model(
    train_data_shape)
nomal_machine_learning_score = simulation(
    'nomal_machine_learning_model', 100, nomal_machine_learning_model)
print(nomal_machine_learning_score)

base_model = get_base_model(train_data_shape)
base_score = simulation('base_model', 100, base_model)
print(base_score)

img_trans_model = get_img_trans_model(train_data_shape)
img_trans_score = simulation('img_tran_model', 100, img_trans_model)
print(img_trans_score)

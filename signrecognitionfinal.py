# !git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset

# pip install fpdf

# !unzip Dataset.zip

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import random

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from keras import backend as K
from os import listdir
from datetime import date
from datetime import datetime
from fpdf import FPDF
from matplotlib import pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# %matplotlib inline

# Function For Reading The Dataset
num_of_classes = 10
image_size = 100


def get_data(dataset_path='Sign-Language-Digits-Dataset/Dataset', is_color=0, is_NN=1):
    digits = "0123456789"
    X = []
    Y = []
    for digit in digits:
        images = dataset_path + '/' + digit
        for image in listdir(images):
            img = cv2.imread(images + '/' + image, is_color)
            img = cv2.resize(img, (image_size, image_size))
            X.append(img)
            Y.append(digit)
    X = np.array(X)
    Y = np.array(Y)
    if is_color == 1:
        Avg = np.average(X)
        X = X - Avg
    X = (X / 255)
    # Conver simple output to NN output
    if is_NN == 1:
        Y = tf.keras.utils.to_categorical(Y, num_of_classes)
    return X, Y


# Measurements (recall, precision, fscore)
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# PDF Report
class PDF(FPDF):
    def __init__(self):
        super().__init__()

    def header(self):
        self.set_font('Arial', '', 12)
        # self.cell(0, 8, 'KNN Implementation Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', '', 12)
        self.cell(0, 8, f'Page {self.page_no()}', 0, 0, 'C')


# Feedforward Neural Network Architecture

# Reading The Dataset
X, Y = get_data(is_NN=1, is_color=0)

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

print("x train: ", X_train.shape)
print("x test: ", X_test.shape)
print("y train: ", Y_train.shape)
print("y test: ", Y_test.shape)


# Building The Model
def build_FFNN():
    model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(2048, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(num_of_classes, activation=tf.nn.softmax)
    ])

    model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy',
                   metrics=['accuracy', f1_m, precision_m, recall_m])
    return model1


k_folds = KFold(n_splits=3)
classifier = KerasClassifier(build_fn=build_FFNN, epochs=150, batch_size=64)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=k_folds)

model1 = build_FFNN()
model1.fit(X_train, Y_train, epochs=150, batch_size=64)
model1.save('save/FFNN_Saved')
# model = tf.keras.models.load_model('save/savedModel')


# Testing The Model
FFNN_Str = ''
FFNN_Str_Table = ''
print(f"Accuracies: {accuracies}")
print(f"Accuracy Variance: {accuracies.std()}")
print(f"Accuracy Mean: {round(accuracies.mean(), 1) * 100}%")

training_score = model1.evaluate(X_train, Y_train)
testing_score = model1.evaluate(X_test, Y_test)

print(f'Training Accuaracy: {round(training_score[1] * 100, 1)}%')
print(f'Testing Accuaracy: {round(testing_score[1] * 100, 1)}%')
print(f'Precision: {testing_score[3]}')
print(f'Recall: {testing_score[4]}')
print(f'F1 score: {testing_score[2]}')
print(model1.summary())

FFNN_Str += ('Accuracies: ' + str(accuracies) + '\n\n')
FFNN_Str += ('Accuracy Variance: ' + str(accuracies.std()) + '\n\n')
FFNN_Str += ('Accuracy Mean: ' + str(round(accuracies.mean(), 1) * 100) + '%\n\n\n')

FFNN_Str += ('Training Accuaracy: ' + str(round(training_score[1] * 100, 1)) + '%\n\n')
FFNN_Str += ('Testing Accuaracy: ' + str(round(testing_score[1] * 100, 1)) + '%\n\n')
FFNN_Str += ('Precision: ' + str(testing_score[3]) + '\n\n')
FFNN_Str += ('Recall: ' + str(testing_score[4]) + '\n\n')
FFNN_Str += ('F1 score: ' + str(testing_score[2]) + '\n\n')

stringlist = []
model1.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

FFNN_Str_Table += str('\n' + short_model_summary)

# Long Short Term Memory (LSTM) Architecture

# Reading The Dataset
X, Y = get_data(is_NN=1, is_color=0)

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

print("x train: ", X_train.shape)
print("x test: ", X_test.shape)
print("y train: ", Y_train.shape)
print("y test: ", Y_test.shape)


# Building The Model
def build_LSTM():
    model2 = tf.keras.models.Sequential([tf.keras.layers.LSTM(128),
                                         tf.keras.layers.Dense(64, activation="relu"),
                                         tf.keras.layers.Dense(num_of_classes, activation="sigmoid")])
    model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', f1_m, precision_m, recall_m])
    return model2


k_folds = KFold(n_splits=3)
classifier = KerasClassifier(build_fn=build_LSTM, epochs=150, batch_size=64)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=k_folds)

model2 = build_LSTM()
model2.fit(X_train, Y_train, epochs=150, batch_size=64)
model2.save('save/LSTM_Saved')
# model = tf.keras.models.load_model('save/savedModel')


# Testing The Mode
LSTM_Str = ''
LSTM_Str_Table = ''

print(f"Accuracies: {accuracies}")
print(f"Accuracy Variance: {accuracies.std()}")
print(f"Accuracy Mean: {round(accuracies.mean(), 1) * 100}%")

training_score = model2.evaluate(X_train, Y_train)
testing_score = model2.evaluate(X_test, Y_test)

print(f'Training Accuaracy: {round(training_score[1] * 100, 1)}%')
print(f'Testing Accuaracy: {round(testing_score[1] * 100, 1)}%')
print(f'Precision: {testing_score[3]}')
print(f'Recall: {testing_score[4]}')
print(f'F1 score: {testing_score[2]}')

print(model2.summary())
LSTM_Str += ('Accuracies: ' + str(accuracies) + '\n\n')
LSTM_Str += ('Accuracy Variance: ' + str(accuracies.std()) + '\n\n')
LSTM_Str += ('Accuracy Mean: ' + str(round(accuracies.mean(), 1) * 100) + '%\n\n\n')

LSTM_Str += ('Training Accuaracy: ' + str(round(training_score[1] * 100, 1)) + '%\n\n')
LSTM_Str += ('Testing Accuaracy: ' + str(round(testing_score[1] * 100, 1)) + '%\n\n')
LSTM_Str += ('Precision: ' + str(testing_score[3]) + '\n\n')
LSTM_Str += ('Recall: ' + str(testing_score[4]) + '\n\n')
LSTM_Str += ('F1 score: ' + str(testing_score[2]) + '\n\n')

stringlist = []
model2.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

LSTM_Str_Table += str('\n' + short_model_summary)

# Convolutional Neural Network(CNN) Architecture

# Reading The Dataset
X, Y = get_data(is_NN=1, is_color=1)

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

print("x train: ", X_train.shape)
print("x test: ", X_test.shape)
print("y train: ", Y_train.shape)
print("y test: ", Y_test.shape)


# Building The Model
def build_CNN():
    model3 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(100, 100, 3), filters=32, kernel_size=(4, 4), strides=(2)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 4), strides=(1)),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2),

        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.7),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(num_of_classes, activation='softmax')
    ])
    model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                   metrics=['accuracy', f1_m, precision_m, recall_m])
    return model3


k_folds = KFold(n_splits=3)
classifier = KerasClassifier(build_fn=build_CNN, epochs=50, batch_size=64)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=Y_train, cv=k_folds)

model3 = build_CNN()
model3.fit(X_train, Y_train, batch_size=64, epochs=50)
model3.save('save/CNN_Saved')
# model = tf.keras.models.load_model('save/savedModel')


# Testing The Model
CNN_Str = ''
CNN_Str_Table = ''

print(f"Accuracies: {accuracies}")
print(f"Accuracy Variance: {accuracies.std()}")
print(f"Accuracy Mean: {round(accuracies.mean(), 1) * 100}%")

training_score = model3.evaluate(X_train, Y_train)
testing_score = model3.evaluate(X_test, Y_test)

print(f'Training Accuaracy: {round(training_score[1] * 100, 1)}%')
print(f'Testing Accuaracy: {round(testing_score[1] * 100, 1)}%')
print(f'Precision: {testing_score[3]}')
print(f'Recall: {testing_score[4]}')
print(f'F1 score: {testing_score[2]}')

print(model3.summary())
CNN_Str += ('Accuracies: ' + str(accuracies) + '\n\n')
CNN_Str += ('Accuracy Variance: ' + str(accuracies.std()) + '\n\n')
CNN_Str += ('Accuracy Mean: ' + str(round(accuracies.mean(), 1) * 100) + '%\n\n\n')

CNN_Str += ('Training Accuaracy: ' + str(round(training_score[1] * 100, 1)) + '%\n\n')
CNN_Str += ('Testing Accuaracy: ' + str(round(testing_score[1] * 100, 1)) + '%\n\n')
CNN_Str += ('Precision: ' + str(testing_score[3]) + '\n\n')
CNN_Str += ('Recall: ' + str(testing_score[4]) + '\n\n')
CNN_Str += ('F1 score: ' + str(testing_score[2]) + '\n\n')

stringlist = []
model3.summary(print_fn=lambda x: stringlist.append(x))
short_model_summary = "\n".join(stringlist)

CNN_Str_Table += str('\n' + short_model_summary)

# SVM

# Reading The Dataset
X, Y = get_data(is_NN=0, is_color=0)

X, Y = np.array(X), np.array(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=15)

print("x train: ", X_train.shape)
print("x test: ", X_test.shape)
print("y train: ", Y_train.shape)
print("y test: ", Y_test.shape)

# Building The Model
from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1000, gamma=0.001, random_state=0)
X_train = X_train.reshape(X_train.shape[0], 10000)
X_test = X_test.reshape(X_test.shape[0], 10000)
model.fit(X_train, Y_train)

# Testing The Model
SVM_Str = ''
y_pred = model.predict(X_train)
print(f'Training Accuaracy: {round(accuracy_score(Y_train, y_pred), 1) * 100}%')
y_pred2 = model.predict(X_test)
print(f'Testing Accuaracy: {round(accuracy_score(Y_test, y_pred2), 1) * 100}%')

# precision tp / (tp + fp)
precision = precision_score(Y_test, y_pred2, pos_label='positive', average='micro')
print(f'Precision: {precision}')
# recall: tp / (tp + fn)
recall = recall_score(Y_test, y_pred2, pos_label='positive', average='macro')
print(f'Recall: {recall}')
# f1: 2 tp / (2 tp + fp + fn)
f1 = f1_score(Y_test, y_pred2, pos_label='positive', average='weighted')
print(f'F1 score: {f1}')

SVM_Str += ('Training Accuaracy: ' + str(round(accuracy_score(Y_train, y_pred), 1) * 100) + '%\n\n')
SVM_Str += ('Testing Accuaracy: ' + str(round(accuracy_score(Y_test, y_pred2), 1) * 100) + '%\n\n')
SVM_Str += ('Precision: ' + str(precision) + '\n\n')
SVM_Str += ('Recall: ' + str(recall) + '\n\n')
SVM_Str += ('F1 score: ' + str(f1) + '\n\n')

# Generating Report
print(SVM_Str)

# Generating Pdf
margin = 8
pdf = PDF()
pdf.add_page()
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(190, 0, 0)
pdf.cell(w=0, h=20, txt="Experiments Report", ln=1)
pdf.set_text_color(0, 0, 0)
pdf.set_font('Arial', 'B', 16)
pdf.cell(w=30, h=margin, txt="Date: ", ln=0)
pdf.cell(w=30, h=margin, txt=str(date.today().strftime("%d/%m/%Y")), ln=1)
pdf.cell(w=30, h=margin, txt="Time: ", ln=0)
pdf.cell(w=30, h=margin, txt=str(datetime.now().strftime("%H:%M:%S")), ln=1)
pdf.cell(w=30, h=margin, txt="Authors: ", ln=0)
pdf.cell(w=30, h=margin, txt="Khaled Ashraf, Ahmed Sayed, Ahmed Ebrahim", ln=1)
pdf.cell(w=30, h=margin, txt="                   Noura Ashraf, Samaa Khalifa", ln=1)
pdf.ln(margin)
# SVM
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(16, 63, 145)
pdf.cell(0, 8, 'SVM Experiment', 0, 10, 'C')
pdf.ln(margin)
pdf.set_text_color(0, 0, 0)
pdf.set_font('Helvetica', '', 22)
pdf.multi_cell(w=0, h=5, txt=str(SVM_Str + '\n'))
pdf.ln(margin)

# FFNN
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(16, 63, 145)
pdf.cell(0, 8, 'FFNN Experiment', 0, 10, 'C')
pdf.ln(margin)
pdf.set_text_color(0, 0, 0)
pdf.set_font('Helvetica', '', 22)
pdf.multi_cell(w=0, h=5, txt=str(FFNN_Str + '\n'))
pdf.ln(margin + 8)
pdf.set_font('Helvetica', 'B', 14)
pdf.multi_cell(w=0, h=5, txt=str(FFNN_Str_Table + '\n'))
pdf.ln(margin)

# LSTM
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(16, 63, 145)
pdf.cell(0, 8, 'LSTM Experiment', 0, 10, 'C')
pdf.ln(margin)
pdf.set_text_color(0, 0, 0)
pdf.set_font('Helvetica', '', 22)
pdf.multi_cell(w=0, h=5, txt=str(LSTM_Str + '\n'))
pdf.ln(margin * 2 + 12)
pdf.set_font('Helvetica', 'B', 14)
pdf.multi_cell(w=0, h=5, txt=str(LSTM_Str_Table + '\n'))
pdf.ln(margin)

# CNN
pdf.set_font('Arial', 'B', 24)
pdf.set_text_color(16, 63, 145)
pdf.cell(0, 8, 'CNN Experiment', 0, 10, 'C')
pdf.ln(margin)
pdf.set_text_color(0, 0, 0)
pdf.set_font('Helvetica', '', 22)
pdf.multi_cell(w=0, h=5, txt=str(CNN_Str + '\n'))
pdf.ln(margin)
pdf.set_font('Helvetica', 'B', 14)
pdf.multi_cell(w=0, h=5, txt=str(CNN_Str_Table + '\n'))
pdf.ln(margin)

pdf.output(f'./Report.pdf', 'F')
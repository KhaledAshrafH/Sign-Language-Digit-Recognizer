# Sign-Language-Digit-Recognizer
Sign Language Detection Project aims to classify sign language for numbers from 0 to 9 using different Neural Network architectures (FFNN - LSTM - CNN) using Keras in Python.
the dataset of Sign up 'Turkey Ankara Ayranci Anadolu High School's Sign Language Digits' , (https://github.com/ardamavi/Sign-Language-Digits-Dataset).

<div class="cell code" data-execution_count="1" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="T06uVeiAghgQ" data-outputid="ab2db6d7-b716-4003-d13e-c690e506bf08">

<div class="sourceCode" id="cb1">

    !git clone https://github.com/ardamavi/Sign-Language-Digits-Dataset

</div>

<div class="output stream stdout">

    Cloning into 'Sign-Language-Digits-Dataset'...
    remote: Enumerating objects: 2095, done.ote: Counting objects: 100% (6/6), done.ote: Compressing objects: 100% (6/6), done.ote: Total 2095 (delta 2), reused 0 (delta 0), pack-reused 2089

</div>

</div>

<div class="cell code" data-execution_count="2" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="NASNRyxbVLv-" data-outputid="334c1fc2-0124-4d78-91b6-c7c62b201205">

<div class="sourceCode" id="cb3">

    pip install fpdf

</div>

<div class="output stream stdout">

    Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
    Collecting fpdf
      Downloading fpdf-1.7.2.tar.gz (39 kB)
    Building wheels for collected packages: fpdf
      Building wheel for fpdf (setup.py) ... e=fpdf-1.7.2-py2.py3-none-any.whl size=40721 sha256=e329e373f1b6cb7f0f6fb03bcd0a014ca88fff5f2347e1bf52edb9baf2ee9fe9
      Stored in directory: /root/.cache/pip/wheels/b4/7f/00/f90ea7c44f8b921477205baa66a7aaf04be398f743ea946fd5
    Successfully built fpdf
    Installing collected packages: fpdf
    Successfully installed fpdf-1.7.2

</div>

</div>

<div class="cell code" id="yWE83WR5tokN">

<div class="sourceCode" id="cb5">

    !unzip Dataset.zip

</div>

</div>

<div class="cell code" data-execution_count="1" id="c8GIWaNm39Km">

<div class="sourceCode" id="cb6">

    import tensorflow as tf
    import pandas as pd
    import numpy as np
    import cv2
    import random

    from sklearn.model_selection import train_test_split,KFold,cross_val_score
    from keras import backend as K
    from os import listdir
    from datetime import date
    from datetime import datetime
    from fpdf import FPDF
    from matplotlib import pyplot as plt
    from keras.wrappers.scikit_learn import KerasClassifier
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

    %matplotlib inline

</div>

</div>

<section id="function-for-reading-the-dataset" class="cell markdown">

# **Function For Reading The Dataset**

</section>

<div class="cell code" data-execution_count="2" id="XN4w-l7w3_7B">

<div class="sourceCode" id="cb7">

    num_of_classes = 10
    image_size = 100

    def get_data(dataset_path='Sign-Language-Digits-Dataset/Dataset',is_color=0,is_NN=1):
        digits = "0123456789"
        X = []
        Y = []
        for digit in digits:
            images = dataset_path+'/'+digit
            for image in listdir(images):
              img = cv2.imread(images+'/'+image,is_color)
              img = cv2.resize(img, (image_size, image_size))
              X.append(img)
              Y.append(digit)
        X = np.array(X)
        Y = np.array(Y)
        if is_color==1:
          Avg = np.average(X)
          X = X - Avg
        X = (X / 255)
        # Conver simple output to NN output
        if is_NN==1:
          Y = tf.keras.utils.to_categorical(Y, num_of_classes)
        return X, Y

</div>

</div>

<section id="measurements-recall-precision-fscore" class="cell markdown">

# **Measurements (recall, precision, fscore)**

</section>

<div class="cell code" data-execution_count="3" id="DzI0ECYXD8C_">

<div class="sourceCode" id="cb8">

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
        return 2*((precision*recall)/(precision+recall+K.epsilon()))

</div>

</div>

<section id="pdf-report" class="cell markdown">

# **PDF Report**

</section>

<div class="cell code" data-execution_count="4" id="jDVcYwQHXoLg">

<div class="sourceCode" id="cb9">

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

</div>

</div>

<section id="reading-the-dataset" class="cell markdown">

## **Reading The Dataset**

</section>

<div class="cell code" data-execution_count="22" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="j8dOo3nIeBEB" data-outputid="ec9aa67c-8671-438b-cf91-ca9e300df209">

<div class="sourceCode" id="cb10">

    X,Y = get_data(is_NN=1,is_color=1)

    X, Y = np.array(X), np.array(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2 , random_state = 15)

    print("x train: ",X_train.shape)
    print("x test: ",X_test.shape)
    print("y train: ",Y_train.shape)
    print("y test: ",Y_test.shape)

</div>

<div class="output stream stdout">

    x train:  (1649, 100, 100, 3)
    x test:  (413, 100, 100, 3)
    y train:  (1649, 10)
    y test:  (413, 10)

</div>

</div>

<section id="feedforward-neural-network-architecture" class="cell markdown">

# **Feedforward Neural Network Architecture**

</section>

<section id="building-the-model" class="cell markdown">

## **Building The Model**

</section>

<div class="cell code" data-execution_count="11" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="LUfIfRtAhi8H" data-outputid="e0da444d-98ec-4165-d53d-c828d96a5272">

<div class="sourceCode" id="cb12">

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

        model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
        return model1

    k_folds = KFold(n_splits = 3)
    classifier = KerasClassifier(build_fn = build_FFNN, epochs = 150,batch_size=64)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = k_folds)

    model1 = build_FFNN()
    model1.fit(X_train, Y_train, epochs=150,batch_size=64)
    model1.save('save/FFNN_Saved')
    # model = tf.keras.models.load_model('save/savedModel')

</div>

<div class="output stream stdout">

    Epoch 1/150

</div>

<div class="output stream stderr">

    <ipython-input-11-3207a940108a>:16: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      classifier = KerasClassifier(build_fn = build_FFNN, epochs = 150,batch_size=64)

</div>

<div class="output stream stdout">

    18/18 [==============================] - 1s 10ms/step - loss: 2.6540 - accuracy: 0.0965 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 2/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.4476 - accuracy: 0.1256 - f1_m: 0.0016 - precision_m: 0.0139 - recall_m: 8.6806e-04
    Epoch 3/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.3598 - accuracy: 0.1292 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 4/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.3322 - accuracy: 0.1301 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 5/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.3054 - accuracy: 0.1401 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 6/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.2858 - accuracy: 0.1410 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 7/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.2533 - accuracy: 0.1720 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 8/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.2217 - accuracy: 0.1865 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 9/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.1785 - accuracy: 0.2011 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 10/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.1433 - accuracy: 0.2129 - f1_m: 0.0068 - precision_m: 0.2222 - recall_m: 0.0035
    Epoch 11/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.1004 - accuracy: 0.2366 - f1_m: 0.0068 - precision_m: 0.1667 - recall_m: 0.0035
    Epoch 12/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.0326 - accuracy: 0.2639 - f1_m: 0.0149 - precision_m: 0.2222 - recall_m: 0.0078
    Epoch 13/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.9497 - accuracy: 0.3094 - f1_m: 0.0316 - precision_m: 0.4778 - recall_m: 0.0165
    ...
    Epoch 149/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3850 - accuracy: 0.8654 - f1_m: 0.8596 - precision_m: 0.9043 - recall_m: 0.8197
    Epoch 150/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4262 - accuracy: 0.8441 - f1_m: 0.8437 - precision_m: 0.8856 - recall_m: 0.8065

</div>

</div>

<section id="testing-the-model" class="cell markdown">

## **Testing The Model**

</section>

<div class="cell code" data-execution_count="12" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="K7m9UK94hmLg" data-outputid="69d4751e-8185-4e20-939b-1c7ebfeb5b19">

<div class="sourceCode" id="cb16">

    FFNN_Str= ''
    FFNN_Str_Table = ''
    print(f"Accuracies: {accuracies}" )
    print(f"Accuracy Variance: {accuracies.std()}" )
    print(f"Accuracy Mean: {round(accuracies.mean(),1)*100}%")

    training_score = model1.evaluate(X_train, Y_train)
    testing_score = model1.evaluate(X_test, Y_test)

    print(f'Training Accuaracy: {round(training_score[1]*100,1)}%')
    print(f'Testing Accuaracy: {round(testing_score[1]*100,1)}%')
    print(f'Precision: {testing_score[3]}')
    print(f'Recall: {testing_score[4]}')
    print(f'F1 score: {testing_score[2]}')
    print(model1.summary())

    FFNN_Str+=('Accuracies: '+ str(accuracies)+ '\n\n')
    FFNN_Str+=('Accuracy Variance: '+ str(accuracies.std())+ '\n\n')
    FFNN_Str+=('Accuracy Mean: '+ str(round(accuracies.mean(),1)*100)+ '%\n\n\n')

    FFNN_Str+=('Training Accuaracy: '+ str(round(training_score[1]*100,1))+ '%\n\n')
    FFNN_Str+=('Testing Accuaracy: '+ str(round(testing_score[1]*100,1))+ '%\n\n')
    FFNN_Str+=('Precision: '+ str(testing_score[3])+ '\n\n')
    FFNN_Str+=('Recall: '+ str(testing_score[4])+ '\n\n')
    FFNN_Str+=('F1 score: '+ str(testing_score[2])+ '\n\n')

    stringlist = []
    model1.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    FFNN_Str_Table+=str('\n'+short_model_summary)

</div>

<div class="output stream stdout">

    Accuracies: [0.75636363 0.77272725 0.75409836]
    Accuracy Variance: 0.008299499934341415
    Accuracy Mean: 80.0%
    52/52 [==============================] - 0s 4ms/step - loss: 0.2673 - accuracy: 0.9327 - f1_m: 0.9283 - precision_m: 0.9679 - recall_m: 0.8932
    13/13 [==============================] - 0s 4ms/step - loss: 0.5344 - accuracy: 0.8523 - f1_m: 0.8285 - precision_m: 0.8885 - recall_m: 0.7774
    Training Accuaracy: 93.3%
    Testing Accuaracy: 85.2%
    Precision: 0.8884928226470947
    Recall: 0.7773541212081909
    F1 score: 0.8285248279571533
    Model: "sequential_11"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_7 (Flatten)         (None, 10000)             0         

     dense_36 (Dense)            (None, 2048)              20482048  

     dropout_14 (Dropout)        (None, 2048)              0         

     dense_37 (Dense)            (None, 1024)              2098176   

     dropout_15 (Dropout)        (None, 1024)              0         

     dense_38 (Dense)            (None, 512)               524800    

     dense_39 (Dense)            (None, 10)                5130      

    =================================================================
    Total params: 23,110,154
    Trainable params: 23,110,154
    Non-trainable params: 0
    _________________________________________________________________
    None

</div>

</div>

<section id="long-short-term-memory-lstm-architecture" class="cell markdown">

# **Long Short Term Memory (LSTM) Architecture**

</section>

<section id="building-the-model" class="cell markdown">

## **Building The Model**

</section>

<div class="cell code" data-execution_count="13" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="5wpVe4VIWI7v" data-outputid="20cbdfb0-edba-4778-ed0c-9a7914a291bf">

<div class="sourceCode" id="cb18">

    from sklearn.model_selection import KFold

    def build_LSTM():
      model2 = tf.keras.models.Sequential([tf.keras.layers.LSTM(128),
                                         tf.keras.layers.Dense(64, activation="relu"),
                                         tf.keras.layers.Dense(num_of_classes, activation="sigmoid")])
      model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
      return model2

    k_folds = KFold(n_splits = 3)
    classifier = KerasClassifier(build_fn = build_LSTM, epochs = 150,batch_size=64)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = k_folds)

    model2 = build_LSTM()
    model2.fit(X_train, Y_train, epochs=150, batch_size=64)
    model2.save('save/LSTM_Saved')
    # model = tf.keras.models.load_model('save/savedModel')

</div>

<div class="output stream stdout">

    Epoch 1/150

</div>

<div class="output stream stderr">

    <ipython-input-13-a769a75837bd>:12: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      classifier = KerasClassifier(build_fn = build_LSTM, epochs = 150,batch_size=64)

</div>

<div class="output stream stdout">
    
    18/18 [==============================] - 3s 13ms/step - loss: 2.3010 - accuracy: 0.1101 - f1_m: 0.1873 - precision_m: 0.1117 - recall_m: 0.5850
    Epoch 2/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.1983 - accuracy: 0.1620 - f1_m: 0.2156 - precision_m: 0.1260 - recall_m: 0.7581
    Epoch 3/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.1351 - accuracy: 0.1765 - f1_m: 0.2292 - precision_m: 0.1346 - recall_m: 0.7730
    Epoch 4/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0771 - accuracy: 0.2211 - f1_m: 0.2440 - precision_m: 0.1443 - recall_m: 0.7960
    Epoch 5/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.0783 - accuracy: 0.2047 - f1_m: 0.2399 - precision_m: 0.1420 - recall_m: 0.7788
    Epoch 6/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0317 - accuracy: 0.2338 - f1_m: 0.2474 - precision_m: 0.1464 - recall_m: 0.8018
    Epoch 7/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0931 - accuracy: 0.1947 - f1_m: 0.2403 - precision_m: 0.1424 - recall_m: 0.7755
    Epoch 8/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.9730 - accuracy: 0.2584 - f1_m: 0.2677 - precision_m: 0.1595 - recall_m: 0.8326
    Epoch 9/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.9252 - accuracy: 0.2848 - f1_m: 0.2673 - precision_m: 0.1592 - recall_m: 0.8345
    Epoch 10/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8563 - accuracy: 0.2985 - f1_m: 0.2821 - precision_m: 0.1682 - recall_m: 0.8759
    Epoch 11/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8225 - accuracy: 0.3194 - f1_m: 0.2876 - precision_m: 0.1723 - recall_m: 0.8717
    Epoch 12/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.7830 - accuracy: 0.3130 - f1_m: 0.2859 - precision_m: 0.1703 - recall_m: 0.8924
    Epoch 13/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8624 - accuracy: 0.3030 - f1_m: 0.2719 - precision_m: 0.1606 - recall_m: 0.8890
    ...
    Epoch 149/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0184 - accuracy: 0.9958 - f1_m: 0.4898 - precision_m: 0.3244 - recall_m: 1.0000
    Epoch 150/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0348 - accuracy: 0.9909 - f1_m: 0.4986 - precision_m: 0.3322 - recall_m: 1.0000

</div>

<div class="output stream stderr">

    WARNING:absl:Found untraced functions such as lstm_cell_7_layer_call_fn, lstm_cell_7_layer_call_and_return_conditional_losses while saving (showing 2 of 2). These functions will not be directly callable after loading.

</div>

</div>

<section id="testing-the-model" class="cell markdown">

## **Testing The Model**

</section>

<div class="cell code" data-execution_count="14" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="-4_ZJ3Q9WOS6" data-outputid="ab8e0c09-c408-4313-a812-f45ef4a3c232">

<div class="sourceCode" id="cb23">

    LSTM_Str= ''
    LSTM_Str_Table = ''

    print(f"Accuracies: {accuracies}" )
    print(f"Accuracy Variance: {accuracies.std()}" )
    print(f"Accuracy Mean: {round(accuracies.mean(),1)*100}%")

    training_score = model2.evaluate(X_train, Y_train)
    testing_score = model2.evaluate(X_test, Y_test)

    print(f'Training Accuaracy: {round(training_score[1]*100,1)}%')
    print(f'Testing Accuaracy: {round(testing_score[1]*100,1)}%')
    print(f'Precision: {testing_score[3]}')
    print(f'Recall: {testing_score[4]}')
    print(f'F1 score: {testing_score[2]}')

    print(model2.summary())
    LSTM_Str+=('Accuracies: '+ str(accuracies)+ '\n\n')
    LSTM_Str+=('Accuracy Variance: '+ str(accuracies.std())+ '\n\n')
    LSTM_Str+=('Accuracy Mean: '+ str(round(accuracies.mean(),1)*100)+ '%\n\n\n')

    LSTM_Str+=('Training Accuaracy: '+ str(round(training_score[1]*100,1))+ '%\n\n')
    LSTM_Str+=('Testing Accuaracy: '+ str(round(testing_score[1]*100,1))+ '%\n\n')
    LSTM_Str+=('Precision: '+ str(testing_score[3])+ '\n\n')
    LSTM_Str+=('Recall: '+ str(testing_score[4])+ '\n\n')
    LSTM_Str+=('F1 score: '+ str(testing_score[2])+ '\n\n')

    stringlist = []
    model2.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    LSTM_Str_Table+=str('\n'+short_model_summary)

</div>

<div class="output stream stdout">

    Accuracies: [0.89272726 0.87454545 0.87613845]
    Accuracy Variance: 0.008221273435511225
    Accuracy Mean: 90.0%
    52/52 [==============================] - 1s 6ms/step - loss: 0.0083 - accuracy: 0.9994 - f1_m: 0.5023 - precision_m: 0.3356 - recall_m: 1.0000
    13/13 [==============================] - 0s 6ms/step - loss: 0.3520 - accuracy: 0.9153 - f1_m: 0.4964 - precision_m: 0.3315 - recall_m: 0.9904
    Training Accuaracy: 99.9%
    Testing Accuaracy: 91.5%
    Precision: 0.33148959279060364
    Recall: 0.9903846383094788
    F1 score: 0.49642473459243774
    Model: "sequential_15"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     lstm_7 (LSTM)               (None, 128)               117248    

     dense_46 (Dense)            (None, 64)                8256      

     dense_47 (Dense)            (None, 10)                650       

    =================================================================
    Total params: 126,154
    Trainable params: 126,154
    Non-trainable params: 0
    _________________________________________________________________
    None

</div>

</div>

<section id="convolutional-neural-networkcnn-architecture" class="cell markdown">

# **Convolutional Neural Network(CNN) Architecture**

</section>

<section id="building-the-model" class="cell markdown">

## **Building The Model**

</section>

<div class="cell code" data-execution_count="23" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="kq8D7rAEWWJC" data-outputid="2390ce83-a9b3-47e1-c9b4-5041cd127eda">

<div class="sourceCode" id="cb25">

    def build_CNN():
        model3 = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(input_shape=(100, 100, 3), filters=32, kernel_size=(4,4), strides=(2)),
            tf.keras.layers.Activation('relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2),

            tf.keras.layers.Conv2D(filters=64, kernel_size=(4,4), strides=(1)),
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
        model3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy',f1_m,precision_m, recall_m])
        return model3

    k_folds = KFold(n_splits = 3)
    classifier = KerasClassifier(build_fn = build_CNN, epochs = 50,batch_size=64)
    accuracies = cross_val_score(estimator = classifier, X = X_train, y = Y_train, cv = k_folds)

    model3 = build_CNN()
    model3.fit(X_train, Y_train, batch_size=64, epochs=50)
    model3.save('save/CNN_Saved')
    # model = tf.keras.models.load_model('save/savedModel')

</div>

<div class="output stream stderr">

    <ipython-input-23-b7438b769e18>:26: DeprecationWarning: KerasClassifier is deprecated, use Sci-Keras (https://github.com/adriangb/scikeras) instead. See https://www.adriangb.com/scikeras/stable/migration.html for help migrating.
      classifier = KerasClassifier(build_fn = build_CNN, epochs = 50,batch_size=64)

</div>

<div class="output stream stdout">

    Epoch 1/50
    18/18 [==============================] - 1s 12ms/step - loss: 2.4551 - accuracy: 0.2266 - f1_m: 0.1626 - precision_m: 0.3319 - recall_m: 0.1106
    Epoch 2/50
    18/18 [==============================] - 0s 11ms/step - loss: 1.6619 - accuracy: 0.4522 - f1_m: 0.3861 - precision_m: 0.5889 - recall_m: 0.2909
    Epoch 3/50
    18/18 [==============================] - 0s 11ms/step - loss: 1.2966 - accuracy: 0.5596 - f1_m: 0.5100 - precision_m: 0.7157 - recall_m: 0.3980
    Epoch 4/50
    18/18 [==============================] - 0s 10ms/step - loss: 1.1673 - accuracy: 0.6024 - f1_m: 0.5522 - precision_m: 0.7371 - recall_m: 0.4433
    Epoch 5/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.9785 - accuracy: 0.6806 - f1_m: 0.6419 - precision_m: 0.8170 - recall_m: 0.5308
    Epoch 6/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.8958 - accuracy: 0.6915 - f1_m: 0.6634 - precision_m: 0.8097 - recall_m: 0.5636
    Epoch 7/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.8076 - accuracy: 0.7216 - f1_m: 0.6907 - precision_m: 0.8461 - recall_m: 0.5896
    Epoch 8/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.8132 - accuracy: 0.7370 - f1_m: 0.6884 - precision_m: 0.8280 - recall_m: 0.5905
    Epoch 9/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.7934 - accuracy: 0.7425 - f1_m: 0.6995 - precision_m: 0.8117 - recall_m: 0.6164
    Epoch 10/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6817 - accuracy: 0.7771 - f1_m: 0.7451 - precision_m: 0.8625 - recall_m: 0.6581
    Epoch 11/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6647 - accuracy: 0.8044 - f1_m: 0.7737 - precision_m: 0.8794 - recall_m: 0.6916
    Epoch 12/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5735 - accuracy: 0.8189 - f1_m: 0.8020 - precision_m: 0.8940 - recall_m: 0.7288
    Epoch 13/50
    ...
    Epoch 49/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1012 - accuracy: 0.9679 - f1_m: 0.9665 - precision_m: 0.9761 - recall_m: 0.9572
    Epoch 50/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1005 - accuracy: 0.9685 - f1_m: 0.9695 - precision_m: 0.9751 - recall_m: 0.9639

</div>

<div class="output stream stderr">

    WARNING:absl:Found untraced functions such as _jit_compiled_convolution_op, _jit_compiled_convolution_op while saving (showing 2 of 2). These functions will not be directly callable after loading.

</div>

</div>

<section id="testing-the-model" class="cell markdown">

## **Testing The Model**

</section>

<div class="cell code" data-execution_count="24" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="sqHEkWoXYl7J" data-outputid="1e5ddb3f-dbec-4d2d-af5f-fec6e9253dc5">

<div class="sourceCode" id="cb29">

    CNN_Str= ''
    CNN_Str_Table = ''

    print(f"Accuracies: {accuracies}" )
    print(f"Accuracy Variance: {accuracies.std()}" )
    print(f"Accuracy Mean: {round(accuracies.mean(),1)*100}%")

    training_score = model3.evaluate(X_train, Y_train)
    testing_score = model3.evaluate(X_test, Y_test)

    print(f'Training Accuaracy: {round(training_score[1]*100,1)}%')
    print(f'Testing Accuaracy: {round(testing_score[1]*100,1)}%')
    print(f'Precision: {testing_score[3]}')
    print(f'Recall: {testing_score[4]}')
    print(f'F1 score: {testing_score[2]}')

    print(model3.summary())
    CNN_Str+=('Accuracies: '+ str(accuracies)+ '\n\n')
    CNN_Str+=('Accuracy Variance: '+ str(accuracies.std())+ '\n\n')
    CNN_Str+=('Accuracy Mean: '+ str(round(accuracies.mean(),1)*100)+ '%\n\n\n')

    CNN_Str+=('Training Accuaracy: '+ str(round(training_score[1]*100,1))+ '%\n\n')
    CNN_Str+=('Testing Accuaracy: '+ str(round(testing_score[1]*100,1))+ '%\n\n')
    CNN_Str+=('Precision: '+ str(testing_score[3])+ '\n\n')
    CNN_Str+=('Recall: '+ str(testing_score[4])+ '\n\n')
    CNN_Str+=('F1 score: '+ str(testing_score[2])+ '\n\n')

    stringlist = []
    model3.summary(print_fn=lambda x: stringlist.append(x))
    short_model_summary = "\n".join(stringlist)

    CNN_Str_Table+=str('\n'+short_model_summary)

</div>

<div class="output stream stdout">

    Accuracies: [0.92909092 0.9218182  0.90892529]
    Accuracy Variance: 0.008338476416481857
    Accuracy Mean: 90.0%
    52/52 [==============================] - 1s 5ms/step - loss: 0.0059 - accuracy: 1.0000 - f1_m: 1.0000 - precision_m: 1.0000 - recall_m: 1.0000
    13/13 [==============================] - 0s 5ms/step - loss: 0.0967 - accuracy: 0.9734 - f1_m: 0.9707 - precision_m: 0.9754 - recall_m: 0.9661
    Training Accuaracy: 100.0%
    Testing Accuaracy: 97.3%
    Precision: 0.9754027128219604
    Recall: 0.9660974740982056
    F1 score: 0.9706761837005615
    Model: "sequential_23"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     conv2d_14 (Conv2D)          (None, 49, 49, 32)        1568      

     activation_14 (Activation)  (None, 49, 49, 32)        0         

     max_pooling2d_14 (MaxPoolin  (None, 24, 24, 32)       0         
     g2D)                                                            

     conv2d_15 (Conv2D)          (None, 21, 21, 64)        32832     

     activation_15 (Activation)  (None, 21, 21, 64)        0         

     max_pooling2d_15 (MaxPoolin  (None, 10, 10, 64)       0         
     g2D)                                                            

     dropout_30 (Dropout)        (None, 10, 10, 64)        0         

     flatten_15 (Flatten)        (None, 6400)              0         

     batch_normalization_14 (Bat  (None, 6400)             25600     
     chNormalization)                                                

     dense_62 (Dense)            (None, 256)               1638656   

     dropout_31 (Dropout)        (None, 256)               0         

     batch_normalization_15 (Bat  (None, 256)              1024      
     chNormalization)                                                

     dense_63 (Dense)            (None, 10)                2570      

    =================================================================
    Total params: 1,702,250
    Trainable params: 1,688,938
    Non-trainable params: 13,312
    _________________________________________________________________
    None

</div>

</div>

<section id="svm" class="cell markdown">

# **SVM**

</section>

<section id="building-the-model" class="cell markdown">

## **Building The Model**

</section>

<div class="cell code" data-execution_count="16" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="MIt-wgn1S_OA" data-outputid="1aecb2a1-3646-430c-b8f3-88bbc052ccbb">

<div class="sourceCode" id="cb31">

    from sklearn.svm import SVC
    from sklearn.model_selection import LeaveOneOut

    model = SVC(kernel ='rbf', C = 1000, gamma =0.001,random_state = 0)
    X_train = X_train.reshape(X_train.shape[0],10000)
    X_test = X_test.reshape(X_test.shape[0],10000)
    model.fit(X_train,Y_train)

</div>

<div class="output execute_result" data-execution_count="16">

    SVC(C=1000, gamma=0.001, random_state=0)

</div>

</div>

<section id="testing-the-model" class="cell markdown">

## **Testing The Model**

</section>

<div class="cell code" data-execution_count="17" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;}" id="xTNZ1EifTVis" data-outputid="63236a1d-b1a4-42f5-ccd3-5bdcceea0a2d">

<div class="sourceCode" id="cb33">

    SVM_Str = ''
    y_pred = model.predict(X_train)
    print(f'Training Accuaracy: {round(accuracy_score(Y_train,y_pred),1)*100}%')
    y_pred2 = model.predict(X_test)
    print(f'Testing Accuaracy: {round(accuracy_score(Y_test,y_pred2),1)*100}%')

    # precision tp / (tp + fp)
    precision = precision_score(Y_test, y_pred2,pos_label='positive',average='micro')
    print(f'Precision: {precision}')
    # recall: tp / (tp + fn)
    recall = recall_score(Y_test, y_pred2,pos_label='positive',average='macro')
    print(f'Recall: {recall}')
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(Y_test, y_pred2,pos_label='positive',average='weighted')
    print(f'F1 score: {f1}')

    SVM_Str+=('Training Accuaracy: '+ str(round(accuracy_score(Y_train,y_pred),1)*100)+ '%\n\n')
    SVM_Str+=('Testing Accuaracy: '+ str(round(accuracy_score(Y_test,y_pred2),1)*100)+ '%\n\n')
    SVM_Str+=('Precision: '+ str(precision)+ '\n\n')
    SVM_Str+=('Recall: '+ str(recall)+ '\n\n')
    SVM_Str+=('F1 score: '+ str(f1)+ '\n\n')

</div>

<div class="output stream stdout">

    Training Accuaracy: 100.0%
    Testing Accuaracy: 90.0%
    Precision: 0.8571428571428571
    Recall: 0.8595429845379904
    F1 score: 0.8568435049952878

</div>

<div class="output stream stderr">

    /usr/local/lib/python3.8/dist-packages/sklearn/metrics/_classification.py:1370: UserWarning: Note that pos_label (set to 'positive') is ignored when average != 'binary' (got 'micro'). You may use labels=[pos_label] to specify a single positive class.
      warnings.warn(
    /usr/local/lib/python3.8/dist-packages/sklearn/metrics/_classification.py:1370: UserWarning: Note that pos_label (set to 'positive') is ignored when average != 'binary' (got 'macro'). You may use labels=[pos_label] to specify a single positive class.
      warnings.warn(
    /usr/local/lib/python3.8/dist-packages/sklearn/metrics/_classification.py:1370: UserWarning: Note that pos_label (set to 'positive') is ignored when average != 'binary' (got 'weighted'). You may use labels=[pos_label] to specify a single positive class.
      warnings.warn(

</div>

</div>

<section id="generating-report" class="cell markdown">

# **Generating Report**

</section>

<div class="cell code" data-execution_count="30" data-colab="{&quot;base_uri&quot;:&quot;https://localhost:8080/&quot;,&quot;height&quot;:226}" id="IKFVA1WyYFx6" data-outputid="e336b860-3cc6-4527-a16d-82e755f0a831">

<div class="sourceCode" id="cb36">

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
    pdf.multi_cell(w=0, h=5, txt=str(SVM_Str+'\n'))
    pdf.ln(margin)

    # FFNN
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'FFNN Experiment', 0, 10, 'C')
    pdf.ln(margin)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 22)
    pdf.multi_cell(w=0, h=5, txt=str(FFNN_Str+'\n'))
    pdf.ln(margin+8)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.multi_cell(w=0, h=5, txt=str(FFNN_Str_Table+'\n'))
    pdf.ln(margin)

    # LSTM
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'LSTM Experiment', 0, 10, 'C')
    pdf.ln(margin)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 22)
    pdf.multi_cell(w=0, h=5, txt=str(LSTM_Str+'\n'))
    pdf.ln(margin*2+12)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.multi_cell(w=0, h=5, txt=str(LSTM_Str_Table+'\n'))
    pdf.ln(margin)

    # CNN
    pdf.set_font('Arial', 'B', 24)
    pdf.set_text_color(16, 63, 145)
    pdf.cell(0, 8, 'CNN Experiment', 0, 10, 'C')
    pdf.ln(margin)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Helvetica', '', 22)
    pdf.multi_cell(w=0, h=5, txt=str(CNN_Str+'\n'))
    pdf.ln(margin)
    pdf.set_font('Helvetica', 'B', 14)
    pdf.multi_cell(w=0, h=5, txt=str(CNN_Str_Table+'\n'))
    pdf.ln(margin)

    pdf.output(f'./Report.pdf', 'F')

</div>

<div class="output stream stdout">

    Training Accuaracy: 100.0%

    Testing Accuaracy: 90.0%

    Precision: 0.8571428571428571

    Recall: 0.8595429845379904

    F1 score: 0.8568435049952878

</div>

<div class="output execute_result" data-execution_count="30">

<div class="sourceCode" id="cb38">

    {"type":"string"}

</div>

</div>

</div>

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
    Epoch 14/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.8874 - accuracy: 0.3185 - f1_m: 0.0823 - precision_m: 0.7894 - recall_m: 0.0441
    Epoch 15/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.8235 - accuracy: 0.3485 - f1_m: 0.0741 - precision_m: 0.6988 - recall_m: 0.0399
    Epoch 16/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.7415 - accuracy: 0.3758 - f1_m: 0.1156 - precision_m: 0.9115 - recall_m: 0.0623
    Epoch 17/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.6787 - accuracy: 0.4158 - f1_m: 0.1539 - precision_m: 0.8237 - recall_m: 0.0858
    Epoch 18/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.6174 - accuracy: 0.4177 - f1_m: 0.1711 - precision_m: 0.7803 - recall_m: 0.0981
    Epoch 19/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.5726 - accuracy: 0.4268 - f1_m: 0.2061 - precision_m: 0.8233 - recall_m: 0.1195
    Epoch 20/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4765 - accuracy: 0.4768 - f1_m: 0.2571 - precision_m: 0.8461 - recall_m: 0.1535
    Epoch 21/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4638 - accuracy: 0.4904 - f1_m: 0.2611 - precision_m: 0.7684 - recall_m: 0.1597
    Epoch 22/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4757 - accuracy: 0.4559 - f1_m: 0.2914 - precision_m: 0.7785 - recall_m: 0.1810
    Epoch 23/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4058 - accuracy: 0.5041 - f1_m: 0.3281 - precision_m: 0.7959 - recall_m: 0.2087
    Epoch 24/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.2926 - accuracy: 0.5560 - f1_m: 0.3872 - precision_m: 0.8418 - recall_m: 0.2530
    Epoch 25/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.3695 - accuracy: 0.5086 - f1_m: 0.3723 - precision_m: 0.7557 - recall_m: 0.2498
    Epoch 26/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3119 - accuracy: 0.5505 - f1_m: 0.3518 - precision_m: 0.7961 - recall_m: 0.2289
    Epoch 27/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.3170 - accuracy: 0.5205 - f1_m: 0.3877 - precision_m: 0.7644 - recall_m: 0.2652
    Epoch 28/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.2604 - accuracy: 0.5560 - f1_m: 0.4334 - precision_m: 0.8083 - recall_m: 0.3000
    Epoch 29/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1841 - accuracy: 0.5814 - f1_m: 0.4646 - precision_m: 0.8120 - recall_m: 0.3323
    Epoch 30/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1608 - accuracy: 0.6087 - f1_m: 0.5155 - precision_m: 0.8505 - recall_m: 0.3735
    Epoch 31/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1271 - accuracy: 0.6142 - f1_m: 0.5229 - precision_m: 0.8018 - recall_m: 0.3893
    Epoch 32/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1876 - accuracy: 0.5823 - f1_m: 0.4912 - precision_m: 0.7850 - recall_m: 0.3610
    Epoch 33/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1752 - accuracy: 0.6051 - f1_m: 0.5141 - precision_m: 0.8160 - recall_m: 0.3775
    Epoch 34/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1512 - accuracy: 0.6151 - f1_m: 0.5116 - precision_m: 0.8032 - recall_m: 0.3782
    Epoch 35/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0133 - accuracy: 0.6670 - f1_m: 0.5897 - precision_m: 0.8292 - recall_m: 0.4612
    Epoch 36/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9925 - accuracy: 0.6570 - f1_m: 0.6043 - precision_m: 0.8424 - recall_m: 0.4740
    Epoch 37/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0096 - accuracy: 0.6642 - f1_m: 0.6140 - precision_m: 0.8375 - recall_m: 0.4883
    Epoch 38/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.0740 - accuracy: 0.6133 - f1_m: 0.5827 - precision_m: 0.7959 - recall_m: 0.4617
    Epoch 39/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9831 - accuracy: 0.6579 - f1_m: 0.6109 - precision_m: 0.8189 - recall_m: 0.4885
    Epoch 40/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8984 - accuracy: 0.7125 - f1_m: 0.6714 - precision_m: 0.8660 - recall_m: 0.5506
    Epoch 41/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9490 - accuracy: 0.6788 - f1_m: 0.6326 - precision_m: 0.8184 - recall_m: 0.5190
    Epoch 42/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9151 - accuracy: 0.6897 - f1_m: 0.6477 - precision_m: 0.8069 - recall_m: 0.5425
    Epoch 43/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9589 - accuracy: 0.6606 - f1_m: 0.6481 - precision_m: 0.8017 - recall_m: 0.5449
    Epoch 44/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8874 - accuracy: 0.6961 - f1_m: 0.6624 - precision_m: 0.8410 - recall_m: 0.5500
    Epoch 45/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9334 - accuracy: 0.6861 - f1_m: 0.6535 - precision_m: 0.8159 - recall_m: 0.5461
    Epoch 46/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8663 - accuracy: 0.6970 - f1_m: 0.6621 - precision_m: 0.8293 - recall_m: 0.5530
    Epoch 47/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8818 - accuracy: 0.7097 - f1_m: 0.6477 - precision_m: 0.8471 - recall_m: 0.5279
    Epoch 48/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8336 - accuracy: 0.7234 - f1_m: 0.6738 - precision_m: 0.8165 - recall_m: 0.5769
    Epoch 49/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8403 - accuracy: 0.7243 - f1_m: 0.6909 - precision_m: 0.8385 - recall_m: 0.5886
    Epoch 50/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8485 - accuracy: 0.7161 - f1_m: 0.6930 - precision_m: 0.8394 - recall_m: 0.5918
    Epoch 51/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8311 - accuracy: 0.7298 - f1_m: 0.6847 - precision_m: 0.8520 - recall_m: 0.5747
    Epoch 52/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7649 - accuracy: 0.7561 - f1_m: 0.7060 - precision_m: 0.8409 - recall_m: 0.6112
    Epoch 53/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7759 - accuracy: 0.7516 - f1_m: 0.7188 - precision_m: 0.8499 - recall_m: 0.6255
    Epoch 54/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8137 - accuracy: 0.7006 - f1_m: 0.6897 - precision_m: 0.8251 - recall_m: 0.5956
    Epoch 55/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7641 - accuracy: 0.7561 - f1_m: 0.7062 - precision_m: 0.8430 - recall_m: 0.6095
    Epoch 56/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8643 - accuracy: 0.7216 - f1_m: 0.6748 - precision_m: 0.8355 - recall_m: 0.5697
    Epoch 57/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7842 - accuracy: 0.7343 - f1_m: 0.7193 - precision_m: 0.8514 - recall_m: 0.6249
    Epoch 58/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7817 - accuracy: 0.7434 - f1_m: 0.7095 - precision_m: 0.8617 - recall_m: 0.6050
    Epoch 59/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7822 - accuracy: 0.7316 - f1_m: 0.7055 - precision_m: 0.8398 - recall_m: 0.6102
    Epoch 60/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7854 - accuracy: 0.7270 - f1_m: 0.7086 - precision_m: 0.8419 - recall_m: 0.6145
    Epoch 61/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7455 - accuracy: 0.7534 - f1_m: 0.7344 - precision_m: 0.8603 - recall_m: 0.6444
    Epoch 62/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6810 - accuracy: 0.7743 - f1_m: 0.7577 - precision_m: 0.8643 - recall_m: 0.6757
    Epoch 63/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6531 - accuracy: 0.7834 - f1_m: 0.7690 - precision_m: 0.8654 - recall_m: 0.6932
    Epoch 64/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7314 - accuracy: 0.7379 - f1_m: 0.7309 - precision_m: 0.8365 - recall_m: 0.6499
    Epoch 65/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6758 - accuracy: 0.7652 - f1_m: 0.7603 - precision_m: 0.8685 - recall_m: 0.6781
    Epoch 66/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6732 - accuracy: 0.7780 - f1_m: 0.7672 - precision_m: 0.8674 - recall_m: 0.6890
    Epoch 67/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7482 - accuracy: 0.7425 - f1_m: 0.7418 - precision_m: 0.8351 - recall_m: 0.6687
    Epoch 68/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6549 - accuracy: 0.7689 - f1_m: 0.7713 - precision_m: 0.8682 - recall_m: 0.6958
    Epoch 69/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6966 - accuracy: 0.7634 - f1_m: 0.7412 - precision_m: 0.8491 - recall_m: 0.6586
    Epoch 70/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6651 - accuracy: 0.7652 - f1_m: 0.7695 - precision_m: 0.8588 - recall_m: 0.6989
    Epoch 71/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6607 - accuracy: 0.7816 - f1_m: 0.7613 - precision_m: 0.8617 - recall_m: 0.6831
    Epoch 72/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6633 - accuracy: 0.7816 - f1_m: 0.7605 - precision_m: 0.8733 - recall_m: 0.6760
    Epoch 73/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6352 - accuracy: 0.7934 - f1_m: 0.7796 - precision_m: 0.8658 - recall_m: 0.7104
    Epoch 74/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5909 - accuracy: 0.7980 - f1_m: 0.7872 - precision_m: 0.8731 - recall_m: 0.7175
    Epoch 75/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6641 - accuracy: 0.7762 - f1_m: 0.7649 - precision_m: 0.8627 - recall_m: 0.6887
    Epoch 76/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6150 - accuracy: 0.7962 - f1_m: 0.7651 - precision_m: 0.8531 - recall_m: 0.6945
    Epoch 77/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6419 - accuracy: 0.7862 - f1_m: 0.7686 - precision_m: 0.8632 - recall_m: 0.6951
    Epoch 78/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5983 - accuracy: 0.7971 - f1_m: 0.7879 - precision_m: 0.8818 - recall_m: 0.7130
    Epoch 79/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6287 - accuracy: 0.7825 - f1_m: 0.7770 - precision_m: 0.8575 - recall_m: 0.7111
    Epoch 80/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6015 - accuracy: 0.8053 - f1_m: 0.7866 - precision_m: 0.8688 - recall_m: 0.7192
    Epoch 81/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6110 - accuracy: 0.7862 - f1_m: 0.7743 - precision_m: 0.8595 - recall_m: 0.7055
    Epoch 82/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6273 - accuracy: 0.7862 - f1_m: 0.7663 - precision_m: 0.8639 - recall_m: 0.6900
    Epoch 83/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6299 - accuracy: 0.7853 - f1_m: 0.7670 - precision_m: 0.8660 - recall_m: 0.6897
    Epoch 84/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6214 - accuracy: 0.7889 - f1_m: 0.7809 - precision_m: 0.8694 - recall_m: 0.7097
    Epoch 85/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6137 - accuracy: 0.7880 - f1_m: 0.7792 - precision_m: 0.8676 - recall_m: 0.7088
    Epoch 86/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6294 - accuracy: 0.7743 - f1_m: 0.7575 - precision_m: 0.8516 - recall_m: 0.6829
    Epoch 87/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6095 - accuracy: 0.7834 - f1_m: 0.7712 - precision_m: 0.8539 - recall_m: 0.7045
    Epoch 88/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6161 - accuracy: 0.7789 - f1_m: 0.7891 - precision_m: 0.8697 - recall_m: 0.7236
    Epoch 89/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6054 - accuracy: 0.8007 - f1_m: 0.7800 - precision_m: 0.8820 - recall_m: 0.7012
    Epoch 90/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5787 - accuracy: 0.7989 - f1_m: 0.7907 - precision_m: 0.8706 - recall_m: 0.7253
    Epoch 91/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5880 - accuracy: 0.8098 - f1_m: 0.8036 - precision_m: 0.8798 - recall_m: 0.7408
    Epoch 92/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5497 - accuracy: 0.8098 - f1_m: 0.7844 - precision_m: 0.8740 - recall_m: 0.7138
    Epoch 93/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5763 - accuracy: 0.7925 - f1_m: 0.7823 - precision_m: 0.8585 - recall_m: 0.7195
    Epoch 94/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6110 - accuracy: 0.7843 - f1_m: 0.7899 - precision_m: 0.8702 - recall_m: 0.7243
    Epoch 95/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5902 - accuracy: 0.7862 - f1_m: 0.7689 - precision_m: 0.8530 - recall_m: 0.7016
    Epoch 96/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5578 - accuracy: 0.8144 - f1_m: 0.8071 - precision_m: 0.8834 - recall_m: 0.7453
    Epoch 97/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5261 - accuracy: 0.8162 - f1_m: 0.8193 - precision_m: 0.8999 - recall_m: 0.7545
    Epoch 98/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5144 - accuracy: 0.8289 - f1_m: 0.8226 - precision_m: 0.8817 - recall_m: 0.7720
    Epoch 99/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5610 - accuracy: 0.8035 - f1_m: 0.8054 - precision_m: 0.8740 - recall_m: 0.7476
    Epoch 100/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5448 - accuracy: 0.8098 - f1_m: 0.8039 - precision_m: 0.8705 - recall_m: 0.7477
    Epoch 101/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5193 - accuracy: 0.8126 - f1_m: 0.8194 - precision_m: 0.8827 - recall_m: 0.7665
    Epoch 102/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5725 - accuracy: 0.7962 - f1_m: 0.7712 - precision_m: 0.8491 - recall_m: 0.7084
    Epoch 103/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5733 - accuracy: 0.7971 - f1_m: 0.7827 - precision_m: 0.8611 - recall_m: 0.7185
    Epoch 104/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5555 - accuracy: 0.8035 - f1_m: 0.8063 - precision_m: 0.8820 - recall_m: 0.7441
    Epoch 105/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5935 - accuracy: 0.7862 - f1_m: 0.7770 - precision_m: 0.8590 - recall_m: 0.7116
    Epoch 106/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5751 - accuracy: 0.7998 - f1_m: 0.7846 - precision_m: 0.8598 - recall_m: 0.7220
    Epoch 107/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5483 - accuracy: 0.8126 - f1_m: 0.8026 - precision_m: 0.8776 - recall_m: 0.7401
    Epoch 108/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4932 - accuracy: 0.8308 - f1_m: 0.8006 - precision_m: 0.8740 - recall_m: 0.7405
    Epoch 109/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5517 - accuracy: 0.8116 - f1_m: 0.7959 - precision_m: 0.8661 - recall_m: 0.7367
    Epoch 110/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5003 - accuracy: 0.8198 - f1_m: 0.8242 - precision_m: 0.8824 - recall_m: 0.7738
    Epoch 111/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5015 - accuracy: 0.8335 - f1_m: 0.8196 - precision_m: 0.8909 - recall_m: 0.7602
    Epoch 112/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5526 - accuracy: 0.8080 - f1_m: 0.8002 - precision_m: 0.8739 - recall_m: 0.7393
    Epoch 113/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5278 - accuracy: 0.8153 - f1_m: 0.8014 - precision_m: 0.8673 - recall_m: 0.7456
    Epoch 114/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4845 - accuracy: 0.8362 - f1_m: 0.8198 - precision_m: 0.8902 - recall_m: 0.7609
    Epoch 115/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5133 - accuracy: 0.8253 - f1_m: 0.8323 - precision_m: 0.8970 - recall_m: 0.7778
    Epoch 116/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4510 - accuracy: 0.8417 - f1_m: 0.8434 - precision_m: 0.8990 - recall_m: 0.7955
    Epoch 117/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5219 - accuracy: 0.8189 - f1_m: 0.8116 - precision_m: 0.8810 - recall_m: 0.7529
    Epoch 118/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4736 - accuracy: 0.8308 - f1_m: 0.8290 - precision_m: 0.8828 - recall_m: 0.7823
    Epoch 119/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4741 - accuracy: 0.8362 - f1_m: 0.8370 - precision_m: 0.8933 - recall_m: 0.7884
    Epoch 120/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4837 - accuracy: 0.8353 - f1_m: 0.8358 - precision_m: 0.8868 - recall_m: 0.7910
    Epoch 121/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4429 - accuracy: 0.8490 - f1_m: 0.8512 - precision_m: 0.8999 - recall_m: 0.8085
    Epoch 122/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5173 - accuracy: 0.8098 - f1_m: 0.7985 - precision_m: 0.8642 - recall_m: 0.7428
    Epoch 123/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5174 - accuracy: 0.8189 - f1_m: 0.8069 - precision_m: 0.8729 - recall_m: 0.7522
    Epoch 124/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5497 - accuracy: 0.8071 - f1_m: 0.7947 - precision_m: 0.8649 - recall_m: 0.7364
    Epoch 125/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5401 - accuracy: 0.7998 - f1_m: 0.7949 - precision_m: 0.8734 - recall_m: 0.7307
    Epoch 126/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4992 - accuracy: 0.8326 - f1_m: 0.8242 - precision_m: 0.8823 - recall_m: 0.7739
    Epoch 127/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5592 - accuracy: 0.8098 - f1_m: 0.8034 - precision_m: 0.8707 - recall_m: 0.7470
    Epoch 128/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4894 - accuracy: 0.8344 - f1_m: 0.8169 - precision_m: 0.8820 - recall_m: 0.7616
    Epoch 129/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4546 - accuracy: 0.8490 - f1_m: 0.8266 - precision_m: 0.8961 - recall_m: 0.7682
    Epoch 130/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4745 - accuracy: 0.8335 - f1_m: 0.8131 - precision_m: 0.8777 - recall_m: 0.7584
    Epoch 131/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4626 - accuracy: 0.8380 - f1_m: 0.8305 - precision_m: 0.8834 - recall_m: 0.7843
    Epoch 132/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5229 - accuracy: 0.8262 - f1_m: 0.8163 - precision_m: 0.8676 - recall_m: 0.7720
    Epoch 133/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4838 - accuracy: 0.8262 - f1_m: 0.8308 - precision_m: 0.8807 - recall_m: 0.7873
    Epoch 134/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4628 - accuracy: 0.8462 - f1_m: 0.8387 - precision_m: 0.8870 - recall_m: 0.7963
    Epoch 135/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5802 - accuracy: 0.7980 - f1_m: 0.7922 - precision_m: 0.8525 - recall_m: 0.7409
    Epoch 136/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5480 - accuracy: 0.7962 - f1_m: 0.7749 - precision_m: 0.8474 - recall_m: 0.7173
    Epoch 137/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5743 - accuracy: 0.7898 - f1_m: 0.7841 - precision_m: 0.8584 - recall_m: 0.7232
    Epoch 138/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4863 - accuracy: 0.8226 - f1_m: 0.8287 - precision_m: 0.8831 - recall_m: 0.7814
    Epoch 139/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4655 - accuracy: 0.8317 - f1_m: 0.8279 - precision_m: 0.8913 - recall_m: 0.7736
    Epoch 140/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4616 - accuracy: 0.8344 - f1_m: 0.8311 - precision_m: 0.8842 - recall_m: 0.7852
    Epoch 141/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5189 - accuracy: 0.8308 - f1_m: 0.8171 - precision_m: 0.8812 - recall_m: 0.7635
    Epoch 142/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4915 - accuracy: 0.8371 - f1_m: 0.8298 - precision_m: 0.8996 - recall_m: 0.7711
    Epoch 143/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4845 - accuracy: 0.8335 - f1_m: 0.8130 - precision_m: 0.8722 - recall_m: 0.7621
    Epoch 144/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5153 - accuracy: 0.8144 - f1_m: 0.8016 - precision_m: 0.8689 - recall_m: 0.7454
    Epoch 145/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4621 - accuracy: 0.8362 - f1_m: 0.8321 - precision_m: 0.8883 - recall_m: 0.7835
    Epoch 146/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4842 - accuracy: 0.8271 - f1_m: 0.8302 - precision_m: 0.8821 - recall_m: 0.7852
    Epoch 147/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4634 - accuracy: 0.8380 - f1_m: 0.8392 - precision_m: 0.8960 - recall_m: 0.7901
    Epoch 148/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4614 - accuracy: 0.8371 - f1_m: 0.8229 - precision_m: 0.8792 - recall_m: 0.7746
    Epoch 149/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4421 - accuracy: 0.8526 - f1_m: 0.8447 - precision_m: 0.8957 - recall_m: 0.7998
    Epoch 150/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4277 - accuracy: 0.8517 - f1_m: 0.8556 - precision_m: 0.8994 - recall_m: 0.8168
    9/9 [==============================] - 0s 4ms/step - loss: 0.7047 - accuracy: 0.7564 - f1_m: 0.7726 - precision_m: 0.8403 - recall_m: 0.7156
    Epoch 1/150
    18/18 [==============================] - 1s 8ms/step - loss: 2.5749 - accuracy: 0.1165 - f1_m: 0.0064 - precision_m: 0.0329 - recall_m: 0.0052
    Epoch 2/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.4471 - accuracy: 0.1228 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 3/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.3245 - accuracy: 0.1447 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 4/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.3272 - accuracy: 0.1574 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 5/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.3003 - accuracy: 0.1419 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 6/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.2416 - accuracy: 0.1720 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 7/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.1977 - accuracy: 0.1993 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 8/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.1467 - accuracy: 0.2320 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 9/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.0884 - accuracy: 0.2520 - f1_m: 0.0102 - precision_m: 0.2778 - recall_m: 0.0052
    Epoch 10/150
    18/18 [==============================] - 0s 8ms/step - loss: 2.0343 - accuracy: 0.2875 - f1_m: 0.0538 - precision_m: 0.7865 - recall_m: 0.0285
    Epoch 11/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.9526 - accuracy: 0.3003 - f1_m: 0.0717 - precision_m: 0.8639 - recall_m: 0.0380
    Epoch 12/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.8978 - accuracy: 0.3312 - f1_m: 0.0921 - precision_m: 0.8282 - recall_m: 0.0495
    Epoch 13/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.7976 - accuracy: 0.3676 - f1_m: 0.1124 - precision_m: 0.8897 - recall_m: 0.0606
    Epoch 14/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.7517 - accuracy: 0.3849 - f1_m: 0.1364 - precision_m: 0.7871 - recall_m: 0.0764
    Epoch 15/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.6582 - accuracy: 0.4368 - f1_m: 0.1735 - precision_m: 0.7941 - recall_m: 0.1002
    Epoch 16/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.5996 - accuracy: 0.4504 - f1_m: 0.2064 - precision_m: 0.7864 - recall_m: 0.1207
    Epoch 17/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.6167 - accuracy: 0.4158 - f1_m: 0.2547 - precision_m: 0.8381 - recall_m: 0.1532
    Epoch 18/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4683 - accuracy: 0.4904 - f1_m: 0.2786 - precision_m: 0.7948 - recall_m: 0.1727
    Epoch 19/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.4183 - accuracy: 0.5096 - f1_m: 0.3007 - precision_m: 0.8078 - recall_m: 0.1884
    Epoch 20/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.3678 - accuracy: 0.5105 - f1_m: 0.3849 - precision_m: 0.8319 - recall_m: 0.2536
    Epoch 21/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3297 - accuracy: 0.5205 - f1_m: 0.3587 - precision_m: 0.7903 - recall_m: 0.2351
    Epoch 22/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.2727 - accuracy: 0.5405 - f1_m: 0.4315 - precision_m: 0.8173 - recall_m: 0.2955
    Epoch 23/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.2520 - accuracy: 0.5560 - f1_m: 0.4612 - precision_m: 0.8296 - recall_m: 0.3256
    Epoch 24/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.2051 - accuracy: 0.5778 - f1_m: 0.4622 - precision_m: 0.8212 - recall_m: 0.3243
    Epoch 25/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.2308 - accuracy: 0.5751 - f1_m: 0.4708 - precision_m: 0.8341 - recall_m: 0.3320
    Epoch 26/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1954 - accuracy: 0.5732 - f1_m: 0.4722 - precision_m: 0.7896 - recall_m: 0.3382
    Epoch 27/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1662 - accuracy: 0.5842 - f1_m: 0.5108 - precision_m: 0.8055 - recall_m: 0.3759
    Epoch 28/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1499 - accuracy: 0.6033 - f1_m: 0.4974 - precision_m: 0.7874 - recall_m: 0.3655
    Epoch 29/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.0648 - accuracy: 0.6142 - f1_m: 0.5788 - precision_m: 0.8325 - recall_m: 0.4451
    Epoch 30/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0259 - accuracy: 0.6542 - f1_m: 0.5867 - precision_m: 0.8139 - recall_m: 0.4609
    Epoch 31/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.0024 - accuracy: 0.6515 - f1_m: 0.6049 - precision_m: 0.8610 - recall_m: 0.4694
    Epoch 32/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9436 - accuracy: 0.6870 - f1_m: 0.6411 - precision_m: 0.8525 - recall_m: 0.5163
    Epoch 33/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9374 - accuracy: 0.6724 - f1_m: 0.6133 - precision_m: 0.8121 - recall_m: 0.4949
    Epoch 34/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9306 - accuracy: 0.6870 - f1_m: 0.6310 - precision_m: 0.8365 - recall_m: 0.5099
    Epoch 35/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9696 - accuracy: 0.6733 - f1_m: 0.6444 - precision_m: 0.8330 - recall_m: 0.5267
    Epoch 36/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8766 - accuracy: 0.7015 - f1_m: 0.6508 - precision_m: 0.8160 - recall_m: 0.5428
    Epoch 37/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9526 - accuracy: 0.6697 - f1_m: 0.6241 - precision_m: 0.8207 - recall_m: 0.5051
    Epoch 38/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.9312 - accuracy: 0.6970 - f1_m: 0.6400 - precision_m: 0.8166 - recall_m: 0.5287
    Epoch 39/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8495 - accuracy: 0.7097 - f1_m: 0.6837 - precision_m: 0.8199 - recall_m: 0.5878
    Epoch 40/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8560 - accuracy: 0.7052 - f1_m: 0.6658 - precision_m: 0.8218 - recall_m: 0.5610
    Epoch 41/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8611 - accuracy: 0.7188 - f1_m: 0.6833 - precision_m: 0.8338 - recall_m: 0.5808
    Epoch 42/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.8580 - accuracy: 0.7116 - f1_m: 0.6935 - precision_m: 0.8475 - recall_m: 0.5893
    Epoch 43/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7892 - accuracy: 0.7425 - f1_m: 0.7032 - precision_m: 0.8467 - recall_m: 0.6031
    Epoch 44/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7936 - accuracy: 0.7343 - f1_m: 0.7159 - precision_m: 0.8511 - recall_m: 0.6201
    Epoch 45/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7324 - accuracy: 0.7389 - f1_m: 0.7079 - precision_m: 0.8452 - recall_m: 0.6102
    Epoch 46/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8259 - accuracy: 0.7288 - f1_m: 0.7092 - precision_m: 0.8421 - recall_m: 0.6142
    Epoch 47/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7539 - accuracy: 0.7270 - f1_m: 0.7190 - precision_m: 0.8426 - recall_m: 0.6281
    Epoch 48/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7530 - accuracy: 0.7343 - f1_m: 0.7316 - precision_m: 0.8535 - recall_m: 0.6418
    Epoch 49/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7771 - accuracy: 0.7325 - f1_m: 0.7052 - precision_m: 0.8350 - recall_m: 0.6117
    Epoch 50/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7600 - accuracy: 0.7234 - f1_m: 0.7220 - precision_m: 0.8489 - recall_m: 0.6312
    Epoch 51/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7307 - accuracy: 0.7379 - f1_m: 0.7275 - precision_m: 0.8449 - recall_m: 0.6402
    Epoch 52/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7104 - accuracy: 0.7561 - f1_m: 0.7467 - precision_m: 0.8583 - recall_m: 0.6626
    Epoch 53/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6907 - accuracy: 0.7680 - f1_m: 0.7579 - precision_m: 0.8615 - recall_m: 0.6774
    Epoch 54/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6677 - accuracy: 0.7798 - f1_m: 0.7663 - precision_m: 0.8816 - recall_m: 0.6795
    Epoch 55/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7234 - accuracy: 0.7443 - f1_m: 0.7058 - precision_m: 0.8147 - recall_m: 0.6245
    Epoch 56/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7028 - accuracy: 0.7753 - f1_m: 0.7406 - precision_m: 0.8500 - recall_m: 0.6577
    Epoch 57/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6920 - accuracy: 0.7534 - f1_m: 0.7391 - precision_m: 0.8367 - recall_m: 0.6631
    Epoch 58/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6523 - accuracy: 0.7753 - f1_m: 0.7577 - precision_m: 0.8592 - recall_m: 0.6795
    Epoch 59/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7024 - accuracy: 0.7507 - f1_m: 0.7331 - precision_m: 0.8372 - recall_m: 0.6534
    Epoch 60/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6900 - accuracy: 0.7753 - f1_m: 0.7569 - precision_m: 0.8680 - recall_m: 0.6727
    Epoch 61/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6602 - accuracy: 0.7771 - f1_m: 0.7486 - precision_m: 0.8313 - recall_m: 0.6825
    Epoch 62/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7818 - accuracy: 0.7279 - f1_m: 0.7086 - precision_m: 0.8360 - recall_m: 0.6170
    Epoch 63/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7325 - accuracy: 0.7507 - f1_m: 0.7344 - precision_m: 0.8565 - recall_m: 0.6458
    Epoch 64/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7442 - accuracy: 0.7470 - f1_m: 0.7376 - precision_m: 0.8590 - recall_m: 0.6479
    Epoch 65/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6884 - accuracy: 0.7634 - f1_m: 0.7610 - precision_m: 0.8632 - recall_m: 0.6810
    Epoch 66/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6811 - accuracy: 0.7652 - f1_m: 0.7528 - precision_m: 0.8413 - recall_m: 0.6829
    Epoch 67/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6841 - accuracy: 0.7552 - f1_m: 0.7378 - precision_m: 0.8300 - recall_m: 0.6654
    Epoch 68/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6108 - accuracy: 0.7916 - f1_m: 0.7799 - precision_m: 0.8577 - recall_m: 0.7159
    Epoch 69/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.7117 - accuracy: 0.7389 - f1_m: 0.7301 - precision_m: 0.8337 - recall_m: 0.6499
    Epoch 70/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6416 - accuracy: 0.7898 - f1_m: 0.7624 - precision_m: 0.8682 - recall_m: 0.6810
    Epoch 71/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6402 - accuracy: 0.7944 - f1_m: 0.7807 - precision_m: 0.8863 - recall_m: 0.6991
    Epoch 72/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6234 - accuracy: 0.7962 - f1_m: 0.7889 - precision_m: 0.8766 - recall_m: 0.7182
    Epoch 73/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5893 - accuracy: 0.8016 - f1_m: 0.7902 - precision_m: 0.8643 - recall_m: 0.7295
    Epoch 74/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6173 - accuracy: 0.7816 - f1_m: 0.7777 - precision_m: 0.8649 - recall_m: 0.7071
    Epoch 75/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6169 - accuracy: 0.7871 - f1_m: 0.7706 - precision_m: 0.8569 - recall_m: 0.7015
    Epoch 76/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6422 - accuracy: 0.7680 - f1_m: 0.7638 - precision_m: 0.8511 - recall_m: 0.6932
    Epoch 77/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6295 - accuracy: 0.7925 - f1_m: 0.7854 - precision_m: 0.8698 - recall_m: 0.7172
    Epoch 78/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5614 - accuracy: 0.8080 - f1_m: 0.7977 - precision_m: 0.8651 - recall_m: 0.7409
    Epoch 79/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5576 - accuracy: 0.8035 - f1_m: 0.7979 - precision_m: 0.8828 - recall_m: 0.7307
    Epoch 80/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5358 - accuracy: 0.8189 - f1_m: 0.8127 - precision_m: 0.8815 - recall_m: 0.7547
    Epoch 81/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5567 - accuracy: 0.8053 - f1_m: 0.7965 - precision_m: 0.8745 - recall_m: 0.7322
    Epoch 82/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5239 - accuracy: 0.8289 - f1_m: 0.8304 - precision_m: 0.8979 - recall_m: 0.7729
    Epoch 83/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5395 - accuracy: 0.8198 - f1_m: 0.8197 - precision_m: 0.8890 - recall_m: 0.7614
    Epoch 84/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5762 - accuracy: 0.7889 - f1_m: 0.7796 - precision_m: 0.8722 - recall_m: 0.7065
    Epoch 85/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5776 - accuracy: 0.7934 - f1_m: 0.7854 - precision_m: 0.8561 - recall_m: 0.7262
    Epoch 86/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5860 - accuracy: 0.7925 - f1_m: 0.7945 - precision_m: 0.8747 - recall_m: 0.7300
    Epoch 87/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5719 - accuracy: 0.7889 - f1_m: 0.7741 - precision_m: 0.8456 - recall_m: 0.7149
    Epoch 88/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5788 - accuracy: 0.8080 - f1_m: 0.7847 - precision_m: 0.8705 - recall_m: 0.7162
    Epoch 89/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5395 - accuracy: 0.8144 - f1_m: 0.8141 - precision_m: 0.8751 - recall_m: 0.7622
    Epoch 90/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5503 - accuracy: 0.8116 - f1_m: 0.7977 - precision_m: 0.8726 - recall_m: 0.7357
    Epoch 91/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5376 - accuracy: 0.8062 - f1_m: 0.8111 - precision_m: 0.8791 - recall_m: 0.7536
    Epoch 92/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4968 - accuracy: 0.8335 - f1_m: 0.8259 - precision_m: 0.8948 - recall_m: 0.7677
    Epoch 93/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5041 - accuracy: 0.8235 - f1_m: 0.8104 - precision_m: 0.8731 - recall_m: 0.7577
    Epoch 94/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5127 - accuracy: 0.8335 - f1_m: 0.8246 - precision_m: 0.8962 - recall_m: 0.7649
    Epoch 95/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5197 - accuracy: 0.8189 - f1_m: 0.8085 - precision_m: 0.8835 - recall_m: 0.7463
    Epoch 96/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5347 - accuracy: 0.8226 - f1_m: 0.8165 - precision_m: 0.8955 - recall_m: 0.7522
    Epoch 97/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6463 - accuracy: 0.7853 - f1_m: 0.7700 - precision_m: 0.8532 - recall_m: 0.7026
    Epoch 98/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5541 - accuracy: 0.8116 - f1_m: 0.7971 - precision_m: 0.8779 - recall_m: 0.7315
    Epoch 99/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5952 - accuracy: 0.7971 - f1_m: 0.7918 - precision_m: 0.8872 - recall_m: 0.7173
    Epoch 100/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5388 - accuracy: 0.8162 - f1_m: 0.7963 - precision_m: 0.8819 - recall_m: 0.7279
    Epoch 101/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6394 - accuracy: 0.7725 - f1_m: 0.7664 - precision_m: 0.8505 - recall_m: 0.6984
    Epoch 102/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5881 - accuracy: 0.7962 - f1_m: 0.7964 - precision_m: 0.8743 - recall_m: 0.7326
    Epoch 103/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4746 - accuracy: 0.8526 - f1_m: 0.8358 - precision_m: 0.9028 - recall_m: 0.7797
    Epoch 104/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4698 - accuracy: 0.8353 - f1_m: 0.8211 - precision_m: 0.8862 - recall_m: 0.7664
    Epoch 105/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4830 - accuracy: 0.8262 - f1_m: 0.8275 - precision_m: 0.8935 - recall_m: 0.7713
    Epoch 106/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4750 - accuracy: 0.8362 - f1_m: 0.8372 - precision_m: 0.8984 - recall_m: 0.7856
    Epoch 107/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4643 - accuracy: 0.8408 - f1_m: 0.8290 - precision_m: 0.8895 - recall_m: 0.7775
    Epoch 108/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4745 - accuracy: 0.8426 - f1_m: 0.8306 - precision_m: 0.8912 - recall_m: 0.7783
    Epoch 109/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4490 - accuracy: 0.8526 - f1_m: 0.8302 - precision_m: 0.8912 - recall_m: 0.7777
    Epoch 110/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5247 - accuracy: 0.8080 - f1_m: 0.7970 - precision_m: 0.8809 - recall_m: 0.7291
    Epoch 111/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4942 - accuracy: 0.8280 - f1_m: 0.8199 - precision_m: 0.8802 - recall_m: 0.7680
    Epoch 112/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4645 - accuracy: 0.8417 - f1_m: 0.8294 - precision_m: 0.9042 - recall_m: 0.7671
    Epoch 113/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4680 - accuracy: 0.8271 - f1_m: 0.8234 - precision_m: 0.8705 - recall_m: 0.7817
    Epoch 114/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5111 - accuracy: 0.8298 - f1_m: 0.8194 - precision_m: 0.8893 - recall_m: 0.7607
    Epoch 115/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5031 - accuracy: 0.8280 - f1_m: 0.8158 - precision_m: 0.8787 - recall_m: 0.7616
    Epoch 116/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4440 - accuracy: 0.8362 - f1_m: 0.8317 - precision_m: 0.8860 - recall_m: 0.7845
    Epoch 117/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4813 - accuracy: 0.8198 - f1_m: 0.8287 - precision_m: 0.8760 - recall_m: 0.7875
    Epoch 118/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5379 - accuracy: 0.8016 - f1_m: 0.8052 - precision_m: 0.8725 - recall_m: 0.7484
    Epoch 119/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5014 - accuracy: 0.8189 - f1_m: 0.8189 - precision_m: 0.8911 - recall_m: 0.7581
    Epoch 120/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4688 - accuracy: 0.8335 - f1_m: 0.8329 - precision_m: 0.8928 - recall_m: 0.7814
    Epoch 121/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4774 - accuracy: 0.8235 - f1_m: 0.8186 - precision_m: 0.8786 - recall_m: 0.7668
    Epoch 122/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5361 - accuracy: 0.7989 - f1_m: 0.8048 - precision_m: 0.8559 - recall_m: 0.7599
    Epoch 123/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5151 - accuracy: 0.8180 - f1_m: 0.8134 - precision_m: 0.8779 - recall_m: 0.7590
    Epoch 124/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4713 - accuracy: 0.8262 - f1_m: 0.8300 - precision_m: 0.8896 - recall_m: 0.7788
    Epoch 125/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4314 - accuracy: 0.8380 - f1_m: 0.8348 - precision_m: 0.8878 - recall_m: 0.7885
    Epoch 126/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4980 - accuracy: 0.8226 - f1_m: 0.8239 - precision_m: 0.8700 - recall_m: 0.7833
    Epoch 127/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4327 - accuracy: 0.8508 - f1_m: 0.8415 - precision_m: 0.9019 - recall_m: 0.7894
    Epoch 128/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4798 - accuracy: 0.8362 - f1_m: 0.8322 - precision_m: 0.8942 - recall_m: 0.7791
    Epoch 129/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4506 - accuracy: 0.8417 - f1_m: 0.8394 - precision_m: 0.8949 - recall_m: 0.7911
    Epoch 130/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4837 - accuracy: 0.8399 - f1_m: 0.8333 - precision_m: 0.8899 - recall_m: 0.7842
    Epoch 131/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4513 - accuracy: 0.8389 - f1_m: 0.8330 - precision_m: 0.8805 - recall_m: 0.7911
    Epoch 132/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4604 - accuracy: 0.8317 - f1_m: 0.8357 - precision_m: 0.8957 - recall_m: 0.7847
    Epoch 133/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4695 - accuracy: 0.8298 - f1_m: 0.8310 - precision_m: 0.8975 - recall_m: 0.7749
    Epoch 134/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4920 - accuracy: 0.8217 - f1_m: 0.8261 - precision_m: 0.8782 - recall_m: 0.7807
    Epoch 135/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4736 - accuracy: 0.8380 - f1_m: 0.8164 - precision_m: 0.8758 - recall_m: 0.7657
    Epoch 136/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5417 - accuracy: 0.7980 - f1_m: 0.7934 - precision_m: 0.8635 - recall_m: 0.7347
    Epoch 137/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4928 - accuracy: 0.8289 - f1_m: 0.8295 - precision_m: 0.8872 - recall_m: 0.7797
    Epoch 138/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4625 - accuracy: 0.8353 - f1_m: 0.8253 - precision_m: 0.8924 - recall_m: 0.7687
    Epoch 139/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4254 - accuracy: 0.8553 - f1_m: 0.8513 - precision_m: 0.9030 - recall_m: 0.8060
    Epoch 140/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.4883 - accuracy: 0.8198 - f1_m: 0.8238 - precision_m: 0.8750 - recall_m: 0.7788
    Epoch 141/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5114 - accuracy: 0.8226 - f1_m: 0.8178 - precision_m: 0.8841 - recall_m: 0.7622
    Epoch 142/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4878 - accuracy: 0.8253 - f1_m: 0.8294 - precision_m: 0.8818 - recall_m: 0.7840
    Epoch 143/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5011 - accuracy: 0.8226 - f1_m: 0.8282 - precision_m: 0.8972 - recall_m: 0.7701
    Epoch 144/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5033 - accuracy: 0.8071 - f1_m: 0.8058 - precision_m: 0.8702 - recall_m: 0.7515
    Epoch 145/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5652 - accuracy: 0.7925 - f1_m: 0.7916 - precision_m: 0.8693 - recall_m: 0.7285
    Epoch 146/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4895 - accuracy: 0.8217 - f1_m: 0.8272 - precision_m: 0.8872 - recall_m: 0.7755
    Epoch 147/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4443 - accuracy: 0.8408 - f1_m: 0.8190 - precision_m: 0.8866 - recall_m: 0.7621
    Epoch 148/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5372 - accuracy: 0.8107 - f1_m: 0.8040 - precision_m: 0.8827 - recall_m: 0.7397
    Epoch 149/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5480 - accuracy: 0.8153 - f1_m: 0.8029 - precision_m: 0.8852 - recall_m: 0.7367
    Epoch 150/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5005 - accuracy: 0.8317 - f1_m: 0.8269 - precision_m: 0.8864 - recall_m: 0.7756
    9/9 [==============================] - 0s 5ms/step - loss: 0.8060 - accuracy: 0.7727 - f1_m: 0.7693 - precision_m: 0.8152 - recall_m: 0.7294
    Epoch 1/150
    18/18 [==============================] - 1s 9ms/step - loss: 2.5424 - accuracy: 0.1009 - f1_m: 0.0034 - precision_m: 0.0556 - recall_m: 0.0017
    Epoch 2/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.4492 - accuracy: 0.1382 - f1_m: 0.0051 - precision_m: 0.1389 - recall_m: 0.0026
    Epoch 3/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.3740 - accuracy: 0.1373 - f1_m: 0.0017 - precision_m: 0.0278 - recall_m: 8.6806e-04
    Epoch 4/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.2978 - accuracy: 0.1473 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 5/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.2845 - accuracy: 0.1536 - f1_m: 0.0017 - precision_m: 0.0556 - recall_m: 8.6806e-04
    Epoch 6/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.2171 - accuracy: 0.1764 - f1_m: 0.0051 - precision_m: 0.1667 - recall_m: 0.0026
    Epoch 7/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.1891 - accuracy: 0.1982 - f1_m: 0.0017 - precision_m: 0.0556 - recall_m: 8.6806e-04    
    Epoch 8/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.1368 - accuracy: 0.2282 - f1_m: 0.0085 - precision_m: 0.2778 - recall_m: 0.0043
    Epoch 9/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.1083 - accuracy: 0.2464 - f1_m: 0.0101 - precision_m: 0.1667 - recall_m: 0.0052
    Epoch 10/150
    18/18 [==============================] - 0s 9ms/step - loss: 2.0427 - accuracy: 0.2664 - f1_m: 0.0481 - precision_m: 0.6741 - recall_m: 0.0252
    Epoch 11/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.9992 - accuracy: 0.2973 - f1_m: 0.0495 - precision_m: 0.7679 - recall_m: 0.0260
    Epoch 12/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.9200 - accuracy: 0.3336 - f1_m: 0.0808 - precision_m: 0.8532 - recall_m: 0.0434
    Epoch 13/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.8536 - accuracy: 0.3636 - f1_m: 0.0907 - precision_m: 0.8583 - recall_m: 0.0489
    Epoch 14/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.7689 - accuracy: 0.3791 - f1_m: 0.1329 - precision_m: 0.8461 - recall_m: 0.0735
    Epoch 15/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.6888 - accuracy: 0.4127 - f1_m: 0.1844 - precision_m: 0.8569 - recall_m: 0.1053
    Epoch 16/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.6428 - accuracy: 0.4100 - f1_m: 0.1848 - precision_m: 0.8784 - recall_m: 0.1045
    Epoch 17/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.5619 - accuracy: 0.4727 - f1_m: 0.2360 - precision_m: 0.8443 - recall_m: 0.1398
    Epoch 18/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4674 - accuracy: 0.4973 - f1_m: 0.2906 - precision_m: 0.8339 - recall_m: 0.1808
    Epoch 19/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4066 - accuracy: 0.5164 - f1_m: 0.3236 - precision_m: 0.8397 - recall_m: 0.2052
    Epoch 20/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3964 - accuracy: 0.5282 - f1_m: 0.3577 - precision_m: 0.8851 - recall_m: 0.2269
    Epoch 21/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3560 - accuracy: 0.5309 - f1_m: 0.3799 - precision_m: 0.8257 - recall_m: 0.2491
    Epoch 22/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3276 - accuracy: 0.5491 - f1_m: 0.4046 - precision_m: 0.8133 - recall_m: 0.2714
    Epoch 23/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.3101 - accuracy: 0.5418 - f1_m: 0.4224 - precision_m: 0.8137 - recall_m: 0.2879
    Epoch 24/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.2398 - accuracy: 0.5827 - f1_m: 0.4523 - precision_m: 0.8387 - recall_m: 0.3139
    Epoch 25/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1694 - accuracy: 0.5982 - f1_m: 0.5056 - precision_m: 0.8423 - recall_m: 0.3643
    Epoch 26/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1153 - accuracy: 0.6264 - f1_m: 0.5304 - precision_m: 0.8586 - recall_m: 0.3857
    Epoch 27/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.1059 - accuracy: 0.6364 - f1_m: 0.5300 - precision_m: 0.8575 - recall_m: 0.3857
    Epoch 28/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1140 - accuracy: 0.6200 - f1_m: 0.5358 - precision_m: 0.8532 - recall_m: 0.3938
    Epoch 29/150
    18/18 [==============================] - 0s 8ms/step - loss: 1.0581 - accuracy: 0.6445 - f1_m: 0.5733 - precision_m: 0.8511 - recall_m: 0.4363
    Epoch 30/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0437 - accuracy: 0.6345 - f1_m: 0.5816 - precision_m: 0.8198 - recall_m: 0.4531
    Epoch 31/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0741 - accuracy: 0.6273 - f1_m: 0.5613 - precision_m: 0.8299 - recall_m: 0.4268
    Epoch 32/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1220 - accuracy: 0.6191 - f1_m: 0.5271 - precision_m: 0.7802 - recall_m: 0.4008
    Epoch 33/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9975 - accuracy: 0.6609 - f1_m: 0.6177 - precision_m: 0.8473 - recall_m: 0.4884
    Epoch 34/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9658 - accuracy: 0.6745 - f1_m: 0.6325 - precision_m: 0.8534 - recall_m: 0.5046
    Epoch 35/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9596 - accuracy: 0.6709 - f1_m: 0.6243 - precision_m: 0.8446 - recall_m: 0.4983
    Epoch 36/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9977 - accuracy: 0.6455 - f1_m: 0.5888 - precision_m: 0.8026 - recall_m: 0.4670
    Epoch 37/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9147 - accuracy: 0.6891 - f1_m: 0.6520 - precision_m: 0.8393 - recall_m: 0.5356
    Epoch 38/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9568 - accuracy: 0.6936 - f1_m: 0.6420 - precision_m: 0.8257 - recall_m: 0.5286
    Epoch 39/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8538 - accuracy: 0.7118 - f1_m: 0.6649 - precision_m: 0.8563 - recall_m: 0.5460
    Epoch 40/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8810 - accuracy: 0.6927 - f1_m: 0.6665 - precision_m: 0.8337 - recall_m: 0.5567
    Epoch 41/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8941 - accuracy: 0.6827 - f1_m: 0.6490 - precision_m: 0.8183 - recall_m: 0.5391
    Epoch 42/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8387 - accuracy: 0.7209 - f1_m: 0.6828 - precision_m: 0.8523 - recall_m: 0.5715
    Epoch 43/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8038 - accuracy: 0.7464 - f1_m: 0.7010 - precision_m: 0.8593 - recall_m: 0.5935
    Epoch 44/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8166 - accuracy: 0.7191 - f1_m: 0.6964 - precision_m: 0.8334 - recall_m: 0.6001
    Epoch 45/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7497 - accuracy: 0.7536 - f1_m: 0.7322 - precision_m: 0.8668 - recall_m: 0.6351
    Epoch 46/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8327 - accuracy: 0.7064 - f1_m: 0.7020 - precision_m: 0.8407 - recall_m: 0.6045
    Epoch 47/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7587 - accuracy: 0.7518 - f1_m: 0.7255 - precision_m: 0.8529 - recall_m: 0.6328
    Epoch 48/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8013 - accuracy: 0.7182 - f1_m: 0.6993 - precision_m: 0.8460 - recall_m: 0.5981
    Epoch 49/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7311 - accuracy: 0.7618 - f1_m: 0.7468 - precision_m: 0.8652 - recall_m: 0.6580
    Epoch 50/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.6996 - accuracy: 0.7673 - f1_m: 0.7410 - precision_m: 0.8647 - recall_m: 0.6505
    Epoch 51/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7022 - accuracy: 0.7727 - f1_m: 0.7457 - precision_m: 0.8551 - recall_m: 0.6623
    Epoch 52/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7236 - accuracy: 0.7482 - f1_m: 0.7365 - precision_m: 0.8617 - recall_m: 0.6444
    Epoch 53/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7112 - accuracy: 0.7455 - f1_m: 0.7358 - precision_m: 0.8560 - recall_m: 0.6467
    Epoch 54/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7486 - accuracy: 0.7436 - f1_m: 0.7247 - precision_m: 0.8371 - recall_m: 0.6403
    Epoch 55/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7461 - accuracy: 0.7436 - f1_m: 0.7321 - precision_m: 0.8522 - recall_m: 0.6427
    Epoch 56/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6890 - accuracy: 0.7709 - f1_m: 0.7548 - precision_m: 0.8728 - recall_m: 0.6658
    Epoch 57/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7021 - accuracy: 0.7591 - f1_m: 0.7428 - precision_m: 0.8538 - recall_m: 0.6600
    Epoch 58/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6558 - accuracy: 0.7836 - f1_m: 0.7672 - precision_m: 0.8839 - recall_m: 0.6788
    Epoch 59/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6919 - accuracy: 0.7727 - f1_m: 0.7399 - precision_m: 0.8427 - recall_m: 0.6606
    Epoch 60/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6713 - accuracy: 0.7591 - f1_m: 0.7549 - precision_m: 0.8637 - recall_m: 0.6727
    Epoch 61/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6238 - accuracy: 0.7864 - f1_m: 0.7637 - precision_m: 0.8558 - recall_m: 0.6910
    Epoch 62/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6908 - accuracy: 0.7609 - f1_m: 0.7487 - precision_m: 0.8609 - recall_m: 0.6638
    Epoch 63/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6400 - accuracy: 0.7855 - f1_m: 0.7593 - precision_m: 0.8581 - recall_m: 0.6843
    Epoch 64/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6918 - accuracy: 0.7655 - f1_m: 0.7539 - precision_m: 0.8480 - recall_m: 0.6814
    Epoch 65/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6829 - accuracy: 0.7764 - f1_m: 0.7654 - precision_m: 0.8601 - recall_m: 0.6907
    Epoch 66/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6411 - accuracy: 0.7836 - f1_m: 0.7799 - precision_m: 0.8724 - recall_m: 0.7069
    Epoch 67/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6923 - accuracy: 0.7545 - f1_m: 0.7462 - precision_m: 0.8290 - recall_m: 0.6800
    Epoch 68/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6276 - accuracy: 0.7882 - f1_m: 0.7745 - precision_m: 0.8795 - recall_m: 0.6939
    Epoch 69/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5705 - accuracy: 0.8191 - f1_m: 0.8100 - precision_m: 0.8921 - recall_m: 0.7425
    Epoch 70/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7102 - accuracy: 0.7491 - f1_m: 0.7459 - precision_m: 0.8431 - recall_m: 0.6704
    Epoch 71/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6581 - accuracy: 0.7700 - f1_m: 0.7742 - precision_m: 0.8706 - recall_m: 0.6979
    Epoch 72/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5877 - accuracy: 0.8036 - f1_m: 0.7907 - precision_m: 0.8812 - recall_m: 0.7188
    Epoch 73/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5800 - accuracy: 0.8009 - f1_m: 0.7899 - precision_m: 0.8737 - recall_m: 0.7219
    Epoch 74/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5886 - accuracy: 0.7973 - f1_m: 0.7935 - precision_m: 0.8818 - recall_m: 0.7234
    Epoch 75/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5646 - accuracy: 0.8073 - f1_m: 0.7976 - precision_m: 0.8813 - recall_m: 0.7303
    Epoch 76/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5350 - accuracy: 0.8264 - f1_m: 0.8257 - precision_m: 0.8940 - recall_m: 0.7677
    Epoch 77/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5149 - accuracy: 0.8245 - f1_m: 0.8141 - precision_m: 0.8845 - recall_m: 0.7567
    Epoch 78/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5672 - accuracy: 0.7964 - f1_m: 0.8011 - precision_m: 0.8625 - recall_m: 0.7491
    Epoch 79/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5713 - accuracy: 0.8036 - f1_m: 0.8052 - precision_m: 0.8919 - recall_m: 0.7355
    Epoch 80/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5697 - accuracy: 0.8136 - f1_m: 0.8012 - precision_m: 0.8755 - recall_m: 0.7396
    Epoch 81/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5407 - accuracy: 0.8173 - f1_m: 0.8057 - precision_m: 0.8890 - recall_m: 0.7378
    Epoch 82/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6431 - accuracy: 0.7636 - f1_m: 0.7758 - precision_m: 0.8467 - recall_m: 0.7173
    Epoch 83/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5789 - accuracy: 0.8082 - f1_m: 0.8068 - precision_m: 0.8793 - recall_m: 0.7462
    Epoch 84/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5461 - accuracy: 0.8200 - f1_m: 0.8148 - precision_m: 0.8939 - recall_m: 0.7494
    Epoch 85/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5230 - accuracy: 0.8355 - f1_m: 0.8253 - precision_m: 0.9057 - recall_m: 0.7601
    Epoch 86/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5307 - accuracy: 0.8391 - f1_m: 0.8388 - precision_m: 0.9057 - recall_m: 0.7815
    Epoch 87/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5171 - accuracy: 0.8191 - f1_m: 0.8172 - precision_m: 0.8829 - recall_m: 0.7613
    Epoch 88/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5022 - accuracy: 0.8191 - f1_m: 0.8275 - precision_m: 0.8867 - recall_m: 0.7769
    Epoch 89/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5023 - accuracy: 0.8409 - f1_m: 0.8258 - precision_m: 0.8918 - recall_m: 0.7697
    Epoch 90/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5680 - accuracy: 0.8036 - f1_m: 0.7878 - precision_m: 0.8582 - recall_m: 0.7295
    Epoch 91/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5391 - accuracy: 0.8136 - f1_m: 0.8052 - precision_m: 0.8743 - recall_m: 0.7471
    Epoch 92/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5298 - accuracy: 0.8309 - f1_m: 0.8202 - precision_m: 0.9032 - recall_m: 0.7523
    Epoch 93/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5162 - accuracy: 0.8245 - f1_m: 0.8210 - precision_m: 0.8952 - recall_m: 0.7590
    Epoch 94/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4757 - accuracy: 0.8409 - f1_m: 0.8377 - precision_m: 0.9042 - recall_m: 0.7818
    Epoch 95/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5087 - accuracy: 0.8218 - f1_m: 0.8185 - precision_m: 0.8974 - recall_m: 0.7535
    Epoch 96/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5211 - accuracy: 0.8127 - f1_m: 0.8078 - precision_m: 0.8710 - recall_m: 0.7541
    Epoch 97/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6116 - accuracy: 0.7818 - f1_m: 0.7805 - precision_m: 0.8407 - recall_m: 0.7292
    Epoch 98/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5499 - accuracy: 0.7973 - f1_m: 0.7840 - precision_m: 0.8578 - recall_m: 0.7231
    Epoch 99/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5370 - accuracy: 0.8109 - f1_m: 0.7906 - precision_m: 0.8667 - recall_m: 0.7283
    Epoch 100/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5006 - accuracy: 0.8309 - f1_m: 0.8206 - precision_m: 0.8969 - recall_m: 0.7584
    Epoch 101/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4430 - accuracy: 0.8591 - f1_m: 0.8520 - precision_m: 0.9091 - recall_m: 0.8024
    Epoch 102/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5129 - accuracy: 0.8082 - f1_m: 0.8045 - precision_m: 0.8682 - recall_m: 0.7503
    Epoch 103/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5051 - accuracy: 0.8273 - f1_m: 0.8285 - precision_m: 0.8925 - recall_m: 0.7737
    Epoch 104/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5186 - accuracy: 0.8182 - f1_m: 0.8094 - precision_m: 0.8887 - recall_m: 0.7454
    Epoch 105/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4600 - accuracy: 0.8418 - f1_m: 0.8346 - precision_m: 0.8877 - recall_m: 0.7879
    Epoch 106/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4777 - accuracy: 0.8400 - f1_m: 0.8163 - precision_m: 0.8864 - recall_m: 0.7575
    Epoch 107/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5156 - accuracy: 0.8155 - f1_m: 0.8088 - precision_m: 0.8659 - recall_m: 0.7604
    Epoch 108/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6081 - accuracy: 0.7864 - f1_m: 0.7882 - precision_m: 0.8688 - recall_m: 0.7228
    Epoch 109/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4700 - accuracy: 0.8327 - f1_m: 0.8354 - precision_m: 0.8956 - recall_m: 0.7841
    Epoch 110/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4470 - accuracy: 0.8491 - f1_m: 0.8439 - precision_m: 0.9012 - recall_m: 0.7943
    Epoch 111/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4732 - accuracy: 0.8355 - f1_m: 0.8246 - precision_m: 0.8890 - recall_m: 0.7700
    Epoch 112/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4222 - accuracy: 0.8491 - f1_m: 0.8527 - precision_m: 0.9039 - recall_m: 0.8076
    Epoch 113/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4678 - accuracy: 0.8382 - f1_m: 0.8403 - precision_m: 0.9034 - recall_m: 0.7867
    Epoch 114/150
    18/18 [==============================] - 0s 8ms/step - loss: 0.5150 - accuracy: 0.8255 - f1_m: 0.8090 - precision_m: 0.8768 - recall_m: 0.7529
    Epoch 115/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4705 - accuracy: 0.8391 - f1_m: 0.8362 - precision_m: 0.8835 - recall_m: 0.7946
    Epoch 116/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5633 - accuracy: 0.7955 - f1_m: 0.7914 - precision_m: 0.8601 - recall_m: 0.7352
    Epoch 117/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5042 - accuracy: 0.8245 - f1_m: 0.8261 - precision_m: 0.8907 - recall_m: 0.7711
    Epoch 118/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4402 - accuracy: 0.8591 - f1_m: 0.8556 - precision_m: 0.9052 - recall_m: 0.8122
    Epoch 119/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4611 - accuracy: 0.8355 - f1_m: 0.8384 - precision_m: 0.8951 - recall_m: 0.7891
    Epoch 120/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4547 - accuracy: 0.8427 - f1_m: 0.8372 - precision_m: 0.9019 - recall_m: 0.7850
    Epoch 121/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5159 - accuracy: 0.8236 - f1_m: 0.8209 - precision_m: 0.8849 - recall_m: 0.7662
    Epoch 122/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4288 - accuracy: 0.8673 - f1_m: 0.8449 - precision_m: 0.9008 - recall_m: 0.7960
    Epoch 123/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4512 - accuracy: 0.8373 - f1_m: 0.8409 - precision_m: 0.8927 - recall_m: 0.7960
    Epoch 124/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4032 - accuracy: 0.8764 - f1_m: 0.8612 - precision_m: 0.9132 - recall_m: 0.8160
    Epoch 125/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4052 - accuracy: 0.8618 - f1_m: 0.8478 - precision_m: 0.9005 - recall_m: 0.8018
    Epoch 126/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4040 - accuracy: 0.8545 - f1_m: 0.8463 - precision_m: 0.8978 - recall_m: 0.8009
    Epoch 127/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4094 - accuracy: 0.8627 - f1_m: 0.8541 - precision_m: 0.9026 - recall_m: 0.8108
    Epoch 128/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4370 - accuracy: 0.8436 - f1_m: 0.8442 - precision_m: 0.8887 - recall_m: 0.8050
    Epoch 129/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4433 - accuracy: 0.8400 - f1_m: 0.8446 - precision_m: 0.8918 - recall_m: 0.8030
    Epoch 130/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4589 - accuracy: 0.8500 - f1_m: 0.8367 - precision_m: 0.8996 - recall_m: 0.7830
    Epoch 131/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4224 - accuracy: 0.8445 - f1_m: 0.8370 - precision_m: 0.8852 - recall_m: 0.7943
    Epoch 132/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4246 - accuracy: 0.8527 - f1_m: 0.8496 - precision_m: 0.8997 - recall_m: 0.8058
    Epoch 133/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4933 - accuracy: 0.8255 - f1_m: 0.8134 - precision_m: 0.8732 - recall_m: 0.7622
    Epoch 134/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4596 - accuracy: 0.8409 - f1_m: 0.8360 - precision_m: 0.8878 - recall_m: 0.7911
    Epoch 135/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3632 - accuracy: 0.8836 - f1_m: 0.8804 - precision_m: 0.9263 - recall_m: 0.8397
    Epoch 136/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3690 - accuracy: 0.8727 - f1_m: 0.8651 - precision_m: 0.9123 - recall_m: 0.8238
    Epoch 137/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4135 - accuracy: 0.8518 - f1_m: 0.8430 - precision_m: 0.9047 - recall_m: 0.7928
    Epoch 138/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4373 - accuracy: 0.8391 - f1_m: 0.8415 - precision_m: 0.8933 - recall_m: 0.7963
    Epoch 139/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4009 - accuracy: 0.8709 - f1_m: 0.8565 - precision_m: 0.8980 - recall_m: 0.8192
    Epoch 140/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3879 - accuracy: 0.8691 - f1_m: 0.8682 - precision_m: 0.9159 - recall_m: 0.8261
    Epoch 141/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4402 - accuracy: 0.8482 - f1_m: 0.8424 - precision_m: 0.8913 - recall_m: 0.7995
    Epoch 142/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4325 - accuracy: 0.8582 - f1_m: 0.8517 - precision_m: 0.9002 - recall_m: 0.8090
    Epoch 143/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3826 - accuracy: 0.8591 - f1_m: 0.8677 - precision_m: 0.9123 - recall_m: 0.8281
    Epoch 144/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3644 - accuracy: 0.8800 - f1_m: 0.8812 - precision_m: 0.9186 - recall_m: 0.8472
    Epoch 145/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3308 - accuracy: 0.8827 - f1_m: 0.8817 - precision_m: 0.9206 - recall_m: 0.8466
    Epoch 146/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3914 - accuracy: 0.8645 - f1_m: 0.8602 - precision_m: 0.9157 - recall_m: 0.8116
    Epoch 147/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3906 - accuracy: 0.8645 - f1_m: 0.8531 - precision_m: 0.9033 - recall_m: 0.8087
    Epoch 148/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4082 - accuracy: 0.8627 - f1_m: 0.8621 - precision_m: 0.8974 - recall_m: 0.8299
    Epoch 149/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3909 - accuracy: 0.8564 - f1_m: 0.8568 - precision_m: 0.8977 - recall_m: 0.8200
    Epoch 150/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4081 - accuracy: 0.8409 - f1_m: 0.8342 - precision_m: 0.8875 - recall_m: 0.7894
    9/9 [==============================] - 0s 6ms/step - loss: 0.7451 - accuracy: 0.7541 - f1_m: 0.7571 - precision_m: 0.8008 - recall_m: 0.7184
    Epoch 1/150
    26/26 [==============================] - 1s 11ms/step - loss: 2.5574 - accuracy: 0.1061 - f1_m: 0.0024 - precision_m: 0.0769 - recall_m: 0.0012
    Epoch 2/150
    26/26 [==============================] - 0s 10ms/step - loss: 2.4000 - accuracy: 0.1231 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 3/150
    26/26 [==============================] - 0s 9ms/step - loss: 2.2963 - accuracy: 0.1571 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 4/150
    26/26 [==============================] - 0s 9ms/step - loss: 2.2228 - accuracy: 0.1965 - f1_m: 0.0000e+00 - precision_m: 0.0000e+00 - recall_m: 0.0000e+00
    Epoch 5/150
    26/26 [==============================] - 0s 10ms/step - loss: 2.1500 - accuracy: 0.2250 - f1_m: 0.0012 - precision_m: 0.0385 - recall_m: 6.0096e-04
    Epoch 6/150
    26/26 [==============================] - 0s 9ms/step - loss: 2.0403 - accuracy: 0.2650 - f1_m: 0.0175 - precision_m: 0.3718 - recall_m: 0.0090
    Epoch 7/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.9466 - accuracy: 0.3050 - f1_m: 0.0459 - precision_m: 0.7404 - recall_m: 0.0240
    Epoch 8/150
    26/26 [==============================] - 0s 9ms/step - loss: 1.8263 - accuracy: 0.3560 - f1_m: 0.0788 - precision_m: 0.8777 - recall_m: 0.0420
    Epoch 9/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.7112 - accuracy: 0.3948 - f1_m: 0.1347 - precision_m: 0.7985 - recall_m: 0.0751
    Epoch 10/150
    26/26 [==============================] - 0s 9ms/step - loss: 1.5975 - accuracy: 0.4354 - f1_m: 0.2008 - precision_m: 0.8364 - recall_m: 0.1161
    Epoch 11/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.5242 - accuracy: 0.4548 - f1_m: 0.2658 - precision_m: 0.8203 - recall_m: 0.1606
    Epoch 12/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.4673 - accuracy: 0.4663 - f1_m: 0.3030 - precision_m: 0.8203 - recall_m: 0.1875
    Epoch 13/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.3526 - accuracy: 0.5428 - f1_m: 0.3703 - precision_m: 0.8481 - recall_m: 0.2388
    Epoch 14/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.3714 - accuracy: 0.5233 - f1_m: 0.3898 - precision_m: 0.8191 - recall_m: 0.2600
    Epoch 15/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.2275 - accuracy: 0.5846 - f1_m: 0.4446 - precision_m: 0.8357 - recall_m: 0.3060
    Epoch 16/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.1720 - accuracy: 0.5973 - f1_m: 0.4944 - precision_m: 0.8112 - recall_m: 0.3573
    Epoch 17/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.1499 - accuracy: 0.5949 - f1_m: 0.5110 - precision_m: 0.8234 - recall_m: 0.3725
    Epoch 18/150
    26/26 [==============================] - 0s 9ms/step - loss: 1.1440 - accuracy: 0.6010 - f1_m: 0.5283 - precision_m: 0.8256 - recall_m: 0.3902
    Epoch 19/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.1046 - accuracy: 0.6149 - f1_m: 0.5572 - precision_m: 0.8073 - recall_m: 0.4269
    Epoch 20/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.0445 - accuracy: 0.6440 - f1_m: 0.5843 - precision_m: 0.8096 - recall_m: 0.4586
    Epoch 21/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.0376 - accuracy: 0.6416 - f1_m: 0.5794 - precision_m: 0.8008 - recall_m: 0.4559
    Epoch 22/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.9986 - accuracy: 0.6586 - f1_m: 0.6213 - precision_m: 0.8270 - recall_m: 0.5003
    Epoch 23/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.9616 - accuracy: 0.6834 - f1_m: 0.6294 - precision_m: 0.8236 - recall_m: 0.5119
    Epoch 24/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8968 - accuracy: 0.7083 - f1_m: 0.6653 - precision_m: 0.8487 - recall_m: 0.5483
    Epoch 25/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.8967 - accuracy: 0.7071 - f1_m: 0.6653 - precision_m: 0.8436 - recall_m: 0.5506
    Epoch 26/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.8907 - accuracy: 0.6998 - f1_m: 0.6736 - precision_m: 0.8431 - recall_m: 0.5629
    Epoch 27/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.9559 - accuracy: 0.6762 - f1_m: 0.6394 - precision_m: 0.7951 - recall_m: 0.5364
    Epoch 28/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8635 - accuracy: 0.7186 - f1_m: 0.6890 - precision_m: 0.8446 - recall_m: 0.5829
    Epoch 29/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8052 - accuracy: 0.7295 - f1_m: 0.7122 - precision_m: 0.8516 - recall_m: 0.6139
    Epoch 30/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.8380 - accuracy: 0.7229 - f1_m: 0.7034 - precision_m: 0.8303 - recall_m: 0.6112
    Epoch 31/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8099 - accuracy: 0.7283 - f1_m: 0.7086 - precision_m: 0.8328 - recall_m: 0.6181
    Epoch 32/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.7856 - accuracy: 0.7417 - f1_m: 0.7085 - precision_m: 0.8328 - recall_m: 0.6179
    Epoch 33/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.7821 - accuracy: 0.7429 - f1_m: 0.7166 - precision_m: 0.8486 - recall_m: 0.6226
    Epoch 34/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7634 - accuracy: 0.7441 - f1_m: 0.7236 - precision_m: 0.8478 - recall_m: 0.6323
    Epoch 35/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7093 - accuracy: 0.7720 - f1_m: 0.7485 - precision_m: 0.8618 - recall_m: 0.6630
    Epoch 36/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7791 - accuracy: 0.7465 - f1_m: 0.7209 - precision_m: 0.8335 - recall_m: 0.6367
    Epoch 37/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8041 - accuracy: 0.7247 - f1_m: 0.7081 - precision_m: 0.8218 - recall_m: 0.6231
    Epoch 38/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7072 - accuracy: 0.7611 - f1_m: 0.7556 - precision_m: 0.8656 - recall_m: 0.6714
    Epoch 39/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7242 - accuracy: 0.7495 - f1_m: 0.7377 - precision_m: 0.8538 - recall_m: 0.6509
    Epoch 40/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.6702 - accuracy: 0.7738 - f1_m: 0.7581 - precision_m: 0.8471 - recall_m: 0.6871
    Epoch 41/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6586 - accuracy: 0.7896 - f1_m: 0.7779 - precision_m: 0.8614 - recall_m: 0.7103
    Epoch 42/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6267 - accuracy: 0.7938 - f1_m: 0.7812 - precision_m: 0.8732 - recall_m: 0.7078
    Epoch 43/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.6548 - accuracy: 0.7823 - f1_m: 0.7727 - precision_m: 0.8629 - recall_m: 0.7010
    Epoch 44/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6300 - accuracy: 0.7823 - f1_m: 0.7698 - precision_m: 0.8602 - recall_m: 0.6979
    Epoch 45/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7180 - accuracy: 0.7489 - f1_m: 0.7481 - precision_m: 0.8362 - recall_m: 0.6776
    Epoch 46/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.6774 - accuracy: 0.7708 - f1_m: 0.7583 - precision_m: 0.8414 - recall_m: 0.6909
    Epoch 47/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6594 - accuracy: 0.7732 - f1_m: 0.7637 - precision_m: 0.8531 - recall_m: 0.6926
    Epoch 48/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.7125 - accuracy: 0.7617 - f1_m: 0.7516 - precision_m: 0.8540 - recall_m: 0.6727
    Epoch 49/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6127 - accuracy: 0.7908 - f1_m: 0.7824 - precision_m: 0.8670 - recall_m: 0.7140
    Epoch 50/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6181 - accuracy: 0.7871 - f1_m: 0.7701 - precision_m: 0.8641 - recall_m: 0.6960
    Epoch 51/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6260 - accuracy: 0.7968 - f1_m: 0.7832 - precision_m: 0.8612 - recall_m: 0.7197
    Epoch 52/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.6130 - accuracy: 0.7932 - f1_m: 0.7823 - precision_m: 0.8726 - recall_m: 0.7104
    Epoch 53/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5924 - accuracy: 0.7999 - f1_m: 0.7938 - precision_m: 0.8723 - recall_m: 0.7290
    Epoch 54/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6153 - accuracy: 0.7847 - f1_m: 0.7786 - precision_m: 0.8623 - recall_m: 0.7103
    Epoch 55/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.6037 - accuracy: 0.7890 - f1_m: 0.7826 - precision_m: 0.8624 - recall_m: 0.7173
    Epoch 56/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5583 - accuracy: 0.8114 - f1_m: 0.8051 - precision_m: 0.8806 - recall_m: 0.7424
    Epoch 57/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5465 - accuracy: 0.8150 - f1_m: 0.8154 - precision_m: 0.8897 - recall_m: 0.7536
    Epoch 58/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5659 - accuracy: 0.8108 - f1_m: 0.7983 - precision_m: 0.8741 - recall_m: 0.7356
    Epoch 59/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5433 - accuracy: 0.8199 - f1_m: 0.8081 - precision_m: 0.8769 - recall_m: 0.7500
    Epoch 60/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5622 - accuracy: 0.8029 - f1_m: 0.8050 - precision_m: 0.8698 - recall_m: 0.7499
    Epoch 61/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.6217 - accuracy: 0.7975 - f1_m: 0.7913 - precision_m: 0.8591 - recall_m: 0.7340
    Epoch 62/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5651 - accuracy: 0.8217 - f1_m: 0.8092 - precision_m: 0.8786 - recall_m: 0.7510
    Epoch 63/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5087 - accuracy: 0.8369 - f1_m: 0.8268 - precision_m: 0.8922 - recall_m: 0.7712
    Epoch 64/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5052 - accuracy: 0.8260 - f1_m: 0.8243 - precision_m: 0.8932 - recall_m: 0.7661
    Epoch 65/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5319 - accuracy: 0.8144 - f1_m: 0.8174 - precision_m: 0.8852 - recall_m: 0.7598
    Epoch 66/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5093 - accuracy: 0.8326 - f1_m: 0.8195 - precision_m: 0.8897 - recall_m: 0.7602
    Epoch 67/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5447 - accuracy: 0.8138 - f1_m: 0.8129 - precision_m: 0.8814 - recall_m: 0.7554
    Epoch 68/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5625 - accuracy: 0.8144 - f1_m: 0.8124 - precision_m: 0.8749 - recall_m: 0.7588
    Epoch 69/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5421 - accuracy: 0.8235 - f1_m: 0.8139 - precision_m: 0.8836 - recall_m: 0.7552
    Epoch 70/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5955 - accuracy: 0.7835 - f1_m: 0.7837 - precision_m: 0.8702 - recall_m: 0.7138
    Epoch 71/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5576 - accuracy: 0.8072 - f1_m: 0.8024 - precision_m: 0.8725 - recall_m: 0.7446
    Epoch 72/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5208 - accuracy: 0.8175 - f1_m: 0.8178 - precision_m: 0.8875 - recall_m: 0.7590
    Epoch 73/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4942 - accuracy: 0.8272 - f1_m: 0.8245 - precision_m: 0.8900 - recall_m: 0.7687
    Epoch 74/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4789 - accuracy: 0.8387 - f1_m: 0.8329 - precision_m: 0.8929 - recall_m: 0.7813
    Epoch 75/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4653 - accuracy: 0.8357 - f1_m: 0.8353 - precision_m: 0.8907 - recall_m: 0.7872
    Epoch 76/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4829 - accuracy: 0.8344 - f1_m: 0.8310 - precision_m: 0.8860 - recall_m: 0.7835
    Epoch 77/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4663 - accuracy: 0.8363 - f1_m: 0.8388 - precision_m: 0.8968 - recall_m: 0.7893
    Epoch 78/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5271 - accuracy: 0.8084 - f1_m: 0.8159 - precision_m: 0.8763 - recall_m: 0.7643
    Epoch 79/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4993 - accuracy: 0.8199 - f1_m: 0.8224 - precision_m: 0.8796 - recall_m: 0.7728
    Epoch 80/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4937 - accuracy: 0.8351 - f1_m: 0.8248 - precision_m: 0.8937 - recall_m: 0.7671
    Epoch 81/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4856 - accuracy: 0.8369 - f1_m: 0.8329 - precision_m: 0.8905 - recall_m: 0.7828
    Epoch 82/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5009 - accuracy: 0.8308 - f1_m: 0.8309 - precision_m: 0.8948 - recall_m: 0.7769
    Epoch 83/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5016 - accuracy: 0.8344 - f1_m: 0.8194 - precision_m: 0.8805 - recall_m: 0.7672
    Epoch 84/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4862 - accuracy: 0.8308 - f1_m: 0.8194 - precision_m: 0.8794 - recall_m: 0.7676
    Epoch 85/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4956 - accuracy: 0.8150 - f1_m: 0.8186 - precision_m: 0.8733 - recall_m: 0.7710
    Epoch 86/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4771 - accuracy: 0.8417 - f1_m: 0.8390 - precision_m: 0.8916 - recall_m: 0.7931
    Epoch 87/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4598 - accuracy: 0.8441 - f1_m: 0.8455 - precision_m: 0.8984 - recall_m: 0.7992
    Epoch 88/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4857 - accuracy: 0.8332 - f1_m: 0.8296 - precision_m: 0.8836 - recall_m: 0.7832
    Epoch 89/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4887 - accuracy: 0.8266 - f1_m: 0.8322 - precision_m: 0.8861 - recall_m: 0.7852
    Epoch 90/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4674 - accuracy: 0.8454 - f1_m: 0.8341 - precision_m: 0.8883 - recall_m: 0.7875
    Epoch 91/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4440 - accuracy: 0.8563 - f1_m: 0.8495 - precision_m: 0.9020 - recall_m: 0.8038
    Epoch 92/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4307 - accuracy: 0.8532 - f1_m: 0.8449 - precision_m: 0.9025 - recall_m: 0.7950
    Epoch 93/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4495 - accuracy: 0.8454 - f1_m: 0.8407 - precision_m: 0.8909 - recall_m: 0.7968
    Epoch 94/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4675 - accuracy: 0.8344 - f1_m: 0.8335 - precision_m: 0.8882 - recall_m: 0.7860
    Epoch 95/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4293 - accuracy: 0.8593 - f1_m: 0.8534 - precision_m: 0.9005 - recall_m: 0.8114
    Epoch 96/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4653 - accuracy: 0.8381 - f1_m: 0.8386 - precision_m: 0.8962 - recall_m: 0.7887
    Epoch 97/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4391 - accuracy: 0.8435 - f1_m: 0.8307 - precision_m: 0.8863 - recall_m: 0.7825
    Epoch 98/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4714 - accuracy: 0.8381 - f1_m: 0.8281 - precision_m: 0.8811 - recall_m: 0.7814
    Epoch 99/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4248 - accuracy: 0.8563 - f1_m: 0.8503 - precision_m: 0.8971 - recall_m: 0.8090
    Epoch 100/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4740 - accuracy: 0.8417 - f1_m: 0.8391 - precision_m: 0.8875 - recall_m: 0.7965
    Epoch 101/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4888 - accuracy: 0.8260 - f1_m: 0.8219 - precision_m: 0.8782 - recall_m: 0.7728
    Epoch 102/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4199 - accuracy: 0.8526 - f1_m: 0.8553 - precision_m: 0.9070 - recall_m: 0.8098
    Epoch 103/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4284 - accuracy: 0.8545 - f1_m: 0.8522 - precision_m: 0.9026 - recall_m: 0.8078
    Epoch 104/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4334 - accuracy: 0.8557 - f1_m: 0.8542 - precision_m: 0.8950 - recall_m: 0.8177
    Epoch 105/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4499 - accuracy: 0.8441 - f1_m: 0.8405 - precision_m: 0.8943 - recall_m: 0.7940
    Epoch 106/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4233 - accuracy: 0.8545 - f1_m: 0.8479 - precision_m: 0.9000 - recall_m: 0.8024
    Epoch 107/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4180 - accuracy: 0.8611 - f1_m: 0.8561 - precision_m: 0.8982 - recall_m: 0.8186
    Epoch 108/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4218 - accuracy: 0.8539 - f1_m: 0.8493 - precision_m: 0.8956 - recall_m: 0.8084
    Epoch 109/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4208 - accuracy: 0.8520 - f1_m: 0.8506 - precision_m: 0.8965 - recall_m: 0.8097
    Epoch 110/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4216 - accuracy: 0.8587 - f1_m: 0.8576 - precision_m: 0.9029 - recall_m: 0.8178
    Epoch 111/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4060 - accuracy: 0.8539 - f1_m: 0.8539 - precision_m: 0.9022 - recall_m: 0.8110
    Epoch 112/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.5105 - accuracy: 0.8187 - f1_m: 0.8178 - precision_m: 0.8708 - recall_m: 0.7720
    Epoch 113/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4379 - accuracy: 0.8545 - f1_m: 0.8439 - precision_m: 0.8930 - recall_m: 0.8006
    Epoch 114/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4396 - accuracy: 0.8411 - f1_m: 0.8481 - precision_m: 0.8943 - recall_m: 0.8071
    Epoch 115/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4302 - accuracy: 0.8478 - f1_m: 0.8473 - precision_m: 0.8976 - recall_m: 0.8028
    Epoch 116/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4338 - accuracy: 0.8508 - f1_m: 0.8485 - precision_m: 0.9021 - recall_m: 0.8021
    Epoch 117/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4345 - accuracy: 0.8563 - f1_m: 0.8518 - precision_m: 0.9057 - recall_m: 0.8048
    Epoch 118/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4459 - accuracy: 0.8284 - f1_m: 0.8342 - precision_m: 0.8890 - recall_m: 0.7870
    Epoch 119/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4013 - accuracy: 0.8569 - f1_m: 0.8553 - precision_m: 0.9028 - recall_m: 0.8132
    Epoch 120/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4143 - accuracy: 0.8520 - f1_m: 0.8545 - precision_m: 0.8935 - recall_m: 0.8196
    Epoch 121/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4562 - accuracy: 0.8320 - f1_m: 0.8297 - precision_m: 0.8848 - recall_m: 0.7819
    Epoch 122/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4211 - accuracy: 0.8532 - f1_m: 0.8488 - precision_m: 0.9007 - recall_m: 0.8034
    Epoch 123/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4543 - accuracy: 0.8429 - f1_m: 0.8343 - precision_m: 0.8856 - recall_m: 0.7896
    Epoch 124/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4145 - accuracy: 0.8617 - f1_m: 0.8503 - precision_m: 0.8984 - recall_m: 0.8077
    Epoch 125/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4225 - accuracy: 0.8557 - f1_m: 0.8520 - precision_m: 0.8985 - recall_m: 0.8107
    Epoch 126/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3879 - accuracy: 0.8672 - f1_m: 0.8619 - precision_m: 0.9067 - recall_m: 0.8219
    Epoch 127/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4249 - accuracy: 0.8496 - f1_m: 0.8518 - precision_m: 0.8909 - recall_m: 0.8166
    Epoch 128/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4216 - accuracy: 0.8526 - f1_m: 0.8549 - precision_m: 0.9023 - recall_m: 0.8128
    Epoch 129/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4146 - accuracy: 0.8478 - f1_m: 0.8406 - precision_m: 0.8882 - recall_m: 0.7985
    Epoch 130/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3763 - accuracy: 0.8720 - f1_m: 0.8607 - precision_m: 0.9026 - recall_m: 0.8231
    Epoch 131/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.3885 - accuracy: 0.8569 - f1_m: 0.8537 - precision_m: 0.8914 - recall_m: 0.8197
    Epoch 132/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4470 - accuracy: 0.8484 - f1_m: 0.8421 - precision_m: 0.8951 - recall_m: 0.7963
    Epoch 133/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4640 - accuracy: 0.8253 - f1_m: 0.8234 - precision_m: 0.8854 - recall_m: 0.7706
    Epoch 134/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4417 - accuracy: 0.8460 - f1_m: 0.8413 - precision_m: 0.9013 - recall_m: 0.7896
    Epoch 135/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4381 - accuracy: 0.8369 - f1_m: 0.8346 - precision_m: 0.8834 - recall_m: 0.7915
    Epoch 136/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4004 - accuracy: 0.8684 - f1_m: 0.8582 - precision_m: 0.9127 - recall_m: 0.8109
    Epoch 137/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3747 - accuracy: 0.8733 - f1_m: 0.8691 - precision_m: 0.9115 - recall_m: 0.8313
    Epoch 138/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4367 - accuracy: 0.8454 - f1_m: 0.8411 - precision_m: 0.8852 - recall_m: 0.8022
    Epoch 139/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4264 - accuracy: 0.8460 - f1_m: 0.8421 - precision_m: 0.8912 - recall_m: 0.7987
    Epoch 140/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4784 - accuracy: 0.8320 - f1_m: 0.8256 - precision_m: 0.8771 - recall_m: 0.7802
    Epoch 141/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.5025 - accuracy: 0.8302 - f1_m: 0.8239 - precision_m: 0.8778 - recall_m: 0.7767
    Epoch 142/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4461 - accuracy: 0.8332 - f1_m: 0.8306 - precision_m: 0.8835 - recall_m: 0.7845
    Epoch 143/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.4371 - accuracy: 0.8484 - f1_m: 0.8437 - precision_m: 0.9036 - recall_m: 0.7921
    Epoch 144/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3999 - accuracy: 0.8648 - f1_m: 0.8628 - precision_m: 0.9090 - recall_m: 0.8217
    Epoch 145/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4758 - accuracy: 0.8284 - f1_m: 0.8311 - precision_m: 0.8818 - recall_m: 0.7873
    Epoch 146/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.4234 - accuracy: 0.8526 - f1_m: 0.8476 - precision_m: 0.9007 - recall_m: 0.8012
    Epoch 147/150
    26/26 [==============================] - 0s 9ms/step - loss: 0.3984 - accuracy: 0.8611 - f1_m: 0.8557 - precision_m: 0.9043 - recall_m: 0.8129
    Epoch 148/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3934 - accuracy: 0.8617 - f1_m: 0.8588 - precision_m: 0.9060 - recall_m: 0.8175
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
    Epoch 14/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.7161 - accuracy: 0.3731 - f1_m: 0.2907 - precision_m: 0.1725 - recall_m: 0.9246
    Epoch 15/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8398 - accuracy: 0.2994 - f1_m: 0.2778 - precision_m: 0.1651 - recall_m: 0.8759
    Epoch 16/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.7153 - accuracy: 0.3667 - f1_m: 0.2984 - precision_m: 0.1789 - recall_m: 0.9003
    Epoch 17/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.5959 - accuracy: 0.4204 - f1_m: 0.3103 - precision_m: 0.1864 - recall_m: 0.9264
    Epoch 18/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.5433 - accuracy: 0.4204 - f1_m: 0.3172 - precision_m: 0.1917 - recall_m: 0.9213
    Epoch 19/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.5037 - accuracy: 0.4286 - f1_m: 0.3223 - precision_m: 0.1945 - recall_m: 0.9401
    Epoch 20/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4170 - accuracy: 0.4677 - f1_m: 0.3229 - precision_m: 0.1950 - recall_m: 0.9394
    Epoch 21/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.5617 - accuracy: 0.4058 - f1_m: 0.3097 - precision_m: 0.1869 - recall_m: 0.9040
    Epoch 22/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.6078 - accuracy: 0.4013 - f1_m: 0.3107 - precision_m: 0.1874 - recall_m: 0.9106
    Epoch 23/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4890 - accuracy: 0.4468 - f1_m: 0.3170 - precision_m: 0.1906 - recall_m: 0.9410
    Epoch 24/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4322 - accuracy: 0.4604 - f1_m: 0.3217 - precision_m: 0.1937 - recall_m: 0.9479
    Epoch 25/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3653 - accuracy: 0.4823 - f1_m: 0.3245 - precision_m: 0.1959 - recall_m: 0.9455
    Epoch 26/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4193 - accuracy: 0.4631 - f1_m: 0.3311 - precision_m: 0.2010 - recall_m: 0.9401
    Epoch 27/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3327 - accuracy: 0.4823 - f1_m: 0.3296 - precision_m: 0.1991 - recall_m: 0.9566
    Epoch 28/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.2965 - accuracy: 0.5032 - f1_m: 0.3392 - precision_m: 0.2062 - recall_m: 0.9575
    Epoch 29/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.2445 - accuracy: 0.5350 - f1_m: 0.3441 - precision_m: 0.2099 - recall_m: 0.9542
    Epoch 30/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1773 - accuracy: 0.5696 - f1_m: 0.3480 - precision_m: 0.2123 - recall_m: 0.9635
    Epoch 31/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1508 - accuracy: 0.5614 - f1_m: 0.3446 - precision_m: 0.2096 - recall_m: 0.9688
    Epoch 32/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1420 - accuracy: 0.5596 - f1_m: 0.3500 - precision_m: 0.2141 - recall_m: 0.9594
    Epoch 33/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0804 - accuracy: 0.6078 - f1_m: 0.3501 - precision_m: 0.2137 - recall_m: 0.9696
    Epoch 34/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1164 - accuracy: 0.5569 - f1_m: 0.3524 - precision_m: 0.2155 - recall_m: 0.9663
    Epoch 35/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.0183 - accuracy: 0.6251 - f1_m: 0.3536 - precision_m: 0.2163 - recall_m: 0.9705
    Epoch 36/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1490 - accuracy: 0.5669 - f1_m: 0.3568 - precision_m: 0.2188 - recall_m: 0.9679
    Epoch 37/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0793 - accuracy: 0.6124 - f1_m: 0.3586 - precision_m: 0.2197 - recall_m: 0.9757
    Epoch 38/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9607 - accuracy: 0.6561 - f1_m: 0.3597 - precision_m: 0.2206 - recall_m: 0.9731
    Epoch 39/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8803 - accuracy: 0.6806 - f1_m: 0.3723 - precision_m: 0.2301 - recall_m: 0.9750
    Epoch 40/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8973 - accuracy: 0.6733 - f1_m: 0.3648 - precision_m: 0.2242 - recall_m: 0.9792
    Epoch 41/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8859 - accuracy: 0.6606 - f1_m: 0.3750 - precision_m: 0.2321 - recall_m: 0.9766
    Epoch 42/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8586 - accuracy: 0.6988 - f1_m: 0.3737 - precision_m: 0.2315 - recall_m: 0.9724
    Epoch 43/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9314 - accuracy: 0.6533 - f1_m: 0.3825 - precision_m: 0.2378 - recall_m: 0.9783
    Epoch 44/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8855 - accuracy: 0.6606 - f1_m: 0.3760 - precision_m: 0.2329 - recall_m: 0.9759
    Epoch 45/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.9216 - accuracy: 0.6706 - f1_m: 0.3737 - precision_m: 0.2318 - recall_m: 0.9654
    Epoch 46/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8518 - accuracy: 0.6906 - f1_m: 0.3782 - precision_m: 0.2344 - recall_m: 0.9800
    Epoch 47/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7465 - accuracy: 0.7452 - f1_m: 0.3758 - precision_m: 0.2327 - recall_m: 0.9767
    Epoch 48/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7163 - accuracy: 0.7389 - f1_m: 0.3815 - precision_m: 0.2368 - recall_m: 0.9819
    Epoch 49/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.7299 - accuracy: 0.7334 - f1_m: 0.3871 - precision_m: 0.2405 - recall_m: 0.9931
    Epoch 50/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6863 - accuracy: 0.7525 - f1_m: 0.3899 - precision_m: 0.2432 - recall_m: 0.9828
    Epoch 51/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.0031 - accuracy: 0.6269 - f1_m: 0.3746 - precision_m: 0.2326 - recall_m: 0.9635
    Epoch 52/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6964 - accuracy: 0.7525 - f1_m: 0.3851 - precision_m: 0.2394 - recall_m: 0.9863
    Epoch 53/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6180 - accuracy: 0.7807 - f1_m: 0.3920 - precision_m: 0.2445 - recall_m: 0.9896
    Epoch 54/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6314 - accuracy: 0.7789 - f1_m: 0.3954 - precision_m: 0.2472 - recall_m: 0.9878
    Epoch 55/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7369 - accuracy: 0.7361 - f1_m: 0.4020 - precision_m: 0.2527 - recall_m: 0.9844
    Epoch 56/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.6329 - accuracy: 0.7771 - f1_m: 0.3996 - precision_m: 0.2504 - recall_m: 0.9905
    Epoch 57/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5865 - accuracy: 0.7925 - f1_m: 0.3964 - precision_m: 0.2480 - recall_m: 0.9878
    Epoch 58/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5384 - accuracy: 0.8062 - f1_m: 0.4032 - precision_m: 0.2529 - recall_m: 0.9948
    Epoch 59/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4881 - accuracy: 0.8308 - f1_m: 0.4019 - precision_m: 0.2520 - recall_m: 0.9922
    Epoch 60/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5658 - accuracy: 0.7862 - f1_m: 0.4027 - precision_m: 0.2528 - recall_m: 0.9905
    Epoch 61/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4633 - accuracy: 0.8408 - f1_m: 0.4083 - precision_m: 0.2568 - recall_m: 0.9965
    Epoch 62/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5669 - accuracy: 0.7925 - f1_m: 0.4015 - precision_m: 0.2517 - recall_m: 0.9922
    Epoch 63/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5084 - accuracy: 0.8135 - f1_m: 0.4065 - precision_m: 0.2555 - recall_m: 0.9957
    Epoch 64/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4224 - accuracy: 0.8562 - f1_m: 0.4065 - precision_m: 0.2555 - recall_m: 0.9948
    Epoch 65/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3817 - accuracy: 0.8653 - f1_m: 0.4059 - precision_m: 0.2549 - recall_m: 0.9965
    Epoch 66/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3510 - accuracy: 0.8799 - f1_m: 0.4083 - precision_m: 0.2568 - recall_m: 0.9965
    Epoch 67/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3426 - accuracy: 0.8726 - f1_m: 0.4080 - precision_m: 0.2565 - recall_m: 0.9983
    Epoch 68/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3297 - accuracy: 0.8881 - f1_m: 0.4064 - precision_m: 0.2554 - recall_m: 0.9957
    Epoch 69/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3580 - accuracy: 0.8872 - f1_m: 0.4093 - precision_m: 0.2576 - recall_m: 0.9957
    Epoch 70/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.2902 - accuracy: 0.8972 - f1_m: 0.4078 - precision_m: 0.2563 - recall_m: 0.9983
    Epoch 71/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.2939 - accuracy: 0.9026 - f1_m: 0.4116 - precision_m: 0.2594 - recall_m: 0.9965
    Epoch 72/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3241 - accuracy: 0.8899 - f1_m: 0.4162 - precision_m: 0.2632 - recall_m: 0.9948
    Epoch 73/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2959 - accuracy: 0.8917 - f1_m: 0.4112 - precision_m: 0.2589 - recall_m: 0.9991
    Epoch 74/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.2114 - accuracy: 0.9381 - f1_m: 0.4137 - precision_m: 0.2609 - recall_m: 0.9991
    Epoch 75/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2607 - accuracy: 0.9117 - f1_m: 0.4161 - precision_m: 0.2629 - recall_m: 0.9974
    Epoch 76/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3144 - accuracy: 0.8926 - f1_m: 0.4170 - precision_m: 0.2639 - recall_m: 0.9948
    Epoch 77/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3250 - accuracy: 0.8899 - f1_m: 0.4163 - precision_m: 0.2631 - recall_m: 0.9974
    Epoch 78/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2797 - accuracy: 0.9163 - f1_m: 0.4143 - precision_m: 0.2615 - recall_m: 0.9974
    Epoch 79/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1930 - accuracy: 0.9445 - f1_m: 0.4140 - precision_m: 0.2612 - recall_m: 0.9983
    Epoch 80/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.0647 - accuracy: 0.6397 - f1_m: 0.3972 - precision_m: 0.2502 - recall_m: 0.9635
    Epoch 81/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8685 - accuracy: 0.6624 - f1_m: 0.3990 - precision_m: 0.2514 - recall_m: 0.9679
    Epoch 82/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5580 - accuracy: 0.7989 - f1_m: 0.4152 - precision_m: 0.2627 - recall_m: 0.9905
    Epoch 83/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3357 - accuracy: 0.8926 - f1_m: 0.4190 - precision_m: 0.2653 - recall_m: 0.9965
    Epoch 84/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2788 - accuracy: 0.9136 - f1_m: 0.4215 - precision_m: 0.2674 - recall_m: 0.9957
    Epoch 85/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1934 - accuracy: 0.9427 - f1_m: 0.4233 - precision_m: 0.2689 - recall_m: 0.9965
    Epoch 86/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1685 - accuracy: 0.9436 - f1_m: 0.4246 - precision_m: 0.2697 - recall_m: 0.9991
    Epoch 87/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1617 - accuracy: 0.9509 - f1_m: 0.4259 - precision_m: 0.2707 - recall_m: 0.9991
    Epoch 88/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1608 - accuracy: 0.9445 - f1_m: 0.4212 - precision_m: 0.2672 - recall_m: 0.9941
    Epoch 89/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6442 - accuracy: 0.7853 - f1_m: 0.4158 - precision_m: 0.2638 - recall_m: 0.9818
    Epoch 90/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3400 - accuracy: 0.8735 - f1_m: 0.4215 - precision_m: 0.2674 - recall_m: 0.9948
    Epoch 91/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2172 - accuracy: 0.9354 - f1_m: 0.4184 - precision_m: 0.2649 - recall_m: 0.9965
    Epoch 92/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1561 - accuracy: 0.9472 - f1_m: 0.4191 - precision_m: 0.2653 - recall_m: 0.9983
    Epoch 93/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1891 - accuracy: 0.9418 - f1_m: 0.4222 - precision_m: 0.2677 - recall_m: 0.9991
    Epoch 94/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1755 - accuracy: 0.9381 - f1_m: 0.4275 - precision_m: 0.2721 - recall_m: 0.9974
    Epoch 95/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1152 - accuracy: 0.9691 - f1_m: 0.4260 - precision_m: 0.2708 - recall_m: 0.9991
    Epoch 96/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1668 - accuracy: 0.9372 - f1_m: 0.4297 - precision_m: 0.2738 - recall_m: 0.9983
    Epoch 97/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1794 - accuracy: 0.9327 - f1_m: 0.4286 - precision_m: 0.2728 - recall_m: 1.0000
    Epoch 98/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1316 - accuracy: 0.9545 - f1_m: 0.4288 - precision_m: 0.2730 - recall_m: 1.0000
    Epoch 99/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1197 - accuracy: 0.9591 - f1_m: 0.4318 - precision_m: 0.2755 - recall_m: 1.0000
    Epoch 100/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0864 - accuracy: 0.9745 - f1_m: 0.4374 - precision_m: 0.2800 - recall_m: 1.0000
    Epoch 101/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0953 - accuracy: 0.9736 - f1_m: 0.4322 - precision_m: 0.2758 - recall_m: 1.0000
    Epoch 102/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0646 - accuracy: 0.9809 - f1_m: 0.4384 - precision_m: 0.2808 - recall_m: 1.0000
    Epoch 103/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0544 - accuracy: 0.9818 - f1_m: 0.4389 - precision_m: 0.2813 - recall_m: 1.0000
    Epoch 104/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0445 - accuracy: 0.9918 - f1_m: 0.4361 - precision_m: 0.2789 - recall_m: 1.0000
    Epoch 105/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0494 - accuracy: 0.9854 - f1_m: 0.4393 - precision_m: 0.2816 - recall_m: 1.0000
    Epoch 106/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1488 - accuracy: 0.9454 - f1_m: 0.4349 - precision_m: 0.2779 - recall_m: 1.0000
    Epoch 107/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2222 - accuracy: 0.9254 - f1_m: 0.4378 - precision_m: 0.2805 - recall_m: 0.9983
    Epoch 108/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2832 - accuracy: 0.9081 - f1_m: 0.4329 - precision_m: 0.2766 - recall_m: 0.9965
    Epoch 109/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.2878 - accuracy: 0.9081 - f1_m: 0.4262 - precision_m: 0.2710 - recall_m: 0.9983
    Epoch 110/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1486 - accuracy: 0.9427 - f1_m: 0.4401 - precision_m: 0.2822 - recall_m: 1.0000
    Epoch 111/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.1179 - accuracy: 0.9600 - f1_m: 0.4351 - precision_m: 0.2782 - recall_m: 0.9991
    Epoch 112/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1853 - accuracy: 0.9363 - f1_m: 0.4322 - precision_m: 0.2759 - recall_m: 0.9983
    Epoch 113/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0958 - accuracy: 0.9709 - f1_m: 0.4414 - precision_m: 0.2833 - recall_m: 1.0000
    Epoch 114/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0671 - accuracy: 0.9818 - f1_m: 0.4330 - precision_m: 0.2764 - recall_m: 1.0000
    Epoch 115/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0537 - accuracy: 0.9854 - f1_m: 0.4360 - precision_m: 0.2788 - recall_m: 1.0000
    Epoch 116/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0352 - accuracy: 0.9891 - f1_m: 0.4344 - precision_m: 0.2775 - recall_m: 1.0000
    Epoch 117/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0258 - accuracy: 0.9945 - f1_m: 0.4371 - precision_m: 0.2797 - recall_m: 1.0000
    Epoch 118/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0146 - accuracy: 0.9991 - f1_m: 0.4392 - precision_m: 0.2816 - recall_m: 1.0000
    Epoch 119/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0085 - accuracy: 1.0000 - f1_m: 0.4378 - precision_m: 0.2804 - recall_m: 1.0000
    Epoch 120/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0065 - accuracy: 1.0000 - f1_m: 0.4409 - precision_m: 0.2828 - recall_m: 1.0000
    Epoch 121/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0048 - accuracy: 1.0000 - f1_m: 0.4414 - precision_m: 0.2833 - recall_m: 1.0000
    Epoch 122/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0039 - accuracy: 1.0000 - f1_m: 0.4444 - precision_m: 0.2859 - recall_m: 1.0000
    Epoch 123/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0035 - accuracy: 1.0000 - f1_m: 0.4432 - precision_m: 0.2847 - recall_m: 1.0000
    Epoch 124/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0031 - accuracy: 1.0000 - f1_m: 0.4421 - precision_m: 0.2839 - recall_m: 1.0000
    Epoch 125/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0030 - accuracy: 1.0000 - f1_m: 0.4436 - precision_m: 0.2850 - recall_m: 1.0000
    Epoch 126/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0027 - accuracy: 1.0000 - f1_m: 0.4451 - precision_m: 0.2864 - recall_m: 1.0000
    Epoch 127/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0024 - accuracy: 1.0000 - f1_m: 0.4473 - precision_m: 0.2883 - recall_m: 1.0000
    Epoch 128/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0023 - accuracy: 1.0000 - f1_m: 0.4454 - precision_m: 0.2866 - recall_m: 1.0000
    Epoch 129/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0021 - accuracy: 1.0000 - f1_m: 0.4475 - precision_m: 0.2883 - recall_m: 1.0000
    Epoch 130/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0020 - accuracy: 1.0000 - f1_m: 0.4448 - precision_m: 0.2861 - recall_m: 1.0000
    Epoch 131/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0019 - accuracy: 1.0000 - f1_m: 0.4446 - precision_m: 0.2859 - recall_m: 1.0000
    Epoch 132/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0018 - accuracy: 1.0000 - f1_m: 0.4450 - precision_m: 0.2862 - recall_m: 1.0000
    Epoch 133/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0016 - accuracy: 1.0000 - f1_m: 0.4458 - precision_m: 0.2869 - recall_m: 1.0000
    Epoch 134/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0016 - accuracy: 1.0000 - f1_m: 0.4471 - precision_m: 0.2880 - recall_m: 1.0000
    Epoch 135/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0015 - accuracy: 1.0000 - f1_m: 0.4474 - precision_m: 0.2882 - recall_m: 1.0000
    Epoch 136/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0014 - accuracy: 1.0000 - f1_m: 0.4471 - precision_m: 0.2880 - recall_m: 1.0000
    Epoch 137/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0013 - accuracy: 1.0000 - f1_m: 0.4497 - precision_m: 0.2902 - recall_m: 1.0000
    Epoch 138/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0013 - accuracy: 1.0000 - f1_m: 0.4489 - precision_m: 0.2896 - recall_m: 1.0000
    Epoch 139/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.0012 - accuracy: 1.0000 - f1_m: 0.4463 - precision_m: 0.2874 - recall_m: 1.0000
    Epoch 140/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0012 - accuracy: 1.0000 - f1_m: 0.4485 - precision_m: 0.2892 - recall_m: 1.0000
    Epoch 141/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0011 - accuracy: 1.0000 - f1_m: 0.4498 - precision_m: 0.2902 - recall_m: 1.0000
    Epoch 142/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0011 - accuracy: 1.0000 - f1_m: 0.4489 - precision_m: 0.2895 - recall_m: 1.0000
    Epoch 143/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0010 - accuracy: 1.0000 - f1_m: 0.4510 - precision_m: 0.2913 - recall_m: 1.0000
    Epoch 144/150
    18/18 [==============================] - 0s 9ms/step - loss: 9.8520e-04 - accuracy: 1.0000 - f1_m: 0.4495 - precision_m: 0.2900 - recall_m: 1.0000
    Epoch 145/150
    18/18 [==============================] - 0s 10ms/step - loss: 9.5382e-04 - accuracy: 1.0000 - f1_m: 0.4486 - precision_m: 0.2893 - recall_m: 1.0000
    Epoch 146/150
    18/18 [==============================] - 0s 10ms/step - loss: 9.2817e-04 - accuracy: 1.0000 - f1_m: 0.4503 - precision_m: 0.2907 - recall_m: 1.0000
    Epoch 147/150
    18/18 [==============================] - 0s 10ms/step - loss: 8.8480e-04 - accuracy: 1.0000 - f1_m: 0.4510 - precision_m: 0.2912 - recall_m: 1.0000
    Epoch 148/150
    18/18 [==============================] - 0s 9ms/step - loss: 8.4455e-04 - accuracy: 1.0000 - f1_m: 0.4496 - precision_m: 0.2901 - recall_m: 1.0000
    Epoch 149/150
    18/18 [==============================] - 0s 9ms/step - loss: 8.1946e-04 - accuracy: 1.0000 - f1_m: 0.4519 - precision_m: 0.2920 - recall_m: 1.0000
    Epoch 150/150
    18/18 [==============================] - 0s 10ms/step - loss: 7.9559e-04 - accuracy: 1.0000 - f1_m: 0.4524 - precision_m: 0.2924 - recall_m: 1.0000
    9/9 [==============================] - 1s 7ms/step - loss: 0.5133 - accuracy: 0.8927 - f1_m: 0.4508 - precision_m: 0.2921 - recall_m: 0.9878
    Epoch 1/150
    18/18 [==============================] - 2s 13ms/step - loss: 2.2718 - accuracy: 0.1365 - f1_m: 0.1780 - precision_m: 0.1126 - recall_m: 0.4315
    Epoch 2/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.2010 - accuracy: 0.1793 - f1_m: 0.2083 - precision_m: 0.1343 - recall_m: 0.4765
    Epoch 3/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.1459 - accuracy: 0.2066 - f1_m: 0.2206 - precision_m: 0.1332 - recall_m: 0.6508
    Epoch 4/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0437 - accuracy: 0.2530 - f1_m: 0.2464 - precision_m: 0.1470 - recall_m: 0.7612
    Epoch 5/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0611 - accuracy: 0.2548 - f1_m: 0.2380 - precision_m: 0.1397 - recall_m: 0.8059
    Epoch 6/150
    18/18 [==============================] - 0s 12ms/step - loss: 1.9491 - accuracy: 0.2693 - f1_m: 0.2577 - precision_m: 0.1526 - recall_m: 0.8296
    Epoch 7/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.8905 - accuracy: 0.3066 - f1_m: 0.2549 - precision_m: 0.1501 - recall_m: 0.8449
    Epoch 8/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.9469 - accuracy: 0.3012 - f1_m: 0.2595 - precision_m: 0.1545 - recall_m: 0.8119
    Epoch 9/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.9575 - accuracy: 0.2803 - f1_m: 0.2656 - precision_m: 0.1613 - recall_m: 0.7626
    Epoch 10/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8251 - accuracy: 0.3185 - f1_m: 0.2832 - precision_m: 0.1690 - recall_m: 0.8743
    Epoch 11/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.8371 - accuracy: 0.3139 - f1_m: 0.2804 - precision_m: 0.1673 - recall_m: 0.8689
    Epoch 12/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.8018 - accuracy: 0.3312 - f1_m: 0.2874 - precision_m: 0.1728 - recall_m: 0.8562
    Epoch 13/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.7227 - accuracy: 0.3449 - f1_m: 0.3005 - precision_m: 0.1813 - recall_m: 0.8814
    Epoch 14/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.6966 - accuracy: 0.3740 - f1_m: 0.2983 - precision_m: 0.1788 - recall_m: 0.9012
    Epoch 15/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.8386 - accuracy: 0.3139 - f1_m: 0.2894 - precision_m: 0.1743 - recall_m: 0.8568
    Epoch 16/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.6309 - accuracy: 0.3667 - f1_m: 0.3070 - precision_m: 0.1842 - recall_m: 0.9219
    Epoch 17/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.6198 - accuracy: 0.3722 - f1_m: 0.3104 - precision_m: 0.1870 - recall_m: 0.9160
    Epoch 18/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.5487 - accuracy: 0.4158 - f1_m: 0.3187 - precision_m: 0.1924 - recall_m: 0.9298
    Epoch 19/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.5145 - accuracy: 0.4204 - f1_m: 0.3173 - precision_m: 0.1908 - recall_m: 0.9427
    Epoch 20/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4596 - accuracy: 0.4459 - f1_m: 0.3315 - precision_m: 0.2010 - recall_m: 0.9470
    Epoch 21/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4625 - accuracy: 0.4449 - f1_m: 0.3302 - precision_m: 0.2003 - recall_m: 0.9394
    Epoch 22/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4504 - accuracy: 0.4295 - f1_m: 0.3289 - precision_m: 0.1999 - recall_m: 0.9281
    Epoch 23/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.4210 - accuracy: 0.4586 - f1_m: 0.3379 - precision_m: 0.2064 - recall_m: 0.9340
    Epoch 24/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.4098 - accuracy: 0.4677 - f1_m: 0.3369 - precision_m: 0.2054 - recall_m: 0.9385
    Epoch 25/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4483 - accuracy: 0.4413 - f1_m: 0.3335 - precision_m: 0.2033 - recall_m: 0.9297
    Epoch 26/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4074 - accuracy: 0.4540 - f1_m: 0.3374 - precision_m: 0.2060 - recall_m: 0.9314
    Epoch 27/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3094 - accuracy: 0.5023 - f1_m: 0.3550 - precision_m: 0.2183 - recall_m: 0.9497
    Epoch 28/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3151 - accuracy: 0.4859 - f1_m: 0.3540 - precision_m: 0.2176 - recall_m: 0.9488
    Epoch 29/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.2455 - accuracy: 0.5287 - f1_m: 0.3526 - precision_m: 0.2169 - recall_m: 0.9430
    Epoch 30/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.2198 - accuracy: 0.5132 - f1_m: 0.3595 - precision_m: 0.2214 - recall_m: 0.9566
    Epoch 31/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1939 - accuracy: 0.5296 - f1_m: 0.3655 - precision_m: 0.2263 - recall_m: 0.9514
    Epoch 32/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1908 - accuracy: 0.5323 - f1_m: 0.3667 - precision_m: 0.2274 - recall_m: 0.9514
    Epoch 33/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1924 - accuracy: 0.5323 - f1_m: 0.3721 - precision_m: 0.2314 - recall_m: 0.9505
    Epoch 34/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1556 - accuracy: 0.5541 - f1_m: 0.3737 - precision_m: 0.2331 - recall_m: 0.9429
    Epoch 35/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1146 - accuracy: 0.5587 - f1_m: 0.3712 - precision_m: 0.2306 - recall_m: 0.9514
    Epoch 36/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.2342 - accuracy: 0.5050 - f1_m: 0.3600 - precision_m: 0.2229 - recall_m: 0.9369
    Epoch 37/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1271 - accuracy: 0.5705 - f1_m: 0.3757 - precision_m: 0.2346 - recall_m: 0.9446
    Epoch 38/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.1744 - accuracy: 0.5396 - f1_m: 0.3707 - precision_m: 0.2309 - recall_m: 0.9436
    Epoch 39/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.0715 - accuracy: 0.5887 - f1_m: 0.3856 - precision_m: 0.2415 - recall_m: 0.9575
    Epoch 40/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9616 - accuracy: 0.6424 - f1_m: 0.3850 - precision_m: 0.2406 - recall_m: 0.9630
    Epoch 41/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.1186 - accuracy: 0.5805 - f1_m: 0.3812 - precision_m: 0.2386 - recall_m: 0.9498
    Epoch 42/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.0450 - accuracy: 0.5905 - f1_m: 0.3900 - precision_m: 0.2450 - recall_m: 0.9575
    Epoch 43/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9646 - accuracy: 0.6197 - f1_m: 0.3961 - precision_m: 0.2493 - recall_m: 0.9653
    Epoch 44/150
    18/18 [==============================] - 0s 9ms/step - loss: 1.0355 - accuracy: 0.6005 - f1_m: 0.3979 - precision_m: 0.2513 - recall_m: 0.9557
    Epoch 45/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9339 - accuracy: 0.6342 - f1_m: 0.3928 - precision_m: 0.2466 - recall_m: 0.9665
    Epoch 46/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8809 - accuracy: 0.6597 - f1_m: 0.4066 - precision_m: 0.2568 - recall_m: 0.9774
    Epoch 47/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8870 - accuracy: 0.6588 - f1_m: 0.4108 - precision_m: 0.2605 - recall_m: 0.9722
    Epoch 48/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8074 - accuracy: 0.6924 - f1_m: 0.4120 - precision_m: 0.2608 - recall_m: 0.9818
    Epoch 49/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9267 - accuracy: 0.6397 - f1_m: 0.4087 - precision_m: 0.2595 - recall_m: 0.9628
    Epoch 50/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.8751 - accuracy: 0.6633 - f1_m: 0.4098 - precision_m: 0.2602 - recall_m: 0.9653
    Epoch 51/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.8426 - accuracy: 0.6879 - f1_m: 0.4141 - precision_m: 0.2632 - recall_m: 0.9722
    Epoch 52/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.7619 - accuracy: 0.6988 - f1_m: 0.4282 - precision_m: 0.2746 - recall_m: 0.9732
    Epoch 53/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7551 - accuracy: 0.7170 - f1_m: 0.4270 - precision_m: 0.2735 - recall_m: 0.9748
    Epoch 54/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7058 - accuracy: 0.7252 - f1_m: 0.4274 - precision_m: 0.2732 - recall_m: 0.9809
    Epoch 55/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6903 - accuracy: 0.7398 - f1_m: 0.4331 - precision_m: 0.2778 - recall_m: 0.9826
    Epoch 56/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7028 - accuracy: 0.7307 - f1_m: 0.4327 - precision_m: 0.2778 - recall_m: 0.9783
    Epoch 57/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6953 - accuracy: 0.7334 - f1_m: 0.4331 - precision_m: 0.2782 - recall_m: 0.9774
    Epoch 58/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6758 - accuracy: 0.7652 - f1_m: 0.4295 - precision_m: 0.2756 - recall_m: 0.9732
    Epoch 59/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7306 - accuracy: 0.7270 - f1_m: 0.4370 - precision_m: 0.2818 - recall_m: 0.9740
    Epoch 60/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8146 - accuracy: 0.6988 - f1_m: 0.4237 - precision_m: 0.2716 - recall_m: 0.9644
    Epoch 61/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6612 - accuracy: 0.7561 - f1_m: 0.4338 - precision_m: 0.2782 - recall_m: 0.9844
    Epoch 62/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5580 - accuracy: 0.7980 - f1_m: 0.4457 - precision_m: 0.2876 - recall_m: 0.9905
    Epoch 63/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5345 - accuracy: 0.8189 - f1_m: 0.4474 - precision_m: 0.2893 - recall_m: 0.9878
    Epoch 64/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6797 - accuracy: 0.7425 - f1_m: 0.4445 - precision_m: 0.2882 - recall_m: 0.9714
    Epoch 65/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6805 - accuracy: 0.7416 - f1_m: 0.4383 - precision_m: 0.2832 - recall_m: 0.9706
    Epoch 66/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7183 - accuracy: 0.7352 - f1_m: 0.4505 - precision_m: 0.2931 - recall_m: 0.9748
    Epoch 67/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.5298 - accuracy: 0.8189 - f1_m: 0.4498 - precision_m: 0.2911 - recall_m: 0.9896
    Epoch 68/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.6293 - accuracy: 0.7671 - f1_m: 0.4534 - precision_m: 0.2946 - recall_m: 0.9852
    Epoch 69/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4735 - accuracy: 0.8399 - f1_m: 0.4477 - precision_m: 0.2892 - recall_m: 0.9913
    Epoch 70/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4865 - accuracy: 0.8189 - f1_m: 0.4534 - precision_m: 0.2944 - recall_m: 0.9870
    Epoch 71/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.5579 - accuracy: 0.8053 - f1_m: 0.4544 - precision_m: 0.2959 - recall_m: 0.9800
    Epoch 72/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4811 - accuracy: 0.8526 - f1_m: 0.4565 - precision_m: 0.2972 - recall_m: 0.9852
    Epoch 73/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4487 - accuracy: 0.8462 - f1_m: 0.4584 - precision_m: 0.2983 - recall_m: 0.9905
    Epoch 74/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4263 - accuracy: 0.8544 - f1_m: 0.4647 - precision_m: 0.3044 - recall_m: 0.9828
    Epoch 75/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.4206 - accuracy: 0.8499 - f1_m: 0.4692 - precision_m: 0.3075 - recall_m: 0.9905
    Epoch 76/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3629 - accuracy: 0.8753 - f1_m: 0.4688 - precision_m: 0.3070 - recall_m: 0.9913
    Epoch 77/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2837 - accuracy: 0.9108 - f1_m: 0.4771 - precision_m: 0.3139 - recall_m: 0.9948
    Epoch 78/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2511 - accuracy: 0.9190 - f1_m: 0.4742 - precision_m: 0.3113 - recall_m: 0.9974
    Epoch 79/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3140 - accuracy: 0.8917 - f1_m: 0.4811 - precision_m: 0.3173 - recall_m: 0.9957
    Epoch 80/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2541 - accuracy: 0.9190 - f1_m: 0.4815 - precision_m: 0.3176 - recall_m: 0.9957
    Epoch 81/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2718 - accuracy: 0.9054 - f1_m: 0.4819 - precision_m: 0.3180 - recall_m: 0.9957
    Epoch 82/150
    18/18 [==============================] - 0s 9ms/step - loss: 0.3197 - accuracy: 0.8926 - f1_m: 0.4819 - precision_m: 0.3179 - recall_m: 0.9965
    Epoch 83/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3644 - accuracy: 0.8735 - f1_m: 0.4756 - precision_m: 0.3133 - recall_m: 0.9896
    Epoch 84/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3652 - accuracy: 0.8690 - f1_m: 0.4866 - precision_m: 0.3227 - recall_m: 0.9913
    Epoch 85/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3057 - accuracy: 0.8954 - f1_m: 0.4945 - precision_m: 0.3288 - recall_m: 0.9974
    Epoch 86/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3147 - accuracy: 0.8963 - f1_m: 0.4820 - precision_m: 0.3183 - recall_m: 0.9939
    Epoch 87/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2238 - accuracy: 0.9154 - f1_m: 0.4851 - precision_m: 0.3206 - recall_m: 0.9991
    Epoch 88/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2934 - accuracy: 0.8990 - f1_m: 0.4827 - precision_m: 0.3189 - recall_m: 0.9948
    Epoch 89/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3213 - accuracy: 0.8844 - f1_m: 0.4775 - precision_m: 0.3147 - recall_m: 0.9905
    Epoch 90/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2018 - accuracy: 0.9227 - f1_m: 0.4878 - precision_m: 0.3229 - recall_m: 0.9983
    Epoch 91/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2386 - accuracy: 0.9126 - f1_m: 0.4965 - precision_m: 0.3311 - recall_m: 0.9939
    Epoch 92/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1854 - accuracy: 0.9381 - f1_m: 0.4871 - precision_m: 0.3222 - recall_m: 0.9983
    Epoch 93/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2023 - accuracy: 0.9290 - f1_m: 0.4893 - precision_m: 0.3243 - recall_m: 0.9974
    Epoch 94/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1595 - accuracy: 0.9390 - f1_m: 0.4953 - precision_m: 0.3295 - recall_m: 0.9983
    Epoch 95/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1211 - accuracy: 0.9654 - f1_m: 0.4930 - precision_m: 0.3274 - recall_m: 0.9983
    Epoch 96/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1399 - accuracy: 0.9500 - f1_m: 0.5002 - precision_m: 0.3337 - recall_m: 0.9991
    Epoch 97/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2442 - accuracy: 0.9181 - f1_m: 0.4972 - precision_m: 0.3316 - recall_m: 0.9939
    Epoch 98/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2045 - accuracy: 0.9327 - f1_m: 0.5022 - precision_m: 0.3358 - recall_m: 0.9965
    Epoch 99/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1516 - accuracy: 0.9463 - f1_m: 0.5178 - precision_m: 0.3498 - recall_m: 0.9991
    Epoch 100/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1305 - accuracy: 0.9591 - f1_m: 0.5081 - precision_m: 0.3414 - recall_m: 0.9941
    Epoch 101/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3302 - accuracy: 0.8817 - f1_m: 0.5031 - precision_m: 0.3374 - recall_m: 0.9896
    Epoch 102/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4686 - accuracy: 0.8362 - f1_m: 0.5024 - precision_m: 0.3374 - recall_m: 0.9844
    Epoch 103/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.3131 - accuracy: 0.8990 - f1_m: 0.4930 - precision_m: 0.3281 - recall_m: 0.9922
    Epoch 104/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2127 - accuracy: 0.9145 - f1_m: 0.5055 - precision_m: 0.3388 - recall_m: 0.9965
    Epoch 105/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1239 - accuracy: 0.9591 - f1_m: 0.5066 - precision_m: 0.3396 - recall_m: 0.9983
    Epoch 106/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1345 - accuracy: 0.9600 - f1_m: 0.5033 - precision_m: 0.3364 - recall_m: 1.0000
    Epoch 107/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1252 - accuracy: 0.9627 - f1_m: 0.5119 - precision_m: 0.3443 - recall_m: 0.9991
    Epoch 108/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1705 - accuracy: 0.9409 - f1_m: 0.5036 - precision_m: 0.3370 - recall_m: 0.9983
    Epoch 109/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0986 - accuracy: 0.9700 - f1_m: 0.5191 - precision_m: 0.3509 - recall_m: 0.9983
    Epoch 110/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1635 - accuracy: 0.9481 - f1_m: 0.5127 - precision_m: 0.3450 - recall_m: 0.9991
    Epoch 111/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1944 - accuracy: 0.9299 - f1_m: 0.5204 - precision_m: 0.3523 - recall_m: 0.9965
    Epoch 112/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1382 - accuracy: 0.9518 - f1_m: 0.5192 - precision_m: 0.3507 - recall_m: 1.0000
    Epoch 113/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0905 - accuracy: 0.9709 - f1_m: 0.5113 - precision_m: 0.3437 - recall_m: 0.9991
    Epoch 114/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0628 - accuracy: 0.9864 - f1_m: 0.5164 - precision_m: 0.3482 - recall_m: 1.0000
    Epoch 115/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0346 - accuracy: 0.9945 - f1_m: 0.5196 - precision_m: 0.3511 - recall_m: 1.0000
    Epoch 116/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9955 - f1_m: 0.5235 - precision_m: 0.3547 - recall_m: 1.0000
    Epoch 117/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0321 - accuracy: 0.9936 - f1_m: 0.5236 - precision_m: 0.3549 - recall_m: 1.0000
    Epoch 118/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0522 - accuracy: 0.9836 - f1_m: 0.5246 - precision_m: 0.3558 - recall_m: 1.0000
    Epoch 119/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0552 - accuracy: 0.9818 - f1_m: 0.5216 - precision_m: 0.3530 - recall_m: 0.9991
    Epoch 120/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1786 - accuracy: 0.9436 - f1_m: 0.5138 - precision_m: 0.3465 - recall_m: 0.9948
    Epoch 121/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1213 - accuracy: 0.9627 - f1_m: 0.5167 - precision_m: 0.3487 - recall_m: 0.9991
    Epoch 122/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2250 - accuracy: 0.9227 - f1_m: 0.5154 - precision_m: 0.3479 - recall_m: 0.9957
    Epoch 123/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1925 - accuracy: 0.9290 - f1_m: 0.5085 - precision_m: 0.3415 - recall_m: 0.9965
    Epoch 124/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2362 - accuracy: 0.9217 - f1_m: 0.5103 - precision_m: 0.3429 - recall_m: 0.9974
    Epoch 125/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1465 - accuracy: 0.9500 - f1_m: 0.5179 - precision_m: 0.3497 - recall_m: 0.9983
    Epoch 126/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0938 - accuracy: 0.9682 - f1_m: 0.5131 - precision_m: 0.3453 - recall_m: 0.9991
    Epoch 127/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0533 - accuracy: 0.9845 - f1_m: 0.5158 - precision_m: 0.3477 - recall_m: 1.0000
    Epoch 128/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0258 - accuracy: 0.9982 - f1_m: 0.5156 - precision_m: 0.3474 - recall_m: 1.0000
    Epoch 129/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0204 - accuracy: 0.9964 - f1_m: 0.5246 - precision_m: 0.3557 - recall_m: 1.0000
    Epoch 130/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0173 - accuracy: 0.9973 - f1_m: 0.5286 - precision_m: 0.3595 - recall_m: 1.0000
    Epoch 131/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0109 - accuracy: 0.9991 - f1_m: 0.5308 - precision_m: 0.3614 - recall_m: 1.0000
    Epoch 132/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0097 - accuracy: 0.9991 - f1_m: 0.5304 - precision_m: 0.3610 - recall_m: 1.0000
    Epoch 133/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0084 - accuracy: 0.9991 - f1_m: 0.5284 - precision_m: 0.3593 - recall_m: 1.0000
    Epoch 134/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0074 - accuracy: 0.9991 - f1_m: 0.5285 - precision_m: 0.3592 - recall_m: 1.0000
    Epoch 135/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0067 - accuracy: 0.9991 - f1_m: 0.5299 - precision_m: 0.3606 - recall_m: 1.0000
    Epoch 136/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0062 - accuracy: 0.9991 - f1_m: 0.5313 - precision_m: 0.3620 - recall_m: 1.0000
    Epoch 137/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0060 - accuracy: 0.9991 - f1_m: 0.5332 - precision_m: 0.3636 - recall_m: 1.0000
    Epoch 138/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0055 - accuracy: 0.9991 - f1_m: 0.5340 - precision_m: 0.3644 - recall_m: 1.0000
    Epoch 139/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0061 - accuracy: 0.9991 - f1_m: 0.5346 - precision_m: 0.3651 - recall_m: 1.0000
    Epoch 140/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0059 - accuracy: 0.9991 - f1_m: 0.5357 - precision_m: 0.3663 - recall_m: 1.0000
    Epoch 141/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0057 - accuracy: 0.9991 - f1_m: 0.5319 - precision_m: 0.3626 - recall_m: 1.0000
    Epoch 142/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0050 - accuracy: 0.9991 - f1_m: 0.5368 - precision_m: 0.3670 - recall_m: 1.0000
    Epoch 143/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0048 - accuracy: 0.9991 - f1_m: 0.5373 - precision_m: 0.3675 - recall_m: 1.0000
    Epoch 144/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0048 - accuracy: 0.9991 - f1_m: 0.5329 - precision_m: 0.3634 - recall_m: 1.0000
    Epoch 145/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0044 - accuracy: 0.9991 - f1_m: 0.5362 - precision_m: 0.3664 - recall_m: 1.0000
    Epoch 146/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0046 - accuracy: 0.9991 - f1_m: 0.5359 - precision_m: 0.3663 - recall_m: 1.0000
    Epoch 147/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0139 - accuracy: 0.9982 - f1_m: 0.5347 - precision_m: 0.3650 - recall_m: 1.0000
    Epoch 148/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0062 - accuracy: 0.9991 - f1_m: 0.5335 - precision_m: 0.3640 - recall_m: 1.0000
    Epoch 149/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0069 - accuracy: 0.9991 - f1_m: 0.5351 - precision_m: 0.3654 - recall_m: 1.0000
    Epoch 150/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0055 - accuracy: 0.9991 - f1_m: 0.5362 - precision_m: 0.3664 - recall_m: 1.0000
    9/9 [==============================] - 1s 7ms/step - loss: 0.6086 - accuracy: 0.8745 - f1_m: 0.5064 - precision_m: 0.3440 - recall_m: 0.9601
    Epoch 1/150
    18/18 [==============================] - 2s 14ms/step - loss: 2.3049 - accuracy: 0.1364 - f1_m: 0.1782 - precision_m: 0.1023 - recall_m: 0.7017
    Epoch 2/150
    18/18 [==============================] - 0s 12ms/step - loss: 2.2594 - accuracy: 0.1527 - f1_m: 0.1961 - precision_m: 0.1128 - recall_m: 0.7532
    Epoch 3/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.1951 - accuracy: 0.1791 - f1_m: 0.2028 - precision_m: 0.1193 - recall_m: 0.6800
    Epoch 4/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.1401 - accuracy: 0.2027 - f1_m: 0.2156 - precision_m: 0.1239 - recall_m: 0.8313
    Epoch 5/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.1334 - accuracy: 0.2036 - f1_m: 0.2206 - precision_m: 0.1283 - recall_m: 0.7885
    Epoch 6/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.0843 - accuracy: 0.2018 - f1_m: 0.2368 - precision_m: 0.1382 - recall_m: 0.8278
    Epoch 7/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.0963 - accuracy: 0.2309 - f1_m: 0.2222 - precision_m: 0.1286 - recall_m: 0.8166
    Epoch 8/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0760 - accuracy: 0.2036 - f1_m: 0.2290 - precision_m: 0.1329 - recall_m: 0.8275
    Epoch 9/150
    18/18 [==============================] - 0s 11ms/step - loss: 2.0134 - accuracy: 0.2518 - f1_m: 0.2442 - precision_m: 0.1427 - recall_m: 0.8446
    Epoch 10/150
    18/18 [==============================] - 0s 10ms/step - loss: 2.0078 - accuracy: 0.2445 - f1_m: 0.2391 - precision_m: 0.1386 - recall_m: 0.8701
    Epoch 11/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.9992 - accuracy: 0.2427 - f1_m: 0.2382 - precision_m: 0.1376 - recall_m: 0.8880
    Epoch 12/150
    18/18 [==============================] - 0s 12ms/step - loss: 2.0362 - accuracy: 0.2427 - f1_m: 0.2356 - precision_m: 0.1357 - recall_m: 0.8981
    Epoch 13/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.9883 - accuracy: 0.2555 - f1_m: 0.2383 - precision_m: 0.1374 - recall_m: 0.8996
    Epoch 14/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.9506 - accuracy: 0.2955 - f1_m: 0.2419 - precision_m: 0.1398 - recall_m: 0.8984
    Epoch 15/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.9947 - accuracy: 0.2709 - f1_m: 0.2392 - precision_m: 0.1382 - recall_m: 0.8895
    Epoch 16/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.9404 - accuracy: 0.2873 - f1_m: 0.2413 - precision_m: 0.1393 - recall_m: 0.9016
    Epoch 17/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.9218 - accuracy: 0.2782 - f1_m: 0.2447 - precision_m: 0.1413 - recall_m: 0.9129
    Epoch 18/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.9225 - accuracy: 0.2727 - f1_m: 0.2456 - precision_m: 0.1422 - recall_m: 0.9039
    Epoch 19/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.8222 - accuracy: 0.3264 - f1_m: 0.2561 - precision_m: 0.1485 - recall_m: 0.9288
    Epoch 20/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.8384 - accuracy: 0.3282 - f1_m: 0.2590 - precision_m: 0.1524 - recall_m: 0.8631
    Epoch 21/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.8424 - accuracy: 0.3264 - f1_m: 0.2565 - precision_m: 0.1497 - recall_m: 0.8964
    Epoch 22/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.8080 - accuracy: 0.3218 - f1_m: 0.2620 - precision_m: 0.1533 - recall_m: 0.9036
    Epoch 23/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.7199 - accuracy: 0.3673 - f1_m: 0.2738 - precision_m: 0.1610 - recall_m: 0.9155
    Epoch 24/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.6575 - accuracy: 0.3864 - f1_m: 0.2717 - precision_m: 0.1589 - recall_m: 0.9384
    Epoch 25/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.6451 - accuracy: 0.3955 - f1_m: 0.2737 - precision_m: 0.1609 - recall_m: 0.9155
    Epoch 26/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.6355 - accuracy: 0.3764 - f1_m: 0.2875 - precision_m: 0.1700 - recall_m: 0.9332
    Epoch 27/150
    18/18 [==============================] - 0s 12ms/step - loss: 1.5322 - accuracy: 0.4191 - f1_m: 0.2929 - precision_m: 0.1731 - recall_m: 0.9502
    Epoch 28/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.5239 - accuracy: 0.4136 - f1_m: 0.2976 - precision_m: 0.1762 - recall_m: 0.9575
    Epoch 29/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.5464 - accuracy: 0.4100 - f1_m: 0.2987 - precision_m: 0.1773 - recall_m: 0.9479
    Epoch 30/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.4750 - accuracy: 0.4391 - f1_m: 0.2996 - precision_m: 0.1778 - recall_m: 0.9531
    Epoch 31/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.4460 - accuracy: 0.4582 - f1_m: 0.3033 - precision_m: 0.1803 - recall_m: 0.9554
    Epoch 32/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.4296 - accuracy: 0.4564 - f1_m: 0.3079 - precision_m: 0.1832 - recall_m: 0.9653
    Epoch 33/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3911 - accuracy: 0.4564 - f1_m: 0.3085 - precision_m: 0.1837 - recall_m: 0.9627
    Epoch 34/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3896 - accuracy: 0.4855 - f1_m: 0.3101 - precision_m: 0.1848 - recall_m: 0.9644
    Epoch 35/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.5276 - accuracy: 0.4236 - f1_m: 0.3086 - precision_m: 0.1845 - recall_m: 0.9462
    Epoch 36/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3532 - accuracy: 0.4909 - f1_m: 0.3202 - precision_m: 0.1919 - recall_m: 0.9688
    Epoch 37/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.3212 - accuracy: 0.4909 - f1_m: 0.3241 - precision_m: 0.1947 - recall_m: 0.9688
    Epoch 38/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.2686 - accuracy: 0.5273 - f1_m: 0.3208 - precision_m: 0.1924 - recall_m: 0.9653
    Epoch 39/150
    18/18 [==============================] - 0s 10ms/step - loss: 1.3000 - accuracy: 0.5036 - f1_m: 0.3167 - precision_m: 0.1893 - recall_m: 0.9696
    Epoch 40/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.2019 - accuracy: 0.5527 - f1_m: 0.3221 - precision_m: 0.1932 - recall_m: 0.9676
    Epoch 41/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.2202 - accuracy: 0.5391 - f1_m: 0.3210 - precision_m: 0.1923 - recall_m: 0.9714
    Epoch 42/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.2130 - accuracy: 0.5391 - f1_m: 0.3209 - precision_m: 0.1928 - recall_m: 0.9586
    Epoch 43/150
    18/18 [==============================] - 0s 12ms/step - loss: 1.1795 - accuracy: 0.5500 - f1_m: 0.3308 - precision_m: 0.1996 - recall_m: 0.9670
    Epoch 44/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.1357 - accuracy: 0.5718 - f1_m: 0.3332 - precision_m: 0.2011 - recall_m: 0.9722
    Epoch 45/150
    18/18 [==============================] - 0s 12ms/step - loss: 1.1695 - accuracy: 0.5555 - f1_m: 0.3305 - precision_m: 0.1993 - recall_m: 0.9676
    Epoch 46/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.0720 - accuracy: 0.5836 - f1_m: 0.3390 - precision_m: 0.2049 - recall_m: 0.9826
    Epoch 47/150
    18/18 [==============================] - 0s 12ms/step - loss: 1.0258 - accuracy: 0.5855 - f1_m: 0.3401 - precision_m: 0.2059 - recall_m: 0.9783
    Epoch 48/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.0879 - accuracy: 0.5827 - f1_m: 0.3420 - precision_m: 0.2073 - recall_m: 0.9783
    Epoch 49/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.9878 - accuracy: 0.6309 - f1_m: 0.3450 - precision_m: 0.2095 - recall_m: 0.9774
    Epoch 50/150
    18/18 [==============================] - 0s 11ms/step - loss: 1.0262 - accuracy: 0.6118 - f1_m: 0.3492 - precision_m: 0.2127 - recall_m: 0.9745
    Epoch 51/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.9951 - accuracy: 0.6118 - f1_m: 0.3440 - precision_m: 0.2084 - recall_m: 0.9852
    Epoch 52/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.9255 - accuracy: 0.6545 - f1_m: 0.3531 - precision_m: 0.2151 - recall_m: 0.9852
    Epoch 53/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.8992 - accuracy: 0.6600 - f1_m: 0.3597 - precision_m: 0.2201 - recall_m: 0.9835
    Epoch 54/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.9194 - accuracy: 0.6464 - f1_m: 0.3568 - precision_m: 0.2183 - recall_m: 0.9771
    Epoch 55/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.8885 - accuracy: 0.6682 - f1_m: 0.3551 - precision_m: 0.2165 - recall_m: 0.9878
    Epoch 56/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.9381 - accuracy: 0.6536 - f1_m: 0.3559 - precision_m: 0.2174 - recall_m: 0.9818
    Epoch 57/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.8381 - accuracy: 0.6864 - f1_m: 0.3588 - precision_m: 0.2193 - recall_m: 0.9861
    Epoch 58/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7385 - accuracy: 0.7218 - f1_m: 0.3687 - precision_m: 0.2265 - recall_m: 0.9905
    Epoch 59/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7505 - accuracy: 0.7200 - f1_m: 0.3667 - precision_m: 0.2254 - recall_m: 0.9832
    Epoch 60/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7336 - accuracy: 0.7164 - f1_m: 0.3751 - precision_m: 0.2314 - recall_m: 0.9913
    Epoch 61/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7908 - accuracy: 0.7082 - f1_m: 0.3612 - precision_m: 0.2213 - recall_m: 0.9835
    Epoch 62/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7102 - accuracy: 0.7345 - f1_m: 0.3716 - precision_m: 0.2288 - recall_m: 0.9896
    Epoch 63/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.7192 - accuracy: 0.7409 - f1_m: 0.3725 - precision_m: 0.2297 - recall_m: 0.9861
    Epoch 64/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7189 - accuracy: 0.7318 - f1_m: 0.3705 - precision_m: 0.2281 - recall_m: 0.9878
    Epoch 65/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.7594 - accuracy: 0.7182 - f1_m: 0.3699 - precision_m: 0.2277 - recall_m: 0.9870
    Epoch 66/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6569 - accuracy: 0.7627 - f1_m: 0.3637 - precision_m: 0.2227 - recall_m: 0.9922
    Epoch 67/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6469 - accuracy: 0.7736 - f1_m: 0.3797 - precision_m: 0.2350 - recall_m: 0.9887
    Epoch 68/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.5352 - accuracy: 0.8064 - f1_m: 0.3748 - precision_m: 0.2313 - recall_m: 0.9876
    Epoch 69/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.6925 - accuracy: 0.7382 - f1_m: 0.3787 - precision_m: 0.2343 - recall_m: 0.9878
    Epoch 70/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.6669 - accuracy: 0.7527 - f1_m: 0.3778 - precision_m: 0.2335 - recall_m: 0.9896
    Epoch 71/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.5272 - accuracy: 0.8209 - f1_m: 0.3783 - precision_m: 0.2337 - recall_m: 0.9931
    Epoch 72/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.5064 - accuracy: 0.8291 - f1_m: 0.3800 - precision_m: 0.2350 - recall_m: 0.9931
    Epoch 73/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4534 - accuracy: 0.8500 - f1_m: 0.3854 - precision_m: 0.2393 - recall_m: 0.9905
    Epoch 74/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4601 - accuracy: 0.8209 - f1_m: 0.3800 - precision_m: 0.2349 - recall_m: 0.9948
    Epoch 75/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.6260 - accuracy: 0.7827 - f1_m: 0.3779 - precision_m: 0.2339 - recall_m: 0.9835
    Epoch 76/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.5198 - accuracy: 0.8155 - f1_m: 0.3766 - precision_m: 0.2326 - recall_m: 0.9905
    Epoch 77/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4033 - accuracy: 0.8636 - f1_m: 0.3791 - precision_m: 0.2346 - recall_m: 0.9884
    Epoch 78/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.5598 - accuracy: 0.8027 - f1_m: 0.3834 - precision_m: 0.2379 - recall_m: 0.9884
    Epoch 79/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3746 - accuracy: 0.8673 - f1_m: 0.3911 - precision_m: 0.2432 - recall_m: 0.9983
    Epoch 80/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3724 - accuracy: 0.8655 - f1_m: 0.3948 - precision_m: 0.2463 - recall_m: 0.9948
    Epoch 81/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4317 - accuracy: 0.8391 - f1_m: 0.3852 - precision_m: 0.2389 - recall_m: 0.9948
    Epoch 82/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3742 - accuracy: 0.8836 - f1_m: 0.3902 - precision_m: 0.2429 - recall_m: 0.9922
    Epoch 83/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3485 - accuracy: 0.8927 - f1_m: 0.3861 - precision_m: 0.2396 - recall_m: 0.9939
    Epoch 84/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2409 - accuracy: 0.9200 - f1_m: 0.3898 - precision_m: 0.2424 - recall_m: 0.9965
    Epoch 85/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2319 - accuracy: 0.9255 - f1_m: 0.3939 - precision_m: 0.2454 - recall_m: 0.9983
    Epoch 86/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1914 - accuracy: 0.9373 - f1_m: 0.3954 - precision_m: 0.2467 - recall_m: 0.9974
    Epoch 87/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1750 - accuracy: 0.9482 - f1_m: 0.3924 - precision_m: 0.2443 - recall_m: 0.9983
    Epoch 88/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.2038 - accuracy: 0.9364 - f1_m: 0.4009 - precision_m: 0.2509 - recall_m: 0.9983
    Epoch 89/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1510 - accuracy: 0.9573 - f1_m: 0.3934 - precision_m: 0.2449 - recall_m: 0.9991
    Epoch 90/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1328 - accuracy: 0.9591 - f1_m: 0.3904 - precision_m: 0.2427 - recall_m: 0.9991
    Epoch 91/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1700 - accuracy: 0.9364 - f1_m: 0.4050 - precision_m: 0.2540 - recall_m: 1.0000
    Epoch 92/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1886 - accuracy: 0.9427 - f1_m: 0.3998 - precision_m: 0.2499 - recall_m: 1.0000
    Epoch 93/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1709 - accuracy: 0.9518 - f1_m: 0.4042 - precision_m: 0.2534 - recall_m: 0.9991
    Epoch 94/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1469 - accuracy: 0.9518 - f1_m: 0.3971 - precision_m: 0.2479 - recall_m: 0.9983
    Epoch 95/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1572 - accuracy: 0.9564 - f1_m: 0.3976 - precision_m: 0.2482 - recall_m: 0.9991
    Epoch 96/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.1750 - accuracy: 0.9445 - f1_m: 0.4005 - precision_m: 0.2507 - recall_m: 0.9974
    Epoch 97/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.5877 - accuracy: 0.8127 - f1_m: 0.3994 - precision_m: 0.2513 - recall_m: 0.9740
    Epoch 98/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.4549 - accuracy: 0.8418 - f1_m: 0.4076 - precision_m: 0.2567 - recall_m: 0.9896
    Epoch 99/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.2537 - accuracy: 0.9164 - f1_m: 0.4015 - precision_m: 0.2513 - recall_m: 0.9983
    Epoch 100/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1623 - accuracy: 0.9491 - f1_m: 0.4062 - precision_m: 0.2550 - recall_m: 0.9991
    Epoch 101/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1251 - accuracy: 0.9691 - f1_m: 0.4009 - precision_m: 0.2507 - recall_m: 1.0000
    Epoch 102/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1163 - accuracy: 0.9618 - f1_m: 0.4104 - precision_m: 0.2585 - recall_m: 1.0000
    Epoch 103/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0869 - accuracy: 0.9755 - f1_m: 0.4102 - precision_m: 0.2581 - recall_m: 0.9991
    Epoch 104/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0561 - accuracy: 0.9909 - f1_m: 0.4052 - precision_m: 0.2542 - recall_m: 1.0000
    Epoch 105/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0556 - accuracy: 0.9873 - f1_m: 0.4063 - precision_m: 0.2550 - recall_m: 1.0000
    Epoch 106/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0551 - accuracy: 0.9855 - f1_m: 0.4086 - precision_m: 0.2569 - recall_m: 1.0000
    Epoch 107/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0732 - accuracy: 0.9791 - f1_m: 0.4097 - precision_m: 0.2577 - recall_m: 1.0000
    Epoch 108/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0704 - accuracy: 0.9864 - f1_m: 0.4113 - precision_m: 0.2589 - recall_m: 1.0000
    Epoch 109/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0973 - accuracy: 0.9700 - f1_m: 0.4077 - precision_m: 0.2561 - recall_m: 1.0000
    Epoch 110/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0947 - accuracy: 0.9700 - f1_m: 0.4137 - precision_m: 0.2609 - recall_m: 0.9991
    Epoch 111/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1103 - accuracy: 0.9627 - f1_m: 0.4146 - precision_m: 0.2617 - recall_m: 1.0000
    Epoch 112/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1844 - accuracy: 0.9418 - f1_m: 0.4087 - precision_m: 0.2571 - recall_m: 0.9965
    Epoch 113/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.4630 - accuracy: 0.8527 - f1_m: 0.4070 - precision_m: 0.2564 - recall_m: 0.9870
    Epoch 114/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3601 - accuracy: 0.8945 - f1_m: 0.4089 - precision_m: 0.2578 - recall_m: 0.9893
    Epoch 115/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.3230 - accuracy: 0.8836 - f1_m: 0.4096 - precision_m: 0.2578 - recall_m: 0.9965
    Epoch 116/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1682 - accuracy: 0.9445 - f1_m: 0.4073 - precision_m: 0.2559 - recall_m: 0.9991
    Epoch 117/150
    18/18 [==============================] - 0s 13ms/step - loss: 0.0771 - accuracy: 0.9782 - f1_m: 0.4159 - precision_m: 0.2626 - recall_m: 1.0000
    Epoch 118/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0429 - accuracy: 0.9936 - f1_m: 0.4159 - precision_m: 0.2627 - recall_m: 1.0000
    Epoch 119/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0326 - accuracy: 0.9936 - f1_m: 0.4122 - precision_m: 0.2597 - recall_m: 1.0000
    Epoch 120/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0285 - accuracy: 0.9945 - f1_m: 0.4154 - precision_m: 0.2624 - recall_m: 1.0000
    Epoch 121/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1822 - accuracy: 0.9445 - f1_m: 0.4123 - precision_m: 0.2598 - recall_m: 1.0000
    Epoch 122/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1714 - accuracy: 0.9427 - f1_m: 0.4086 - precision_m: 0.2569 - recall_m: 0.9991
    Epoch 123/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.1016 - accuracy: 0.9691 - f1_m: 0.4086 - precision_m: 0.2568 - recall_m: 0.9991
    Epoch 124/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.1275 - accuracy: 0.9682 - f1_m: 0.4104 - precision_m: 0.2583 - recall_m: 0.9983
    Epoch 125/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0622 - accuracy: 0.9818 - f1_m: 0.4171 - precision_m: 0.2636 - recall_m: 1.0000
    Epoch 126/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0734 - accuracy: 0.9764 - f1_m: 0.4192 - precision_m: 0.2653 - recall_m: 1.0000
    Epoch 127/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0421 - accuracy: 0.9909 - f1_m: 0.4208 - precision_m: 0.2665 - recall_m: 1.0000
    Epoch 128/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0517 - accuracy: 0.9855 - f1_m: 0.4256 - precision_m: 0.2704 - recall_m: 1.0000
    Epoch 129/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0411 - accuracy: 0.9900 - f1_m: 0.4237 - precision_m: 0.2689 - recall_m: 1.0000
    Epoch 130/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0455 - accuracy: 0.9845 - f1_m: 0.4190 - precision_m: 0.2651 - recall_m: 1.0000
    Epoch 131/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0434 - accuracy: 0.9891 - f1_m: 0.4233 - precision_m: 0.2686 - recall_m: 1.0000
    Epoch 132/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0341 - accuracy: 0.9873 - f1_m: 0.4178 - precision_m: 0.2641 - recall_m: 1.0000
    Epoch 133/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0351 - accuracy: 0.9900 - f1_m: 0.4186 - precision_m: 0.2648 - recall_m: 1.0000
    Epoch 134/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0331 - accuracy: 0.9918 - f1_m: 0.4190 - precision_m: 0.2651 - recall_m: 1.0000
    Epoch 135/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0479 - accuracy: 0.9855 - f1_m: 0.4228 - precision_m: 0.2681 - recall_m: 1.0000
    Epoch 136/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0493 - accuracy: 0.9827 - f1_m: 0.4236 - precision_m: 0.2688 - recall_m: 1.0000
    Epoch 137/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0469 - accuracy: 0.9855 - f1_m: 0.4263 - precision_m: 0.2709 - recall_m: 1.0000
    Epoch 138/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0438 - accuracy: 0.9891 - f1_m: 0.4296 - precision_m: 0.2736 - recall_m: 1.0000
    Epoch 139/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0240 - accuracy: 0.9945 - f1_m: 0.4231 - precision_m: 0.2684 - recall_m: 1.0000
    Epoch 140/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0173 - accuracy: 0.9982 - f1_m: 0.4267 - precision_m: 0.2713 - recall_m: 1.0000
    Epoch 141/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0079 - accuracy: 1.0000 - f1_m: 0.4211 - precision_m: 0.2668 - recall_m: 1.0000
    Epoch 142/150
    18/18 [==============================] - 0s 12ms/step - loss: 0.0050 - accuracy: 1.0000 - f1_m: 0.4238 - precision_m: 0.2690 - recall_m: 1.0000
    Epoch 143/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0037 - accuracy: 1.0000 - f1_m: 0.4239 - precision_m: 0.2690 - recall_m: 1.0000
    Epoch 144/150
    18/18 [==============================] - 0s 14ms/step - loss: 0.0032 - accuracy: 1.0000 - f1_m: 0.4225 - precision_m: 0.2679 - recall_m: 1.0000
    Epoch 145/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0028 - accuracy: 1.0000 - f1_m: 0.4233 - precision_m: 0.2686 - recall_m: 1.0000
    Epoch 146/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0026 - accuracy: 1.0000 - f1_m: 0.4244 - precision_m: 0.2694 - recall_m: 1.0000
    Epoch 147/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0024 - accuracy: 1.0000 - f1_m: 0.4218 - precision_m: 0.2673 - recall_m: 1.0000
    Epoch 148/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0022 - accuracy: 1.0000 - f1_m: 0.4222 - precision_m: 0.2677 - recall_m: 1.0000
    Epoch 149/150
    18/18 [==============================] - 0s 11ms/step - loss: 0.0020 - accuracy: 1.0000 - f1_m: 0.4237 - precision_m: 0.2688 - recall_m: 1.0000
    Epoch 150/150
    18/18 [==============================] - 0s 10ms/step - loss: 0.0019 - accuracy: 1.0000 - f1_m: 0.4223 - precision_m: 0.2677 - recall_m: 1.0000
    9/9 [==============================] - 1s 8ms/step - loss: 0.6051 - accuracy: 0.8761 - f1_m: 0.4142 - precision_m: 0.2625 - recall_m: 0.9814
    Epoch 1/150
    26/26 [==============================] - 2s 13ms/step - loss: 2.3040 - accuracy: 0.1231 - f1_m: 0.1769 - precision_m: 0.1069 - recall_m: 0.5249
    Epoch 2/150
    26/26 [==============================] - 0s 11ms/step - loss: 2.2075 - accuracy: 0.1686 - f1_m: 0.2091 - precision_m: 0.1267 - recall_m: 0.6057
    Epoch 3/150
    26/26 [==============================] - 0s 10ms/step - loss: 2.1397 - accuracy: 0.1928 - f1_m: 0.2239 - precision_m: 0.1330 - recall_m: 0.7116
    Epoch 4/150
    26/26 [==============================] - 0s 11ms/step - loss: 2.0517 - accuracy: 0.2268 - f1_m: 0.2436 - precision_m: 0.1443 - recall_m: 0.7834
    Epoch 5/150
    26/26 [==============================] - 0s 11ms/step - loss: 2.0022 - accuracy: 0.2365 - f1_m: 0.2521 - precision_m: 0.1487 - recall_m: 0.8286
    Epoch 6/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.9371 - accuracy: 0.2638 - f1_m: 0.2619 - precision_m: 0.1551 - recall_m: 0.8434
    Epoch 7/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.8679 - accuracy: 0.2978 - f1_m: 0.2779 - precision_m: 0.1653 - recall_m: 0.8757
    Epoch 8/150
    26/26 [==============================] - 0s 17ms/step - loss: 1.7857 - accuracy: 0.3269 - f1_m: 0.2816 - precision_m: 0.1667 - recall_m: 0.9069
    Epoch 9/150
    26/26 [==============================] - 0s 15ms/step - loss: 1.7348 - accuracy: 0.3384 - f1_m: 0.2839 - precision_m: 0.1684 - recall_m: 0.9044
    Epoch 10/150
    26/26 [==============================] - 0s 13ms/step - loss: 1.7153 - accuracy: 0.3445 - f1_m: 0.2892 - precision_m: 0.1717 - recall_m: 0.9181
    Epoch 11/150
    26/26 [==============================] - 0s 13ms/step - loss: 1.6543 - accuracy: 0.3845 - f1_m: 0.2929 - precision_m: 0.1741 - recall_m: 0.9252
    Epoch 12/150
    26/26 [==============================] - 0s 13ms/step - loss: 1.5964 - accuracy: 0.3948 - f1_m: 0.2995 - precision_m: 0.1785 - recall_m: 0.9317
    Epoch 13/150
    26/26 [==============================] - 0s 15ms/step - loss: 1.5355 - accuracy: 0.4075 - f1_m: 0.3026 - precision_m: 0.1799 - recall_m: 0.9517
    Epoch 14/150
    26/26 [==============================] - 0s 13ms/step - loss: 1.5749 - accuracy: 0.3942 - f1_m: 0.3033 - precision_m: 0.1812 - recall_m: 0.9320
    Epoch 15/150
    26/26 [==============================] - 0s 12ms/step - loss: 1.4741 - accuracy: 0.4360 - f1_m: 0.3115 - precision_m: 0.1857 - recall_m: 0.9666
    Epoch 16/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.4517 - accuracy: 0.4209 - f1_m: 0.3145 - precision_m: 0.1879 - recall_m: 0.9656
    Epoch 17/150
    26/26 [==============================] - 0s 12ms/step - loss: 1.4229 - accuracy: 0.4372 - f1_m: 0.3153 - precision_m: 0.1887 - recall_m: 0.9608
    Epoch 18/150
    26/26 [==============================] - 0s 12ms/step - loss: 1.3872 - accuracy: 0.4536 - f1_m: 0.3193 - precision_m: 0.1915 - recall_m: 0.9614
    Epoch 19/150
    26/26 [==============================] - 0s 13ms/step - loss: 1.3389 - accuracy: 0.4827 - f1_m: 0.3183 - precision_m: 0.1904 - recall_m: 0.9700
    Epoch 20/150
    26/26 [==============================] - 0s 12ms/step - loss: 1.2776 - accuracy: 0.5124 - f1_m: 0.3219 - precision_m: 0.1931 - recall_m: 0.9680
    Epoch 21/150
    26/26 [==============================] - 0s 12ms/step - loss: 1.2455 - accuracy: 0.5124 - f1_m: 0.3265 - precision_m: 0.1962 - recall_m: 0.9730
    Epoch 22/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.2106 - accuracy: 0.5343 - f1_m: 0.3310 - precision_m: 0.1995 - recall_m: 0.9730
    Epoch 23/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.2101 - accuracy: 0.5379 - f1_m: 0.3307 - precision_m: 0.1994 - recall_m: 0.9698
    Epoch 24/150
    26/26 [==============================] - 0s 10ms/step - loss: 1.1720 - accuracy: 0.5415 - f1_m: 0.3353 - precision_m: 0.2026 - recall_m: 0.9728
    Epoch 25/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.1998 - accuracy: 0.5288 - f1_m: 0.3379 - precision_m: 0.2048 - recall_m: 0.9652
    Epoch 26/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.1102 - accuracy: 0.5761 - f1_m: 0.3467 - precision_m: 0.2109 - recall_m: 0.9766
    Epoch 27/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.0218 - accuracy: 0.5998 - f1_m: 0.3495 - precision_m: 0.2128 - recall_m: 0.9790
    Epoch 28/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.1111 - accuracy: 0.5852 - f1_m: 0.3484 - precision_m: 0.2122 - recall_m: 0.9736
    Epoch 29/150
    26/26 [==============================] - 0s 11ms/step - loss: 1.0299 - accuracy: 0.6028 - f1_m: 0.3595 - precision_m: 0.2205 - recall_m: 0.9738
    Epoch 30/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.9722 - accuracy: 0.6392 - f1_m: 0.3569 - precision_m: 0.2183 - recall_m: 0.9792
    Epoch 31/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.9216 - accuracy: 0.6598 - f1_m: 0.3619 - precision_m: 0.2218 - recall_m: 0.9826
    Epoch 32/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.9324 - accuracy: 0.6458 - f1_m: 0.3681 - precision_m: 0.2268 - recall_m: 0.9776
    Epoch 33/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.9313 - accuracy: 0.6380 - f1_m: 0.3674 - precision_m: 0.2266 - recall_m: 0.9724
    Epoch 34/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8013 - accuracy: 0.7065 - f1_m: 0.3746 - precision_m: 0.2314 - recall_m: 0.9838
    Epoch 35/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.7687 - accuracy: 0.7162 - f1_m: 0.3844 - precision_m: 0.2388 - recall_m: 0.9856
    Epoch 36/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.8259 - accuracy: 0.7047 - f1_m: 0.3750 - precision_m: 0.2320 - recall_m: 0.9782
    Epoch 37/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.6701 - accuracy: 0.7647 - f1_m: 0.3904 - precision_m: 0.2434 - recall_m: 0.9870
    Epoch 38/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.8077 - accuracy: 0.7065 - f1_m: 0.3876 - precision_m: 0.2414 - recall_m: 0.9838
    Epoch 39/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.7657 - accuracy: 0.7362 - f1_m: 0.3810 - precision_m: 0.2366 - recall_m: 0.9796
    Epoch 40/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.5989 - accuracy: 0.7981 - f1_m: 0.3921 - precision_m: 0.2447 - recall_m: 0.9866
    Epoch 41/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.5366 - accuracy: 0.8132 - f1_m: 0.4011 - precision_m: 0.2513 - recall_m: 0.9932
    Epoch 42/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.5009 - accuracy: 0.8326 - f1_m: 0.4038 - precision_m: 0.2534 - recall_m: 0.9938
    Epoch 43/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.5692 - accuracy: 0.8084 - f1_m: 0.4048 - precision_m: 0.2546 - recall_m: 0.9886
    Epoch 44/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.4930 - accuracy: 0.8344 - f1_m: 0.4057 - precision_m: 0.2548 - recall_m: 0.9958
    Epoch 45/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.4581 - accuracy: 0.8460 - f1_m: 0.4080 - precision_m: 0.2568 - recall_m: 0.9928
    Epoch 46/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.3765 - accuracy: 0.8745 - f1_m: 0.4174 - precision_m: 0.2641 - recall_m: 0.9958
    Epoch 47/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.3284 - accuracy: 0.8842 - f1_m: 0.4191 - precision_m: 0.2654 - recall_m: 0.9974
    Epoch 48/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.3422 - accuracy: 0.8860 - f1_m: 0.4176 - precision_m: 0.2643 - recall_m: 0.9956
    Epoch 49/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.4334 - accuracy: 0.8496 - f1_m: 0.4161 - precision_m: 0.2632 - recall_m: 0.9938
    Epoch 50/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.3508 - accuracy: 0.8836 - f1_m: 0.4267 - precision_m: 0.2715 - recall_m: 0.9970
    Epoch 51/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.3139 - accuracy: 0.8908 - f1_m: 0.4293 - precision_m: 0.2739 - recall_m: 0.9940
    Epoch 52/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2782 - accuracy: 0.9084 - f1_m: 0.4302 - precision_m: 0.2744 - recall_m: 0.9964
    Epoch 53/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2707 - accuracy: 0.9060 - f1_m: 0.4386 - precision_m: 0.2811 - recall_m: 0.9982
    Epoch 54/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2211 - accuracy: 0.9218 - f1_m: 0.4411 - precision_m: 0.2833 - recall_m: 0.9970
    Epoch 55/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.2024 - accuracy: 0.9333 - f1_m: 0.4406 - precision_m: 0.2828 - recall_m: 0.9982
    Epoch 56/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2142 - accuracy: 0.9327 - f1_m: 0.4416 - precision_m: 0.2838 - recall_m: 0.9964
    Epoch 57/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2407 - accuracy: 0.9163 - f1_m: 0.4354 - precision_m: 0.2786 - recall_m: 0.9970
    Epoch 58/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.2253 - accuracy: 0.9315 - f1_m: 0.4410 - precision_m: 0.2832 - recall_m: 0.9970
    Epoch 59/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1591 - accuracy: 0.9503 - f1_m: 0.4395 - precision_m: 0.2819 - recall_m: 0.9976
    Epoch 60/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.2212 - accuracy: 0.9242 - f1_m: 0.4404 - precision_m: 0.2826 - recall_m: 0.9976
    Epoch 61/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.3227 - accuracy: 0.8921 - f1_m: 0.4428 - precision_m: 0.2847 - recall_m: 0.9964
    Epoch 62/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.2011 - accuracy: 0.9357 - f1_m: 0.4442 - precision_m: 0.2856 - recall_m: 0.9988
    Epoch 63/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.2187 - accuracy: 0.9254 - f1_m: 0.4442 - precision_m: 0.2859 - recall_m: 0.9970
    Epoch 64/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1414 - accuracy: 0.9533 - f1_m: 0.4450 - precision_m: 0.2864 - recall_m: 0.9988
    Epoch 65/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1175 - accuracy: 0.9660 - f1_m: 0.4449 - precision_m: 0.2862 - recall_m: 0.9988
    Epoch 66/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1493 - accuracy: 0.9497 - f1_m: 0.4501 - precision_m: 0.2906 - recall_m: 0.9994
    Epoch 67/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.1518 - accuracy: 0.9527 - f1_m: 0.4447 - precision_m: 0.2862 - recall_m: 0.9982
    Epoch 68/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1721 - accuracy: 0.9430 - f1_m: 0.4514 - precision_m: 0.2917 - recall_m: 0.9986
    Epoch 69/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1263 - accuracy: 0.9618 - f1_m: 0.4534 - precision_m: 0.2933 - recall_m: 0.9988
    Epoch 70/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0763 - accuracy: 0.9751 - f1_m: 0.4614 - precision_m: 0.2999 - recall_m: 1.0000
    Epoch 71/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1035 - accuracy: 0.9594 - f1_m: 0.4616 - precision_m: 0.3002 - recall_m: 0.9994
    Epoch 72/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.1492 - accuracy: 0.9503 - f1_m: 0.4661 - precision_m: 0.3040 - recall_m: 1.0000
    Epoch 73/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1552 - accuracy: 0.9509 - f1_m: 0.4569 - precision_m: 0.2963 - recall_m: 0.9982
    Epoch 74/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1596 - accuracy: 0.9442 - f1_m: 0.4571 - precision_m: 0.2965 - recall_m: 0.9988
    Epoch 75/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.1418 - accuracy: 0.9503 - f1_m: 0.4622 - precision_m: 0.3008 - recall_m: 0.9982
    Epoch 76/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0700 - accuracy: 0.9788 - f1_m: 0.4640 - precision_m: 0.3022 - recall_m: 1.0000
    Epoch 77/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0718 - accuracy: 0.9812 - f1_m: 0.4580 - precision_m: 0.2971 - recall_m: 1.0000
    Epoch 78/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1313 - accuracy: 0.9545 - f1_m: 0.4573 - precision_m: 0.2966 - recall_m: 0.9994
    Epoch 79/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0787 - accuracy: 0.9763 - f1_m: 0.4642 - precision_m: 0.3023 - recall_m: 1.0000
    Epoch 80/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0592 - accuracy: 0.9806 - f1_m: 0.4684 - precision_m: 0.3059 - recall_m: 1.0000
    Epoch 81/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0446 - accuracy: 0.9897 - f1_m: 0.4700 - precision_m: 0.3073 - recall_m: 1.0000
    Epoch 82/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0551 - accuracy: 0.9818 - f1_m: 0.4670 - precision_m: 0.3047 - recall_m: 1.0000
    Epoch 83/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0849 - accuracy: 0.9751 - f1_m: 0.4750 - precision_m: 0.3116 - recall_m: 0.9994
    Epoch 84/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0927 - accuracy: 0.9654 - f1_m: 0.4767 - precision_m: 0.3130 - recall_m: 1.0000
    Epoch 85/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1050 - accuracy: 0.9618 - f1_m: 0.4788 - precision_m: 0.3150 - recall_m: 0.9988
    Epoch 86/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0906 - accuracy: 0.9679 - f1_m: 0.4716 - precision_m: 0.3087 - recall_m: 0.9994
    Epoch 87/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0827 - accuracy: 0.9751 - f1_m: 0.4844 - precision_m: 0.3197 - recall_m: 1.0000
    Epoch 88/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.0536 - accuracy: 0.9830 - f1_m: 0.4668 - precision_m: 0.3046 - recall_m: 1.0000
    Epoch 89/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.0504 - accuracy: 0.9848 - f1_m: 0.4722 - precision_m: 0.3092 - recall_m: 1.0000
    Epoch 90/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0503 - accuracy: 0.9861 - f1_m: 0.4785 - precision_m: 0.3147 - recall_m: 1.0000
    Epoch 91/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0920 - accuracy: 0.9697 - f1_m: 0.4829 - precision_m: 0.3185 - recall_m: 0.9994
    Epoch 92/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0900 - accuracy: 0.9721 - f1_m: 0.4839 - precision_m: 0.3193 - recall_m: 0.9994
    Epoch 93/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0957 - accuracy: 0.9673 - f1_m: 0.4793 - precision_m: 0.3153 - recall_m: 1.0000
    Epoch 94/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0494 - accuracy: 0.9836 - f1_m: 0.4805 - precision_m: 0.3162 - recall_m: 1.0000
    Epoch 95/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0574 - accuracy: 0.9782 - f1_m: 0.4752 - precision_m: 0.3117 - recall_m: 1.0000
    Epoch 96/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0387 - accuracy: 0.9897 - f1_m: 0.4799 - precision_m: 0.3158 - recall_m: 1.0000
    Epoch 97/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0287 - accuracy: 0.9915 - f1_m: 0.4886 - precision_m: 0.3233 - recall_m: 1.0000
    Epoch 98/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0481 - accuracy: 0.9867 - f1_m: 0.4878 - precision_m: 0.3227 - recall_m: 1.0000
    Epoch 99/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1307 - accuracy: 0.9576 - f1_m: 0.4802 - precision_m: 0.3162 - recall_m: 0.9988
    Epoch 100/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1031 - accuracy: 0.9630 - f1_m: 0.4754 - precision_m: 0.3120 - recall_m: 0.9994
    Epoch 101/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0654 - accuracy: 0.9776 - f1_m: 0.4803 - precision_m: 0.3163 - recall_m: 0.9994
    Epoch 102/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.0694 - accuracy: 0.9806 - f1_m: 0.4852 - precision_m: 0.3206 - recall_m: 0.9980
    Epoch 103/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0466 - accuracy: 0.9867 - f1_m: 0.4875 - precision_m: 0.3224 - recall_m: 0.9994
    Epoch 104/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0369 - accuracy: 0.9909 - f1_m: 0.5002 - precision_m: 0.3336 - recall_m: 0.9994
    Epoch 105/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0217 - accuracy: 0.9958 - f1_m: 0.5052 - precision_m: 0.3381 - recall_m: 1.0000
    Epoch 106/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0230 - accuracy: 0.9939 - f1_m: 0.5030 - precision_m: 0.3361 - recall_m: 1.0000
    Epoch 107/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0220 - accuracy: 0.9945 - f1_m: 0.4977 - precision_m: 0.3314 - recall_m: 1.0000
    Epoch 108/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0351 - accuracy: 0.9897 - f1_m: 0.4914 - precision_m: 0.3258 - recall_m: 1.0000
    Epoch 109/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0226 - accuracy: 0.9939 - f1_m: 0.5001 - precision_m: 0.3335 - recall_m: 1.0000
    Epoch 110/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0405 - accuracy: 0.9873 - f1_m: 0.4961 - precision_m: 0.3300 - recall_m: 1.0000
    Epoch 111/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0643 - accuracy: 0.9776 - f1_m: 0.4999 - precision_m: 0.3334 - recall_m: 1.0000
    Epoch 112/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2443 - accuracy: 0.9260 - f1_m: 0.4943 - precision_m: 0.3291 - recall_m: 0.9938
    Epoch 113/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.2145 - accuracy: 0.9357 - f1_m: 0.4766 - precision_m: 0.3133 - recall_m: 0.9976
    Epoch 114/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1905 - accuracy: 0.9436 - f1_m: 0.4858 - precision_m: 0.3213 - recall_m: 0.9964
    Epoch 115/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0631 - accuracy: 0.9770 - f1_m: 0.4989 - precision_m: 0.3325 - recall_m: 1.0000
    Epoch 116/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0348 - accuracy: 0.9921 - f1_m: 0.5002 - precision_m: 0.3336 - recall_m: 1.0000
    Epoch 117/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0271 - accuracy: 0.9897 - f1_m: 0.5006 - precision_m: 0.3340 - recall_m: 1.0000
    Epoch 118/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0231 - accuracy: 0.9939 - f1_m: 0.5022 - precision_m: 0.3354 - recall_m: 1.0000
    Epoch 119/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0240 - accuracy: 0.9945 - f1_m: 0.5003 - precision_m: 0.3337 - recall_m: 1.0000
    Epoch 120/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.0242 - accuracy: 0.9945 - f1_m: 0.5035 - precision_m: 0.3366 - recall_m: 1.0000
    Epoch 121/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0327 - accuracy: 0.9897 - f1_m: 0.5031 - precision_m: 0.3362 - recall_m: 1.0000
    Epoch 122/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0223 - accuracy: 0.9939 - f1_m: 0.4990 - precision_m: 0.3325 - recall_m: 1.0000
    Epoch 123/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0267 - accuracy: 0.9915 - f1_m: 0.4944 - precision_m: 0.3285 - recall_m: 1.0000
    Epoch 124/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0300 - accuracy: 0.9909 - f1_m: 0.4953 - precision_m: 0.3292 - recall_m: 1.0000
    Epoch 125/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0277 - accuracy: 0.9897 - f1_m: 0.4994 - precision_m: 0.3329 - recall_m: 1.0000
    Epoch 126/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0115 - accuracy: 0.9982 - f1_m: 0.4949 - precision_m: 0.3289 - recall_m: 1.0000
    Epoch 127/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0066 - accuracy: 0.9988 - f1_m: 0.4938 - precision_m: 0.3279 - recall_m: 1.0000
    Epoch 128/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0061 - accuracy: 0.9982 - f1_m: 0.4960 - precision_m: 0.3299 - recall_m: 1.0000
    Epoch 129/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0060 - accuracy: 0.9994 - f1_m: 0.4966 - precision_m: 0.3304 - recall_m: 1.0000
    Epoch 130/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0085 - accuracy: 0.9976 - f1_m: 0.5024 - precision_m: 0.3356 - recall_m: 1.0000
    Epoch 131/150
    26/26 [==============================] - 0s 10ms/step - loss: 0.0334 - accuracy: 0.9915 - f1_m: 0.4976 - precision_m: 0.3313 - recall_m: 1.0000
    Epoch 132/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0249 - accuracy: 0.9927 - f1_m: 0.4957 - precision_m: 0.3296 - recall_m: 1.0000
    Epoch 133/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0363 - accuracy: 0.9885 - f1_m: 0.4963 - precision_m: 0.3301 - recall_m: 1.0000
    Epoch 134/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0275 - accuracy: 0.9909 - f1_m: 0.5010 - precision_m: 0.3343 - recall_m: 1.0000
    Epoch 135/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0184 - accuracy: 0.9951 - f1_m: 0.4971 - precision_m: 0.3309 - recall_m: 1.0000
    Epoch 136/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0278 - accuracy: 0.9897 - f1_m: 0.4966 - precision_m: 0.3305 - recall_m: 1.0000
    Epoch 137/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0306 - accuracy: 0.9915 - f1_m: 0.5021 - precision_m: 0.3354 - recall_m: 1.0000
    Epoch 138/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0531 - accuracy: 0.9800 - f1_m: 0.5016 - precision_m: 0.3349 - recall_m: 1.0000
    Epoch 139/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.1176 - accuracy: 0.9600 - f1_m: 0.4884 - precision_m: 0.3232 - recall_m: 1.0000
    Epoch 140/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0770 - accuracy: 0.9739 - f1_m: 0.4870 - precision_m: 0.3220 - recall_m: 1.0000
    Epoch 141/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0575 - accuracy: 0.9788 - f1_m: 0.5079 - precision_m: 0.3405 - recall_m: 1.0000
    Epoch 142/150
    26/26 [==============================] - 0s 13ms/step - loss: 0.1022 - accuracy: 0.9679 - f1_m: 0.4978 - precision_m: 0.3317 - recall_m: 0.9982
    Epoch 143/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.1205 - accuracy: 0.9709 - f1_m: 0.5001 - precision_m: 0.3337 - recall_m: 0.9982
    Epoch 144/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0935 - accuracy: 0.9697 - f1_m: 0.5035 - precision_m: 0.3368 - recall_m: 0.9994
    Epoch 145/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0547 - accuracy: 0.9836 - f1_m: 0.5037 - precision_m: 0.3368 - recall_m: 1.0000
    Epoch 146/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0672 - accuracy: 0.9776 - f1_m: 0.5003 - precision_m: 0.3337 - recall_m: 1.0000
    Epoch 147/150
    26/26 [==============================] - 0s 12ms/step - loss: 0.0519 - accuracy: 0.9818 - f1_m: 0.4970 - precision_m: 0.3308 - recall_m: 1.0000
    Epoch 148/150
    26/26 [==============================] - 0s 11ms/step - loss: 0.0408 - accuracy: 0.9867 - f1_m: 0.5046 - precision_m: 0.3376 - recall_m: 1.0000
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
    18/18 [==============================] - 0s 10ms/step - loss: 0.5412 - accuracy: 0.8326 - f1_m: 0.8123 - precision_m: 0.8992 - recall_m: 0.7418
    Epoch 14/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5393 - accuracy: 0.8180 - f1_m: 0.8132 - precision_m: 0.9021 - recall_m: 0.7428
    Epoch 15/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.5200 - accuracy: 0.8189 - f1_m: 0.8198 - precision_m: 0.8920 - recall_m: 0.7592
    Epoch 16/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5523 - accuracy: 0.8135 - f1_m: 0.7949 - precision_m: 0.8721 - recall_m: 0.7318
    Epoch 17/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4916 - accuracy: 0.8362 - f1_m: 0.8142 - precision_m: 0.8851 - recall_m: 0.7544
    Epoch 18/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5294 - accuracy: 0.8189 - f1_m: 0.8021 - precision_m: 0.8755 - recall_m: 0.7411
    Epoch 19/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4258 - accuracy: 0.8662 - f1_m: 0.8444 - precision_m: 0.9148 - recall_m: 0.7855
    Epoch 20/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.4331 - accuracy: 0.8644 - f1_m: 0.8442 - precision_m: 0.9042 - recall_m: 0.7923
    Epoch 21/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4802 - accuracy: 0.8480 - f1_m: 0.8267 - precision_m: 0.8971 - recall_m: 0.7690
    Epoch 22/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4187 - accuracy: 0.8817 - f1_m: 0.8634 - precision_m: 0.9314 - recall_m: 0.8071
    Epoch 23/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4849 - accuracy: 0.8317 - f1_m: 0.8222 - precision_m: 0.8814 - recall_m: 0.7713
    Epoch 24/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4558 - accuracy: 0.8635 - f1_m: 0.8216 - precision_m: 0.8845 - recall_m: 0.7695
    Epoch 25/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4114 - accuracy: 0.8690 - f1_m: 0.8666 - precision_m: 0.9228 - recall_m: 0.8190
    Epoch 26/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3735 - accuracy: 0.8781 - f1_m: 0.8659 - precision_m: 0.9158 - recall_m: 0.8217
    Epoch 27/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3498 - accuracy: 0.8972 - f1_m: 0.8747 - precision_m: 0.9188 - recall_m: 0.8355
    Epoch 28/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3471 - accuracy: 0.8908 - f1_m: 0.8833 - precision_m: 0.9293 - recall_m: 0.8425
    Epoch 29/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3399 - accuracy: 0.8826 - f1_m: 0.8726 - precision_m: 0.9153 - recall_m: 0.8350
    Epoch 30/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3237 - accuracy: 0.9017 - f1_m: 0.8995 - precision_m: 0.9408 - recall_m: 0.8623
    Epoch 31/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3235 - accuracy: 0.9072 - f1_m: 0.8910 - precision_m: 0.9298 - recall_m: 0.8558
    Epoch 32/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3011 - accuracy: 0.9008 - f1_m: 0.8853 - precision_m: 0.9218 - recall_m: 0.8522
    Epoch 33/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3101 - accuracy: 0.8935 - f1_m: 0.8917 - precision_m: 0.9297 - recall_m: 0.8571
    Epoch 34/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3083 - accuracy: 0.9026 - f1_m: 0.8987 - precision_m: 0.9329 - recall_m: 0.8678
    Epoch 35/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4098 - accuracy: 0.8753 - f1_m: 0.8505 - precision_m: 0.8988 - recall_m: 0.8093
    Epoch 36/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.4193 - accuracy: 0.8662 - f1_m: 0.8614 - precision_m: 0.9136 - recall_m: 0.8161
    Epoch 37/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3726 - accuracy: 0.8854 - f1_m: 0.8774 - precision_m: 0.9326 - recall_m: 0.8295
    Epoch 38/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.3300 - accuracy: 0.8963 - f1_m: 0.8865 - precision_m: 0.9268 - recall_m: 0.8513
    Epoch 39/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3493 - accuracy: 0.8808 - f1_m: 0.8666 - precision_m: 0.9044 - recall_m: 0.8333
    Epoch 40/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3169 - accuracy: 0.8990 - f1_m: 0.8982 - precision_m: 0.9308 - recall_m: 0.8682
    Epoch 41/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.3005 - accuracy: 0.9072 - f1_m: 0.9012 - precision_m: 0.9270 - recall_m: 0.8772
    Epoch 42/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2599 - accuracy: 0.9126 - f1_m: 0.9084 - precision_m: 0.9355 - recall_m: 0.8831
    Epoch 43/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2825 - accuracy: 0.9117 - f1_m: 0.9001 - precision_m: 0.9366 - recall_m: 0.8678
    Epoch 44/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2323 - accuracy: 0.9363 - f1_m: 0.9298 - precision_m: 0.9547 - recall_m: 0.9064
    Epoch 45/50
    18/18 [==============================] - 0s 9ms/step - loss: 0.2391 - accuracy: 0.9217 - f1_m: 0.9143 - precision_m: 0.9393 - recall_m: 0.8913
    Epoch 46/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2422 - accuracy: 0.9327 - f1_m: 0.9262 - precision_m: 0.9480 - recall_m: 0.9057
    Epoch 47/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2727 - accuracy: 0.9190 - f1_m: 0.9136 - precision_m: 0.9378 - recall_m: 0.8909
    Epoch 48/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2746 - accuracy: 0.9154 - f1_m: 0.8903 - precision_m: 0.9222 - recall_m: 0.8612
    Epoch 49/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2720 - accuracy: 0.9136 - f1_m: 0.9055 - precision_m: 0.9319 - recall_m: 0.8814
    Epoch 50/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2581 - accuracy: 0.9290 - f1_m: 0.9219 - precision_m: 0.9494 - recall_m: 0.8961
    9/9 [==============================] - 0s 7ms/step - loss: 0.2432 - accuracy: 0.9291 - f1_m: 0.9217 - precision_m: 0.9495 - recall_m: 0.8957
    Epoch 1/50
    18/18 [==============================] - 1s 10ms/step - loss: 2.3960 - accuracy: 0.2448 - f1_m: 0.1848 - precision_m: 0.3866 - recall_m: 0.1238
    Epoch 2/50
    18/18 [==============================] - 0s 10ms/step - loss: 1.5519 - accuracy: 0.4868 - f1_m: 0.4191 - precision_m: 0.6803 - recall_m: 0.3051
    Epoch 3/50
    18/18 [==============================] - 0s 11ms/step - loss: 1.2996 - accuracy: 0.5469 - f1_m: 0.5026 - precision_m: 0.7195 - recall_m: 0.3874
    Epoch 4/50
    18/18 [==============================] - 0s 10ms/step - loss: 1.0475 - accuracy: 0.6506 - f1_m: 0.5905 - precision_m: 0.7750 - recall_m: 0.4785
    Epoch 5/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.9582 - accuracy: 0.6861 - f1_m: 0.6452 - precision_m: 0.8135 - recall_m: 0.5367
    Epoch 6/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.8490 - accuracy: 0.7243 - f1_m: 0.6807 - precision_m: 0.8263 - recall_m: 0.5799
    Epoch 7/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.7558 - accuracy: 0.7516 - f1_m: 0.7290 - precision_m: 0.8594 - recall_m: 0.6355
    Epoch 8/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.7397 - accuracy: 0.7507 - f1_m: 0.7008 - precision_m: 0.8301 - recall_m: 0.6118
    Epoch 9/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6851 - accuracy: 0.7834 - f1_m: 0.7594 - precision_m: 0.8592 - recall_m: 0.6813
    Epoch 10/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6524 - accuracy: 0.7843 - f1_m: 0.7540 - precision_m: 0.8780 - recall_m: 0.6731
    Epoch 11/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5992 - accuracy: 0.8044 - f1_m: 0.8028 - precision_m: 0.8978 - recall_m: 0.7272
    Epoch 12/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5642 - accuracy: 0.8226 - f1_m: 0.8122 - precision_m: 0.8935 - recall_m: 0.7453
    Epoch 13/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5303 - accuracy: 0.8207 - f1_m: 0.8112 - precision_m: 0.8872 - recall_m: 0.7477
    Epoch 14/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5037 - accuracy: 0.8308 - f1_m: 0.8254 - precision_m: 0.8951 - recall_m: 0.7668
    Epoch 15/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5169 - accuracy: 0.8271 - f1_m: 0.8084 - precision_m: 0.8856 - recall_m: 0.7447
    Epoch 16/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4426 - accuracy: 0.8581 - f1_m: 0.8440 - precision_m: 0.9156 - recall_m: 0.7846
    Epoch 17/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4801 - accuracy: 0.8435 - f1_m: 0.8238 - precision_m: 0.8960 - recall_m: 0.7637
    Epoch 18/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4823 - accuracy: 0.8362 - f1_m: 0.8138 - precision_m: 0.8671 - recall_m: 0.7674
    Epoch 19/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4400 - accuracy: 0.8517 - f1_m: 0.8349 - precision_m: 0.8936 - recall_m: 0.7845
    Epoch 20/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4380 - accuracy: 0.8526 - f1_m: 0.8292 - precision_m: 0.8892 - recall_m: 0.7787
    Epoch 21/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4523 - accuracy: 0.8562 - f1_m: 0.8386 - precision_m: 0.8987 - recall_m: 0.7871
    Epoch 22/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3891 - accuracy: 0.8699 - f1_m: 0.8622 - precision_m: 0.9112 - recall_m: 0.8190
    Epoch 23/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3797 - accuracy: 0.8835 - f1_m: 0.8750 - precision_m: 0.9205 - recall_m: 0.8345
    Epoch 24/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3363 - accuracy: 0.8972 - f1_m: 0.8891 - precision_m: 0.9317 - recall_m: 0.8509
    Epoch 25/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3253 - accuracy: 0.8872 - f1_m: 0.8848 - precision_m: 0.9302 - recall_m: 0.8442
    Epoch 26/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3294 - accuracy: 0.8990 - f1_m: 0.8917 - precision_m: 0.9330 - recall_m: 0.8545
    Epoch 27/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3399 - accuracy: 0.8944 - f1_m: 0.8853 - precision_m: 0.9242 - recall_m: 0.8503
    Epoch 28/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3603 - accuracy: 0.8844 - f1_m: 0.8619 - precision_m: 0.9128 - recall_m: 0.8176
    Epoch 29/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3287 - accuracy: 0.8944 - f1_m: 0.8944 - precision_m: 0.9384 - recall_m: 0.8554
    Epoch 30/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2955 - accuracy: 0.9154 - f1_m: 0.9076 - precision_m: 0.9452 - recall_m: 0.8736
    Epoch 31/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2933 - accuracy: 0.9099 - f1_m: 0.8979 - precision_m: 0.9341 - recall_m: 0.8651
    Epoch 32/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2929 - accuracy: 0.9117 - f1_m: 0.8945 - precision_m: 0.9364 - recall_m: 0.8576
    Epoch 33/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2576 - accuracy: 0.9236 - f1_m: 0.9217 - precision_m: 0.9556 - recall_m: 0.8908
    Epoch 34/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2800 - accuracy: 0.9172 - f1_m: 0.9041 - precision_m: 0.9344 - recall_m: 0.8765
    Epoch 35/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2522 - accuracy: 0.9190 - f1_m: 0.9096 - precision_m: 0.9390 - recall_m: 0.8826
    Epoch 36/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2553 - accuracy: 0.9136 - f1_m: 0.9039 - precision_m: 0.9318 - recall_m: 0.8781
    Epoch 37/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2758 - accuracy: 0.9081 - f1_m: 0.9051 - precision_m: 0.9368 - recall_m: 0.8762
    Epoch 38/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2617 - accuracy: 0.9181 - f1_m: 0.9119 - precision_m: 0.9432 - recall_m: 0.8833
    Epoch 39/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2149 - accuracy: 0.9336 - f1_m: 0.9229 - precision_m: 0.9478 - recall_m: 0.8996
    Epoch 40/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2212 - accuracy: 0.9327 - f1_m: 0.9212 - precision_m: 0.9501 - recall_m: 0.8947
    Epoch 41/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2368 - accuracy: 0.9172 - f1_m: 0.9062 - precision_m: 0.9288 - recall_m: 0.8852
    Epoch 42/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2187 - accuracy: 0.9327 - f1_m: 0.9213 - precision_m: 0.9448 - recall_m: 0.8998
    Epoch 43/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2231 - accuracy: 0.9327 - f1_m: 0.9344 - precision_m: 0.9512 - recall_m: 0.9184
    Epoch 44/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2002 - accuracy: 0.9427 - f1_m: 0.9340 - precision_m: 0.9551 - recall_m: 0.9144
    Epoch 45/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2036 - accuracy: 0.9390 - f1_m: 0.9314 - precision_m: 0.9454 - recall_m: 0.9180
    Epoch 46/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.1936 - accuracy: 0.9436 - f1_m: 0.9362 - precision_m: 0.9479 - recall_m: 0.9248
    Epoch 47/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.1784 - accuracy: 0.9527 - f1_m: 0.9498 - precision_m: 0.9711 - recall_m: 0.9298
    Epoch 48/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2193 - accuracy: 0.9336 - f1_m: 0.9203 - precision_m: 0.9394 - recall_m: 0.9024
    Epoch 49/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2234 - accuracy: 0.9227 - f1_m: 0.9225 - precision_m: 0.9480 - recall_m: 0.8986
    Epoch 50/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2205 - accuracy: 0.9327 - f1_m: 0.9281 - precision_m: 0.9485 - recall_m: 0.9090
    9/9 [==============================] - 0s 8ms/step - loss: 0.2709 - accuracy: 0.9218 - f1_m: 0.9222 - precision_m: 0.9378 - recall_m: 0.9073
    Epoch 1/50
    18/18 [==============================] - 1s 13ms/step - loss: 2.4140 - accuracy: 0.2373 - f1_m: 0.1768 - precision_m: 0.3454 - recall_m: 0.1201
    Epoch 2/50
    18/18 [==============================] - 0s 13ms/step - loss: 1.5774 - accuracy: 0.4773 - f1_m: 0.4148 - precision_m: 0.6436 - recall_m: 0.3073
    Epoch 3/50
    18/18 [==============================] - 0s 11ms/step - loss: 1.3379 - accuracy: 0.5609 - f1_m: 0.5181 - precision_m: 0.7088 - recall_m: 0.4100
    Epoch 4/50
    18/18 [==============================] - 0s 11ms/step - loss: 1.2141 - accuracy: 0.5936 - f1_m: 0.5482 - precision_m: 0.7343 - recall_m: 0.4398
    Epoch 5/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.9929 - accuracy: 0.6564 - f1_m: 0.6145 - precision_m: 0.7995 - recall_m: 0.5026
    Epoch 6/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.9217 - accuracy: 0.7018 - f1_m: 0.6687 - precision_m: 0.8152 - recall_m: 0.5689
    Epoch 7/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.8651 - accuracy: 0.7127 - f1_m: 0.6837 - precision_m: 0.8196 - recall_m: 0.5877
    Epoch 8/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.7849 - accuracy: 0.7345 - f1_m: 0.7135 - precision_m: 0.8447 - recall_m: 0.6201
    Epoch 9/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.7811 - accuracy: 0.7518 - f1_m: 0.7360 - precision_m: 0.8629 - recall_m: 0.6438
    Epoch 10/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.7174 - accuracy: 0.7682 - f1_m: 0.7498 - precision_m: 0.8532 - recall_m: 0.6698
    Epoch 11/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6344 - accuracy: 0.8018 - f1_m: 0.7821 - precision_m: 0.8684 - recall_m: 0.7121
    Epoch 12/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.6100 - accuracy: 0.8055 - f1_m: 0.7758 - precision_m: 0.8941 - recall_m: 0.6916
    Epoch 13/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5554 - accuracy: 0.8182 - f1_m: 0.8156 - precision_m: 0.9045 - recall_m: 0.7436
    Epoch 14/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5681 - accuracy: 0.8245 - f1_m: 0.7991 - precision_m: 0.8983 - recall_m: 0.7208
    Epoch 15/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.5310 - accuracy: 0.8255 - f1_m: 0.8136 - precision_m: 0.8949 - recall_m: 0.7465
    Epoch 16/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.4709 - accuracy: 0.8509 - f1_m: 0.8390 - precision_m: 0.9035 - recall_m: 0.7839
    Epoch 17/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4972 - accuracy: 0.8400 - f1_m: 0.8257 - precision_m: 0.8888 - recall_m: 0.7714
    Epoch 18/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.4470 - accuracy: 0.8509 - f1_m: 0.8416 - precision_m: 0.9115 - recall_m: 0.7827
    Epoch 19/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4523 - accuracy: 0.8536 - f1_m: 0.8384 - precision_m: 0.9105 - recall_m: 0.7778
    Epoch 20/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4393 - accuracy: 0.8718 - f1_m: 0.8356 - precision_m: 0.9072 - recall_m: 0.7792
    Epoch 21/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3992 - accuracy: 0.8727 - f1_m: 0.8681 - precision_m: 0.9134 - recall_m: 0.8284
    Epoch 22/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4070 - accuracy: 0.8736 - f1_m: 0.8607 - precision_m: 0.9148 - recall_m: 0.8134
    Epoch 23/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.4272 - accuracy: 0.8673 - f1_m: 0.8555 - precision_m: 0.9092 - recall_m: 0.8087
    Epoch 24/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3346 - accuracy: 0.9045 - f1_m: 0.8821 - precision_m: 0.9250 - recall_m: 0.8435
    Epoch 25/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.4164 - accuracy: 0.8645 - f1_m: 0.8632 - precision_m: 0.9162 - recall_m: 0.8166
    Epoch 26/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3746 - accuracy: 0.8809 - f1_m: 0.8831 - precision_m: 0.9234 - recall_m: 0.8469
    Epoch 27/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3528 - accuracy: 0.8864 - f1_m: 0.8777 - precision_m: 0.9175 - recall_m: 0.8417
    Epoch 28/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3558 - accuracy: 0.8864 - f1_m: 0.8738 - precision_m: 0.9181 - recall_m: 0.8339
    Epoch 29/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3633 - accuracy: 0.8900 - f1_m: 0.8688 - precision_m: 0.9182 - recall_m: 0.8255
    Epoch 30/50
    18/18 [==============================] - 0s 12ms/step - loss: 0.3646 - accuracy: 0.8891 - f1_m: 0.8789 - precision_m: 0.9229 - recall_m: 0.8400
    Epoch 31/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3161 - accuracy: 0.8991 - f1_m: 0.8932 - precision_m: 0.9294 - recall_m: 0.8605
    Epoch 32/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.3385 - accuracy: 0.9045 - f1_m: 0.8841 - precision_m: 0.9321 - recall_m: 0.8435
    Epoch 33/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3570 - accuracy: 0.8909 - f1_m: 0.8859 - precision_m: 0.9192 - recall_m: 0.8556
    Epoch 34/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3020 - accuracy: 0.9036 - f1_m: 0.8901 - precision_m: 0.9370 - recall_m: 0.8513
    Epoch 35/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3074 - accuracy: 0.9036 - f1_m: 0.8928 - precision_m: 0.9273 - recall_m: 0.8611
    Epoch 36/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.3076 - accuracy: 0.9036 - f1_m: 0.8932 - precision_m: 0.9204 - recall_m: 0.8678
    Epoch 37/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2591 - accuracy: 0.9273 - f1_m: 0.9176 - precision_m: 0.9440 - recall_m: 0.8929
    Epoch 38/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2494 - accuracy: 0.9264 - f1_m: 0.9137 - precision_m: 0.9497 - recall_m: 0.8817
    Epoch 39/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2621 - accuracy: 0.9164 - f1_m: 0.9132 - precision_m: 0.9412 - recall_m: 0.8874
    Epoch 40/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2501 - accuracy: 0.9227 - f1_m: 0.9077 - precision_m: 0.9339 - recall_m: 0.8834
    Epoch 41/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2563 - accuracy: 0.9173 - f1_m: 0.9083 - precision_m: 0.9370 - recall_m: 0.8819
    Epoch 42/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2498 - accuracy: 0.9300 - f1_m: 0.9097 - precision_m: 0.9389 - recall_m: 0.8831
    Epoch 43/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2505 - accuracy: 0.9145 - f1_m: 0.9060 - precision_m: 0.9327 - recall_m: 0.8811
    Epoch 44/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2817 - accuracy: 0.9091 - f1_m: 0.8878 - precision_m: 0.9166 - recall_m: 0.8617
    Epoch 45/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2664 - accuracy: 0.9173 - f1_m: 0.9009 - precision_m: 0.9297 - recall_m: 0.8750
    Epoch 46/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2624 - accuracy: 0.9173 - f1_m: 0.9008 - precision_m: 0.9331 - recall_m: 0.8721
    Epoch 47/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2874 - accuracy: 0.9127 - f1_m: 0.9136 - precision_m: 0.9485 - recall_m: 0.8817
    Epoch 48/50
    18/18 [==============================] - 0s 10ms/step - loss: 0.2700 - accuracy: 0.9145 - f1_m: 0.9094 - precision_m: 0.9343 - recall_m: 0.8863
    Epoch 49/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2467 - accuracy: 0.9309 - f1_m: 0.9248 - precision_m: 0.9526 - recall_m: 0.8990
    Epoch 50/50
    18/18 [==============================] - 0s 11ms/step - loss: 0.2393 - accuracy: 0.9255 - f1_m: 0.9217 - precision_m: 0.9462 - recall_m: 0.8987
    9/9 [==============================] - 0s 7ms/step - loss: 0.2527 - accuracy: 0.9089 - f1_m: 0.9042 - precision_m: 0.9200 - recall_m: 0.8892
    Epoch 1/50
    26/26 [==============================] - 1s 15ms/step - loss: 2.2808 - accuracy: 0.2644 - f1_m: 0.1981 - precision_m: 0.3960 - recall_m: 0.1344
    Epoch 2/50
    26/26 [==============================] - 0s 13ms/step - loss: 1.3217 - accuracy: 0.5403 - f1_m: 0.5068 - precision_m: 0.7184 - recall_m: 0.3937
    Epoch 3/50
    26/26 [==============================] - 0s 12ms/step - loss: 1.1026 - accuracy: 0.6374 - f1_m: 0.5962 - precision_m: 0.7739 - recall_m: 0.4863
    Epoch 4/50
    26/26 [==============================] - 0s 13ms/step - loss: 0.9025 - accuracy: 0.6883 - f1_m: 0.6622 - precision_m: 0.8110 - recall_m: 0.5610
    Epoch 5/50
    26/26 [==============================] - 0s 13ms/step - loss: 0.7684 - accuracy: 0.7508 - f1_m: 0.7352 - precision_m: 0.8539 - recall_m: 0.6473
    Epoch 6/50
    26/26 [==============================] - 0s 13ms/step - loss: 0.7019 - accuracy: 0.7720 - f1_m: 0.7477 - precision_m: 0.8678 - recall_m: 0.6579
    Epoch 7/50
    26/26 [==============================] - 0s 13ms/step - loss: 0.5997 - accuracy: 0.8023 - f1_m: 0.7881 - precision_m: 0.8804 - recall_m: 0.7148
    Epoch 8/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.5368 - accuracy: 0.8405 - f1_m: 0.8187 - precision_m: 0.8956 - recall_m: 0.7548
    Epoch 9/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.4949 - accuracy: 0.8387 - f1_m: 0.8265 - precision_m: 0.9035 - recall_m: 0.7625
    Epoch 10/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.4424 - accuracy: 0.8563 - f1_m: 0.8504 - precision_m: 0.9186 - recall_m: 0.7924
    Epoch 11/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.4180 - accuracy: 0.8660 - f1_m: 0.8626 - precision_m: 0.9195 - recall_m: 0.8129
    Epoch 12/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.3962 - accuracy: 0.8787 - f1_m: 0.8669 - precision_m: 0.9202 - recall_m: 0.8199
    Epoch 13/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.3587 - accuracy: 0.8854 - f1_m: 0.8811 - precision_m: 0.9294 - recall_m: 0.8379
    Epoch 14/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.3194 - accuracy: 0.9018 - f1_m: 0.8906 - precision_m: 0.9345 - recall_m: 0.8512
    Epoch 15/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.3170 - accuracy: 0.9024 - f1_m: 0.8995 - precision_m: 0.9398 - recall_m: 0.8628
    Epoch 16/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.2833 - accuracy: 0.9206 - f1_m: 0.9095 - precision_m: 0.9439 - recall_m: 0.8779
    Epoch 17/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.2756 - accuracy: 0.9230 - f1_m: 0.9185 - precision_m: 0.9491 - recall_m: 0.8903
    Epoch 18/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.2698 - accuracy: 0.9193 - f1_m: 0.9155 - precision_m: 0.9461 - recall_m: 0.8874
    Epoch 19/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.2545 - accuracy: 0.9284 - f1_m: 0.9254 - precision_m: 0.9523 - recall_m: 0.9003
    Epoch 20/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.2296 - accuracy: 0.9345 - f1_m: 0.9286 - precision_m: 0.9547 - recall_m: 0.9044
    Epoch 21/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.2217 - accuracy: 0.9394 - f1_m: 0.9315 - precision_m: 0.9552 - recall_m: 0.9094
    Epoch 22/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.2181 - accuracy: 0.9333 - f1_m: 0.9303 - precision_m: 0.9537 - recall_m: 0.9083
    Epoch 23/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.2049 - accuracy: 0.9424 - f1_m: 0.9413 - precision_m: 0.9605 - recall_m: 0.9231
    Epoch 24/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.2039 - accuracy: 0.9388 - f1_m: 0.9370 - precision_m: 0.9571 - recall_m: 0.9179
    Epoch 25/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1920 - accuracy: 0.9436 - f1_m: 0.9450 - precision_m: 0.9652 - recall_m: 0.9259
    Epoch 26/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1759 - accuracy: 0.9509 - f1_m: 0.9462 - precision_m: 0.9659 - recall_m: 0.9275
    Epoch 27/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1867 - accuracy: 0.9400 - f1_m: 0.9388 - precision_m: 0.9577 - recall_m: 0.9210
    Epoch 28/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1638 - accuracy: 0.9527 - f1_m: 0.9474 - precision_m: 0.9624 - recall_m: 0.9330
    Epoch 29/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1570 - accuracy: 0.9527 - f1_m: 0.9510 - precision_m: 0.9619 - recall_m: 0.9404
    Epoch 30/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1558 - accuracy: 0.9594 - f1_m: 0.9553 - precision_m: 0.9683 - recall_m: 0.9428
    Epoch 31/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1467 - accuracy: 0.9588 - f1_m: 0.9578 - precision_m: 0.9697 - recall_m: 0.9464
    Epoch 32/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1419 - accuracy: 0.9636 - f1_m: 0.9588 - precision_m: 0.9685 - recall_m: 0.9496
    Epoch 33/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1570 - accuracy: 0.9545 - f1_m: 0.9519 - precision_m: 0.9659 - recall_m: 0.9384
    Epoch 34/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1340 - accuracy: 0.9612 - f1_m: 0.9604 - precision_m: 0.9720 - recall_m: 0.9493
    Epoch 35/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1331 - accuracy: 0.9612 - f1_m: 0.9612 - precision_m: 0.9726 - recall_m: 0.9504
    Epoch 36/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1232 - accuracy: 0.9642 - f1_m: 0.9618 - precision_m: 0.9696 - recall_m: 0.9541
    Epoch 37/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1404 - accuracy: 0.9588 - f1_m: 0.9573 - precision_m: 0.9665 - recall_m: 0.9484
    Epoch 38/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1106 - accuracy: 0.9703 - f1_m: 0.9689 - precision_m: 0.9782 - recall_m: 0.9598
    Epoch 39/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1093 - accuracy: 0.9709 - f1_m: 0.9689 - precision_m: 0.9754 - recall_m: 0.9626
    Epoch 40/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1124 - accuracy: 0.9721 - f1_m: 0.9686 - precision_m: 0.9776 - recall_m: 0.9598
    Epoch 41/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1153 - accuracy: 0.9660 - f1_m: 0.9658 - precision_m: 0.9738 - recall_m: 0.9580
    Epoch 42/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.1004 - accuracy: 0.9739 - f1_m: 0.9733 - precision_m: 0.9800 - recall_m: 0.9668
    Epoch 43/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.0996 - accuracy: 0.9679 - f1_m: 0.9694 - precision_m: 0.9738 - recall_m: 0.9651
    Epoch 44/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1093 - accuracy: 0.9666 - f1_m: 0.9671 - precision_m: 0.9737 - recall_m: 0.9608
    Epoch 45/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.0840 - accuracy: 0.9782 - f1_m: 0.9765 - precision_m: 0.9826 - recall_m: 0.9706
    Epoch 46/50
    26/26 [==============================] - 0s 10ms/step - loss: 0.1044 - accuracy: 0.9709 - f1_m: 0.9738 - precision_m: 0.9791 - recall_m: 0.9686
    Epoch 47/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.0994 - accuracy: 0.9721 - f1_m: 0.9732 - precision_m: 0.9785 - recall_m: 0.9680
    Epoch 48/50
    26/26 [==============================] - 0s 11ms/step - loss: 0.0965 - accuracy: 0.9721 - f1_m: 0.9736 - precision_m: 0.9810 - recall_m: 0.9663
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

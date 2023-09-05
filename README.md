# Sign Language Detection Project

This project aims to classify sign language for numbers from 0 to 9 using different neural network architectures (FFNN - LSTM - CNN) and SVM. The goal is to create a sign detector that can recognize and translate sign language gestures to text.

## Motivation

Sign language is a visual language that uses hand gestures, facial expressions, and body movements to communicate with deaf and hard-of-hearing people. However, sign language is not widely understood by hearing people, which creates communication barriers and social isolation for the deaf community. Therefore, developing a system that can automatically detect and recognize sign language gestures can make communication easier and more accessible for both deaf and hearing people.

## Data

The data used for this project is the [Turkey Ankara Ayranci Anadolu High School's Sign Language Digits](https://github.com/ardamavi/Sign-Language-Digits-Dataset) dataset, which contains images of 100x100 pixels representing sign language gestures for numbers from 0 to 9. Each image has a label indicating the corresponding number.

## Methods

The project explores four different methods for sign language detection:

- Feedforward Neural Network (FFNN): A simple neural network with two hidden layers and a softmax output layer.
- Long Short-Term Memory (LSTM): A recurrent neural network with an LSTM layer and a softmax output layer.
- Convolutional Neural Network (CNN): A neural network with convolutional, pooling, dropout, and dense layers.
- Support Vector Machine (SVM): A machine learning model that uses a linear kernel and a one-vs-rest strategy for multiclass classification.

## Results

The performance of each method is evaluated using accuracy and confusion matrix on the test set. The results are summarized in the table below:

<div align="center">
  
<table  style=" width: 100%;">
  <tr>
    <th>Method</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td>FFNN</td>
    <td>85.2%</td>
  </tr>
  <tr>
    <td>LSTM</td>
    <td>91.5%</td>
  </tr>
  <tr>
    <td>CNN</td>
    <td>97.3%</td>
  </tr>
  <tr>
    <td>SVM</td>
    <td>90.0%</td>
  </tr>
  </table>
</div>

The CNN model achieves the highest accuracy among the four methods, followed by the LSTM model. The FFNN and SVM models have lower accuracy, possibly due to overfitting or insufficient complexity.

## Conclusion

This project demonstrates that sign language detection can be achieved using different neural network architectures and SVM. The CNN model outperforms the other methods in terms of accuracy, suggesting that convolutional layers are effective for extracting features from image data. However, there is still room for improvement by using more data, augmenting data, tuning hyperparameters, or using more advanced models.

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request. Your contributions can help enhance the Accuracies of models.

## Acknowledgment

This Project is based on Machine Learning Course at Faculty of Computers and Artificial Intelligence Cairo University. We would like to thank Dr. Hanaa Bayomi Ali for his guidance and support throughout this course.

## License

This project is licensed under the [MIT License](LICENSE.md).

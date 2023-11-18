(work in progress)
A comprehensive pipeline for music genre classification using machine learning. This repository includes data loading, audio feature extraction, model training with a convolutional neural network (CNN), and prediction for classifying music genres. Features include Short-Time Fourier Transform (STFT), Spectral Rolloff, Chroma features, and more. The trained model is saved for easy integration into other applications. Explore and enhance your understanding of audio signal processing and deep learning in the context of music genre classification.

The main aim is to create a machine learning model, which classifies music samples into different genres. It aims to predict the genre using an audio signal as its input.

Librosa is a python package for music and audio analysis. It provides the building blocks necessary to create music information retrieval systems. By using Librosa, we can extract certain key features from the audio samples such as Tempo, Chroma Energy Normalized,
Mel-Freqency Cepstral Coefficients, Spectral Centroid, Spectral Contrast, Spectral Rolloff, and Zero Crossing Rate.

Dataset used: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

Methods used for visualizing the audio files:

1. Plot Raw Wave Files: Waveforms are visual representations of sound as time on the x-axis and amplitude on the y-axis.
   They are great for allowing us to quickly scan the audio data and visually compare and contrast which genres might be more similar than others.

2. Spectrograms:
   A spectrogram is a visual way of representing the signal loudness of a signal over time at various frequencies present in a particular waveform. Not only can one see whether there is more or less energy at, for example, 2 Hz vs 10 Hz, but one can also see how energy levels vary over time.
   Spectrograms are sometimes called sonographs, voiceprints, or voicegrams. When the data is represented in a 3D plot, they may be called waterfalls. In 2-dimensional arrays, the first axis is frequency while the second axis is time.

3. Spectral Rolloff:
   Spectral Rolloff is the frequency below which a specified percentage of the total spectral energy lies ex: below 85%
   'librosa.feature.spectral_rolloff' computes the rolloff frequency for each frame in a signal

4. Chroma Feature:
   It is a powerful tool for analyzing music features whose pitches can be meaningfully categorized and whose tuning approximates to the equal-tempered scale.
   One main property of chroma features is that they capture harmonic and melodic characteristics of music while being robust to changes in timbre and instrumentation.

5. Zero Crossing Rate:
   Zero crossing is said to occur if successive samples have different algebraic signs. The rate at which zero-crossings occur is a simple measure of the frequency content of a signal.
   Zero-crossing rate is a measure of the number of times in a given time interval/frame that the amplitude of the speech signals passes through a value of zero.

For the CNN model, we had used the Adam optimizer for training the model. The epoch that was chosen for the training model is 600.
All of the hidden layers are using the RELU activation function and the output layer uses the softmax function. The loss is calculated using the sparse_categorical_crossentropy function.
Dropout is used to prevent overfitting.
We chose the Adam optimizer because it gave us the best results after evaluating other optimizers.
The model accuracy can be increased by further increasing the epochs but after a certain period, we may achieve a threshold, so the value should be determined accordingly.
The accuracy we achieved for the test set is 92.93 percent which is very decent.
So we come to the conclusion that Neural Networks are very effective in machine learning models. Tensorflow is very useful in implementing Convolutional Neural Network (CNN) that helps in the classifying process.

explanation:
the code should identify the bird songs and and map it to the respective bird

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import IPython.display as ipd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load audio file
filepath = "C:/Users/RUTUJA/MSc sem 3 project/train audio dataset/amekes/XC16111.mp3"
ipd.Audio(filepath)

# Load audio data using librosa
data, sample_rate = librosa.load(filepath)

# Display waveform
plt.figure(figsize=(10, 5))
librosa.display.waveshow(data, sr=sample_rate)

# Load metadata
metadata = pd.read_csv('C:/Users/RUTUJA/MSc sem 3 project/train data.csv')
metadata.head(10)

# Extract MFCC features
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Extract features from all audio files
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = os.path.join("C:/Users/RUTUJA/MSc sem 3 project/train audio dataset", '/',
                             row["primary_label"], row["filename"])
    final_class_labels = row["primary_label"]
    #data = features_extractor(file_name)
    extracted_features.append([data, final_class_labels])

# Convert extracted_features to Pandas dataframe
extracted_features_df = pd.DataFrame(extracted_features, columns=['feature', 'primary_label'])

# Split dataset into independent and dependent datasets
x_mfcc = np.array(extracted_features_df['feature'].tolist())
y = np.array(extracted_features_df['primary_label'])

# Flatten the 2D MFCC matrix into 1D vectors
x_flat = [mfcc.flatten() for mfcc in x_mfcc]

# Convert the flattened features to a NumPy array
x = np.array(x_flat)

# Label Encoding
labelencoder = LabelEncoder()
y = to_categorical(labelencoder.fit_transform(y))

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# No of classes
num_labels = y.shape[1]

# Define the model
model = Sequential()
model.add(Dense(100, input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

model.summary()

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='./audio_classification.hdf5', verbose=1, save_best_only=True)

model.fit(x_train, y_train, batch_size=num_batch_size, epochs=num_epochs, callbacks=[checkpointer], verbose=1)

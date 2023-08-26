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


#Load metadata
metadata = pd.read_csv('C:/Users/RUTUJA/MSc sem 3 project/train data.csv')
print(metadata.head(10))


# Extract MFCC features
def features_extractor(file):
    audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features

# Extract features from all audio files
extracted_features = []
for index_num, row in tqdm(metadata.iterrows()):
    file_name = "C:/Users/RUTUJA/MSc sem 3 project/train audio dataset"+ "/"+ row["primary_label"]+ "/" + row["filename"]
    final_class_labels = row["primary_label"]

    from os.path import exists
    file_exists = exists(file_name)
    #print(file_name, "exist")
    if (file_exists):
        data = features_extractor(file_name)
        extracted_features.append([data, final_class_labels])

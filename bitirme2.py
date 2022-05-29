import pandas as pad 
import numpy as nup
import glob
import soundfile
import os
import sys

import librosa
import librosa.display
import seaborn as sbn
import matplotlib.pyplot as mplt
from sklearn import metrics

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from IPython.display import Audio

import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

RavdessData = "D:\\dataset\\audio_speech_actors_01-24\\"

ravdessDirectoryList = os.listdir(RavdessData)
fileEmotion = []
filePath = []
for dir in ravdessDirectoryList:
    actor = os.listdir(RavdessData + dir)
    for file in actor:
        part = file.split('.')[0]
        part = file.split('-')
        fileEmotion.append(int(part[2]))
        filePath.append(RavdessData + dir + '/' + file)
emotion_df = pad.DataFrame(fileEmotion, columns=['Emotions'])
path_df = pad.DataFrame(filePath, columns=['Path'])
Ravdess_df = pad.concat([emotion_df, path_df], axis = 1)

Ravdess_df.Emotions.replace({1:'neutral', 2:'calm', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8: 'surprised'})
Ravdess_df.head();

dataPath = pad.concat([Ravdess_df], axis = 0)
dataPath.to_csv("data_path.csv",index=False)
dataPath.head()

mplt.title('Count of Emotions', size=16)
sbn.countplot(dataPath.Emotions)
mplt.ylabel('Count', size=12)
mplt.xlabel('Emotions', size=12)
sbn.despine(top=True, right=True, left=False, bottom= False)
mplt.show()


# def createWaveplot(data, sr, e):
#     mplt.figure(figsize=(10, 3))
#     mplt.title('WAveplot for audio with {} emotion'.format(e), size = 15)
#     librosa.display.waveplot(data, sr=sr)
#     mplt.show()
    
# def createSpectrogram(data, sr, e):
#     X = librosa.stft(data)
#     Xdb = librosa.amplitude_to_db(abs(X))
#     mplt.figure(figsize = (12, 3))
#     mplt.title('Spectrogram for audio with {} emotion'.format(e), size = 15)
#     librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
#     mplt.colorbar()
    
# emotion = 'fear'
# path = nup.array(dataPath.Path[dataPath.Emotions==emotion])[1]
# data, samplingRate = librosa.load(path)
# createWaveplot(data, samplingRate, emotion)
# createSpectrogram(data, samplingRate, emotion)
# Audio(path)

def noise(data):
    noiseAmp = 0.035*nup.random.uniform()*nup.amax(data)
    data = data + noiseAmp*nup.random.normal(size = data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shiftRange = int(nup.random.uniform(low=-5, high = 5)*1000)
    return nup.roll(data, shiftRange)

def pitch(data, samplingRate, pitchFactor = 0.7):
    return librosa.effects.pitch_shift(data, samplingRate, pitchFactor)

path= nup.array(dataPath.Path)[1]
data, sampleRate = librosa.load(path)

x = noise(data)
mplt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sampleRate)
Audio(x, rate= sampleRate)

x = stretch(data)
mplt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sampleRate)
Audio(x, rate= sampleRate)

x = shift(data)
mplt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sampleRate)
Audio(x, rate= sampleRate)

x = pitch(data, sampleRate)
mplt.figure(figsize=(14,4))
librosa.display.waveshow(y=x, sr=sampleRate)
Audio(x, rate= sampleRate)

def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate= sound_file.samplerate
        if chroma:
            stft=nup.abs(librosa.stft(X))
        result=nup.array([])
       
        if mfcc:
            mfccs=nup.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=10).T, axis=0)
            result=nup.hstack((result, mfccs))
        if chroma:
            chroma=nup.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=nup.hstack((result, chroma))
        if mel:
            mel=nup.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=nup.hstack((result, mel))
    return result

#DataFlair - Emotions in the RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
#DataFlair - Emotions to observe
observed_emotions=['calm', 'happy', 'fearful', 'disgust', 'angry',
                   'sad', 'neutral', 'surprised']

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob("D:\\dataset\\Actor_*\\*.wav"):

        file_name=os.path.basename(file)
        emotion1=emotions[file_name.split("-")[2]]
        print(file_name)
        if emotion1 not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
     
        x.append(feature)
        y.append(emotion1)
    return train_test_split(nup.array(x), y, test_size=test_size, random_state=9)

#DataFlair - Get the shape of the training and testing datasets

xTrain,xTest,yTrain,yTest = load_data(test_size = 0.25)
print((xTrain.shape[0], xTest.shape[0]))

print('Features extracted:', {xTrain.shape[1]})

model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), 
                    learning_rate='adaptive', max_iter=500)
print(xTrain)
print(yTrain)
model.fit(xTrain, yTrain)

expected_Of_y = yTest
yPred = model.predict(xTest)

print(metrics.confusion_matrix(expected_Of_y, yPred))

print(classification_report(yTest, yPred))

accuracy = accuracy_score(y_true= yTest, y_pred = yPred)
print("Accuracy: {:.2f}%".format(accuracy*100))
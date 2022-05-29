# https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/
# librosa is python library for analyzing audio and music.
# libs: librosa, soundfile, sklearn to build a model using an MLPClassifier.

# pip install librosa soundfile numpy sklearn pyaudio

import librosa
import soundfile
import os, glob, pickle 
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import random

#Öznitelik Çıkarımı
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        return result
    
#Duygular
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fear',
  '07':'disgust',
  '08':'surprise'
}

# Gözlemlenecek Duygular
observed_emotions=['calm', 'happy', 'fear', 'disgust', 'angry',
                   'sad', 'neutral', 'surprise']

# Her ses dosyası için verileri yükleyip öznitelik çıkarımı yapıyoruz.
def load_data(test_size=0.2):
    x,y=[],[]
    # RAVDESS DATASET
    for file in glob.glob("D:\\dataset\\Actor_*\\*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
        
    # TESS DATASET
    Tess = 'D:\dataset\TESS Toronto emotional speech set data\\'
    tess_directory_list = os.listdir(Tess)
    print(tess_directory_list)
    for dir in tess_directory_list:
        directories = os.listdir(Tess + dir)
        
        for file in directories:
            part = file.split('.')[0]
            part = part.split('_')[2]
            if part=='ps':
                y.append('surprise')
            else:
                y.append(part)
            feature=extract_feature(Tess +dir+ '\\' + file, mfcc=True, chroma=True, mel=True)
            x.append(feature)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


x_train,x_test,y_train,y_test=load_data()

#DataFlair - Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))

#DataFlair - Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

#Multi Layer Perceptron Classifier ile modelimizi kuruyoruz.
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, 
                    hidden_layer_sizes=(300,), learning_rate='adaptive', 
                    max_iter=500)


#DataFlair - Train the model
model.fit(x_train,y_train)

#DataFlair - Predict for the test set
y_pred=model.predict(x_test)

print(metrics.confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))

#DataFlair - Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
#DataFlair - Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))


# Model'i Kaydediyoruz
Pkl_Filename = "karisik.pkl"  

with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(model, file)
    

for i in range(1, 5):
    index = random.randint(0, 846)
    predict = model.predict(x_test)
    print("index:", index)
    print("Tahmin: ",predict[index])
    print("Cevap: ", y_test[index])
    print("----------------------")
    

# Load the Model back from file
# with open(Pkl_Filename, 'rb') as file:  
#     Pickled_LR_Model = pickle.load(file)

# Pickled_LR_Model


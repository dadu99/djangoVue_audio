from django.shortcuts import render
from django.http import JsonResponse
from fileuploads.forms import UploadForm
import json
from django.http import JsonResponse, HttpResponse
import librosa
import numpy as np
from keras.models import load_model
import os
import cv2


# Create your views here.
def index(request):
    form = UploadForm()
    
    return render(request, 'document/index.html', {'form' : form})

def upload(request):
    if request.FILES:
        form = UploadForm(request.POST, request.FILES)

        if form.is_valid():
            form.save()
    
    return JsonResponse({'success': True})

def proceedAudio(request):
    selectionObj = json.loads(request.body)
    path="D:/1Disertatie/SoundClassificationSystem/djangoVue_audio/audio/media/uploads/"+selectionObj["filename"]
    if(selectionObj["sound_type"] == 'environmental'):
        if(selectionObj["sound_representation"] == 'raw'):
           model= load_model('C:/Users/User/OneDrive/Desktop/Disertatie/1D_CNN/raw_audio3.h5')
           classindex= predict_audio_classRaw(path,model)         
        if(selectionObj["sound_representation"] == 'mfcc'):
           model= load_model('C:/Users/User/OneDrive/Desktop/Disertatie/CNN/BestModels/Urban/1DCNN_mffc_100Epochs91.76acc.h5')
           classindex= predict_audio_classMFCC(path,model)
        if(selectionObj["sound_representation"] == 'vgg'):
           model= load_model('D:/1Disertatie/SoundClassificationSystem/models/Urban/VGG_UrbanPretrainedIMageNet3.h5')
           classindex= predict_audio_classVGG(path,model)
        if(selectionObj["sound_representation"] == 'imageProcesing'):
           if(selectionObj["image_representation"] == 'mel'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/Urban/CNN_melSpec_100Acc0.87.h5')
              classindex= predict_audio_classIMG_MEL(path,model)
           if(selectionObj["image_representation"] == 'chroma'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/Urban/CNN_chromaCQT_100Acc0.62.h5')
              classindex= predict_audio_classIMG_CQT(path,model)
           if(selectionObj["image_representation"] == 'chromaCens'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/Urban/CNN_chromaCENS_100Acc0.5787.h5')
              classindex= predict_audio_classIMG_CENS(path,model)
        return JsonResponse({'environmental' : str(classindex)})
    if(selectionObj["sound_type"] == 'music'):        
        if(selectionObj["sound_representation"] == 'mfcc'):
           model= load_model('D:/1Disertatie/SoundClassificationSystem/models/fma/fma_mfcc.h5')
           classindex= predict_audio_classMFCC(path,model)
        if(selectionObj["sound_representation"] == 'imageProcesing'):
           if(selectionObj["image_representation"] == 'mel'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/fma/fma_melspec3.h5')
              classindex= predict_audio_classIMG_MEL2(path,model)
           if(selectionObj["image_representation"] == 'chroma'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/fma/CNN_fma_cqt_Acc0.2962.h5')
              classindex= predict_audio_classIMG_CQT(path,model)
           if(selectionObj["image_representation"] == 'chromaCens'):
              model= load_model('D:/1Disertatie/SoundClassificationSystem/models/fma/CNN_fma_cens_Acc0.2812.h5')
              classindex= predict_audio_classIMG_CENS(path,model)
        return JsonResponse({'music' : str(classindex)})

def predict_audio_classRaw(file_path, modelRaw):
        TARGET_SR=8000
        AUDIO_LENGTH=32000
        rawData=[];
        audioRaw, _ = librosa.load(file_path, sr=TARGET_SR, mono=True)
        audioRaw = audioRaw.reshape(-1, 1)
         # normalize mean 0, variance 1
        audioRaw = (audioRaw - np.mean(audioRaw)) / np.std(audioRaw)
        original_length = len(audioRaw)
        if  original_length < AUDIO_LENGTH:
            audioRaw=np.concatenate((audioRaw, np.zeros(shape=(AUDIO_LENGTH - original_length, 1))))
            print('PAD New length =', len(audioRaw))
        
        else:
            if  original_length > AUDIO_LENGTH:
                audioRaw = audioRaw[0:AUDIO_LENGTH]
                print('CUT New length =', len(audioRaw))

        rawData.append(audioRaw);
        rawData= np.array(rawData)
        predicted_vector = modelRaw.predict(rawData)
        predicted_class_index = np.argmax(predicted_vector, axis=-1)
        return predicted_class_index[0]

def predict_audio_classMFCC(file_path, modelMFCC):
    # Load the audio file and resample it
    target_sr = 22050
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr,n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    # Reshape the features to fit the input shape of the model
    features = mfccs_scaled.reshape(1, mfccs_scaled.shape[0], 1)

    # Predict the class
    predicted_vector = modelMFCC.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)

    # Decode the class index to its corresponding label
    #predicted_class = le.inverse_transform(predicted_class_index)


    return predicted_class_index[0]

def predict_audio_classVGG(file_path, model):
    # Load the audio file and resample it
    target_sr = 22050
    features = [];
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')
    spectogram = librosa.feature.melspectrogram(y=audio, sr=sr,n_mels=128)
    spectogram = librosa.power_to_db(spectogram,ref=np.max)
    spectrogram = (255 * (spectogram - np.min(spectogram)) / np.ptp(spectogram)).astype(np.uint8)
    spectogram = cv2.resize(spectogram, (256,256)) 
    spectrogram_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    spectrogram_rgb[:, :, 0] = spectogram
    spectrogram_rgb[:, :, 1] = spectogram
    spectrogram_rgb[:, :, 2] = spectogram

    features.append(spectrogram_rgb)
    features=np.array(features)
    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)

    # Decode the class index to its corresponding label
    #predicted_class = le.inverse_transform(predicted_class_index)


    return predicted_class_index[0]

def predict_audio_classIMG_MEL(file_path, model):
    # Load the audio file and resample it
    features=[]
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')

    # Extract MFCC features
    melspec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    melspec_scaled = np.mean(melspec.T, axis=0)
    # Reshape the features to fit the input shape of the model
    features.append(melspec_scaled)
    features =  np.array([x.reshape( (16, 8, 1) ) for x in features])
    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)

    return predicted_class_index[0]

def predict_audio_classIMG_MEL2(file_path, model):
     # Load the audio file and resample it
    target_sr = 22050
    features = [];
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')
    spectogram = librosa.feature.melspectrogram(y=audio, sr=sr,n_mels=128)
    spectogram = librosa.power_to_db(spectogram,ref=np.max)
    spectrogram = (255 * (spectogram - np.min(spectogram)) / np.ptp(spectogram)).astype(np.uint8)
    spectogram = cv2.resize(spectogram, (256,256)) 
    spectrogram_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
    spectrogram_rgb[:, :, 0] = spectogram
    spectrogram_rgb[:, :, 1] = spectogram
    spectrogram_rgb[:, :, 2] = spectogram

    features.append(spectrogram_rgb)
    features=np.array(features)
    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)
    return predicted_class_index[0]

def predict_audio_classIMG_CQT(file_path, model):
    # Load the audio file and resample it
    features=[]
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')

    # Extract MFCC features
    cqt = librosa.feature.chroma_cqt(y=audio, sr=sr)
    cqt_scaled = np.mean(cqt.T, axis=0)
    # Reshape the features to fit the input shape of the model
    features.append(cqt_scaled)
    features =  np.array([x.reshape( (12, 1, 1) ) for x in features])
    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)


    return predicted_class_index[0]
def predict_audio_classIMG_CENS(file_path, model):
    # Load the audio file and resample it
    features=[]
    audio, sr = librosa.load(file_path,res_type='kaiser_fast')

    # Extract MFCC features
    cqt = librosa.feature.chroma_cens(y=audio, sr=sr)
    cqt_scaled = np.mean(cqt.T, axis=0)
    # Reshape the features to fit the input shape of the model
    features.append(cqt_scaled)
    features =  np.array([x.reshape( (12, 1, 1) ) for x in features])
    # Predict the class
    predicted_vector = model.predict(features)
    predicted_class_index = np.argmax(predicted_vector, axis=-1)


    return predicted_class_index[0]
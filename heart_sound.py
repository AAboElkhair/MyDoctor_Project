import numpy as np
import librosa
import librosa.display
import tensorflow as tf

CLASSES = ['artifact','murmur','normal']
NB_CLASSES=len(CLASSES)
label_to_int = {k:v for v,k in enumerate(CLASSES)}
int_to_label = {v:k for k,v in label_to_int.items()}

model = None
def load_model():
    model = tf.keras.models.load_model("heart_sounds1-Copy1.h5")
    print("Model loaded")
    return model

def preprocessing (file_path, duration=10, sr=22050):
  input_length=sr*duration
  process_file=[]
  X, sr = librosa.load(file_path, sr=sr, duration=duration) 
  dur = librosa.get_duration(y=X, sr=sr)
  # pad audio file same duration
  if (round(dur) < duration):
    y = librosa.util.fix_length(X, input_length)                
  mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40, n_fft=512,hop_length=2048).T,axis=0)
  feature = np.array(mfccs).reshape([-1,1])
  process_file.append(feature)
  process_file_array = np.asarray(process_file)
  return process_file_array

def predict(file_path):
    global model
    if model is None:
      model = load_model()
    process_audio = preprocessing(file_path)
    pred = np.asarray(model.predict(process_audio, batch_size=32))
    prediction_val = np.argmax(pred,axis=1)
    prediction = "prediction test return :"+ str(prediction_val)+ "-"+str(int_to_label[prediction_val[0]])
    return prediction
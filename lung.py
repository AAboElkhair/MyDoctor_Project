import numpy as np
import librosa
import librosa.display
import tensorflow as tf

CLASSES = ['Healthy','Unhealthy']
NB_CLASSES=len(CLASSES)
label_to_int = {k:v for v,k in enumerate(CLASSES)}
int_to_label = {v:k for k,v in label_to_int.items()}

model = None
def load_model_():
    model_ = tf.keras.models.load_model("lung_sound1.h5")
    print("Model loaded")
    return model_

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

def predict_lung(file_path):
    global model
    if model_ is None:
      model_ = load_model_()
    process_audio = preprocessing(file_path)
    pred = np.asarray(model.predict(process_audio, batch_size=32))
    prediction_val_ = np.argmax(pred,axis=1)
    if prediction_val_[0] == 0:
        result_ = "Lung sound is normal, Don't worry"
    elif prediction_val_[0] == 1:
        result_ = "The lung sound is not normal please contact with doctor"
    return result_
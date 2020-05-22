from __future__ import unicode_literals
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import ntpath
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import youtube_dl
import os
import librosa
from keras.models import load_model
from sklearn.metrics import accuracy_score
import math as math
from matplotlib.pyplot import step, show
import matplotlib.pyplot as plt




def gen_chunks(audio_filename, output_folder):
  myaudio = AudioSegment.from_file(audio_filename) 
  chunk_length_ms = 1000 # pydub calculates in millisec
  chunks = make_chunks(myaudio, chunk_length_ms) #Make chunks of one sec
  chunk_list = []

  base_filename = ntpath.basename(audio_filename)

  if len(base_filename) > 7:
    vid_title = base_filename[0:6]
  else:
    vid_title = base_filename

  #Export all of the individual chunks as wav files

  for i, chunk in enumerate(chunks):
      chunk_name = f"{vid_title}chunk{i}.wav"
      print ("exporting", chunk_name)
      chunk.export(f'{output_folder}/{chunk_name}', format="wav")
      chunk_list.append(chunk_name)
  
  chunk_list.pop()

  return vid_title, chunk_list

def delete_path(path):
  for root, dirs, files in os.walk(path):
      for f in files:
          os.unlink(os.path.join(root, f))
      for d in dirs:
          shutil.rmtree(os.path.join(root, d))

def predict(url):
    chunk_path = '/home/watch_my_set/chunks'
    prediction_path = '/home/watch_my_set/bucket/predictions'
    model_path = '/home/watch_my_set/bucket/model'
    url  = url
    yt_url = f'https://youtu.be/{url}'
    output_path = f'/home/watch_my_set/youtube/{url}'
    ydl_opts = {
        'outtmpl': os.path.join(output_path, '%(title)s-%(id)s.%(ext)s'),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
        'postprocessor_args': [
            '-ar', '16000'
        ],
        'prefer_ffmpeg': True,
        'keepvideo': True
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
      ydl.download([yt_url])

    for filename in os.listdir(output_path):
        if filename.endswith(".wav"):
          title, chunkname_list = gen_chunks(f'{output_path}/{filename}', chunk_path)

    shutil.rmtree(output_path)

    X_pred_list = []
    _min, _max = float('inf'), -float('inf')

    model_name = 'WMS_model_5.5k.model'
    model = load_model(f'{model_path}/{model_name}')

    for i in tqdm(chunkname_list):
      signal, rate = librosa.load(f'{chunk_path}/{i}', sr=16000)
      mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=512)
      X_pred_list.append(mel)
      _min = min(np.amin(X_pred_list), _min)
      _max = max(np.amax(X_pred_list), _max)

    X_pred = np.array(X_pred_list)
    X_pred = (X_pred - _min) / (_max - _min)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[1], X_pred.shape[2])

    y_pred = model.predict(X_pred)

    y_guess = np.argmax(y_pred, axis=1)

    np.savetxt(f"{prediction_path}/{model_name}_{title}_csv_predict.csv", y_guess, delimiter=",")

    delete_path(chunk_path)

    numLaughter = np.count_nonzero(y_guess)
    laughPercent =  round((numLaughter / y_guess.size) * 100, 1)
    laughTimes = []
    laughTimesList = []
    numLaughs = 0
    check = 0

    for i in tqdm(range(y_guess.size)):
      if (y_guess[i]==1 and check==0):
        numLaughs=numLaughs+1
        check = 1
        laughTimes.append(i)
      else:
        check = y_guess[i]

    for i in tqdm(range(len(laughTimes))):
      seconds = np.mod(laughTimes[i], 60)
      minutes = np.floor_divide(laughTimes[i], 60)
      laughTimesList.append('{0} minutes '.format(minutes) + 'and {0} seconds'.format(seconds))

    laughsPerMin = numLaughs / (y_guess.size/60)
    laughsPerMin = round(laughsPerMin, 1)

    print('\n{0} stats'.format(title))
    print('Laugh Percentage = {0}%'.format(laughPercent))
    print('Num Laughs = {0}'.format(numLaughs))
    print("\n".join(laughTimesList))

    plt.rcParams["figure.figsize"] = (3,1)
    x_axis = []
    y_axis = []
    for i in range(y_guess.size-1):
      minutes = str(np.floor_divide(i, 60))
      seconds = str(np.mod(i, 60))
      if minutes=='0':
        x_axis.append('' + seconds)
      else:
        x_axis.append('' + minutes + ':' + seconds) 
      if y_guess[i]==0:
        y_axis.append('silence')
      else:
        y_axis.append('laughter')

    fig, ax = plt.subplots()
    ax.step(x_axis, y_axis)


    plt.xlim([0, y_guess.shape[0]]);
    plt.ylim([-.05,1.05]);
    plt.yticks(fontsize=16);
    plt.xticks(fontsize=8)
    plt.xticks(np.arange(0, y_guess.shape[0], 10)); 
    plt.fill_between(x_axis, y_axis, step="pre", alpha=0.2)
    plt.savefig(f'/home/watch_my_set/bucket/plots/{url}.png', dpi=300)
    return laughPercent, numLaughs, laughsPerMin, laughTimesList, laughTimes



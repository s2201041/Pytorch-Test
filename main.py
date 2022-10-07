
import os
import pandas as pd
from tqdm import tqdm
import librosa
import librosa.display # インポートしないでlibrosa.display(〜〜)で実行しようとするとエラーになりました
import matplotlib.pyplot as plt
import IPython.display as ipd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# とりあえず1つの音源ファイルを指定してlibrosaを使ってみる
#librosa.loadで音源の波形データ（第1戻り値）とサンプルレート（第2戻り値）を取得できます。
waveform, sample_rate = librosa.load(drive_dir + "audio/1-100032-A-0.wav") # 犬の鳴き声の音源データを指定

# メルスペクトログラムの取得
# librosa.feature.melspectrogramに上で取得した波形データとサンプルレートを渡せば一発でメルスペクトログラムを取得できます。
feature_melspec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)

# MFCCの取得
# librosa.feature.mfccでOK
feature_mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)

# 波形データ、メルスペクトログラム、MFCCはnumpyで取得されます。
print("波形データのデータタイプ", type(waveform))
print("メルスペクトログラムのデータタイプ", type(feature_melspec))
print("MFCCのデータタイプ", type(feature_mfcc))
print("サンプルレート", sample_rate)
print("波形データの形状", waveform.shape)
print("メルスペクトログラムの形状", feature_melspec.shape)
print("MFCCの形状", feature_mfcc.shape)

# ランダムに3つほど音源データをピックアップして、波形、メルスペクトログラム、MFCCをそれぞれ可視化してみます。
# ついでにnotebook上で音源の再生ができるようにもします。
for row in meta_df.sample(frac=1)[['filename', 'category']][:3].iterrows():
    filename = row[1][0] # wavファイル名
    category = row[1][1] # そのファイルのカテゴリ

    # 波形データとサンプルレートを取得
    waveform, sample_rate = librosa.load(drive_dir + "audio/" + filename)

    # メルスペクトログラムを取得
    feature_melspec = librosa.feature.melspectrogram(y=waveform, sr=sample_rate)

    # MFCCを取得
    feature_mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate)

    # 可視化してみる
    print("category : " + category)
    plt.figure(figsize=(20, 5))

    # librosa.display.waveplotで波形データを可視化できます
    plt.subplot(1,3,1)
    plt.title("wave form")
    librosa.display.waveplot(waveform, sr=sample_rate, color='blue')

    # librosa.display.specshowでメルスペクトログラム、MFCCを可視化できます
    plt.subplot(1,3,2)
    plt.title("mel spectrogram")
    librosa.display.specshow(feature_melspec, sr=sample_rate, x_axis='time', y_axis='hz')
    plt.colorbar()

    plt.subplot(1,3,3)
    plt.title("MFCC")
    librosa.display.specshow(feature_mfcc, sr=sample_rate, x_axis='time')
    plt.colorbar()

    plt.tight_layout()
    plt.show()
    print()

    # 音源の再生はlibrosaで取得できた波形データとサンプルレートをIPython.display.Audioに以下のようにして渡します。
    display(ipd.Audio(waveform, rate=sample_rate))
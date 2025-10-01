from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import librosa
import pandas as pd
import numpy as np
import os

#
# label -> 'valence_mean', 'arousal_mean' 값으로 이루어진 배열
#
def per_data_extractor(audio_path, label):
    y_full, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Segment into 5-second chunks (for consistency with CNN)
    segment_samples = 5 * sr
    segments = [y_full[i:i+segment_samples] for i in range(0, len(y_full), segment_samples)
                if len(y_full[i:i+segment_samples]) == segment_samples]

    mel_specs = [] # 실제로 사용될 데이터
    for segment in segments:
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_specs.append(mel_spec_db)

    return mel_specs, [label] * len(mel_specs) # 변환된 데이터, Raw labels (1-9 scale)

def total_data_extractor(dataframe_static_annotations, audio_dir):
    # Process all songs
    X_by_song = {}
    y_by_song = {}

    for song_id in dataframe_static_annotations['song_id'].values:
        audio_path = os.path.join(audio_dir, f"{song_id}.mp3")
        try:
            X_by_song[song_id], y_by_song[song_id] \
                = per_data_extractor(audio_path, dataframe_static_annotations[dataframe_static_annotations['song_id'] == song_id][[' valence_mean', ' arousal_mean']].values[0])

        except Exception as e:
            print(f"Error processing song {song_id}: {e}")

    print("Data loading and mel-spectrogram extraction complete.")
    return X_by_song, y_by_song
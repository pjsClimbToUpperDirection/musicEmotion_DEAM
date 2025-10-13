import librosa
import numpy as np
import os
import pandas as pd
from tensorflow.keras import models
import joblib


#
# label -> 'valence_mean', 'arousal_mean' 값으로 이루어진 배열
# 경로에 존재하는 각 .mp3 포멧의 파일을 모델에서 사용가능한 형식으로 변환
def per_data_extractor(audio_path, label=None):
    y_full, sr = librosa.load(audio_path, sr=44100, mono=True)

    # Segment into 5-second chunks (for consistency with CNN)
    segment_samples = 5 * sr
    segments = [y_full[i:i+segment_samples] for i in range(0, len(y_full), segment_samples)
                if len(y_full[i:i+segment_samples]) == segment_samples]

    mel_specs = [] # 실제로 사용될 데이터
    for segment in segments:
        # 각 segment 단위로 특징 추출
        mel_spec = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_specs.append(mel_spec_db)

    if label is not None:
        # 변환된 데이터, Raw labels (1-9 scaling) -> mel_specs의 각각의 값에 label이 대응되도록 list로서의 label 데이터를 mel_specs의 길이만큼 반복
        # 'valence_mean', 'arousal_mean' 두 데이터가 존재하므로 label의 길이는 2 -> (2,) shape의 데이터를 위에 서술한 바와 같이 scaling
        return mel_specs, [label] * len(mel_specs)
    else:
        return mel_specs # 단순 데이터 변환만 희망하는 경우 label 인자를 전달하지 않음

def total_data_extractor(static_csv_path, audio_dir, do_not_print_scaled_deg=False):
    # Process all songs
    X_by_song = {}
    y_by_song = {}

    dataframe_static_annotations = pd.read_csv(static_csv_path)


    for song_id in dataframe_static_annotations['song_id'].values:
        audio_path = os.path.join(audio_dir, f"{song_id}.mp3")

        # 'valence_mean', 'arousal_mean' 두 데이터만 사용
        label = dataframe_static_annotations[dataframe_static_annotations['song_id'] == song_id][[' valence_mean', ' arousal_mean']].values[0]
        try:
            X_by_song[song_id], y_by_song[song_id] \
                = per_data_extractor(audio_path, label)

            if not do_not_print_scaled_deg and song_id % 20 == 0:
                print(f"the raw label that is co_related by song_id: {song_id} has been scaled about {len(y_by_song[song_id])} times")
        except Exception as e:
            print(f"Error processing song {song_id}: {e}")

    print("Data loading and mel-spectrogram extraction complete.")
    return X_by_song, y_by_song

# todo 출력값은 정규화된 라벨에 의하여 학습되었으므로 inverse_transform으로 역변환하여 실제 예측 값을 확인할 수 있다.
def test_only_one_music(audio_path, model_path, fitted_scalar_path):
    mel_specs_as_array = []
    mel_specs = per_data_extractor(audio_path)
    print("mel_specs: ", np.array(mel_specs).shape)

    mel_specs_as_array.extend(mel_specs)
    processed = np.array(mel_specs_as_array)
    processed = np.expand_dims(processed, axis=-1) # Expand dimensions for CNN or LSTM
    print("processed music data's shape: ", processed.shape)

    model = models.load_model(model_path)
    loaded_scaler = joblib.load(fitted_scalar_path)  # Load the saved scaler
    y_pred = model.predict(processed)
    transformed_pred = loaded_scaler.inverse_transform(y_pred)

    print(y_pred)
    print(np.array(y_pred).shape)
    print(transformed_pred)
    print(np.array(transformed_pred).shape)

    mean_valence = np.mean(np.abs(transformed_pred[:, 0]))
    mean_arousal = np.mean(np.abs(transformed_pred[:, 1]))
    print(f"Test(Mean) - Valence: {mean_valence:.4f}, Arousal: {mean_arousal:.4f}")

# x_song = []
# x_as_song = per_data_extractor('1002.mp3')
# x_song.extend(x_as_song)
# print("np.array(X_train).shape: ", np.array(x_song).shape)
test_only_one_music('1002.mp3', 'best_cnn_model.keras', 'minmax_scaler.pkl')
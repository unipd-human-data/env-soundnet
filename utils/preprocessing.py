import os
import numpy as np
import librosa
import librosa.display
from tqdm import tqdm
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SAMPLE_RATE = 44100
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
DURATION = 5.0


def load_audio_file(file_path, sr=SAMPLE_RATE, duration=DURATION):
    y, sr = librosa.load(file_path, sr=sr, duration=duration)
    y = librosa.util.normalize(y)
    return y


def center_crop(signal, target_len):
    if len(signal) < target_len:
        pad_left = (target_len - len(signal)) // 2
        pad_right = target_len - len(signal) - pad_left
        return np.pad(signal, (pad_left, pad_right), mode='constant')
    else:
        start = (len(signal) - target_len) // 2
        return signal[start:start + target_len]


def naa(y, sr):
    augmented = []
    target_len = int(sr * DURATION)
    augmented.append(center_crop(y, target_len))
    augmented.append(center_crop(librosa.effects.pitch_shift(y, sr=sr, n_steps=+2), target_len))
    augmented.append(center_crop(librosa.effects.pitch_shift(y, sr=sr, n_steps=-2), target_len))
    for rate in [0.7, 1.2]:
        y_stretch = librosa.effects.time_stretch(y, rate=rate)
        augmented.append(center_crop(y_stretch, target_len))
    return augmented


def preprocess_audio(y, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    y_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    y_mel = librosa.power_to_db(y_mel, ref=np.max)
    return y_mel


def pad_to_multiple_of(mel, multiple=50, value=-80.0):
    current_len = mel.shape[1]
    target_len = ((current_len + multiple - 1) // multiple) * multiple
    pad_width = target_len - current_len
    return np.pad(mel, ((0, 0), (0, pad_width)), mode='constant', constant_values=value)


def batch_logmel(X):
    mel_list = []
    for x in tqdm(X, desc="Log-Mel"):
        mel = preprocess_audio(x)
        mel_list.append(mel)
    return mel_list


def preprocess_dataset(X_audio, y_labels, apply_naa=True, apply_taa=True, augmentations_per_sample=4):
    X_aug, y_aug = [], []

    if apply_naa:
        for i in tqdm(range(len(X_audio)), desc="NAA"):
            augmented = naa(X_audio[i], sr=SAMPLE_RATE)
            X_aug.extend(augmented)
            y_aug.extend([y_labels[i]] * len(augmented))
    else:
        X_aug = X_audio
        y_aug = y_labels

    X_mel = batch_logmel(X_aug)
    X_mel = [pad_to_multiple_of(m) for m in X_mel]
    X_mel_np = np.array(X_mel)
    X_mel_np = (X_mel_np + 80.0) / 80.0
    if X_mel_np.ndim == 3:
        X_mel_np = X_mel_np[..., np.newaxis]

    y_aug = np.array(y_aug)

    if apply_taa:
        X_taa, y_taa = [], []
        taa_gen = ImageDataGenerator(
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.25,
            shear_range=0.3,
            fill_mode='nearest'
        )
        for i in tqdm(range(len(X_mel_np)), desc="TAA"):
            sample = np.expand_dims(X_mel_np[i], axis=0)
            gen = taa_gen.flow(sample, batch_size=1)
            for _ in range(augmentations_per_sample):
                aug = next(gen)[0]
                X_taa.append(aug)
                y_taa.append(y_aug[i])
        X_final = np.concatenate([X_mel_np, np.array(X_taa)], axis=0)
        y_final = np.concatenate([y_aug, np.array(y_taa)], axis=0)
    else:
        X_final = X_mel_np
        y_final = y_aug

    X_final_tensor = torch.from_numpy(X_final).float()
    y_final_tensor = torch.from_numpy(y_final).long()
    return X_final_tensor, y_final_tensor

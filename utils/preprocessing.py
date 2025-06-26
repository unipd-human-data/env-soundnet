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
HOP_LENGTH = 512
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


def naa_class_specific(y, sr, class_name, n_augments=4):
    """
    Class-specific augmentation with pitch shift, time stretch.
    Output is always cropped to fixed length.
    """
    target_len = int(sr * DURATION)
    augmented = []
    
    augmented.append(center_crop(y, target_len))
    
    limits = {
        'dog': ((-4, 4), (0.95, 1.1)),
        'sea_waves': ((0, 0), (0.9, 1.2)),
        'crying_baby': ((-3, 6), (0.8, 1.3)),
        'sneezing': ((-4, 4), (0.9, 1.2)),
        'helicopter': ((0, 0), (0.9, 1.2)),
        'chainsaw': ((-4, 2), (0.9, 1.2)),
        'rooster': ((-3, 2), (0.95, 1.1)),
        'clock_tick': ((0, 0), (1.0, 1.0)),
        'crackling_fire': ((-2, 2), (0.9, 1.1)),
    }

    pitch_range, time_range = limits.get(class_name, ((-2, 2), (0.9, 1.1))) 

    for _ in range(n_augments):
        pitch_shift = np.random.randint(pitch_range[0], pitch_range[1] + 1)
        time_stretch = np.random.uniform(time_range[0], time_range[1])

        if pitch_shift == 0 and abs(time_stretch - 1.0) < 0.01:
            continue

        y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch_shift)
        y_aug = librosa.effects.time_stretch(y_aug, rate=time_stretch)

        augmented.append(center_crop(y_aug, target_len))

    return augmented


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


def preprocess_dataset(X_audio, y_labels,
                       naa_mode='none',  # 'none', 'generic', or 'class_specific'
                       naa_augmentations=4,
                       taa_augmentations=2,
                       apply_taa=False,
                       label_index_to_name=None):
    """
    naa_mode: 
        - 'none': no augmentation
        - 'generic': apply standard naa()
        - 'class_specific': apply naa_class_specific()
    """
    X_aug, y_aug = [], []

    if naa_mode == 'generic':
        for i in tqdm(range(len(X_audio)), desc="NAA - Generic"):
            augmented = naa(X_audio[i], sr=SAMPLE_RATE)
            X_aug.extend(augmented)
            y_aug.extend([y_labels[i]] * len(augmented))

    elif naa_mode == 'class_specific':
        if label_index_to_name is None:
            raise ValueError("label_index_to_name is required for 'class_specific' naa_mode.")
        for i in tqdm(range(len(X_audio)), desc="NAA - Class Specific"):
            class_name = label_index_to_name[y_labels[i]]
            augmented = naa_class_specific(X_audio[i], sr=SAMPLE_RATE, class_name=class_name,
                                           n_augments=naa_augmentations)
            X_aug.extend(augmented)
            y_aug.extend([y_labels[i]] * len(augmented))

    else:  # 'none'
        X_aug = X_audio
        y_aug = y_labels

    # log-Mel spectrogram
    X_mel = batch_logmel(X_aug)
    X_mel = [pad_to_multiple_of(m) for m in X_mel]
    X_mel_np = np.array(X_mel)
    X_mel_np = (X_mel_np + 80.0) / 80.0
    if X_mel_np.ndim == 3:
        X_mel_np = X_mel_np[..., np.newaxis]

    y_aug = np.array(y_aug)

    # TAA step
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
            for _ in range(taa_augmentations):
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

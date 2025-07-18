import os
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB

# Default config
TARGET_SAMPLE_RATE = 16000
N_MELS = 80

def load_audio(file_path, target_sr=TARGET_SAMPLE_RATE):
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != target_sr:
        waveform = Resample(orig_freq=sample_rate, new_freq=target_sr)(waveform)
    return waveform, target_sr

def normalize_audio(waveform):
    return (waveform - waveform.mean()) / waveform.std()

def extract_log_mel_spectrogram(waveform, sample_rate=TARGET_SAMPLE_RATE):
    mel_spectrogram = MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=N_MELS,
        hop_length=512,
        n_fft=1024
    )(waveform)
    db_transform = AmplitudeToDB(top_db=80)
    log_mel = db_transform(mel_spectrogram)
    return log_mel

def preprocess_audio_file(file_path):
    waveform, sample_rate = load_audio(file_path)
    waveform = normalize_audio(waveform)
    log_mel = extract_log_mel_spectrogram(waveform, sample_rate)
    return log_mel.squeeze(0)

def batch_preprocess(audio_dir):
    processed = []
    for fname in os.listdir(audio_dir):
        if fname.endswith('.mp3') or fname.endswith('.wav'):
            path = os.path.join(audio_dir, fname)
            features = preprocess_audio_file(path)
            processed.append(features)
    return processed

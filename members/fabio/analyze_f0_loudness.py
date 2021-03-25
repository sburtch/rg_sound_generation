import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pydub
import pickle
from ddsp import spectral_ops
from ddsp import core
import warnings
import glob
import os
import wavio


def _load_audio(audio_path, sample_rate, frame_rate):
    with tf.io.gfile.GFile(audio_path, 'rb') as f:
        # Load audio at original SR
        audio_segment = (pydub.AudioSegment.from_file(f).set_channels(1))
        # Compute expected length at given `sample_rate`
        expected_len = int(audio_segment.duration_seconds * sample_rate)
        # expected_len = int(audio_segment.duration_seconds * frame_rate) * int(sample_rate // frame_rate)
        # Resample to `sample_rate`
        audio_segment = audio_segment.set_frame_rate(sample_rate)
        sample_arr = audio_segment.get_array_of_samples()
        audio = np.array(sample_arr).astype(np.float32)
        # Zero pad missing samples, if any
        audio = spectral_ops.pad_or_trim_to_expected_length(audio, expected_len, len_tolerance=10000)
    # Convert from int to float representation.
    audio /= np.iinfo(sample_arr.typecode).max
    return audio


def normalize_wav(data, sample_width):
    # sample_width in byte (i.e.: 1 = 8bit, 2 = 16bit, 3 = 24bit, 4 = 32bit)
    divisor = 2 ** (8 * sample_width) / 2
    data = data / float(divisor)
    return data


def wav_read(filename, normalize=True):
    # check if file exists
    if not os.path.exists(filename):
        raise NameError("File does not exist:" + filename)

    wav = wavio.read(filename)
    data = wav.data
    rate = wav.rate
    sample_width = wav.sampwidth

    if normalize:
        data = normalize_wav(data, sample_width)

    return data, rate, sample_width


def save_f0_loudness(wav_filename, sample_rate, frame_rate):
    # audio = _load_audio(target_path, sample_rate, frame_rate)
    audio, _, _ = wav_read(wav_filename)
    audio = np.squeeze(audio)
    f0_hz, f0_confidence = spectral_ops.compute_f0(
        audio, sample_rate, frame_rate, len_tolerance=10000)
    mean_loudness_db = spectral_ops.compute_loudness(
        audio, sample_rate, frame_rate, 2048, len_tolerance=10000)

    pickle_filename = os.path.splitext(wav_filename)[0] + '.pickle'

    with open(pickle_filename, 'wb') as f:
        pickle.dump([f0_hz, mean_loudness_db], f)

    return f0_hz, mean_loudness_db


def load_f0_loudness(pickle_filename):
    with open(pickle_filename, 'rb') as f:
        f0_hz, mean_loudness_db = pickle.load(f)

    return f0_hz, mean_loudness_db


def main():
    # sample_rate = 48000
    # frame_rate = 375
    # path = 'wav'

    sample_rate = 16000
    frame_rate = 250
    path = 'nsynth'

    wav_filepattern = os.path.join(path, '*.wav')
    pickle_filepattern = os.path.join(path, '*.pickle')

    wav_files = glob.glob(wav_filepattern)
    pickle_files = glob.glob(pickle_filepattern)

    if len(pickle_files) == 0:
        for wav_filename in wav_files:
            save_f0_loudness(wav_filename, sample_rate, frame_rate)

    pickle_files = glob.glob(pickle_filepattern)

    for pickle_filename in pickle_files:
        f0_hz, mean_loudness_db = load_f0_loudness(pickle_filename)

        root_note = 60.0
        f0_root = core.midi_to_hz(root_note)

        for i in range(3):
            note = core.hz_to_midi(f0_hz)

            f0_hz = np.where(note > root_note + 6, f0_hz / 2.0, f0_hz)
            f0_hz = np.where(note < root_note - 6, f0_hz * 2.0, f0_hz)

        note = core.hz_to_midi(f0_hz)
        f0_hz = np.where(np.abs(note - root_note) > 1.0, f0_root, f0_hz)

        name = os.path.basename(os.path.splitext(pickle_filename)[0])

        plt.figure(1)
        plt.plot(f0_hz, label=name)
        plt.legend()

        plt.figure(2)
        plt.plot(mean_loudness_db, label=name)
        plt.legend()

    plt.show()


if __name__ == '__main__':
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()

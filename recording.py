from typing import List

import numpy as np
import pyaudio
import soundfile as sf
from tqdm import tqdm

AUDIO_FORMAT = pyaudio.paInt16  # 音声フォーマット
CHANNELS = 1  # チャンネル数
RATE = 44100  # サンプリングレート
CHUNK = 2**11  # バッファサイズ
RECORD_SECONDS = 4  # 録音する秒数
iDeviceIndex = 0  # 録音デバイスのインデックス番号


def record_voice(file_name: str):
    """
    A function that records voice and saves it as a wav file

    Parameters
    ----------
    file_name : str
        The name of the wav file to save
    """

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
    )

    frames = []
    print("Now Recording...")
    for _ in tqdm(range(0, int(RATE / CHUNK * RECORD_SECONDS))):
        data = stream.read(CHUNK)
        frames.append(data)
    print("Finished Recording.")

    frames = b"".join(frames)
    frames = np.frombuffer(frames, dtype=np.int16)

    save_recording(frames, file_name)

    stream.stop_stream()
    stream.close()
    audio.terminate()


def save_recording(frames: List, file_name: str):
    """
    A function that writes recorded data in numpy array to a wav file

    Parameters
    ----------
    frames : np.ndarray
        Recorded data
    file_name : str
        The name of the wav file to save
    """

    if frames.size > 0:
        sf.write(file_name, frames, RATE)

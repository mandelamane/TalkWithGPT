import os
import time
import warnings

import numpy as np
import openai
import pyaudio
import torch
import whisper
from espnet2.bin.tts_inference import Text2Speech
from espnet2.utils.types import str_or_none

from recording import record_voice

warnings.simplefilter("ignore")

AUDIO_FORMAT = pyaudio.paFloat32  # music format
CHANNELS = 1  # a number of channels
RATE = 44100  # sampling rate
FRAMES_PER_BUFFER = 1024  # buffer size
WAV_FILE = "tmp/tmp.wav"  # 　path to wave file


def get_t2s_model(
    tag: str = "kan-bayashi/jsut_full_band_vits_prosody",
) -> Text2Speech:
    """
    load text-to-speech model using espnet

    Parameters
    ----------
    tag : str
        model name for text to speech, by default
        "kan-bayashi/jsut_full_band_vits_prosody"

    Returns
    -------
    Text2Speech
        text-to-speech model object
    """

    t2s_model = Text2Speech.from_pretrained(
        model_tag=str_or_none(tag),
        vocoder_tag=None,
        device="cpu",
        speed_control_alpha=1.0,
        noise_scale=0.333,
        noise_scale_dur=0.333,
    )

    return t2s_model


def get_s2t_model(mode: str = "large") -> whisper.model.Whisper:
    """
    load speech-to-text model using whisper

    Parameters
    ----------
    mode : str
        size of model parameter, one of ["base", "large", "multilingual",
        "multilingual-large"], by default "large"

    Returns
    -------
    whisper.model.Whisper
        speech-to-text model object
    """

    s2t_model = whisper.load_model(mode)

    return s2t_model


def load_secret_key(api_key_file: str) -> str:
    """
    Load an API key from a file

    Parameters
    ----------
    api_key_file : str
        Path of the file where the API key was saved

    Returns
    -------
    str
        The API key as a string
    """

    with open(api_key_file, "r") as f:
        api_key = f.readline()

    return api_key


def text2wav_array(t2s_model: Text2Speech, output_text: str) -> np.ndarray:
    """
    Create audio from the output text and return it as a numpy array.

    Parameters
    ----------
    t2s_model : Text2Speech
        A text-to-speech model object
    output_text : str
        Text replied from chatgpt

    Returns
    -------
    np.ndarray
        The audio array as a numpy array
    """

    with torch.no_grad():
        wav = t2s_model(output_text)["wav"]
    wav_array = wav.view(-1).cpu().numpy()

    return wav_array


def get_assistant_reply(
    messages: list[dict], tag: str = "gpt-3.5-turbo"
) -> str:
    """
    Generate and return a reply from the assistant based on the messages.

    Parameters
    ----------
    messages : list[dict]
        A list of dictionaries containing the role and content of each message.
    tag : str
        model name for openai, one of ["gpt-3.5-turbo", "gpt-3.5"],
        by default "gpt-3.5-turbo"

    Returns
    -------
    str
        A reply from the assistant as a string.
    """

    response = openai.ChatCompletion.create(model=tag, messages=messages)
    reply = response["choices"][0]["message"]["content"]
    return reply


def play_assistant_reply(t2s: Text2Speech, stream: pyaudio.Stream, reply: str):
    """
    Convert the reply to audio and play it through the stream.

    Parameters
    ----------
    text_to_speech : Text2Speech
        A text-to-speech model object that can convert text to audio.
    stream : pyaudio.Stream
        A stream object that can write audio data to the output device.
    reply : str
        A reply from the assistant as a string.

    Returns
    -------
    """

    wav_array = text2wav_array(t2s, reply)
    stream.write(wav_array.astype(np.float32).tobytes())


def main():
    if not os.path.exists("tmp"):
        os.mkdir("tmp")

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=AUDIO_FORMAT,
        channels=CHANNELS,
        rate=RATE,
        frames_per_buffer=FRAMES_PER_BUFFER,
        output=True,
    )
    openai.api_key = load_secret_key("api.key")

    text_to_speech = get_t2s_model(
        tag="kan-bayashi/tsukuyomi_full_band_vits_prosody"
    )
    speech_to_text = get_s2t_model(mode="large")

    messages = []
    # chatbotの人格を指定するプロンプト
    with open("system.content", "r") as f:
        system_msg = f.read()

    messages.append({"role": "system", "content": system_msg})

    print("Say hello to your new assistant!")
    while True:
        record_voice(WAV_FILE)
        message = speech_to_text.transcribe(WAV_FILE)["text"]

        print("-" * 80)
        print(f"User: {message}")

        if "やめた" in message:
            break

        messages.append({"role": "user", "content": message})
        reply = get_assistant_reply(messages)

        print(f"Assistant: {reply}")
        print("-" * 80)

        play_assistant_reply(text_to_speech, stream, reply)

        messages.append({"role": "assistant", "content": reply})

        time.sleep(2)

    stream.close()
    audio.terminate()


if __name__ == "__main__":
    main()

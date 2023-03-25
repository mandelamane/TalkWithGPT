# TalkWithGPT

Transcription with whisper, conversation creation with chatbot, speech synthesis with espnet2

## setup

Create openai-api key file by

```
echo "xxxxxxx" > api.key
```

Create virtual environment by

```
python3 -m venv env
../env/bin/activate
```

and install dependent packages by

```
pip install -r requirements.txt
```

## run

For running the chatbot, use

```
python chatbot.py
```

If you want to give chatgpt a personality, rewrite system.content.
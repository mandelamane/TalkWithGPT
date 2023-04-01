# TalkWithGPT

Transcription with whisper, conversation creation with chatbot, speech synthesis with espnet2

## setup

Get access key from [openai api](https://openai.com/product)

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

if you can not install pyopenjtalk, you excute `pip install pyopenjtalk==0.2.0 --no-use-pep517` in terminal.

## run

For running the chatbot, use

```
python chatbot.py --system content.txt --whisper base
```

If you stop this program, you say "やめた".

If you want to give chatgpt a personality, rewrite content.txt
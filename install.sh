#!/bin/bash

python -m venv ~/birdspy
cd ~/birdspy
source ./bin/activate

pip install --upgrade pip setuptools wheel
pip install opencv-python-headless RPi.GPIO python-telegram-bot python.dotenv Pillow pigpio picamera tflite-runtime

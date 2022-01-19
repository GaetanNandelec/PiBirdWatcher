#!/usr/bin/python
# Author : Gaetan Nandelec
# Date   : 19/01/2022

import telegram
from picamera import PiCamera
import time
import RPi.GPIO as GPIO
import time
import os
from dotenv import load_dotenv

load_dotenv()

# Telegram settings
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
bot = telegram.Bot(token=TELEGRAM_BOT_TOKEN)

# camera settings
camera = PiCamera()
time.sleep(1)
camera.resolution = (1280, 720)

# PIR settings
# Use BCM GPIO references
# instead of physical pin numbers
GPIO.setmode(GPIO.BCM)
# Define GPIO to use on Pi
GPIO_PIR = 27

# Set pin as input
GPIO.setup(GPIO_PIR,GPIO.IN)      # Echo
Current_State  = 0
Previous_State = 0

def takePicture(file_name):
    camera.capture(file_name)
    print("Picture Taken.")

def sendTelegram(file_name):
#    bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="From Telegram Bot")
    bot.send_photo(chat_id=TELEGRAM_CHAT_ID, photo=open(file_name, 'rb'))

def main():
    file_name = "img_buffer.jpg"
    takePicture(file_name)
    sendTelegram(file_name)

try:

  print("Waiting for PIR to settle ...")

  # Loop until PIR output is 0
  while GPIO.input(GPIO_PIR)==1:
    Current_State  = 0

  print("  Ready")

  # Loop until users quits with CTRL-C
  while True :

    # Read PIR state
    Current_State = GPIO.input(GPIO_PIR)

    if Current_State==1 and Previous_State==0:
      # PIR is triggered
      start_time=time.time()
      print("  Motion detected!")
      # Record previous state
      Previous_State=1
      main()
      time.sleep(3)
      main()
      time.sleep(5)
      main()
      time.sleep(5)
      main()
    elif Current_State==0 and Previous_State==1:
      # PIR has returned to ready state
      print("  Ready")
      stop_time=time.time()
      elapsed_time=int(stop_time-start_time)
      print(" (Elapsed time : " + str(elapsed_time) + " secs)")
      Previous_State=0

    # Wait for 10 milliseconds
    time.sleep(0.01)

except KeyboardInterrupt:
  print("  Quit")
  # Reset GPIO settings
  GPIO.cleanup()

#!/bin/bash

echo "***************************************************************"
echo "**********Updating and Upgrading the Raspberry Pi OS***********"
echo "***************************************************************"

sudo apt-get update -y
sudo apt-get upgrade -y


echo "***************************************************************"
echo "******Installing Tensorflow Lite and USB Coral Libraries*******"
echo "***************************************************************"

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install python3-tflite-runtime -y
sudo apt-get install libedgetpu1-std -y
sudo python3 -m pip install numpy
sudo python3 -m pip install Pillow

echo "***************************************************************"
echo "********Installing OpenCV**************************************"
echo "***************************************************************"

sudo apt install python3-opencv -y

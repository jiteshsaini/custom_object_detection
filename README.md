# Train and Deploy Custom Object Detection Model on Raspberry Pi
<p align="left">
Read the :- <a href='https://helloworld.co.in/article/custom-object-detection-model' target='_blank'>
   complete article here.
</a> 
Watch Video :- <a href='https://youtu.be/kjuStyfl6yk' target='_blank'>
   on Youtube.
</a> 
</p>

This repo contains a python script and few Object Detection models. These models are placed in two folders i.e. 'custom' and 'pretrained'. The model in 'custom' folder is created using Tensorflow Lite Model maker and trained to detect 3 types of objects as shown below. 

<p align="center">
<img src='https://github.com/jiteshsaini/files/blob/main/img/custom-object-detection.jpg'>
</p>

The models in 'pretrained' folder are downloaded from [coral.ai](https://coral.ai/models/object-detection/) website. These pretrained models are trained with COCO dataset to detect 90 types of objects.

The python script can be used to run a custom as well as pretrained model. It also supports Google Coral USB accelerator to speed up the inferencing process.

## Training the Model with your data

The training is done through a Colab notebook which is an interactive Python notebook accessible through a web browser. It makes use of Tensorflow Lite Model Maker to create custom models through Transfer Learning. The link of original notebook created by [Khanh LeViet](https://github.com/khanhlvg) is [here](https://colab.research.google.com/github/khanhlvg/tflite_raspberry_pi/blob/main/object_detection/Train_custom_model_tutorial.ipynb).

The link of notebook I derived from the original notebook is [here](https://colab.research.google.com/drive/1LT27nDGfTNfTJXBILURDomWyMc0UazNS).

The data set created for this project is [here](https://drive.google.com/file/d/1Tk6PmWxQnH8zdp85bGQyU_Xmzp6LBTJL/view).

The notebook allows you to create and download a custom model using your data.


## Running your custom model

The packages and libraries required to run this file can be installed through bash script by running the command 'sudo sh setup.sh' in terminal. 

Run the python file using the command 'python3 detect.py'

You can use a Pi camera or a USB camera with your Raspberry Pi to run the python file 'detect.py'. The python script also supports Google Coral USB Accelerator. If you want to use Coral Acc, just make sure that you set the variable 'edgetpu' as '1' and provide the path to the model created for Coral Acc. Both folder i.e. 'pretrained' and 'custom' contains models that can run on Coral Acc. The name of these models ends with 'edgetpu'.

# Train and Deploy Custom Object Detection Model 

This project demonstrates how you can train and create your custom model and deploy on Raspberry Pi to detect objects that are not present in the popular 
pre-trained Object Detection Models.

The training is done through a Colab notebook. Which is an interactive Python notebook accessible through a web browser. It makes use of Tensorflow Lite Model Maker to create custom models through Transfer Learning. 
The link of original notebook is :- 

the link of notebook I created from the original notebook:--

The notebook allows you to create a custom model using your data and download it.

The model I created along with label file is placed in the 'models/custom' folder. 

Some other sample pre-trained models are placed in 'models/pretrained' folder.

## detect.py

The python file 'detect.py' is capable of running a custom model or any other pre-trained model.

You need to uncomment the relevant portion to use the file. The file also supports Google Coral USB Accelerator. If you have one and test it out, just make sure that you make the variable 'edgetpu' as '1' and provide the path to the model created for Coral Acc. Both folder i.e. 'pretrained' and 'custom' contains models that can run on Coral Acc. The name of these models ends with 'edgetpu'. So you need to uncomment this model if you are using a Coral USB Acc.

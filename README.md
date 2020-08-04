# MICRO-Expression-Recognition-with-deep-learning
Main stream of the model:

![image](https://github.com/JayShaun/MICRO-Expression-Recognition-with-deep-learning/blob/master/Recognition_Experiments/model.png)

Origin paper: Enriched Long-term Recurrent Convolutional Network for Facial Micro-Expression Recognition
Ref code: https://github.com/IcedDoggie/Micro-Expression-with-Deep-Learning
## Environment：
matlab 2015b

python 3.6

Keras (tensorflow as backend)
## Files:
TIM-temporal-interpolation-model-master contains interpolation programs to preprocess data.

tvl1flow_3 is the package for computing optical flow and optical strain.

Recognition-Experiment contains model and training files.

## How to run:
1. Prepare your data (e.g. CASMEII dataset  http://fu.psych.ac.cn/CASME/casme2-en.php)
2. Detect and crop the faces with opencv or some other packages by yourself.
3. Run interpolation files to interpolate your video into 16 frames(can set by yourself).
4. Compute optical flow image and optical strain image.
5. When your data is ready, run train.py

## Result:
I have trained the model for 1000 epochs and the accuracy is arround 70%.
If you need my pretrained model, contact me. And I will upload it to cloud disk later.

## Updates:
Pretrained model on CASMEII:

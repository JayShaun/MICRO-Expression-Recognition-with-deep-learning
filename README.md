# MICRO-Expression-Recognition-with-deep-learning


## Environment：
matlab 2015b
python 3.6
Keras (tensorflow as backend)
## Files:
TIM-temporal-interpolation-model-master contains interpolation programs to preprocess data.
tvl1flow_3 is the package for computing optical flow and optical strain.
Recognition-Experiment contains model and training files.

## How to run:
1. prepare your data (e.g. CASMEII dataset  http://fu.psych.ac.cn/CASME/casme2-en.php)
2. detect and crop the faces with opencv or other package by yourself.
3. carry out interpolation files to interpolate your video into 16 frames(can set by yourself).
4. compute optical flow image and optical strain image.
5. when your data is ready, run train.py

## Result:
I have train the model for 1000 epoch and the accuracy is arround 70%.
If you need my pretrained model, contact me. And I will upload it to cloud disk later.
by zxl

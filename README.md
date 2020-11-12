# DeepEye
DeepEye: Deep Convolutional Network for Pupil Detection in real Environment

Requirements:

Tensorflow 1.13
OpenCV

The Best way to get it working:

Install Conda for python 3, then:

    conda create --name deepeye_env
    conda activate deepeye_env
    conda install tensorflow=1.13.1
    conda install opencv

Then employ this venv to run the example.py 

The best way to use DeepEye with a webcam is using a eye localization algorithm, such as opencv with cascade or using another neural network. 

Example video of DeepEye:

https://youtu.be/vKiUue0kpHw

This repository is based on the following publication:

Vera-Olmos, F. J., Pardo, E., Melero, H., & Malpica, N. (2019). DeepEye: Deep convolutional network for pupil detection in real environments. Integrated Computer-Aided Engineering, 26(1), 85-95.

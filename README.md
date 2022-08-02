# JETSONSKY

2022-08-02 UPDATE : upload of a PDF file to explain options you can set in the main window of JetsonSky

2022-08-01 UPDATE : New Linux and Windows versions released (with camera control). Version 12_02
Changes :
- added a variable to set Main Window fonts size (from 5 to 7, depending of your system.
Set the good value in line 41 in the program

- added support of ZWO ASI224MC camera
- added support of ZWO ASI290MC and ASI290MM camera


2022-07-30 UPDATE : Video treatment software (no camera control) is uploaded. 2 versions :
- in the main directory : Jetson version
- in Windows directory : Windows 10/11 version


2022-07-29 UPDATE : PC version (with camera control) is uploaded. See below.

*****************************************************************************

This program is mainly dedicated to astronomy live video with live treatments.

This program works with Nvidia Jetson single board computer. It was possible for me to develop this software with the support of NVIDIA. I would like to thank Dustin Franklin from NVIDIA who supported me for years. Many thanks Dusty !

This program require an camera for acquisitions. The supported cameras are :
- ZWO ASI178MC and ASI 178MM
- ZWO ASI485MC and ASI585MC

You will need CUDA installed on the Jetson computer.

You will also need :
- Python (version depends of your Jetson Model)
- openCV library
- numpy
- pillow
- pycuda
- six

In the Lib directroy, you will find ZWO drivers for linux (ArmV8 for ZWO SDK for linux).

In the zwoasi directory, you will find a Python gateway to use ZWO camera library.

You will also have to create 2 directories in you installation directory :
- one for the images you will acquire
- one for the video you will acquire

There is an other directory (zwoefw) which gets Python gateway for ZWO EFmini filter wheel. This is in standby for now, waiting for issues solving.

JetsonSky still have bugs i will solve ASAP.

JetsonSky can also be used on a PC Windows 10/11 64b system. The PC must have a NVIDIA graphic card to be able to run. I will provide some files soon.

For those who do not have a ZWO camera, i will also provide soon a modified version of JetsonSky which will be able to manage videos (no camera required).

IMPORTANT license information :
This softawre and any part of this software are free of use for personal use.
This softawre and any part of this software are NOT free of use for any kind of professional or commercial use.


PC VERSION of JetsonSky :

You will find main files in the "PC Windows version" directory. In fact, it is the same main program as Jetson version.

Your PC must be Window 10/11 64bits system with NVIDIA GPU.

You will need to install CUDA :

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64

You will need Python 3 (i recommend Python 3.10 ; DO NOT USE PYTHON 3.11 beta) :

https://www.python.org/downloads/windows/

You will need to install some Python libraries :
- openCV library
- numpy
- pillow
- pycuda
- six

You can find them here : 

https://www.lfd.uci.edu/~gohlke/pythonlibs/

To get the right version, just take an example wth pyCUDA. You have CUDA 11.6 and Python 3.10 with a 64bits windows 10/11 :

You will need to download : pycuda‑2022.1+cuda116‑cp310‑cp310‑win_amd64.whl

For openCV, you will need to download : opencv_python‑4.5.5‑cp310‑cp310‑win_amd64.whl

And so on ...

To install the packages :
You will have to open a Windows terminal (administrator) and :
py -m pip install your_package.whl

Some explanations here : https://pip.pypa.io/en/latest/user_guide/#installing-from-wheels

Note : maybe you will have to upgrade PIP :
py -m pip install --upgrade pip

As it is ZWO camera, you will have to install ZWO driver (ASI camera). You can find it here :
https://astronomy-imaging-camera.com/software-drivers

The rest is the same as Jetson version. You will have to create Images and Videos and set the good path in the main program.

If everything is ok, the software will run.

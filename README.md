# JETSONSKY

**** 2022-08-22 UPDATE : I have uploaded the latest version of zwoasi library (Pyhon gateway to ZWO SDK library) for both Linux and Windows systems. Many thanks to Steve Marple for this great library. The old version of the library is in the Old_version directory.

I have also fixed a bug with monochrome video DEMO. Now it works. The version is still V14_07 for Linux and Windows systems.


**** 2022-08-21 UPDATE : minor bug fixes. The Pillow version do not need anymore to be set manually in the software. Version is now V14_07 for both Linux and Windows versions

**** 2022-08-19 UPDATE : I made a minor change in JetsonSky (remain V14_06) due to Pillow versions.
For older Pillow version, PIL.Image.Resampling.NEAREST (newest version) must be replaced by PIL.Image.NEAREST (old versions).
You can choose the good version in the software lines 2040-2044.
I have also tested JetsonSky successfully with Jetson Xavier NX.

If you use a Jetson Nano 2GB, you could have some memory issues. Maybe you will have to create a SWAP file.

I really recommend at least a Jetson Xavier NX to run my software. Jetson Nano will be a bit weak with hires cameras.


**** 2022-08-17 UPDATE : New version of JetsonSky for both Linux & Windows. bugs fixed for mono camera treatments. Version is now 14.06

**** 2022-08-16 UPDATE : New version of JetsonSky for both Linux & Windows. Mainly bugs fixed for mono camera treatments. Version is now 14.05

**** 2022-08-14 UPDATE : New version of JetsonSky for both Linux & Windows. Small changes (bugs fix). Version is now 14.03

**** 2022-08-09 UPDATE : New libraries from ZWO SDK (V1.26) uploaded for both Linux & Windows versions

**** 2022-08-07 UPDATE : ABOUT ZWO cameras

If you use a Windows 10/11 PC, you will need ZWO ASI camera driver installed on your computer (as i said previously, see below).

If you use Linux system (such as Nvidia Jetson SBC or x64 system under Linux), yu won't need ZWO ASI camera driver. You will need ZWO SDK for Linux which is in the Lib directory.
You will alsso need to instal asi.rules (which is in Lib directory). To install those rules, just do :

$ sudo install asi.rules /lib/udev/rules.d

or

$ sudo install asi.rules /etc/udev/rules.d

Just check where the rules are to see which version of the command you need.

**** 2022-08-02 UPDATE : upload of a PDF file to explain options you can set in the main window of JetsonSky

**** 2022-08-01 UPDATE : New Linux and Windows versions released (with camera control). Version 12_02
Changes :
- added a variable to set Main Window fonts size (from 5 to 7, depending of your system.
Set the good value in line 41 in the program

- added support of ZWO ASI224MC camera
- added support of ZWO ASI290MC and ASI290MM camera

**** 2022-07-30 UPDATE : Video treatment software (no camera control) is uploaded. 2 versions :
- in the main directory : Jetson version
- in Windows directory : Windows 10/11 version

**** 2022-07-29 UPDATE : PC version (with camera control) is uploaded. See below.

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

NOTE : to install PyCuda, here is a command line you should try (many thanks to jaybdub from NVIDIA) :

sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda

IMPORTANT : Check your system path for cuda location and name. Maybe you will have to modify the include and lib64 PATH, depending of the name of the CUDA directory on your system


In the Lib directroy, you will find ZWO drivers for linux (ZWO SDK ArmV8 for linux).

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


********************************************************

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

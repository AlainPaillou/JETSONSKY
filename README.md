# JETSONSKY

This program is mainly dedicated to astronomy live video with live treatments.
This program require an camera for acquisitions. The supported cameras are :
- ZWO ASI178MC and ASI 178MM
- ZWO ASI485MC and ASI585MC

This program works with Nvidia Jetson single board computer. It was possible for me to develop this software with the support of NVIDIA. I would like to thank Dustin Franklin from NVIDIA who supported me for years. Many thanks Dusty !

You will need CUDA installed on the Jetson computer

You will also need :
- Python (version depends of your Jetson Model)
- openCV library
- numpy
- pillow
- pycuda

In the Lib directroy, you will find ZWO drivers for linux (ArmV8)

In the zwoasi directory, you will find a Python binning to use ZWO camera library.

You will also have to create 2 directories in you installation directory :
- one for the images you will acquire
- one for the video you will acquire

There is an other directory (zwoefw) which gets Python binning for ZWO EFmini filter wheel. This is in standby for now, waiting for issues solving.

JetsonSky still have bugs i will solve ASAP.

JetsonSky can also be used on a PC Windows 10/11 64b system. The PC must have a NVIDIA graphic card to be able to run. Il will provide some files soon.

For those who do not have a ZWO camera, i will provide soon a modified version of JetsonSky which will be able to manage videos (no camera required).

IMPORTANT license information :
This softawre and any part of this software are free of use for personal use.
This softawre and any part of this software are NOT free of use for any kind of professional or commercial use.

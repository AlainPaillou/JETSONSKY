# JETSONSKY

**** 2022-12-03 UPDATE : New version released V17_03 added (JetsonSky).

Bugs removed.

Older version have been removed

**** 2022-11-28 UPDATE : New version released V17_02 added (JetsonSky and JetsonTreatment).

Bugs removed.

Oldest versions have been removed

**** 2022-11-26 UPDATE : New beta version released V17_01d added to Jetsonsky directory

This version add support of Cupy and VPI libraries.


**** 2022-11-15 UPDATE : New version released V15_04. Old V15_03 has been removed.

Bugs fixed and small optimizations.

Also released JetsonTreatment V15_04. For video treatment only (no camera needed).


**** 2022-11-14 UPDATE : New version released V15_03. Old V15_02 has been removed.

Bugs fixed.


**** 2022-11-12 UPDATE : New version released V15_02. Old V15_01 has been removed.

Bugs fixed and small speed enhancement (Minimize the use of PILLOW which is quite slow library).


**** 2022-11-11 UPDATE : Changes in directories organization and new version of JetsonSky

Now, everything is in one directory : JetsonSky

The new version of JetsonSky (V15_01) is the same for Linux and Windows systems.

This new version of the software gets speed improvements for live treatments.

The previous directories are still here for some time. If new JetsonSky version is ok, i will remove the old directories.


**** 2022-11-03 UPDATE : New version of JetsonSky released (V14_20beta) :
Windows and Linux programs are now the same. Il will modify the directories structure later in order to have only 1 directory for both versions.

Changes :
- some improvements to speed up the software
- modified the ghost reduction with a slider to choose the reduction. Ghost reduction will be only active with Low dynamic AADF noise removal filter

This is a beta version because i did not have enough time to test everything.


**** 2022-10-24 UPDATE : Issue solved in V14_19. Version stay the same. Need to upload again V14_19.

**** 2022-10-21 UPDATE : new versions of JetsonSky for both Linux and Windows systems. New version is V14_19.

Added a "ghost reducer" when using adaptative absorber denoise filter (AADF). Just check or uncheck the checkbox "ghost reducer".

Added support for ZWO ASI533MM/MC and ASI533MM/MC Pro (i did not test this support).


**** 2022-09-25 UPDATE : new versions of JetsonSky for both Linux and Windows systems. New version is V14_16.

Added some cameras. Supported cameras :
- ASI178MC, ASI178MM, ASI178MM Pro
- ASI224MC
- ASI290MC, ASI290MM
- ASI294MC, ASI294MM, ASI294MC Pro, ASI294MM Pro
- ASI385MC
- ASI462MC
- ASI482MC
- ASI485MC, ASI585MC
- ASI662MC
- ASI678MC
- ASI1600MC, ASI1600MM


**** 2022-09-20 UPDATE : new versions of JetsonSky and Video treatment only software for both Linux and Windows systems. It is beta versions (V14_13).
Changes :
- satellite detection & tracking have been modified
- contraste CLAHE for night video have been modified
- added support for ZWO ASI678MC, ASI662MC, ASI482MC, ASI462MC, ASI385MC cameras

So, supported cameras are :
- ASI178MM & ASI178MC
- ASI224MC
- ASI290MM & ASI290MC
- ASI385MC
- ASI462MC
- ASI482MC, ASI485MC, ASI585MC
- ASI662MC
- ASI678MC

Those new files ARE NOT in the ZIP files.


**** 2022-09-11 UPDATE : I have changed the organisation of the directory. Now, we have 1 directory for Linux ArmV8 version and 1 directory for Windows version.
In each directory, you will find all the files and i have added 1 zip file with all the files and directories. It will be more simple to download the needed files.

**** 2022-08-24 UPDATE : New version released (V14_09) for Linux and Windows. Some bugs fixed and code optimization to improve live treatments speed. Did not have time to make deep tests so some bugs may occur. Old version (V14_07) still here for download if necessary.

For Windows version only : V14_10 released with the ZWO EFWmini 5 positions control (electrical filter wheel).

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

You will need to download : pycuda???2022.1+cuda116???cp310???cp310???win_amd64.whl

For openCV, you will need to download : opencv_python???4.5.5???cp310???cp310???win_amd64.whl

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

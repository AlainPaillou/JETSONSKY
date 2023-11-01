# JETSONSKY

**** 2023-11-01 UPDATE : 1 new version uploaded V40_26RC. Previous version (V40_25RC) has been removed.

- Some bugs removed.

**** 2023-09-14 UPDATE :
Some useful informations to install JetsonSky can be find here :

https://forums.developer.nvidia.com/t/electronically-assisted-astronomy-with-a-jetson-nano/76861/764




**** 2023-08-27 UPDATE : 1 new version uploaded V40_25RC. Previous version (V40_23RC) has been removed.

- Some bugs removed.
- more simple stars & satellites detection routines.

**** 2023-08-22 UPDATE : 1 new version uploaded V40_23RC. Previous version (V40_22RC) has been removed.

- Some bugs removed.
- some changes with color saturation enhancement.

Donc forget to check if you have the latest ZWO SDK libraries. You can find the SDK for Windows and Linux here :

https://www.zwoastro.com/downloads/developers

If you use Windows system and if you have installed GStreamer, check JetsonSky lines 40 to 41 to set the good GStreamer paths.


**** 2023-07-15 UPDATE : 1 new version uploaded V40_22RC. Previous version (V40_21RC) has been removed.

- Some bugs removed.

**** 2023-07-14 UPDATE : 1 new version uploaded V40_21RC. Previous version (V40_20RC) has been removed.

- Some bugs removed.
- added a checkbox to reverse Red and Blue channels.

For memory :
- V30_03RC uses Pycuda. This version is the last version with Pycuda. No more development with this library
- V40_XXRC use Cupy. This is LT releases.

**** 2023-07-09 UPDATE : 1 new version uploaded V40_20RC. Previous version (V40_19RC) has been removed.

Small update to avoid error with Windows version if GStreamer directory is missing or if you set the wrong paths in the source code (lines 40 to 42).


**** 2023-06-25 UPDATE : 1 new version uploaded V40_19RC.

- added gamma correction (internal camera correction) for camera setup (ZWO cameras)
- added slow or fast image read (camera). Slow option will give better image
  

**** 2023-06-09 UPDATE : a brief documentation about V40_18RC has been uploaded (JETSONSKY_V40_18RC_INFO.pdf)


**** 2023-06-08 UPDATE : 1 new version uploaded V40_18RC. I made some outdoor tests but it can still get some bugs.

- Can now set 3 different ratio for the camera capture (4/3, 16/9 and 1/1)
- can now select Video directory and Images directory. It is useful if you want to use an other drive (for example external SSD drive)
- some bugs fixed


**** 2023-05-28 UPDATE : 2 new versions are uploaded. Old V40_14RC has been removed.

- New V40_15RC is in fact V40_14C with some bugs removed
- New V40_16RC added a trigger for satellite detection (check box under satellite checkbox on the left of the screen). Means when click Start Record, recording will be active only if a satellite is detected. Otherwise, recording wait for satellites. It's a beta version function. Needs some tests in real conditions. Also work with video treatment only (no camera).


**** 2023-05-27 UPDATE :

Bugs fixed in V40_14RC version. It was impossible to make image treatment with this very small bug. The V40_14RC needs to be reloaded.


**** 2023-05-25 UPDATE : New version V40_14RC released.

Bugs fixed. V40_12RC removed.


**** 2023-05-24 UPDATE : New version V40_12RC released.

This is a 3 in 1 version :
- can manage ZWO camera
- can manage video only (no camera)
- can manage image only (no camera)

This is a RC version and i think it may be buggy. It seems to work but i did not make many tests with it.


**** 2023-05-21 UPDATE : New version V40_11RC released.

Bugs fixed


**** 2023-05-18 UPDATE : New version V40_09RC released.

Bugs fixed

Some improvements

GStreamer mp4 video saving with GPU acceleration seems to be ok with Windows plateform


**** 2023-05-14 UPDATE : New version V40_07RC released.

Bug fixed.


Some old versions have been removed.


If you used old V40_0XRC version, i recommand you to use the latest version/


**** 2023-05-11 UPDATE : New version V40_06RC released.

Bug fixed.


**** 2023-05-10 UPDATE : New versions V30_03RC and V40_05RC released. Older versions removed.

Some bugs are solved.

More stable FPS (information displayed). Many thanks to Honey_Patouceul.

Some optimizations.

Python priority set to highest (need to start Python with superuser mode under Windows). Brings higher frame rate.


**** 2023-05-07 UPDATE : New versions V30_02RC and V40_04RC released. Bugs removed.

V30_01RC and V40_03RC have been removed.


**** 2023-05-05 UPDATE : New versions released. It's Release Candidates (tes)t versions.

V30_01RC : it is the last version of JetsonSky using PyCuda.

Improve frame rate, treatments speed.
Can manage a camera (if a camera is plugged). If no camera, it will manage existing videos.

V40_03RC : This version does not use PyCuda. Cupy replaces PyCuda.

Improve frame rate, treatments speed.
Can manage a camera (if a camera is plugged). If no camera, it will manage existing videos.
Some useless filters have been removed.

The future version won't use PyCuda anymore.


**** 2023-04-12 UPDATE : New version released V20_09Beta added (beta version). It's test version. It replace V20_08beta version (removed).

Some small improvements.

Many thanks to Honey_Patouceul !


**** 2023-04-10 UPDATE : New version released V20_08Beta added (beta version). It's test version.

Better frame management.

Added FPS numbrer in TIP.

GSTreamer hardware encoding for compressed colour video supported.



**** 2023-03-31 UPDATE : New versions released V20_05RC added (release candidate). It's test versions.

3 softwares :
- JetsonSky : needs ZWO camera - Acquisition and live treatments
- Jetson Videos : dedicated to video treatments (load an existing video and apply treatments). No camera management
- Jetson Images : dedicated to image treatments (load an existing image and apply treatments). No camera management

Video and Image version software is more user friendly (you can directly load and save Images/Videos)

The 3 software get now relative path. This means you don't need anymore to set your entire path in the code. You just have to respect this directories architecture :

../../YourDirectory/

../../YourDirectory/Videos

../../YourDirectory/Images

../../YourDirectory/Lib

../../YourDirectory/zwoasi

../../YourDirectory/zwoefw


As it is RC software, i guess there are some bugs inside.


**** 2023-03-12 UPDATE : New version released V18_05aRC added (JetsonSky). It's a test version.

V18_05aRC allow GSTREAMER use for Video writing if OpenCV has been compiled with GSTREAMER (hardware encoding). This will give a higher frame rate when writing the video.



**** 2023-02-25 UPDATE : New version released V18_02RC added (JetsonSky).

Some issues are solved from V18_01RC version.

**IMPORTANT** : you must get the new __init__.py file in zwoasi directory (and place it in your zwoasi directory) to get the V18_02RC version works !!!




**** 2023-02-23 UPDATE : New version released V18_01RC added (JetsonSky).

Better management of frames acquisition (frame rate should be better). Don't forget to adjust USB parameter during capture in order to get the best frame rate.

Added HDR function for colour capture only.

**IMPORTANT** : you must get the new __init__.py file in zwoasi directory (and place it in your zwoasi directory) to get the V18_01RC version works !!!

This is a release candidate version because i did not test it in real conditions.


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

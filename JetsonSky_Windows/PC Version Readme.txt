PC VERSION of JetsonSky :

You will find main files in the "PC Windows version" directory. In fact, it is the same main program as Jetson version.

Your PC must be Window 10/11 64bits system with NVIDIA GPU.

You will need to install CUDA :

https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64

You will also need Microsoft C compiler cl.exe : you can install Microsoft Visual studio with C/C++ to get this compiler (it's free for personal use). 
Dont forget to add cl.exe PATH to the windows environment variables !

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

Don't forget to create Lib directory and put in this library the ZWO SDK libraries.

If everything is ok, the software will run.

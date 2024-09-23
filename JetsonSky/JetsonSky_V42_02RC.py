
############################################################################################################
#                                                                                                          #
#                      Images/videos acquisition & treatment software for ZWO cameras                      #
#                                    Images/videos treatment software                                      #
#                                                                                                          #
#                                                                                                          #
#                                   Copyright Alain Paillou 2018-2024                                      #
#                                                                                                          #
############################################################################################################


# Supported camera :  
#                                                                                                         
# - ASI178MC, ASI178MM, ASI178MM Pro
# - ASI224MC
# - ASI290MC, ASI290MM
# - ASI294MC, ASI294MM, ASI294MC Pro, ASI294MM Pro
# - ASI385MC
# - ASI462MC
# - ASI482MC
# - ASI485MC, ASI585MC
# - ASI533MC, ASI533MM, ASI533MC Pro, ASI533MM Pro
# - ASI662MC
# - ASI678MC, ASI678MM
# - ASI1600MC, ASI1600MM

############################################################################################################


# Libraries import

import sys
my_os = sys.platform

import os
import random


# Set your GSTREAMER directories in the following lines
if my_os == "win32" :
    try :
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\bin")
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\lib")
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\lib\\gstreamer-1.0")
    except :
        print("No GSTREAMER directory find")
    
import time
import numpy as np

try :
    import cupy as cp
    flag_cupy = True
except :
    flag_cupy = False

if flag_cupy == True :
    from cupyx.scipy import ndimage

if flag_cupy == True :
    print("Cupy libray OK")

s1 = cp.cuda.Stream(non_blocking=False)

import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw
from PIL import ImageEnhance
from tkinter import *
from tkinter.messagebox import askyesno
from tkinter import filedialog as fd
from tkinter.font import nametofont

import cv2
try :
    CudaEnable = cv2.cuda.getCudaEnabledDeviceCount()
    if CudaEnable == 1 :
        flag_OpenCvCuda = True
    else :
        flag_OpenCvCuda = False
except :
        flag_OpenCvCuda = False
if flag_OpenCvCuda == True :
    print("OpenCV loaded with CUDA")
else :
    print("OpenCV loaded")

from threading import Thread
from datetime import datetime
import argparse
import math
import zwoasi as asi
try :
    import zwoefw as efw
except :
    pass


############################################
#                Main program              #
############################################

JetsonSky_version = "V42_02RC"

# Choose the size of the fonts in the Main Window - It depends of your system - can be set from 5 to 7
MainWindowFontSize = 6

if my_os == "win32" :
    Dev_system = "Windows"
    print("Windows system")
    titre = "JetsonSky " +  JetsonSky_version + " Windows release - CUDA / OPENCV / CUPY / TorchVision - Copyright Alain PAILLOU 2018-2024"

if my_os == "linux" :
    Dev_system = "Linux"
    print("Linux system")
    titre = "JetsonSky " +  JetsonSky_version + " Linux release - CUDA / OPENCV / CUPY / TorchVision - Copyright Alain PAILLOU 2018-2024" 

if Dev_system == "Linux" :
    # Choose your directories for images and videos
    #Internal SSD path
    image_path = os.path.join(os.getcwd(), 'Images')
    video_path = os.path.join(os.getcwd(), 'Videos')

    # Path to librairies ZWO Jetson sbc
    env_filename_camera = os.path.join(os.getcwd(), 'Lib','libASICamera2.so')
    env_filename_efw = os.path.join(os.getcwd(), 'Lib','libEFWFilter.so')

    USB178 = 80
    USB485 = 90    
    
else :
    # Choose your directories for images and videos
    image_path = os.path.join(os.getcwd(), 'Images')
    video_path = os.path.join(os.getcwd(), 'Videos')
    
    # Path to librairies ZWO Windows
    env_filename_camera = os.path.join(os.getcwd(), 'Lib','ASICamera2.dll')
    env_filename_efw = os.path.join(os.getcwd(), 'Lib','EFW_filter.dll')

    USB178 = 95
    USB485 = 95
    

# Setting the Images & videos quality
flag_HQ = 0 # if 1 : save HQ RAW    if 0 : save LQ compressed format

# Variables initialization
exp_min=100 #µs
exp_max=10000 #µs
exp_delta=100 #µs
exp_interval=2000
val_resolution = 1
mode_BIN=1
delta_s = 0
fact_s = 1.0
res_cam_x = 3096
res_cam_y = 2080
cam_displ_x = int(1350 * fact_s)
cam_displ_y = int(1012 * fact_s)
sensor_factor = "4/3"

RES_X_BIN1_4_3 = [3096,2560,1920,1600,1280,1024,800,640,320]
RES_Y_BIN1_4_3 = [2080,1920,1440,1200,960,768,600,480,240]
RES_X_BIN1_16_9 = [3096,2560,1920,1600,1280,1024,800,640,320]
RES_Y_BIN1_16_9 = [2080,1920,1440,1200,960,768,600,480,240]
RES_X_BIN1_1_1 = [3096,2560,1920,1600,1280,1024,800,640,320]
RES_Y_BIN1_1_1 = [2080,1920,1440,1200,960,768,600,480,240]

RES_X_BIN1 = RES_X_BIN1_4_3
RES_Y_BIN1 = RES_Y_BIN1_4_3

RES_X_BIN2_4_3 = [1544,1280,960,800,640,512,400]
RES_Y_BIN2_4_3 = [1040,960,720,600,480,384,300]
RES_X_BIN2_16_9 = [1920,1544,1280,960,800,640,512]
RES_Y_BIN2_16_9 = [1080,1040,960,720,600,480,384]
RES_X_BIN2_1_1 = [1544,1280,960,800,640,512,400]
RES_Y_BIN2_1_1 = [1040,960,720,600,480,384,300]

RES_X_BIN2 = RES_X_BIN2_4_3
RES_Y_BIN2 = RES_Y_BIN2_4_3

val_exposition = 1000
timeoutexp = 1 + 500
exposition = 0
val_gain = 100
val_maxgain = 600
val_denoise = 0.4
val_histo_min = 0
val_histo_max = 255
val_contrast_CLAHE = 1.5
val_reduce_variation = 2
val_phi = 1.0
val_theta = 100
val_heq2 = 1.0
text_info1 = "Test information"
val_nb_captures = 1
nb_cap_video =0
val_nb_capt_video = 100
compteur_images = 0
numero_image = 0
image_camera = 0
image_camera_old = 0
val_nb_darks = 5
dispo_dark = 'Dark NO       '
FlipV = 0
FlipH = 0
ImageNeg = 0
val_red = 63
val_blue = 74
val_FS = 1
compteur_FS = 0
compteur_AADF = 0
compteur_RV = 0
val_denoise_KNN = 0.2
val_USB = 90
val_SGR = 95
val_AGR = 50
val_NGB = 13
val_SAT = 1.0
ASIGAMMA = 50
ASIAUTOMAXBRIGHTNESS = 140
nb_erreur = 0
text_TIP = ""
val_ampl = 1.0
val_deltat = 0
timer1 = 0.0
grad_vignet = 1
type_debayer = cv2.COLOR_BayerBG2RGB
stack_div = 1
val_reds = 100
val_blues = 100
val_greens = 100
val_sdp = 2
val_seuil_denoise = 180
val_Mu = 0.0
val_Ro = 1.0
nb_sat = 0
xdark=1750
ydark=843
xLI1=1475
yLI1=1000
Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')
hauteur = 0.0
azimut = 0.0
val_ghost_reducer = 50
val_grid_CLAHE = 8
total_start = 0.0
total_stop = 0.0
type_gamma="cuda"
video_debayer = 0 # RAW
start_time_video = 0
stop_time_video = 0
type_flip = "none"
time_exec_test = 0.0
total_time = 1.0
font = cv2.FONT_HERSHEY_SIMPLEX
val_sharpen = 1.0
val_sigma_sharpen = 1.0
val_sharpen2 = 1.0
val_sigma_sharpen2 = 3.0
trig_count_down = 0
video_frame_number = 0
video_frame_position = 0
compteur_3FNR  = 0

beta = 1.0
alpha = 5
Corr_cont = np.zeros(255,dtype=np.single)
for x in range(0,255) :
    Corr_cont[x] = (1-math.exp(-1*x/alpha)+beta)/(1+beta)


    
trsf_r = np.zeros(256,dtype=int)
trsf_g = np.zeros(256,dtype=int)
trsf_b = np.zeros(256,dtype=int)
for x in range(1,256) :
    trsf_r[x] = x
    trsf_g[x] = x
    trsf_b[x] = x

Corr_GS = np.zeros(255,dtype=np.single)
for x in range(0,255) :
    Corr_GS[x] = 1.0

Corr_CLL = np.zeros(256,dtype=np.uint8)
val_MuCLL = 0
val_RoCLL = 0.5
val_AmpCLL = 1.0
for x in range(0,256) :
    Corr = np.exp(-0.5*((x*0.0392157-val_MuCLL)/val_RoCLL)**2)
    Corr_CLL[x] = int(x * (1/(1 + val_AmpCLL*Corr)))
    if x> 0 :
        if Corr_CLL[x] <= Corr_CLL[x-1] :
            Corr_CLL[x] = Corr_CLL[x-1]

stars_x = np.zeros(10000,dtype=int)
stars_y = np.zeros(10000,dtype=int)
stars_s = np.zeros(10000,dtype=int)

sat_x = np.zeros(10000,dtype=int)
sat_y = np.zeros(10000,dtype=int)
sat_s = np.zeros(10000,dtype=int)
nb_sat = 0
max_sat = 12

curFPS = 0.0    
fpsQueue = []

curTT = 0.0    
TTQueue = []
 

def quitter() :
    global camera,flag_autorise_acquisition,thread1,fenetre_principale,flag_image_disponible,flag_quitter

    if flag_camera_ok == True :
        flag_autorise_acquisition = False
        flag_quitter = True
        time.sleep(1)
        flag_image_disponible = False
        thread_1.join()
        if flag_filter_wheel == True :
            filter_wheel.close()
        try :
            camera.stop_video_capture()
            camera.close()
        except :
            print("Close camera error")
        fenetre_principale.quit()
    else :
        flag_autorise_acquisition = False
        time.sleep(1)
        fenetre_principale.quit()
        


# Main Window
fenetre_principale = Tk ()
screen_width = fenetre_principale.winfo_screenwidth()
screen_height = fenetre_principale.winfo_screenheight()

if screen_width > 2000 :
    if askyesno("Hires Window", "Choose Hires Window ?") :
        fact_s = 1.33
        delta_s = 460
        w,h=int(1920*(fact_s-0.1)),int(1060*fact_s)
    else :
        fact_s = 1
        delta_s = 0
        w,h=1920,1060
else :
    fact_s = 1
    delta_s = 0
    w,h=1920,1060
    
fenetre_principale.geometry("%dx%d+0+0" % (w, h))
fenetre_principale.protocol("WM_DELETE_WINDOW", quitter)
default_font = nametofont("TkDefaultFont")
default_font.configure(size=MainWindowFontSize)
fenetre_principale.title(titre)


# Création cadre général
cadre = Frame (fenetre_principale, width = w , heigh = h)
cadre.pack ()

gradient_vignetting = IntVar()
gradient_vignetting.set(1) # Initialisation du mode gradient ou vignetting a gradient

choix_FrontBack = IntVar ()
choix_FrontBack.set(1) # Initialisation filtrage histogram en Front

mode_acq = IntVar()
mode_acq.set(2) # Initialisation du mode d'acquisition a Moyen

choix_autoexposure = IntVar ()
choix_autoexposure.set(0) # Initialisation autoexposure camera

choix_autogain = IntVar ()
choix_autogain.set(0) # Initialisation autogain camera

choix_stacking = IntVar ()
choix_stacking.set(1) # Initialisation mode stacking Mean

choix_bin = IntVar ()
choix_bin.set(1) # Initialisation BIN 1 sur choix 1 ou 2

mode_Lin_Gauss = IntVar ()
mode_Lin_Gauss.set(1) # Initialisation choix amplification Lineaire Gaussien

choix_TIP = IntVar ()
choix_TIP.set(0) # Initialisation TIP Inactif

choix_cross = IntVar ()
choix_cross.set(0) # Initialisation Croix centre image

choix_mount = IntVar ()
choix_mount.set(0) # Initialisation mount read pos

choix_hold_picture = IntVar ()
choix_hold_picture.set(0) # Initialisation hold picture

choix_SAT = IntVar () # réglage saturation couleurs
choix_SAT.set(0)

choix_flipV = IntVar ()
choix_flipV.set(0) # Initialisation Flip V inactif

choix_flipH = IntVar ()
choix_flipH.set(0) # Initialisation Flip H inactif

choix_img_Neg = IntVar ()
choix_img_Neg.set(0) # Initialisation image en négatif inactif

choix_2DConv = IntVar()
choix_2DConv.set(0) # intialisation filtre 2D convolution inactif

choix_mode_full_res = IntVar()
choix_mode_full_res.set(0) # Initialisation mode full resolution inactif

choix_sharpen_soft1 = IntVar()
choix_sharpen_soft1.set(0) # initialisation mode sharpen software 1 inactif

choix_sharpen_soft2 = IntVar()
choix_sharpen_soft2.set(0) # initialisation mode sharpen software 2 inactif

choix_denoise_soft = IntVar()
choix_denoise_soft.set(0) # initialisation mode denoise software inactif

choix_histogram_equalize2 = IntVar()
choix_histogram_equalize2.set(0) # initialisation mode histogram equalize 2 inactif

choix_histogram_stretch = IntVar()
choix_histogram_stretch.set(0) # initialisation mode histogram stretch inactif

choix_histogram_phitheta = IntVar()
choix_histogram_phitheta.set(0) # initialisation mode histogram Phi Theta inactif

choix_contrast_CLAHE = IntVar()
choix_contrast_CLAHE.set(0) # initialisation mode contraste CLAHE inactif

choix_CLL = IntVar()
choix_CLL.set(0) # initialisation mode contraste low light inactif

choix_noir_blanc = IntVar()
choix_noir_blanc.set(0) # initialisation mode noir et blanc inactif

choix_reverse_RB = IntVar()
choix_reverse_RB.set(0) # initialisation mode reverse Reb Blue inactif

choix_noir_blanc_estime = IntVar()
choix_noir_blanc_estime.set(0) # initialisation mode noir et blanc estimate inactif

choix_hard_bin = IntVar()
choix_hard_bin.set(0) # initialisation mode hardware bin disable

choix_HQ_capt = IntVar()
choix_HQ_capt.set(0) # initialisation mode capture Low Quality

choix_filtrage_ON = IntVar()
choix_filtrage_ON.set(1) # Initialisation Filtrage ON actif

choix_denoise_KNN = IntVar()
choix_denoise_KNN.set(0) # Initialisation Filtrage Denoise KNN

choix_denoise_Paillou = IntVar()
choix_denoise_Paillou.set(0) # Initialisation Filtrage Denoise Paillou

choix_denoise_Paillou2 = IntVar()
choix_denoise_Paillou2.set(0) # Initialisation Filtrage Denoise Paillou 2

choix_denoise_stacking_Paillou = IntVar()
choix_denoise_stacking_Paillou.set(0) # Initialisation Filtrage Denoise Paillou stacking

choix_3FNR = IntVar()
choix_3FNR.set(0) # Initialisation Filtrage 3 frames noise removal

choix_reduce_variation = IntVar()
choix_reduce_variation.set(0) # Initialisation Filtrage Denoise Paillou stacking

choix_sub_dark = IntVar()
choix_sub_dark.set(0) # Initialisation sub dark inactif

choix_GR = IntVar()
choix_GR.set(0) # Initialisation Filtre Gradient Removal

presence_FW = IntVar()
presence_FW.set(0) # Initialisation absence FW

fw_position_ = IntVar()
fw_position_.set(0) # Position filter wheel

cam_read_speed = IntVar()
cam_read_speed.set(1) # Set cam read speed to slow

sensor_ratio = IntVar()
sensor_ratio.set(0) # Set sensor ratio

choix_AmpSoft = IntVar()
choix_AmpSoft.set(0) # Initialisation amplification software OFF

bayer_sensor = IntVar()
bayer_sensor.set(1) # Position filter wheel

choix_HST = IntVar()
choix_HST.set(0) # Initialisation histogram OFF

choix_HDR = IntVar()
choix_HDR.set(0) # Initialisation HDR OFF

choix_TRSF = IntVar()
choix_TRSF.set(0) # Initialisation histogram OFF

choix_TRGS = IntVar()
choix_TRGS.set(0) # Initialisation histogram transformation gain soft

choix_TRCLL = IntVar()
choix_TRCLL.set(0) # Initialisation histogram contrast Low Light LUT

choix_DEMO = IntVar()
choix_DEMO.set(0) # Initialisation Demo OFF

choix_STAB = IntVar()
choix_STAB.set(0) # Initialisation Stabilization OFF

choix_DETECT_STARS = IntVar()
choix_DETECT_STARS.set(0) # Initialisation Detect Stars OFF

choix_dyn_AADP = IntVar ()
choix_dyn_AADP.set(1) # Initialisation mode dynamique High AADP

choix_ghost_reducer = IntVar ()
choix_ghost_reducer.set(0) # Initialisation ghost reducer

choix_TRKSAT = IntVar()
choix_TRKSAT.set(0) # Initialisation Track Satellites OFF

choix_REMSAT = IntVar()
choix_REMSAT.set(0) # Initialisation Remove Satellites OFF

choix_CONST = IntVar()
choix_CONST.set(0) # Initialisation Reconstruct image OFF

choix_TRIGGER = IntVar()
choix_TRIGGER.set(0) # Initialisation Record video on trigger OFF

# Initialisation des filtres soft
flag_full_res = 0
flag_sharpen_soft1 = 0
flag_sharpen_soft2 = 0
flag_unsharp_mask = 0
flag_denoise_soft = 0
flag_histogram_equalize2 = 0
flag_histogram_stretch = 0
flag_histogram_phitheta = 0
flag_contrast_CLAHE = 0
flag_CLL = 0
flag_noir_blanc = 0
flag_AmpSoft = 0
flag_acquisition_en_cours = False
flag_autorise_acquisition = False
flag_image_disponible = False
flag_premier_demarrage = True
flag_BIN2 = False
flag_BIN3 = False
flag_cap_pic = False
flag_cap_video = False
flag_sub_dark = False
flag_acq_rapide = "MedF"
flag_colour_camera = True
flag_filter_wheel = False
flag_seuillage_PB = False
Im1OK = False
Im2OK = False
Im3OK = False
Im4OK = False
Im5OK = False
Im1fsdnOK = False
Im2fsdnOK = False
Im1rvOK = False
Im2rvOK = False
filter_on = False
flag_filtrage_ON = True
flag_filtre_work = False
flag_denoise_KNN = False
flag_denoise_Paillou = False
flag_denoise_Paillou2 = False
flag_AADF = False
flag_GR = False
flag_TIP = False
flag_cross = False
flag_SAT = False
flag_pause_video = False
flag_hold_picture = 0
flag_quitter = False
flag_autoexposure_exposition = False
flag_autoexposure_gain = False
flag_stacking = "Mean"
flag_HST = 0
flag_TRSF = 0
flag_DEMO = 0
flag_DETECT_STARS = 0
flag_dyn_AADP = 1
flag_front = True
flag_TRGS = 0
flag_lin_gauss = 1
flag_BCap = 0
flag_TRKSAT = 0
flag_REMSAT = 0
flag_CONST = 0
flag_TRIGGER = 0
flag_NB_estime = 0
flag_nouvelle_resolution = False
flag_sat_exist = False
flag_mountpos = True
flag_mount_connect = False
flag_mountpos = False
flag_supported_camera = False
flag_ghost_reducer = 0
flag_BG_Ok = False
flag_HDR = False
flag_new_image = False
flag_torchvision_sharpen = 0
flag_camera_ok = False
flag_image_mode = False
flag_image_video_loaded = False
flag_sat_detected = False
flag_read_speed = "Slow"
flag_reverse_RB = 0
flag_reduce_variation = False
flag_STAB = False
flag_Template = False
FNR_First_Start = False
img1_3FNROK  = False
img2_3FNROK  = False
img3_3FNROK = False
flag_3FNR = False
flag_TRCLL = 0


Contrast_Low_Light_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Low_Light_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, unsigned char *Corr_CLL)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];  
      dest_r[index] = Corr_CLL[vr];
      dest_g[index] = Corr_CLL[vg];
      dest_b[index] = Corr_CLL[vb];
    } 
}
''', 'Contrast_Low_Light_Colour_C')

Contrast_Low_Light_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Low_Light_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, unsigned char *Corr_CLL)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      dest_r[index] = Corr_CLL[vr];
    } 
}
''', 'Contrast_Low_Light_Mono_C')


Contrast_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Contrast_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *luminance, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (luminance[index] > 1.1 *(0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index])) {
          X = luminance[index] / (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
          dest_r[index] = (int)(min(max(int(img_r[index]*X * 0.7), 0), 255));
          dest_g[index] = (int)(min(max(int(img_g[index]*X * 0.7), 0), 255));
          dest_b[index] = (int)(min(max(int(img_b[index]*X * 0.7), 0), 255));
          }
    } 
}
''', 'Contrast_Combine_Colour_C')


reduce_noise_Color = cp.RawKernel(r'''
extern "C" __global__
void reduce_noise_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  int i1,i2,i3,i4,i5,i6,i7,i8,i9;
  int ind1,ind2,ind3,ind4;
  int mini,maxi;
      
  if (i < height && i > 1 && j < width && j >1) {
      i1 = (i-1) * width + (j-1);
      i2 = (i-1) * width + j;
      i3 = (i-1) * width + (j+1);
      i4 = i * width + (j-1);
      i5 = i * width + j;
      i6 = i * width + (j+1);
      i7 = (i+1) * width + (j-1);
      i8 = (i+1) * width + j;
      i9 = (i+1) * width + (j+1);
	  
	  if ((img_r[i5] - img_r[i1]) * (img_r[i9] - img_r[i5]) > 0)
		ind1 = (img_r[i1] + img_r[i5]*5 + img_r[i9]) / 7;
	  else
		ind1 = (img_r[i1] + img_r[i5] + img_r[i9]) / 3;
		
	  if ((img_r[i5] - img_r[i2]) * (img_r[i8] - img_r[i5]) > 0)
		ind2 = (img_r[i2] + img_r[i5]*5 + img_r[i8]) / 7;
	  else
		ind2 = (img_r[i2] + img_r[i5] + img_r[i8]) / 3;
  
	  if ((img_r[i5] - img_r[i3]) * (img_r[i7] - img_r[i5]) > 0)
		ind3 = (img_r[i3] + img_r[i5]*5 + img_r[i7]) / 7;
	  else
		ind3 = (img_r[i3] + img_r[i5] + img_r[i7]) / 3;
		
	  if ((img_r[i5] - img_r[i6]) * (img_r[i4] - img_r[i5]) > 0)
		ind4 = (img_r[i4] + img_r[i5]*5 + img_r[i6]) / 7;
	  else
		ind4 = (img_r[i4] + img_r[i5] + img_r[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_r[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
 

	  if ((img_g[i5] - img_g[i1]) * (img_g[i9] - img_g[i5]) > 0)
		ind1 = (img_g[i1] + img_g[i5]*5 + img_g[i9]) / 7;
	  else
		ind1 = (img_g[i1] + img_g[i5] + img_g[i9]) / 3;
		
	  if ((img_g[i5] - img_g[i2]) * (img_g[i8] - img_g[i5]) > 0)
		ind2 = (img_g[i2] + img_g[i5]*5 + img_g[i8]) / 7;
	  else
		ind2 = (img_g[i2] + img_g[i5] + img_g[i8]) / 3;
  
	  if ((img_g[i5] - img_g[i3]) * (img_g[i7] - img_g[i5]) > 0)
		ind3 = (img_g[i3] + img_g[i5]*5 + img_g[i7]) / 7;
	  else
		ind3 = (img_g[i3] + img_g[i5] + img_g[i7]) / 3;
		
	  if ((img_g[i5] - img_g[i6]) * (img_g[i4] - img_g[i5]) > 0)
		ind4 = (img_g[i4] + img_g[i5]*5 + img_g[i6]) / 7;
	  else
		ind4 = (img_g[i4] + img_g[i5] + img_g[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_g[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));


	  if ((img_b[i5] - img_b[i1]) * (img_b[i9] - img_b[i5]) > 0)
		ind1 = (img_b[i1] + img_b[i5]*5 + img_b[i9]) / 7;
	  else
		ind1 = (img_b[i1] + img_b[i5] + img_b[i9]) / 3;
		
	  if ((img_b[i5] - img_b[i2]) * (img_b[i8] - img_b[i5]) > 0)
		ind2 = (img_b[i2] + img_b[i5]*5 + img_b[i8]) / 7;
	  else
		ind2 = (img_b[i2] + img_b[i5] + img_b[i8]) / 3;
  
	  if ((img_b[i5] - img_b[i3]) * (img_b[i7] - img_b[i5]) > 0)
		ind3 = (img_b[i3] + img_b[i5]*5 + img_b[i7]) / 7;
	  else
		ind3 = (img_b[i3] + img_b[i5] + img_b[i7]) / 3;
		
	  if ((img_b[i5] - img_b[i6]) * (img_b[i4] - img_b[i5]) > 0)
		ind4 = (img_b[i4] + img_b[i5]*5 + img_b[i6]) / 7;
	  else
		ind4 = (img_b[i4] + img_b[i5] + img_b[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_b[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
      }
}
''', 'reduce_noise_Color_C')


reduce_noise_Mono = cp.RawKernel(r'''
extern "C" __global__
void reduce_noise_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  int i1,i2,i3,i4,i5,i6,i7,i8,i9;
  int ind1,ind2,ind3,ind4;
  int mini,maxi;
      
  if (i < height && i > 1 && j < width && j >1) {
      i1 = (i-1) * width + (j-1);
      i2 = (i-1) * width + j;
      i3 = (i-1) * width + (j+1);
      i4 = i * width + (j-1);
      i5 = i * width + j;
      i6 = i * width + (j+1);
      i7 = (i+1) * width + (j-1);
      i8 = (i+1) * width + j;
      i9 = (i+1) * width + (j+1);
	  
	  if ((img_r[i5] - img_r[i1]) * (img_r[i9] - img_r[i5]) > 0)
		ind1 = (img_r[i1] + img_r[i5]*5 + img_r[i9]) / 7;
	  else
		ind1 = (img_r[i1] + img_r[i5] + img_r[i9]) / 3;
		
	  if ((img_r[i5] - img_r[i2]) * (img_r[i8] - img_r[i5]) > 0)
		ind2 = (img_r[i2] + img_r[i5]*5 + img_r[i8]) / 7;
	  else
		ind2 = (img_r[i2] + img_r[i5] + img_r[i8]) / 3;
  
	  if ((img_r[i5] - img_r[i3]) * (img_r[i7] - img_r[i5]) > 0)
		ind3 = (img_r[i3] + img_r[i5]*5 + img_r[i7]) / 7;
	  else
		ind3 = (img_r[i3] + img_r[i5] + img_r[i7]) / 3;
		
	  if ((img_r[i5] - img_r[i6]) * (img_r[i4] - img_r[i5]) > 0)
		ind4 = (img_r[i4] + img_r[i5]*5 + img_r[i6]) / 7;
	  else
		ind4 = (img_r[i4] + img_r[i5] + img_r[i6]) / 3;
       
      mini = int(min(ind1,min(ind2,min(ind3,min(ind4,255)))));
      maxi = int(max(ind1,max(ind2,max(ind3,max(ind4,0)))));
      dest_r[i5] = (int)(min(max(int(mini + (maxi-mini) / 2), 0), 255));
      }
}
''', 'reduce_noise_Mono_C')


Saturation_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *ext_r, unsigned char *ext_g, unsigned char *ext_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]) / (0.299*ext_r[index] + 0.587*ext_g[index] + 0.114*ext_b[index]);
      dest_r[index] = (int)(min(max(int(ext_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(ext_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(ext_b[index]*X), 0), 255));
    } 
}
''', 'Saturation_Combine_Colour_C')


Gaussian_CUDA_Colour = cp.RawKernel(r'''
extern "C" __global__
void Gaussian_CUDA_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, int sigma)
{
    long int j = threadIdx.x + blockIdx.x * blockDim.x;
    long int i = threadIdx.y + blockIdx.y * blockDim.y;
    long int index;
    float red,green,blue;
    float factor;
    int filterX;
    int filterY;
    int imageX;
    int imageY;
    #define filterWidth 7
    #define filterHeight 7

    index = i * width + j;


    float filter[filterHeight][filterWidth] =
    {
      0, 0, 1, 2, 1, 0, 0,
      0, 3, 13, 22, 11, 3, 0,
      1, 13, 59, 97, 59, 13, 1,
      2, 22, 97, 159, 97, 22, 2,
      1, 13, 59, 97, 59, 13, 1,
      0, 3, 13, 22, 11, 3, 0,
      0, 0, 1, 2, 1, 0, 0,
    };
    
    factor = 1.0 / 1003.0;

    red = 0.0;
    green = 0.0;
    blue = 0.0;

    if (i < height && j < width) {
    
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
     }
    dest_r[index] = (int)(min(int(factor * red), 255));
    dest_g[index] = (int)(min(int(factor * green), 255));
    dest_b[index] = (int)(min(int(factor * blue), 255));
    }
}
''', 'Gaussian_CUDA_Colour_C')


Colour_2_Grey_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_2_Grey_GPU_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      dest_r[index] = (int)(0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
    } 
}
''', 'Colour_2_Grey_GPU_C')


grey_estimate_Mono = cp.RawKernel(r'''
extern "C" __global__
void grey_estimate_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index1,index2,index3,index4;
  float colonne,ligne;
  long int nb_pixels = width*height;
  int vr;
  
  index1 = i * width + j;
  index2 = i * width + j+1;
  index3 = (i+1) * width + j;
  index4 = (i+1) * width + (j+1);

  if (i < height && j < width) {
    colonne = (j/2-int(j/2))*2;
    ligne = (i/2-int(i/2))*2;

    if ((colonne == 0 && ligne == 0) || (colonne == 1 && ligne == 1)) {
        dest_r[index1] = (int)(min(max(int(img_r[index1]+(img_r[index2]+img_r[index3])/2+img_r[index4]), 0), 255));  
    }
    else {
        dest_r[index1] = (int)(min(max(int(img_r[index2]+(img_r[index1]+img_r[index4])/2+img_r[index3]), 0), 255));  
    }
  }
}
''', 'grey_estimate_Mono_C')


Mono_ampsoft_GPU = cp.RawKernel(r'''
extern "C" __global__
void Mono_ampsoft_GPU_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, float val_ampl, float *Corr_GS)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      dest_r[index] = (int)(min(max(int(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr])), 0), 255));
    } 
}
''', 'Mono_ampsoft_GPU_C')


Colour_ampsoft_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_ampsoft_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_ampl, float *Corr_GS)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];  
      dest_r[index] = (int)(min(max(int(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr])), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vg])), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vb])), 0), 255));
    } 
}
''', 'Colour_ampsoft_GPU_C')

Colour_contrast_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_contrast_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_ampl, float *Corr_cont)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];  
      dest_r[index] = (int)(img_r[index] *Corr_cont[vr]);
      dest_g[index] = (int)(img_g[index] *Corr_cont[vg]);
      dest_b[index] = (int)(img_b[index] *Corr_cont[vb]);
    } 
}
''', 'Colour_contrast_GPU_C')


Saturation_Color = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, float val_sat)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float P;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      P = __fsqrt_rd(img_r[index]*img_r[index]*0.299+img_g[index]*img_g[index]*0.587+img_b[index]*img_b[index]*0.114);
      dest_r[index] = (int)(min(max(int(P+(img_r[index]-P)*val_sat), 0), 255));
      dest_g[index] = (int)(min(max(int(P+(img_g[index]-P)*val_sat), 0), 255));
      dest_b[index] = (int)(min(max(int(P+(img_b[index]-P)*val_sat), 0), 255));
    } 
}
''', 'Saturation_Color_C')

Saturation_Combine_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Combine_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *ext_r, unsigned char *ext_g, unsigned char *ext_b, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]) / (0.299*ext_r[index] + 0.587*ext_g[index] + 0.114*ext_b[index]);
      dest_r[index] = (int)(min(max(int(ext_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(ext_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(ext_b[index]*X), 0), 255));
    } 
}
''', 'Saturation_Combine_Colour_C')


Saturation_Colour = cp.RawKernel(r'''
extern "C" __global__
void Saturation_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
long int width, long int height, float val_sat, int flag_neg_sat)
{

    long int j = threadIdx.x + blockIdx.x * blockDim.x;
    long int i = threadIdx.y + blockIdx.y * blockDim.y;
    long int index;
    double R1,G1,B1;
    double X1;
    double r,g,b;
    double C,X,m;
    double cmax,cmin,diff,h,s,v;
    double radian;
    double cosA,sinA;
    double m1,m2,m3,m4,m5,m6,m7,m8,m9;

    index = i * width + j;
  
    if (i < height && j < width) {
        r = img_r[index] / 255.0;
        g = img_g[index] / 255.0;
        b = img_b[index] / 255.0;
        cmax = max(r, max(g, b));
        cmin = min(r, min(g, b));
        diff = cmax - cmin;
        h = -1.0;
        s = -1.0;
        if (cmax == cmin) 
            h = 0; 
        else if (cmax == r) 
            h = fmod(60 * ((g - b) / diff) + 360, 360); 
        else if (cmax == g) 
            h = fmod(60 * ((b - r) / diff) + 120, 360); 
        else if (cmax == b) 
            h = fmod(60 * ((r - g) / diff) + 240, 360); 
  
        if (cmax == 0) 
            s = 0; 
        else
            s = (diff / cmax); 

        v = cmax;

        s = s * val_sat;

            
        if (h > 360)
            h = 360;
        if (h < 0)
            h = 0;
        if (s > 1.0)
            s = 1.0;
        if (s < 0)
            s = 0;

        C = s*v;
        X = C*(1-abs(fmod(h/60.0, 2)-1));
        m = v-C;

        if(h >= 0 && h < 60){
            r = C,g = X,b = 0;
        }
        else if(h >= 60 && h < 120){
            r = X,g = C,b = 0;
        }
        else if(h >= 120 && h < 180){
            r = 0,g = C,b = X;
        }
        else if(h >= 180 && h < 240){
            r = 0,g = X,b = C;
        }
        else if(h >= 240 && h < 300){
            r = X,g = 0,b = C;
        }
        else{
            r = C,g = 0,b = X;
        }

        R1 = (int)((r+m)*255);
        G1 = (int)((g+m)*255);
        B1 = (int)((b+m)*255);

        if (flag_neg_sat == 1) {
            radian = 3.141592;
            cosA = cos(radian);
            sinA = sin(radian);
            m1 = cosA + (1.0 - cosA) / 3.0;
            m2 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m3 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m4 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m5 = cosA + 1./3.*(1.0 - cosA);
            m6 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m7 = 1./3. * (1.0 - cosA) - sqrt(1./3.) * sinA;
            m8 = 1./3. * (1.0 - cosA) + sqrt(1./3.) * sinA;
            m9 = cosA + 1./3. * (1.0 - cosA);
            dest_r[index] = (int)(min(max(int(R1 * m1 + G1 * m2 + B1 * m3), 0), 255));
            dest_g[index] = (int)(min(max(int(R1 * m4 + G1 * m5 + B1 * m6), 0), 255));
            dest_b[index] = (int)(min(max(int(R1 * m7 + G1 * m8 + B1 * m9), 0), 255));
        }
        else {
            dest_r[index] = (int)(min(max(int(R1), 0), 255));
            dest_g[index] = (int)(min(max(int(G1), 0), 255));
            dest_b[index] = (int)(min(max(int(B1), 0), 255));
        }
    }
}
''', 'Saturation_Colour_C')




Colour_staramp_GPU = cp.RawKernel(r'''
extern "C" __global__
void Colour_staramp_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
unsigned char *grey_gpu, unsigned char *grey_blur_gpu,long int width, long int height, float val_Mu, float val_Ro, float val_ampl, float *Corr_GS)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta;
  unsigned char index_grey;
  float factor;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      delta =(int)(min(max(int( grey_gpu[index] - grey_blur_gpu[index]), 0), 255));
      index_grey = grey_gpu[index];
      factor = delta*Corr_GS[index_grey]*val_ampl;
      dest_r[index] = (int)(min(max(int(img_r[index] + factor), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index] + factor), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index] + factor), 0), 255));
  }
}
''', 'Colour_staramp_GPU_C')


Mono_staramp_GPU = cp.RawKernel(r'''
extern "C" __global__
void Mono_staramp_GPU_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *grey_blur_gpu,long int width, long int height, float val_Mu, float val_Ro, float val_ampl, float *Corr_GS)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta;
  unsigned char index_grey;
  float factor;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      delta =(int)(min(max(int(img_r[index] - grey_blur_gpu[index]), 0), 255));
      index_grey = img_r[index];
      factor = delta*Corr_GS[index_grey]*val_ampl;
      dest_r[index] = (int)(min(max(int(img_r[index] + factor), 0), 255));
  }
}
''', 'Mono_staramp_GPU_C')


Smooth_Mono_high = cp.RawKernel(r'''
extern "C" __global__
void Smooth_Mono_high_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 7
  #define filterHeight 7

  index = i * width + j;

  if (i < height && j < width) {
    float filter[filterHeight][filterWidth] =
    {
      0, 0, 1, 2, 1, 0, 0,
      0, 3, 13, 22, 11, 3, 0,
      1, 13, 59, 97, 59, 13, 1,
      2, 22, 97, 159, 97, 22, 2,
      1, 13, 59, 97, 59, 13, 1,
      0, 3, 13, 22, 11, 3, 0,
      0, 0, 1, 2, 1, 0, 0,
    };
    
    factor = 1.0 / 1003.0;
      
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
  }
}
''', 'Smooth_Mono_high_C')



FNR_Color = cp.RawKernel(r'''
extern "C" __global__
void FNR_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *im1_r, unsigned char *im1_g, unsigned char *im1_b, unsigned char *im2_r, unsigned char *im2_g, unsigned char *im2_b,
unsigned char *im3_r, unsigned char *im3_g, unsigned char *im3_b,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int D1r,D1g,D1b;
  int D2r,D2g,D2b;
  int Delta_r,Delta_g,Delta_b;
  
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    D1r = im2_r[index] - im1_r[index];
    D1g = im2_g[index] - im1_g[index];
    D1b = im2_b[index] - im1_b[index];
    D2r = im3_r[index] - im2_r[index];
    D2g = im3_g[index] - im2_g[index];
    D2b = im3_b[index] - im2_b[index];
  
    if ((D1r*D2r) < 0) {
        Delta_r = (D1r + D2r) / (2.5 - abs(D2r)/255.0);
    }
    else {
        Delta_r = (D1r + D2r) / 2.0;
    }
    if ((D1g*D2g) < 0) {
        Delta_g = (D1g + D2g) / (2.5 - abs(D2g)/255.0);
    }
    else {
        Delta_g = (D1g + D2g) / 2.0;
    }
    if ((D1b*D2b) < 0) {
        Delta_b = (D1b + D2b) / (2.5 - abs(D2b)/255.0);
    }
    else {
        Delta_b = (D1b + D2b) / 2.0;
    }
    if (abs(D2r) > 40) {
        dest_r[index] = im3_r[index];
    }
    else {
        dest_r[index] = (int)(min(max(int((im1_r[index] + im2_r[index]) / 2.0 + Delta_r), 0), 255));
    }
    if (abs(D2g) > 40) {
        dest_g[index] = im3_g[index];
    }
    else {
        dest_g[index] = (int)(min(max(int((im1_g[index] + im2_g[index]) / 2.0 + Delta_g), 0), 255));
    }
    if (abs(D2b) > 40) {
        dest_b[index] = im3_b[index];
    }
    else {
        dest_b[index] = (int)(min(max(int((im1_b[index] + im2_b[index]) / 2.0 + Delta_b), 0), 255));
    }
  }
}
''', 'FNR_Color_C')

FNR_Mono = cp.RawKernel(r'''
extern "C" __global__
void FNR_Mono_C(unsigned char *dest_b, unsigned char *im1_b, unsigned char *im2_b, unsigned char *im3_b,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int D1b;
  int D2b;
  int Delta_b;
  
 
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
    D1b = im2_b[index] - im1_b[index];
    D2b = im3_b[index] - im2_b[index];
  
    if ((D1b*D2b) < 0) {
        Delta_b = (D1b + D2b) / (2.5 - abs(D2b)/255.0);
    }
    else {
        Delta_b = (D1b + D2b) / 2.0;
    }
    if (abs(D2b) > 40) {
        dest_b[index] = im3_b[index];
    }
    else {
        dest_b[index] = (int)(min(max(int((im1_b[index] + im2_b[index]) / 2.0 + Delta_b), 0), 255));
    }
  }
}
''', 'FNR_Mono_C')



adaptative_absorber_denoise_Color = cp.RawKernel(r'''
extern "C" __global__
void adaptative_absorber_denoise_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int flag_dyn_AADP, int flag_ghost_reducer, int val_ghost_reducer)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;
  int flag_r,flag_g,flag_b;
  float coef_r,coef_g,coef_b;
  
  flag_r = 0;
  flag_g = 0;
  flag_b = 0;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      if (img_r[index] > 190) {
          img_r[index] = (int)((img_r[index-1] + img_r[index+1])/2.0);
          }
      if (img_g[index] > 190) {
          img_g[index] = (int)((img_g[index-1] + img_g[index+1])/2.0);
          }
      if (img_b[index] > 190) {
          img_b[index] = (int)((img_b[index-1] + img_b[index+1])/2.0);
          }
      if (old_r[index] > 190) {
          old_r[index] = (int)((old_r[index-1] + old_r[index+1])/2.0);
          }
      if (old_g[index] > 190) {
          old_g[index] = (int)((old_g[index-1] + old_g[index+1])/2.0);
          }
      if (old_b[index] > 190) {
          old_b[index] = (int)((old_b[index-1] + old_b[index+1])/2.0);
          }
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];
      if (flag_dyn_AADP == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (abs(delta_g) > val_ghost_reducer) {
              flag_g = 1;
              dest_g[index] = img_g[index];
          }
          if (abs(delta_b) > val_ghost_reducer) {
              flag_b = 1;
              dest_b[index] = img_b[index];
          }
          if (delta_r > 0 && flag_dyn_AADP == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AADP == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          if (delta_g > 0 && flag_dyn_AADP == 1 && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.025995987)*1.2669433195)));
          }
          if ((delta_g < 0 || flag_dyn_AADP == 0) && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.54405)*20.8425))); 
          }
          if (delta_b > 0 && flag_dyn_AADP == 1 && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.025995987)*1.2669433195)));
          }
          if ((delta_b < 0 || flag_dyn_AADP == 0) && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AADP == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          if (delta_g > 0 && flag_dyn_AADP == 1) {
              coef_g = __powf(abs(delta_g),-0.025995987)*1.2669433195;
          }
          else {
              coef_g = __powf(abs(delta_g),-0.54405)*20.8425; 
          }
          if (delta_b > 0 && flag_dyn_AADP == 1) {
              coef_b = __powf(abs(delta_b),-0.025995987)*1.2669433195;
          }
          else {
              coef_b = __powf(abs(delta_b),-0.54405)*20.8425;
          }
          dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
          dest_g[index] = (int)((old_g[index] - delta_g / coef_g));
          dest_b[index] = (int)((old_b[index] - delta_b / coef_b));
      } 
      }
}
''', 'adaptative_absorber_denoise_Color_C')


adaptative_absorber_denoise_Mono = cp.RawKernel(r'''
extern "C" __global__
void adaptative_absorber_denoise_Mono_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *old_r, long int width, long int height, int flag_dyn_AADP,
int flag_ghost_reducer, int val_ghost_reducer)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r;
  int flag_r;
  float coef_r;
  
  flag_r = 0;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      if (img_r[index] > 210) {
          img_r[index] = (int)((img_r[index-1] + img_r[index+1])/2.0);
          }
      delta_r = old_r[index] - img_r[index];
      if (flag_dyn_AADP == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (delta_r > 0 && flag_dyn_AADP == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AADP == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AADP == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          dest_r[index] = (int)((old_r[index] - delta_r / coef_r));
      } 
      }
}
''', 'adaptative_absorber_denoise_Mono_C')


reduce_variation_Color = cp.RawKernel(r'''
extern "C" __global__
void reduce_variation_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *old_r, unsigned char *old_g, unsigned char *old_b,
long int width, long int height, int variation)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r,delta_g,delta_b;  
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];

      if (abs(delta_r) > variation) {
          if (delta_r >= 0) {
              dest_r[index] = old_r[index] - variation;
          }
          else {
              dest_r[index] = old_r[index] + variation;          
          }
      }
      else {
          dest_r[index] = img_r[index];
      }
      
      if (abs(delta_g) > variation) {
          if (delta_g >= 0) {
              dest_g[index] = old_g[index] - variation;
          }
          else {
              dest_g[index] = old_g[index] + variation;          
          }
      }
      else {
          dest_g[index] = img_g[index];
      }

      if (abs(delta_b) > variation) {
          if (delta_b >= 0) {
              dest_b[index] = old_b[index] - variation;
          }
          else {
              dest_b[index] = old_b[index] + variation;          
          }
      }
      else {
          dest_b[index] = img_b[index];
      }
      }
}
''', 'reduce_variation_Color_C')

reduce_variation_Mono = cp.RawKernel(r'''
extern "C" __global__
void reduce_variation_Mono_C(unsigned char *dest_r,
unsigned char *img_r, unsigned char *old_r, long int width, long int height, int variation)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta_r;  
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      delta_r = old_r[index] - img_r[index];

      if (abs(delta_r) > variation) {
          if (delta_r >= 0) {
              dest_r[index] = old_r[index] - variation;
          }
          else {
              dest_r[index] = old_r[index] + variation;          
          }
      }
      else {
          dest_r[index] = img_r[index];
      }
      }
}
''', 'reduce_variation_Mono_C')


Denoise_Paillou_Colour = cp.RawKernel(r'''
extern "C" __global__
void Denoise_Paillou_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, int cell_size, int sqr_cell_size)
{    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float3 Grd = {0, 0, 0};
    float3 Mean = {0, 0, 0};
    float3 Delta =  {0, 0, 0};

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        // Dead pixels detection and correction
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd.x += img_r[index1]-img_r[index2];
                Grd.y += img_g[index1]-img_g[index2];
                Grd.z += img_b[index1]-img_b[index2];
                Mean.x += img_r[index2];
                Mean.y += img_g[index2];
                Mean.z += img_b[index2];
                }
        Delta.x = (Grd.x / (sqr_cell_size * (1.0 + Grd.x/Mean.x))*(-0.00392157 * img_r[index1] +1.0));
        Delta.y = (Grd.y / (sqr_cell_size * (1.0 + Grd.y/Mean.y))*(-0.00392157 * img_g[index1] +1.0));
        Delta.z = (Grd.z / (sqr_cell_size * (1.0 + Grd.z/Mean.z))*(-0.00392157 * img_b[index1] +1.0));
        if (dest_r[index1] > abs(Delta.x) && dest_g[index1] > abs(Delta.y) && dest_b[index1] > abs(Delta.z)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta.x), 0), 255));
            dest_g[index1] = (int)(min(max(int(img_g[index1] - Delta.y), 0), 255));
            dest_b[index1] = (int)(min(max(int(img_b[index1] - Delta.z), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
            dest_g[index1] = int((img_g[ix - 1 + iy * imageW] + img_g[ix + 1 + iy * imageW] + img_g[ix + (iy-1) * imageW] + img_g[ix + (iy+1) * imageW])/4.0);
            dest_b[index1] = int((img_b[ix - 1 + iy * imageW] + img_b[ix + 1 + iy * imageW] + img_b[ix + (iy-1) * imageW] + img_b[ix + (iy+1) * imageW])/4.0);
        }
    }
}
''', 'Denoise_Paillou_Colour_C')


Denoise_Paillou_Mono = cp.RawKernel(r'''
extern "C" __global__
void Denoise_Paillou_Mono_C(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, int cell_size, int sqr_cell_size)
{ 
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;
     
    long int index1;
    long int index2;
    int delta;
    //float3 corr = {0, 0, 0};
    float Grd = 0;
    float Mean = 0;
    float Delta =  0;

    delta = (int)(abs(cell_size/2));
    index1 = ix + iy * imageW;
    
    if(ix<=(imageW-cell_size) && ix > delta && iy<=(imageH-cell_size) && iy > delta){
        // Dead pixels detection and correction
        for(float n = -delta; n <= delta; n++)
            for(float m = -delta; m <= delta; m++) {
                index2 = ix + m + (iy + n) * imageW;
                Grd += img_r[index1]-img_r[index2];
                Mean += img_r[index2];
                }
        Delta = (Grd / (sqr_cell_size * (1.0 + Grd/Mean))*(-0.00392157 * img_r[index1] +1.0));
        if (dest_r[index1] > abs(Delta)) {
            dest_r[index1] = (int)(min(max(int(img_r[index1] - Delta), 0), 255));
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
        }
    }
}
''', 'Denoise_Paillou_Mono_C')



Histo_Mono = cp.RawKernel(r'''
extern "C" __global__
void Histo_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2, float val_heq2, int flag_histogram_phitheta,
float val_phi, float val_theta)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
    }
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/(val_histo_max-val_histo_min))), 0), 255));
      img_r[index] = dest_r[index];
    }    
  }
}
''', 'Histo_Mono_C')


Sharp_Mono = cp.RawKernel(r'''
extern "C" __global__
void Sharp_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
 int flag_sharpen_soft1, int flag_unsharp_mask)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 5
  #define filterHeight 5

  index = i * width + j;

  if (i < height && j < width) {
  if (flag_sharpen_soft1 == 1) {
    float filter[filterHeight][filterWidth] =
    {
      -1, -1, -1, -1, -1,
      -1,  2,  2,  2, -1,
      -1,  2,  8,  2, -1,
      -1,  2,  2,  2, -1,
      -1, -1, -1, -1, -1,
    };
    factor = 1.0 / 8.0;
  
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    img_r[index] = dest_r[index];
    }
  if (flag_unsharp_mask == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(img_r[index] * 1.5 - (red * factor * 0.5)), 0), 255));
    }
  }
}
''', 'Sharp_Mono_C')


Smooth_Mono = cp.RawKernel(r'''
extern "C" __global__
void Smooth_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
int flag_2DConv)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;

  #define filterWidth 5
  #define filterHeight 5

  index = i * width + j;

  if (i < height && j < width) {
  if (flag_2DConv == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
    };
    
    factor = 1.0 / 25.0;
      
    red = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    }
  }
}
''', 'Smooth_Mono_C')


Histo_Color = cp.RawKernel(r'''
extern "C" __global__
void Histo_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_histogram_stretch, float val_histo_min, float val_histo_max, int flag_histogram_equalize2,
float val_heq2, int flag_histogram_phitheta, float val_phi, float val_theta)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  long int delta_histo = val_histo_max-val_histo_min;
  
  index = i * width + j;
  
  if (i < height && j < width) {
  if (flag_histogram_phitheta == 1) {
      dest_r[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_r[index]-val_theta)/32.0))));
      dest_g[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_g[index]-val_theta)/32.0))));
      dest_b[index] = (int)(255.0/(1.0+__expf(-1.0*val_phi*((img_b[index]-val_theta)/32.0))));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    }
  if (flag_histogram_equalize2 == 1 ) {
      dest_r[index] = (int)(255.0*__powf(((img_r[index]) / 255.0),val_heq2));
      dest_g[index] = (int)(255.0*__powf(((img_g[index]) / 255.0),val_heq2));
      dest_b[index] = (int)(255.0*__powf(((img_b[index]) / 255.0),val_heq2));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
  if (flag_histogram_stretch == 1 ) {
      dest_r[index] = (int)(min(max(int((img_r[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_g[index] = (int)(min(max(int((img_g[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      dest_b[index] = (int)(min(max(int((img_b[index]-val_histo_min)*(255.0/delta_histo)), 0), 255));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    }
  }
}
''', 'Histo_Color_C')


Set_RGB = cp.RawKernel(r'''
extern "C" __global__
void Set_RGB_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, float mod_red, float mod_green, float mod_blue)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      dest_r[index] = img_r[index];
      dest_g[index] = img_g[index];
      dest_b[index] = img_b[index];      
      if (mod_blue != 1.0) {
          dest_r[index] = (int)(min(max(int(img_r[index] * mod_blue), 0), 255));
          }
      if (mod_green != 1.0) {        
          dest_g[index] = (int)(min(max(int(img_g[index] * mod_green), 0), 255));
          }
      if (mod_red != 1.0) {  
          dest_b[index] = (int)(min(max(int(img_b[index] * mod_red), 0), 255));
          }
    } 
}
''', 'Set_RGB_C')


Smooth_Color = cp.RawKernel(r'''
extern "C" __global__
void Smooth_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_2DConv)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float green;
  float blue;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;
  
  index = i * width + j;

  #define filterWidth 5
  #define filterHeight 5

  if (i < height && j < width) {
  if (flag_2DConv == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
      1, 1, 1, 1, 1,
    };
    
    factor = 1.0 / 25.0;
      
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (int)((j - filterWidth / 2 + filterX + width) % width);
        imageY = (int)((i - filterHeight / 2 + filterY + height) % height);
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    dest_g[index] = (int)(min(max(int(factor * green), 0), 255));
    dest_b[index] = (int)(min(max(int(factor * blue), 0), 255));
    }

  }
}
''', 'Smooth_Color_C')


Sharp_Color = cp.RawKernel(r'''
extern "C" __global__
void Sharp_Color_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, long int width, long int height, int flag_sharpen_soft1, int flag_unsharp_mask)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float red;
  float green;
  float blue;
  float factor;
  int filterX;
  int filterY;
  int imageX;
  int imageY;
  
  index = i * width + j;

  #define filterWidth 5
  #define filterHeight 5

  if (i < height && j < width) {
  if (flag_sharpen_soft1 == 1) {
    float filter[filterHeight][filterWidth] =
    {
      -1, -1, -1, -1, -1,
      -1,  2,  2,  2, -1,
      -1,  2,  8,  2, -1,
      -1,  2,  2,  2, -1,
      -1, -1, -1, -1, -1,
    };
    factor = 1.0 / 8.0;
  
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(factor * red), 0), 255));
    dest_g[index] = (int)(min(max(int(factor * green), 0), 255));
    dest_b[index] = (int)(min(max(int(factor * blue), 0), 255));
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    }
  if (flag_unsharp_mask == 1) {
    float filter[filterHeight][filterWidth] =
    {
      1,  4,  6,  4,  1,
      4, 16, 24, 16,  4,
      6, 24, 36, 24,  6,
      4, 16, 24, 16,  4,
      1,  4,  6,  4,  1,
    };
    factor = 1.0 / 256.0;
  
    red = 0.0;
    green = 0.0;
    blue = 0.0;
    for(filterY = 0; filterY < filterHeight; filterY++)
      for(filterX = 0; filterX < filterWidth; filterX++)
      {
        imageX = (j - filterWidth / 2 + filterX + width) % width;
        imageY = (i - filterHeight / 2 + filterY + height) % height;
        red += img_r[imageY * width + imageX] * filter[filterY][filterX];
        green += img_g[imageY * width + imageX] * filter[filterY][filterX];
        blue += img_b[imageY * width + imageX] * filter[filterY][filterX];
      }
    dest_r[index] = (int)(min(max(int(img_r[index] * 1.9 - (red * factor * 0.9)), 0), 255));
    dest_g[index] = (int)(min(max(int(img_g[index] * 1.9 - (green * factor * 0.9)), 0), 255));
    dest_b[index] = (int)(min(max(int(img_b[index] * 1.9 - (blue * factor * 0.9)), 0), 255));
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    }
  }
}
''', 'Sharp_Color_C')



NLM2_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void NLM2_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = 6;
    const float limxmax = imageW - 6;
    const float limymin = 6;
    const float limymax = imageH - 6;
   
    long int index4;
    long int index5;

    if(x>limxmin && x<limxmax && y>limymin && y<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])
                + (img_g[index2] - img_g[index1]) * (img_g[index2] - img_g[index1])
                + (img_b[index2] - img_b[index1]) * (img_b[index2] - img_b[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0.0, 0.0, 0.0};

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float3 clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index3];
                clrIJ.y = img_g[index3];
                clrIJ.z = img_b[index3];
                
                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float3 clr00 = {0.0, 0.0, 0.0};
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4] / 256.0;
        clr00.y = img_g[index4] / 256.0;
        clr00.z = img_b[index4] / 256.0;
        
        clr.x = clr.x + (clr00.x - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'NLM2_Colour_C')


NLM2_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void NLM2_Mono_C(unsigned char *dest_r, unsigned char *img_r,
int imageW, int imageH, float Noise, float lerpC)
{
    
    #define NLM_WINDOW_RADIUS   3
    #define NLM_BLOCK_RADIUS    3

    #define NLM_WEIGHT_THRESHOLD    0.00039f
    #define NLM_LERP_THRESHOLD      0.10f
    
    __shared__ float fWeights[64];

    const float NLM_WINDOW_AREA = (2.0 * NLM_WINDOW_RADIUS + 1.0) * (2.0 * NLM_WINDOW_RADIUS + 1.0) ;
    const float INV_NLM_WINDOW_AREA = (1.0 / NLM_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float cx = blockDim.x * blockIdx.x + NLM_WINDOW_RADIUS + 1.0f;
    const float cy = blockDim.x * blockIdx.y + NLM_WINDOW_RADIUS + 1.0f;
    const float limxmin = NLM_BLOCK_RADIUS + 3;
    const float limxmax = imageW - NLM_BLOCK_RADIUS - 3;
    const float limymin = NLM_BLOCK_RADIUS + 3;
    const float limymax = imageH - NLM_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Find color distance from current texel to the center of NLM window
        float weight = 0;

        for(float n = -NLM_BLOCK_RADIUS; n <= NLM_BLOCK_RADIUS; n++)
            for(float m = -NLM_BLOCK_RADIUS; m <= NLM_BLOCK_RADIUS; m++) {
                long int index1 = cx + m + (cy + n) * imageW;
                long int index2 = x + m + (y + n) * imageW;
                weight += ((img_r[index2] - img_r[index1]) * (img_r[index2] - img_r[index1])) / (256.0 * 256.0);
                }

        //Geometric distance from current texel to the center of NLM window
        float dist =
            (threadIdx.x - NLM_WINDOW_RADIUS) * (threadIdx.x - NLM_WINDOW_RADIUS) +
            (threadIdx.y - NLM_WINDOW_RADIUS) * (threadIdx.y - NLM_WINDOW_RADIUS);

        //Derive final weight from color and geometric distance
        weight = __expf(-(weight * Noise + dist * INV_NLM_WINDOW_AREA));

        //Write the result to shared memory
        fWeights[threadIdx.y * 8 + threadIdx.x] = weight / 256.0;
        //Wait until all the weights are ready
        __syncthreads();


        //Normalized counter for the NLM weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;

        int idx = 0;

        //Cycle through NLM window, surrounding (x, y) texel
        
        for(float i = -NLM_WINDOW_RADIUS; i <= NLM_WINDOW_RADIUS + 1; i++)
            for(float j = -NLM_WINDOW_RADIUS; j <= NLM_WINDOW_RADIUS + 1; j++)
            {
                //Load precomputed weight
                float weightIJ = fWeights[idx++];

                //Accumulate (x + j, y + i) texel color with computed weight
                float clrIJ ; // Ligne code modifiée
                int index3 = x + j + (y + i) * imageW;
                clrIJ = img_r[index3];
                
                clr += clrIJ * weightIJ;
 
                //Sum of weights for color normalization to [0..1] range
                sumWeights  += weightIJ;

                //Update weight counter, if NLM weight for current window texel
                //exceeds the weight threshold
                fCount += (weightIJ > NLM_WEIGHT_THRESHOLD) ? INV_NLM_WINDOW_AREA : 0;
            }

        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotent basing on how many texels
        //within the NLM window exceeded the weight threshold
        float lerpQ = (fCount > NLM_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;

        //Write final result to global memory
        float clr00 = 0.0;
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4] / 256.0;
        
        clr = clr + (clr00 - clr) * lerpQ;
       
        dest_r[index5] = (int)(clr * 256.0);
    }
}
''', 'NLM2_Mono_C')



KNN_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void KNN_Colour_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
unsigned char *img_b, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float3 clr = {0, 0, 0};
        float3 clr00 = {0, 0, 0};
        float3 clrIJ = {0, 0, 0};
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00.x = img_r[index4];
        clr00.y = img_g[index4];
        clr00.z = img_b[index4];
    
        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ.x = img_r[index2];
                clrIJ.y = img_g[index2];
                clrIJ.z = img_b[index2];
                float distanceIJ = ((clrIJ.x - clr00.x) * (clrIJ.x - clr00.x)
                + (clrIJ.y - clr00.y) * (clrIJ.y - clr00.y)
                + (clrIJ.z - clr00.z) * (clrIJ.z - clr00.z)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr.x += clrIJ.x * weightIJ;
                clr.y += clrIJ.y * weightIJ;
                clr.z += clrIJ.z * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr.x *= sumWeights;
        clr.y *= sumWeights;
        clr.z *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr.x = clr.x + (clr00.x / 256.0 - clr.x) * lerpQ;
        clr.y = clr.y + (clr00.y / 256.0 - clr.y) * lerpQ;
        clr.z = clr.z + (clr00.z / 256.0 - clr.z) * lerpQ;
        
        dest_r[index5] = (int)(clr.x * 256.0);
        dest_g[index5] = (int)(clr.y * 256.0);
        dest_b[index5] = (int)(clr.z * 256.0);
    }
}
''', 'KNN_Colour_C')


KNN_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void KNN_Mono_C(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, float Noise, float lerpC)
{
    
    #define KNN_WINDOW_RADIUS   3
    #define KNN_BLOCK_RADIUS    3

    #define KNN_WEIGHT_THRESHOLD    0.00078125f
    #define KNN_LERP_THRESHOLD      0.79f

    const float KNN_WINDOW_AREA = (2.0 * KNN_WINDOW_RADIUS + 1.0) * (2.0 * KNN_WINDOW_RADIUS + 1.0) ;
    const float INV_KNN_WINDOW_AREA = (1.0 / KNN_WINDOW_AREA);
    
    const long int   ix = blockDim.x * blockIdx.x + threadIdx.x;
    const long int   iy = blockDim.y * blockIdx.y + threadIdx.y;

    const float  x = (float)ix  + 1.0f;
    const float  y = (float)iy  + 1.0f;
    const float limxmin = KNN_BLOCK_RADIUS + 3;
    const float limxmax = imageW - KNN_BLOCK_RADIUS - 3;
    const float limymin = KNN_BLOCK_RADIUS + 3;
    const float limymax = imageH - KNN_BLOCK_RADIUS - 3;
   
    long int index4;
    long int index5;

    if(ix>limxmin && ix<limxmax && iy>limymin && iy<limymax){
        //Normalized counter for the weight threshold
        float fCount = 0;
        //Total sum of pixel weights
        float sumWeights = 0;
        //Result accumulator
        float clr = 0.0;
        float clr00 = 0.0;
        float clrIJ = 0.0;
        //Center of the KNN window
        index4 = x + (y * imageW);
        index5 = imageW * (iy + 1) + ix + 1;
         
        clr00 = img_r[index4];

        for(float i = -KNN_BLOCK_RADIUS; i <= KNN_BLOCK_RADIUS; i++)
            for(float j = -KNN_BLOCK_RADIUS; j <= KNN_BLOCK_RADIUS; j++) {
                long int index2 = x + j + (y + i) * imageW;
                clrIJ = img_r[index2];
                float distanceIJ = ((clrIJ - clr00) * (clrIJ - clr00)) / 65536.0;

                //Derive final weight from color and geometric distance
                float   weightIJ = (__expf(- (distanceIJ * Noise + (i * i + j * j) * INV_KNN_WINDOW_AREA))) / 256.0;

                clr += clrIJ * weightIJ;

                //Sum of weights for color normalization to [0..1] range
                sumWeights     += weightIJ;

                //Update weight counter, if KNN weight for current window texel
                //exceeds the weight threshold
                fCount         += (weightIJ > KNN_WEIGHT_THRESHOLD) ? INV_KNN_WINDOW_AREA : 0;
        }
        
        //Normalize result color by sum of weights
        sumWeights = 0.0039f / sumWeights;
        clr *= sumWeights;

        //Choose LERP quotient basing on how many texels
        //within the KNN window exceeded the weight threshold
        float lerpQ = (fCount > KNN_LERP_THRESHOLD) ? lerpC : 1.0f - lerpC;
        
        clr = clr + (clr00 / 256.0 - clr) * lerpQ;
        
        dest_r[index5] = (int)(clr * 256.0);
    }
}
''', 'KNN_Mono_C')



cv2.setUseOptimized(True)

def init_camera() :
    global camera,flag_colour_camera,format_capture,controls,res_cam_x,res_cam_y, \
           cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,sensor_ratio,sensor_factor, \
           DELAY_BIN1,DELAY_BIN2,DELAY_BIN3,val_delai,val_blue,val_red,val_USB,val_maxgain,val_resolution, \
           flag_supported_camera,flag_camera_ok,RES_X_BIN1_4_3,RES_Y_BIN1_4_3,RES_X_BIN1_16_9,RES_Y_BIN1_16_9,\
           RES_X_BIN1_1_1,RES_Y_BIN1_1_1,RES_X_BIN2_4_3,RES_Y_BIN2_4_3,RES_X_BIN2_16_9,RES_Y_BIN2_16_9,\
           RES_X_BIN2_1_1,RES_Y_BIN2_1_1

    if env_filename_camera:
        asi.init(env_filename_camera)
        time.sleep(1)
    else:
        print('The filename of the SDK library for camera is required')
        sys.exit(1)

    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        print('No cameras found - Video treatment mode activated')
        flag_camera_ok = False
    else :
        flag_camera_ok = True

    if flag_camera_ok == True :
        cameras_found = asi.list_cameras()  # Models names of the connected cameras

        if num_cameras == 1:
            camera_id = 0
            print('Found one camera: %s' % cameras_found[0])
        else:
            print('Found %d cameras' % num_cameras)
            for n in range(num_cameras):
                print('    %d: %s' % (n, cameras_found[n]))
            camera_id = 0
            print('Using #%d: %s' % (camera_id, cameras_found[camera_id]))
        print(cameras_found[0])

        if cameras_found[0] == "ZWO ASI1600MC" or cameras_found[0] == "ZWO ASI1600MM" :
            res_cam_x = 4656
            res_cam_y = 3520
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(1012*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "4_3"
            sensor_ratio.set(0)

            RES_X_BIN1_4_3 = [4656,3600,3000,2400,2000,1600,1280,1024,800]
            RES_Y_BIN1_4_3 = [3520,2700,2250,1800,1500,1200,960,768,600]

            RES_X_BIN1_16_9 = [4656,3600,3000,2400,2000,1600,1280,1024,800]
            RES_Y_BIN1_16_9 = [2620,2026,1688,1350,1126,900,720,576,450]

            RES_X_BIN1_1_1 = [3520,2704,2248,1800,1504,1200,960,768,600]
            RES_Y_BIN1_1_1 = [3520,2704,2248,1800,1504,1200,960,768,600]

            RES_X_BIN2_4_3 = [2328,1800,1504,1200,1000,800,640]
            RES_Y_BIN2_4_3 = [1760,1350,1126,900,750,600,480]
            
            RES_X_BIN2_16_9 = [2328,1800,1504,1200,1000,800,640]
            RES_Y_BIN2_16_9 = [1310,1013,844,674,562,450,360]
            
            RES_X_BIN2_1_1 = [1760,1352,1128,904,752,600,480]
            RES_Y_BIN2_1_1 = [1760,1352,1128,904,752,600,480]

            RES_X_BIN1 = RES_X_BIN1_4_3
            RES_Y_BIN1 = RES_Y_BIN1_4_3
            RES_X_BIN2 = RES_X_BIN2_4_3
            RES_Y_BIN2 = RES_Y_BIN2_4_3            

            RES_X_BIN1 = [4656,3600,3000,2400,2000,1600,1280,1024,800]
            RES_Y_BIN1 = [3520,2700,2250,1800,1500,1200,960,770,600]

            RES_X_BIN2 = [2328,1800,1504,1200,1000,640,400]
            RES_Y_BIN2 = [1760,1350,1130,900,750,480,300]


        if cameras_found[0] == "ZWO ASI294MC" or cameras_found[0] == "ZWO ASI294MC Pro" or cameras_found[0] == "ZWO ASI294MM" or cameras_found[0] == "ZWO ASI294MM Pro" :
            res_cam_x = 4144
            res_cam_y = 2822
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(760*fact_s)
            val_USB = USB485
            flag_supported_camera = True

            sensor_factor = "16_9"
            sensor_ratio.set(1)

            RES_X_BIN1_4_3 = [3760,3240,2880,2400,1800,1536,1200,960,600]
            RES_Y_BIN1_4_3 = [2822,2430,2160,1800,1350,1152,900,720,450]

            RES_X_BIN1_16_9 = [4144,3240,2880,2400,1800,1536,1216,960,608]
            RES_Y_BIN1_16_9 = [2822,1824,1620,1350,1018,864,684,540,342]

            RES_X_BIN1_1_1 = [2816,2432,2160,1800,1352,1152,904,720,448]
            RES_Y_BIN1_1_1 = [2816,2432,2160,1800,1352,1152,904,720,448]

            RES_X_BIN2_4_3 = [1880,1624,1440,1200,900,768,600]
            RES_Y_BIN2_4_3 = [1410,1216,1080,900,674,576,450]
            
            RES_X_BIN2_16_9 = [2072,1620,1440,1200,900,768,608]
            RES_Y_BIN2_16_9 = [1410,912,810,674,510,432,342]
            
            RES_X_BIN2_1_1 = [1408,1216,1080,900,676,576,452]
            RES_Y_BIN2_1_1 = [1408,1216,1080,900,676,576,452]

            RES_X_BIN1 = RES_X_BIN1_16_9
            RES_Y_BIN1 = RES_Y_BIN1_16_9
            RES_X_BIN2 = RES_X_BIN2_16_9
            RES_Y_BIN2 = RES_Y_BIN2_16_9          
 
        if cameras_found[0] == "ZWO ASI178MC" or cameras_found[0] == "ZWO ASI178MM" or cameras_found[0] == "ZWO ASI178MM Pro" :
            res_cam_x = 3096
            res_cam_y = 2080
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(1012*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "4_3"
            sensor_ratio.set(0)

            RES_X_BIN1_4_3 = [3096,2560,1920,1600,1280,1024,800,640,320]
            RES_Y_BIN1_4_3 = [2080,1920,1440,1200,960,768,600,480,240]

            RES_X_BIN1_16_9 = [3096,2560,1920,1600,1280,1024,800,640,320]
            RES_Y_BIN1_16_9 = [1740,1440,1080,900,720,576,450,360,180]

            RES_X_BIN1_1_1 = [2080,1920,1440,1200,960,768,600,480,240]
            RES_Y_BIN1_1_1 = [2080,1920,1440,1200,960,768,600,480,240]

            RES_X_BIN2_4_3 = [1544,1280,960,800,640,512,400]
            RES_Y_BIN2_4_3 = [1040,960,720,600,480,384,300]
            
            RES_X_BIN2_16_9 = [1544,1280,960,800,640,512,400]
            RES_Y_BIN2_16_9 = [870,720,540,450,360,288,225]
            
            RES_X_BIN2_1_1 = [1040,960,720,600,480,384,296]
            RES_Y_BIN2_1_1 = [1040,960,720,600,480,384,296]

            RES_X_BIN1 = RES_X_BIN1_4_3
            RES_Y_BIN1 = RES_Y_BIN1_4_3
            RES_X_BIN2 = RES_X_BIN2_4_3
            RES_Y_BIN2 = RES_Y_BIN2_4_3            

        if cameras_found[0] == "ZWO ASI485MC" or cameras_found[0] == "ZWO ASI585MC" or cameras_found[0] == "ZWO ASI678MC" or cameras_found[0] == "ZWO ASI678MM" :
            res_cam_x = 3840
            res_cam_y = 2160
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(760*fact_s)
            val_USB = USB485
            flag_supported_camera = True

            sensor_factor = "16_9"
            sensor_ratio.set(1)

            RES_X_BIN1 = [3840,3240,2880,2400,1800,1536,1200,960,600]
            RES_Y_BIN1 = [2160,2160,1920,1600,1200,1024,800,640,400]

            RES_X_BIN2 = [1920,1600,1440,1200,904,768,600]
            RES_Y_BIN2 = [1080,1080,960,800,600,512,400]

            RES_X_BIN1_4_3 = [2880,2432,2160,1800,1352,1152,904,720,448]
            RES_Y_BIN1_4_3 = [2080,1920,1440,1200,960,768,600,480,240]

            RES_X_BIN1_16_9 = [3840,3240,2880,2400,1800,1536,1200,960,600]
            RES_Y_BIN1_16_9 = [2160,1822,1620,1350,1012,864,674,540,336]

            RES_X_BIN1_1_1 = [2160,1824,1616,1352,1008,864,680,544,336]
            RES_Y_BIN1_1_1 = [2160,1824,1616,1352,1008,864,680,544,336]

            RES_X_BIN2_4_3 = [1544,1280,960,800,640,512,400]
            RES_Y_BIN2_4_3 = [1040,960,720,600,480,384,300]
            
            RES_X_BIN2_16_9 = [1920,1544,1280,960,800,640,512]
            RES_Y_BIN2_16_9 = [1080,870,720,540,450,360,288]
            
            RES_X_BIN2_1_1 = [1040,960,720,600,480,384,296]
            RES_Y_BIN2_1_1 = [1040,960,720,600,480,384,296]

            RES_X_BIN1 = RES_X_BIN1_16_9
            RES_Y_BIN1 = RES_Y_BIN1_16_9
            RES_X_BIN2 = RES_X_BIN2_16_9
            RES_Y_BIN2 = RES_Y_BIN2_16_9

        if cameras_found[0] == "ZWO ASI290MC" or cameras_found[0] == "ZWO ASI290MM" or cameras_found[0] == "ZWO ASI462MC" or cameras_found[0] == "ZWO ASI385MC" :
            res_cam_x = 1936
            res_cam_y = 1096
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(760*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "16_9"
            sensor_ratio.set(1)

            RES_X_BIN1_4_3 = [1456,1200,960,768,608,480,240,240,240]
            RES_Y_BIN1_4_3 = [1096,900,720,580,460,360,180,180,180]

            RES_X_BIN1_16_9 = [1936,1600,1280,1024,800,640,320,320,320]
            RES_Y_BIN1_16_9 = [1096,900,720,580,460,360,180,180,180]

            RES_X_BIN1_1_1 = [1096,896,720,576,456,360,176,176,176]
            RES_Y_BIN1_1_1 = [1096,896,720,576,456,360,176,176,176]

            RES_X_BIN2_4_3 = [728,600,480,384,304,240,120]
            RES_Y_BIN2_4_3 = [548,450,360,290,230,180,90]
            
            RES_X_BIN2_16_9 = [968,800,640,512,400,320,160]
            RES_Y_BIN2_16_9 = [548,450,360,290,230,180,90]
            
            RES_X_BIN2_1_1 = [548,448,360,288,228,180,88]
            RES_Y_BIN2_1_1 = [548,448,360,288,228,180,88]

            RES_X_BIN1 = RES_X_BIN1_16_9
            RES_Y_BIN1 = RES_Y_BIN1_16_9
            RES_X_BIN2 = RES_X_BIN2_16_9
            RES_Y_BIN2 = RES_Y_BIN2_16_9

        if cameras_found[0] == "ZWO ASI662MC" or cameras_found[0] == "ZWO ASI482MC" :
            res_cam_x = 1920
            res_cam_y = 1080
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(760*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "16_9"
            sensor_ratio.set(1)

            RES_X_BIN1_4_3 = [1456,1200,960,768,608,480,240,240,240]
            RES_Y_BIN1_4_3 = [1080,900,720,580,460,360,180,180,180]

            RES_X_BIN1_16_9 = [1920,1600,1280,1024,800,640,320,320,320]
            RES_Y_BIN1_16_9 = [1080,900,720,580,460,360,180,180,180]

            RES_X_BIN1_1_1 = [1080,896,720,576,456,360,176,176,176]
            RES_Y_BIN1_1_1 = [1080,896,720,576,456,360,176,176,176]

            RES_X_BIN2_4_3 = [728,600,480,384,304,240,120]
            RES_Y_BIN2_4_3 = [540,450,360,290,230,180,90]
            
            RES_X_BIN2_16_9 = [960,800,640,512,400,320,160]
            RES_Y_BIN2_16_9 = [540,450,360,290,230,180,90]
            
            RES_X_BIN2_1_1 = [540,448,360,288,228,180,88]
            RES_Y_BIN2_1_1 = [540,448,360,288,228,180,88]

            RES_X_BIN1 = RES_X_BIN1_16_9
            RES_Y_BIN1 = RES_Y_BIN1_16_9
            RES_X_BIN2 = RES_X_BIN2_16_9
            RES_Y_BIN2 = RES_Y_BIN2_16_9

        if cameras_found[0] == "ZWO ASI224MC" :
            res_cam_x = 1304
            res_cam_y = 976
            cam_displ_x = int(1350*fact_s)
            cam_displ_y = int(1012*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "4_3"
            sensor_ratio.set(0)

            RES_X_BIN1_4_3 = [1304,1280,1024,800,640,320,320,320,320]
            RES_Y_BIN1_4_3 = [976,960,768,600,480,240,240,240,240]

            RES_X_BIN1_16_9 = [1304,1280,1024,800,640,320,320,320,320]
            RES_Y_BIN1_16_9 = [732,720,576,450,360,180,180,180,180]

            RES_X_BIN1_1_1 = [976,960,768,600,480,240,240,240,240]
            RES_Y_BIN1_1_1 = [976,960,768,600,480,240,240,240,240]

            RES_X_BIN2_4_3 = [652,1624,512,400,320,160,160]
            RES_Y_BIN2_4_3 = [488,480,384,300,240,120,120]
            
            RES_X_BIN2_16_9 = [652,640,512,400,320,160,160]
            RES_Y_BIN2_16_9 = [366,360,288,674,510,90,90]
            
            RES_X_BIN2_1_1 = [488,480,384,300,240,120,120]
            RES_Y_BIN2_1_1 = [488,480,384,300,240,120,120]

            RES_X_BIN1 = RES_X_BIN1_4_3
            RES_Y_BIN1 = RES_Y_BIN1_4_3
            RES_X_BIN2 = RES_X_BIN2_4_3
            RES_Y_BIN2 = RES_Y_BIN2_4_3            

        if cameras_found[0] == "ZWO ASI533MC" or cameras_found[0] == "ZWO ASI533MM" or cameras_found[0] == "ZWO ASI533MC Pro" or cameras_found[0] == "ZWO ASI533MM Pro" :
            res_cam_x = 3008
            res_cam_y = 2260
            cam_displ_x = int(1012*fact_s)
            cam_displ_y = int(1012*fact_s)
            val_USB = USB178
            flag_supported_camera = True

            sensor_factor = "1_1"
            sensor_ratio.set(2)

            RES_X_BIN1_4_3 = [3008,2560,1920,1600,1280,1024,800,640,320]
            RES_Y_BIN1_4_3 = [2256,1920,1440,1200,960,768,600,480,240]

            RES_X_BIN1_16_9 = [3008,2560,1920,1600,1280,1024,800,640,320]
            RES_Y_BIN1_16_9 = [1692,1440,1080,900,720,576,450,360,180]

            RES_X_BIN1_1_1 = [3008,2560,1920,1600,1280,1024,800,640,320]
            RES_Y_BIN1_1_1 = [3008,2560,1920,1600,1280,1024,800,640,320]

            RES_X_BIN2_4_3 = [1504,1624,960,800,640,512,400]
            RES_Y_BIN2_4_3 = [1128,960,720,600,480,384,300]
            
            RES_X_BIN2_16_9 = [1504,1280,960,800,640,512,400]
            RES_Y_BIN2_16_9 = [846,720,540,450,360,288,225]
            
            RES_X_BIN2_1_1 = [1504,1280,960,800,640,512,400]
            RES_Y_BIN2_1_1 = [1504,1280,960,800,640,512,400]

            RES_X_BIN1 = RES_X_BIN1_1_1
            RES_Y_BIN1 = RES_Y_BIN1_1_1
            RES_X_BIN2 = RES_X_BIN2_1_1
            RES_Y_BIN2 = RES_Y_BIN2_1_1

        if flag_supported_camera == True :
            camera = asi.Camera(camera_id)
            camera_info = camera.get_camera_property()
            controls = camera.get_controls()

            for cn in sorted(controls.keys()):
                print('    %s:' % cn)
                for k in sorted(controls[cn].keys()):
                    if cn == "Gain" :
                        if k == "MaxValue" :
                            val_maxgain = int(repr(controls[cn][k]))
                    if cn == "WB_B" :
                        if k == "DefaultValue" :
                            val_blue = int(repr(controls[cn][k]))
                    if cn == "WB_R" :
                       if k == "DefaultValue" :
                            val_red = int(repr(controls[cn][k]))
                    print('        %s: %s' % (k, repr(controls[cn][k])))

            if camera_info['IsColorCam']:
                flag_colour_camera = True
            else :
                flag_colour_camera = False
            
            format_capture = asi.ASI_IMG_RAW8

            camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, val_USB)
            camera.disable_dark_subtract()
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
            camera.set_control_value(asi.ASI_MONO_BIN,0)
            camera.set_control_value(asi.ASI_HARDWARE_BIN,0)

            camera.set_roi(None,None,RES_X_BIN1[0], RES_Y_BIN1[0],mode_BIN,format_capture)
            camera.set_control_value(asi.ASI_GAIN, val_gain)
            camera.set_control_value(asi.ASI_EXPOSURE, val_exposition)
            camera.set_control_value(asi.ASI_WB_B, val_blue)
            camera.set_control_value(asi.ASI_WB_R, val_red)
            camera.set_control_value(asi.ASI_FLIP, 0)
            camera.set_control_value(asi.ASI_GAMMA, ASIGAMMA)
            camera.set_control_value(asi.ASI_AUTO_MAX_BRIGHTNESS, ASIAUTOMAXBRIGHTNESS)

            camera.default_timeout = 500

            try:
            # Force any single exposure to be halted
                camera.stop_video_capture()
                camera.stop_exposure()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                pass

            camera.start_video_capture()
        else :
            print("Camera not supported")
            flag_camera_ok = Flase

def init_efw():
    global filter_wheel, fw_position, flag_filter_wheel,presence_FW,fw_position_

    if env_filename_efw:
        efw.init(env_filename_efw)
    else:
        print('The filename of the SDK library for EFW is required')
        sys.exit(1)

    num_efw = efw.get_num_efw()
    if num_efw == 0:
        labelFW.config(text = "FW OFF")
        flag_filter_wheel = False
        print("No filter wheel find")
        fw_position = 0
        presence_FW.set(0)  # Initialisation présence FW
        fw_position_.set(0)
    else :
        labelFW.config(text = "FW ON")
        print("EFW OK")
        flag_filter_wheel = True
        efw_id = 0
        filter_wheel = efw.EFW(efw_id)
        filter_wheel.open()
        fw_position = 0
        filter_wheel.set_position(fw_position)
        fw_position_.set(0)
        presence_FW.set(1)  # Initialisation présence FW

    
def start_acquisition() :
    global flag_autorise_acquisition,thread_1,flag_stop_acquisition
    if flag_camera_ok == True :
        flag_autorise_acquisition = True
        flag_stop_acquisition = False
        thread_1 = acquisition("1")
        thread_1.start()
             
  
class acquisition(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global type_debayer,camera,nb_erreur,image_brute,flag_acquisition_en_cours,flag_autorise_acquisition,image_camera,\
               flag_image_disponible,flag_noir_blanc,flag_hold_picture,flag_sub_dark,Master_Dark,image_camera_old,\
               res_bb1,res_gg1,res_rr1,flag_sub_dark,numero_image,timeoutexp,flag_new_image,s1,flag_filtrage_ON,flag_filtre_work

        while flag_autorise_acquisition == True :
            with s1 :
                if flag_stop_acquisition == False and flag_filtre_work == False :
                    if flag_hold_picture == 0 and flag_HDR == False :
                        flag_acquisition_en_cours = True
                        try :
                            image_brute_cam=camera.capture_video_frame_RAW8(filename=None,timeout=timeoutexp)
                            if flag_filtrage_ON == False and flag_filtre_work == False :
                                image_brute = image_brute_cam
                            else :
                                if flag_noir_blanc == 0 and flag_colour_camera == True :
                                    if flag_OpenCvCuda == True :
                                        gpu_image = cv2.cuda_GpuMat()
                                        gpu_image.upload(image_brute_cam)
                                        gpu_color_image = cv2.cuda.cvtColor(gpu_image, type_debayer)
                                        image_brute = gpu_color_image.download()
                                    else :
                                        image_brute = cv2.cvtColor(image_brute_cam, type_debayer)
                                    Dim = 3
                                else :
                                    image_brute = image_brute_cam
                                    Dim = 1
                                if flag_STAB == True and flag_HDR == False and flag_filtrage_ON == True :
                                    image_brute = Template_tracking(image_brute,Dim)
                            image_camera_old = image_camera   
                            image_camera = image_camera + 1
                            if flag_filtrage_ON == True :
                                if flag_sub_dark == True and dispo_dark == 'Dark disponible' :
                                    image_brute = cv2.subtract(image_brute,Master_Dark)
                                if flag_noir_blanc == 0 and flag_colour_camera == True :
                                    res_rr1,res_gg1,res_bb1 = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                                    flag_image_disponible = False
                                    if type_flip == "vertical" or type_flip == "both" :
                                        res_bb1 = cp.flipud(res_bb1)
                                        res_gg1 = cp.flipud(res_gg1)
                                        res_rr1 = cp.flipud(res_rr1)
                                    if type_flip == "horizontal" or type_flip == "both" :
                                        res_bb1 = cp.fliplr(res_bb1)
                                        res_gg1 = cp.fliplr(res_gg1)
                                        res_rr1 = cp.fliplr(res_rr1)
                                else :
                                    res_bb1 = cp.asarray(image_brute)
                                    if type_flip == "vertical" or type_flip == "both" :
                                        res_bb1 = cp.flipud(res_bb1)
                                    if type_flip == "horizontal" or type_flip == "both" :
                                        res_bb1 = cp.fliplr(res_bb1)
                                    flag_image_disponible = False
                                    res_gg1 = res_bb1
                                    res_rr1 = res_bb1
                            flag_image_disponible = True
                            flag_new_image = True
                            flag_acquisition_en_cours = False
                        except Exception as error :
                            camera.stop_video_capture()
                            camera.stop_exposure()
                            nb_erreur += 1
                            print("erreur acquisition : ",nb_erreur)
                            camera.start_video_capture()
                    else :
                        time.sleep(0.01)
                flag_acquisition_en_cours = False
                if Dev_system == "Windows" :
                    time.sleep(0.002)
                else :
                    time.sleep(0.002)

    def stop(self) :
        global flag_start_acquisition
        flag_start_acquisition = False
        
    def reprise(self) :
        global flag_start_acquisition
        flag_start_acquisition = True
        

def refresh() :
    global video,flag_cap_video,camera,traitement,cadre_image,image_brute,flag_image_disponible,flag_quitter,timeoutexp,flag_new_image,curFPS,fpsQueue,image_brut_read,\
           thread_1,flag_acquisition_en_cours,flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2,image_traitee,val_SAT,exposition,total_start,total_stop,flag_image_video_loaded,\
           val_gain, echelle2,val_exposition,echelle1,imggrey1,imggrey2,flag_DETECT_STARS,flag_TRKSAT,flag_REMSAT,flag_CONST,labelInfo2,Date_hour_image,image_camera,val_nb_capt_video,flag_image_mode,\
           calque_stars,calque_satellites,calque_TIP,flag_nouvelle_resolution,nb_sat,sat_x,sat_y,sat_s,res_bb1,res_gg1,res_rr1,image_camera_old,s1,flag_TRIGGER,flag_filtrage_ON,flag_filtre_work,\
           compteur_images,numero_image,nb_erreur,image_brute_grey,flag_sat_detected,start_time_video,stop_time_video,time_exec_test,total_time,font,image_camera_old,\
           image_camera,res_cam_x,res_cam_y,flag_premier_demarrage,Video_Test,TTQueue,curTT,max_sat,video_frame_number,video_frame_position,image_reconstructed,stars_x,strs_y,stars_s,nb_stars

    with s1 :
        if flag_camera_ok == True :
            if flag_premier_demarrage == True :
                flag_premier_demarrage = False
                start_acquisition()
            if flag_autoexposure_gain == True :
                val_gain = (int)(camera.get_control_value(asi.ASI_GAIN)[0])
                echelle2.set(val_gain)        
            if flag_autoexposure_exposition == True :
                val_exposition = (int)(camera.get_control_value(asi.ASI_EXPOSURE)[0]) // 1000
                echelle1.set(val_exposition)        
            if ((flag_image_disponible == True and flag_new_image == True) or flag_HDR == True) and (image_camera > numero_image):
                if flag_HDR == True and flag_filtrage_ON == True :
                    try :
                        image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutexp)
                        image_brute1=cv2.cvtColor(image_tmp, type_debayer)
                        val_exposition = echelle1.get()
                        if flag_acq_rapide == "Fast" :
                            exposition = val_exposition // 2
                        else :
                            exposition = val_exposition*1000 // 2
                        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                        timeoutHDR = timeoutexp // 2
                        time.sleep(0.1)
                        image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                        image_brute2=cv2.cvtColor(image_tmp, type_debayer)
                        if flag_acq_rapide == "Fast" :
                            exposition = val_exposition // 3
                        else :
                            exposition = val_exposition*1000 // 3
                        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                        timeoutHDR = timeoutexp // 5
                        time.sleep(0.1)
                        image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                        image_brute3=cv2.cvtColor(image_tmp, type_debayer)
                        if flag_acq_rapide == "Fast" :
                            exposition = val_exposition // 4
                        else :
                            exposition = val_exposition*1000 // 4
                        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                        timeoutHDR = timeoutexp // 10
                        time.sleep(0.1)
                        image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                        image_brute4=cv2.cvtColor(image_tmp, type_debayer)
                        val_exposition = echelle1.get()
                        if flag_acq_rapide == "Fast" :
                            exposition = val_exposition
                        else :
                            exposition = val_exposition*1000
                        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                        time.sleep(0.1)
                        compteur_images = compteur_images + 1
                        Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                        img_list = [image_brute1,image_brute2,image_brute3,image_brute4]
                        merge_mertens = cv2.createMergeMertens()
                        res_mertens = merge_mertens.process(img_list)
                        image_brute = np.clip(res_mertens*255, 0, 255).astype('uint8')
                        if flag_STAB == True and flag_HDR == True and flag_filtrage_ON == True :
                            if image_brute.ndim == 3 :
                                Dim = 3
                            else :
                                Dim = 1
                            image_brute = Template_tracking(image_brute,Dim)
                        if image_brute.ndim == 3 :
                            res_rr1,res_gg1,res_bb1 = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                        else :
                            res_bb1 = cp.asarray(image_brute)
                            res_gg1 = res_bb1
                            res_rr1 = res_bb1
                        flag_image_disponible = True
                        flag_acquisition_en_cours = False
                        flag_new_image = True
                        image_camera_old = image_camera   
                        image_camera = image_camera + 1
                        numero_image = image_camera - 1
                    except :
                        nb_erreur += 1
                        print("erreur acquisition : ",nb_erreur)
                        time.sleep(0.01)
                if flag_stop_acquisition == False and flag_filtre_work == False :
                    Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                    compteur_images = compteur_images + 1
                    if flag_filtrage_ON == True :
                        application_filtrage(res_rr1,res_gg1,res_bb1)
                    else :
                        image_traitee = image_brute
                        curTT = 0
                    flag_new_image = False
                    if flag_filtrage_ON == True :
                        if (flag_TRKSAT == 1 and flag_noir_blanc == 0 and flag_colour_camera == True) and flag_REMSAT == 0 :
                            satellites_tracking()
                        flag_sat_detected = False
                        if flag_CONST == 0 :
                            if flag_DETECT_STARS == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                                stars_detection()
                            if (flag_TRKSAT == 1 and nb_sat >= 0 and nb_sat < max_sat and flag_noir_blanc == 0 and flag_colour_camera == True) and flag_REMSAT == 0 :
                                for i in range(nb_sat+1):
                                    centercircle = (sat_x[i],sat_y[i])
                                    cv2.circle(image_traitee, centercircle, 10, (0,255,0), 1, cv2.LINE_AA)
                            if flag_DETECT_STARS == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                                image_traitee = cv2.addWeighted(image_traitee, 1, calque_stars, 1, 0)
                        else :
                            pass
                    total_stop = cv2.getTickCount()
                    total_time= int((total_stop-total_start)/cv2.getTickFrequency()*1000)
                    fpsQueue.append(1000/total_time)
                    if len(fpsQueue) > 10:
                        fpsQueue.pop(0)
                    curFPS = (sum(fpsQueue)/len(fpsQueue))
                    texte_TIP = Date_hour_image + "  Frame nbr : " + str(numero_image) + "  FPS : " + str(int(curFPS*10)/10)
                    if flag_TIP == True :
                        if flag_noir_blanc == 0 and flag_colour_camera == True and flag_filtrage_ON == True :
                            height,width,layers = image_traitee.shape
                        else :
                            height,width = image_traitee.shape 
                        pos = 30
                        if width > 2000 :
                            size = 1
                        else :
                            size = 0.5
                            pos = 20
                        if width < 600 :
                            size = 0.3
                            pos = 15
                        cv2.putText(image_traitee, texte_TIP, (pos,pos), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                    labelInfo2.config(text = str(curTT) + " ms   FPS : " + str(int(curFPS*10)/10) + "    ")
                    total_start = cv2.getTickCount()
                    if flag_cap_pic == True:
                        pic_capture()
                    if flag_cap_video == True :
                        video_capture(image_traitee)
                    if res_cam_x > int(1350*fact_s) and flag_full_res == 0 :
                        image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_NEAREST)
                        cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                    if (res_cam_x > int(1350*fact_s) or res_cam_y > int(1012*fact_s)) and flag_full_res == 1 :
                        rs = (res_cam_y - int(1012*fact_s)) // 2 - 1
                        re = (res_cam_y + int(1012*fact_s)) // 2 + 1
                        cs = (res_cam_x - int(1350*fact_s)) // 2 - 1
                        ce = (res_cam_x + int(1350*fact_s)) // 2 + 1
                        if rs < 0 :
                            rs = 0
                            re = res_cam_y
                        if cs < 0 :
                            cs = 0
                            ce = res_cam_x
                        image_crop = image_traitee[rs:re,cs:ce]
                        cadre_image.im=PIL.Image.fromarray(image_crop)
                    if res_cam_x <= 1350*fact_s :
                        cadre_image.im=PIL.Image.fromarray(image_traitee)                    
                    SX, SY = cadre_image.im.size
                    if flag_cross == True :
                        draw = PIL.ImageDraw.Draw(cadre_image.im)
                        SX, SY = cadre_image.im.size
                        if image_traitee.ndim == 3 :
                            draw.line(((SX/2-100,SY/2),(SX/2+100,SY/2)), fill="red", width=1)
                            draw.line(((SX/2,SY/2-100),(SX/2,SY/2+100)), fill="red", width=1)
                        else :
                            draw.line(((SX/2-100,SY/2),(SX/2+100,SY/2)), fill="white", width=1)
                            draw.line(((SX/2,SY/2-100),(SX/2,SY/2+100)), fill="white", width=1)
                    if flag_filtrage_ON == True :
                        if flag_HST == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                            r,g,b = cadre_image.im.split()
                            hst_r = r.histogram()
                            hst_g = g.histogram()
                            hst_b = b.histogram()
                            histo = PIL.ImageDraw.Draw(cadre_image.im)
                            for x in range(1,256) :
                                histo.line(((x*3,SY),(x*3,SY-hst_r[x]/200)),fill="red")
                                histo.line(((x*3+1,SY),(x*3+1,SY-hst_g[x]/200)),fill="green")
                                histo.line(((x*3+2,SY),(x*3+2,SY-hst_b[x]/200)),fill="blue")
                            histo.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                            histo.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                        if flag_HST == 1 and (flag_noir_blanc == 1 or flag_colour_camera == False) :
                            r = cadre_image.im
                            hst_r = r.histogram()
                            histo = PIL.ImageDraw.Draw(cadre_image.im)
                            for x in range(1,256) :
                                histo.line(((x*3,SY),(x*3,SY-hst_r[x]/200)),fill="white")
                            histo.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                            histo.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                        if flag_TRSF == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                            transform = PIL.ImageDraw.Draw(cadre_image.im)
                            for x in range(2,256) :
                                transform.line((((x-1)*3,SY-trsf_r[x-1]*2),(x*3,SY-trsf_r[x]*2)),fill="red",width=2)
                                transform.line((((x-1)*3,SY-trsf_g[x-1]*2),(x*3,SY-trsf_g[x]*2)),fill="green",width=2)
                                transform.line((((x-1)*3,SY-trsf_b[x-1]*2),(x*3,SY-trsf_b[x]*2)),fill="blue",width=2)
                            transform.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                            transform.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                        if flag_TRSF == 1 and (flag_noir_blanc == 1 or flag_colour_camera == False) :
                            transform = PIL.ImageDraw.Draw(cadre_image.im)
                            for x in range(2,256) :
                                transform.line((((x-1)*3,SY-trsf_r[x-1]*2),(x*3,SY-trsf_r[x]*2)),fill="green",width=2)
                            transform.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                            transform.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                        if flag_TRGS == 1 or flag_TRCLL == 1:
                            transform = PIL.ImageDraw.Draw(cadre_image.im)
                            if flag_TRGS == 1 :
                                for x in range(1,255) :
                                    transform.line((((x-1)*3,cam_displ_y-Corr_GS[x-1]*512),(x*3,cam_displ_y-Corr_GS[x]*512)),fill="blue",width=4)
                            if flag_TRCLL == 1 :
                                for x in range(1,255) :
                                    transform.line((((x-1)*3,cam_displ_y-Corr_CLL[x-1]*2),(x*3,cam_displ_y-Corr_CLL[x]*2)),fill="green",width=4)
                            transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                            transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)  
                    cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
                    cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
                else :
                    application_filtrage(res_rr1,res_gg1,res_bb1)
                    if res_cam_x > cam_displ_x and flag_full_res == 0 :
                        image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_NEAREST)
                        cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                    else :
                        cadre_image.im=PIL.Image.fromarray(image_traitee)
                    cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
                    cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)    
            if flag_quitter == False:
                if flag_HDR == False :
                    fenetre_principale.after(2, refresh)
                else :
                    fenetre_principale.after(20, refresh)
        else :
            if flag_premier_demarrage == True :
                flag_premier_demarrage = False
                if flag_image_mode == True :
                    image_brut_read = cv2.imread(Video_Test,cv2.IMREAD_COLOR)
                    image_brute = image_brut_read
                    flag_image_disponible = True
                    flag_image_mode = True
                    res_cam_y,res_cam_x,layer = image_brut_read.shape
                    flag_image_video_loaded = True
                if flag_image_mode == False :
                    video = cv2.VideoCapture(Video_Test, cv2.CAP_FFMPEG)
                    property_id = int(cv2.CAP_PROP_FRAME_WIDTH)
                    res_cam_x = int(cv2.VideoCapture.get(video, property_id))
                    property_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
                    res_cam_y = int(cv2.VideoCapture.get(video, property_id))
                    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                    video_frame_number = int(cv2.VideoCapture.get(video, property_id))
                    flag_image_mode = False
                    flag_image_video_loaded = True
            if flag_cap_video == True and nb_cap_video == 1 and flag_image_mode == False :
                video.release()
                time.sleep(0.1)
                video = cv2.VideoCapture(Video_Test, cv2.CAP_FFMPEG)
                property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                val_nb_capt_video = int(cv2.VideoCapture.get(video, property_id))
                property_id = int(cv2.CAP_PROP_FRAME_WIDTH)
                res_cam_x = int(cv2.VideoCapture.get(video, property_id))
                property_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
                res_cam_y = int(cv2.VideoCapture.get(video, property_id))
                property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                video_frame_number = int(cv2.VideoCapture.get(video, property_id))
                flag_image_video_loaded = True
                time.sleep(0.1)
            if flag_image_mode == False :
                ret,frame = video.read()
                property_id = int(cv2.CAP_PROP_POS_FRAMES)
                video_frame_position = int(cv2.VideoCapture.get(video, property_id))
                if ret == True :
                    image_brute = np.asarray(frame)
                    if video_debayer != 0 :
                        r = image_brute[:,:,0].copy()
                        image_brute=cv2.cvtColor(r, video_debayer)
                    flag_image_video_loaded = True
                else :
                    flag_image_video_loaded = False
                if flag_STAB == True :
                    image_brute = Template_tracking(image_brute,3)
            if flag_image_mode == True :
                image_brute = image_brut_read
                res_cam_y,res_cam_x,layer = image_brute.shape
                flag_image_video_loaded = True
            if flag_image_video_loaded == True :
                res_rr1,res_gg1,res_bb1 = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                if type_flip == "vertical" or type_flip == "both" :
                    res_bb1 = cp.flipud(res_bb1)
                    res_gg1 = cp.flipud(res_gg1)
                    res_rr1 = cp.flipud(res_rr1)
                if type_flip == "horizontal" or type_flip == "both" :
                    res_bb1 = cp.fliplr(res_bb1)
                    res_gg1 = cp.fliplr(res_gg1)
                    res_rr1 = cp.fliplr(res_rr1)
                flag_image_disponible = True
                if flag_filtrage_ON == True :
                    application_filtrage(res_bb1,res_gg1,res_rr1)
                else :
                    if Dev_system == "Windows" :
                        image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_bb1,res_gg1,res_rr1)
                    else :
                        image_traitee = cv2.merge((res_bb1,res_gg1,res_rr1))
                total_stop = cv2.getTickCount()
                total_time= int((total_stop-total_start)/cv2.getTickFrequency()*1000)
                fpsQueue.append(1000/total_time)
                if len(fpsQueue) > 10:
                    fpsQueue.pop(0)
                curFPS = (sum(fpsQueue)/len(fpsQueue))
                if flag_image_mode == False :
                    labelInfo2.config(text = "FPS : " + str(int(curFPS*10)/10) + "    ")
                else :
                    labelInfo2.config(text = str(curTT) + " ms      ")
                total_start = cv2.getTickCount()
                start_time_video = stop_time_video
                if flag_CONST == 0 :
                    if (flag_TRKSAT == 1 and flag_noir_blanc == 0 and flag_colour_camera == True and flag_image_mode == False) and flag_REMSAT == 0 :
                        satellites_tracking()
                    if (flag_REMSAT == 1 and flag_noir_blanc == 0 and flag_colour_camera == True and flag_image_mode == False) :
                        remove_satellites()
                    flag_sat_detected = False
                    if flag_DETECT_STARS == 1 and flag_noir_blanc == 0 and flag_colour_camera == True and flag_image_mode == False :
                        stars_detection(True)
                    if (flag_TRKSAT == 1  and nb_sat >= 0 and nb_sat < max_sat and flag_noir_blanc == 0 and flag_colour_camera == True and flag_image_mode == False) and flag_REMSAT == 0 :
                        for i in range(nb_sat+1):
                            centercircle = (sat_x[i],sat_y[i])
                            cv2.circle(image_traitee, centercircle, 10, (0,255,0), 1, cv2.LINE_AA)
                    if flag_DETECT_STARS == 1 and flag_noir_blanc == 0 and flag_colour_camera == True and flag_image_mode == False :
                        image_traitee = cv2.addWeighted(image_traitee, 1, calque_stars, 1, 0)
                else :
                    reconstruction_image()
                    image_traitee = image_reconstructed
                if flag_cap_pic == True:
                    pic_capture()
                if flag_cap_video == True and flag_image_mode == False :
                    video_capture(image_traitee)
                if (res_cam_x > int(cam_displ_x*fact_s) or res_cam_y > int(cam_displ_y*fact_s)) and flag_full_res == 0 :
                    image_traitee_resize = cv2.resize(image_traitee,(int(cam_displ_x*fact_s),int(cam_displ_y*fact_s)),interpolation = cv2.INTER_NEAREST)
                    cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                if (res_cam_x > int(1350*fact_s) or res_cam_y > int(1012*fact_s)) and flag_full_res == 1 :
                    rs = (res_cam_y - int(1012*fact_s)) // 2 - 1
                    re = (res_cam_y + int(1012*fact_s)) // 2 + 1
                    cs = (res_cam_x - int(1350*fact_s)) // 2 - 1
                    ce = (res_cam_x + int(1350*fact_s)) // 2 + 1
                    if rs < 0 :
                        rs = 0
                        re = res_cam_y
                    if cs < 0 :
                        cs = 0
                        ce = res_cam_x
                    image_crop = image_traitee[rs:re,cs:ce]
                    cadre_image.im=PIL.Image.fromarray(image_crop)
                if res_cam_x <= int(cam_displ_x*fact_s) :
                    cadre_image.im=PIL.Image.fromarray(image_traitee)                    
                if flag_cross == True :
                    draw = PIL.ImageDraw.Draw(cadre_image.im)
                    SX, SY = cadre_image.im.size
                    draw.line(((SX/2-100,SY/2),(SX/2+100,SY/2)), fill="red", width=1)
                    draw.line(((SX/2,SY/2-100),(SX/2,SY/2+100)), fill="red", width=1)
                if flag_HST == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                    r,g,b = cadre_image.im.split()
                    hst_r = r.histogram()
                    hst_g = g.histogram()
                    hst_b = b.histogram()
                    histo = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(1,256) :
                        histo.line(((x*3,cam_displ_y),(x*3,cam_displ_y-hst_r[x]/100)),fill="red")
                        histo.line(((x*3+1,cam_displ_y),(x*3+1,cam_displ_y-hst_g[x]/100)),fill="green")
                        histo.line(((x*3+2,cam_displ_y),(x*3+2,cam_displ_y-hst_b[x]/100)),fill="blue")
                    histo.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    histo.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_HST == 1 and (flag_noir_blanc == 1 or flag_colour_camera == False) :
                    r = cadre_image.im
                    hst_r = r.histogram()
                    histo = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(1,256) :
                        histo.line(((x*3,cam_displ_y),(x*3,cam_displ_y-hst_r[x]/100)),fill="white")
                    histo.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    histo.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_TRSF == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(2,256) :
                        transform.line((((x-1)*3,cam_displ_y-trsf_r[x-1]*3),(x*3,cam_displ_y-trsf_r[x]*3)),fill="red",width=2)
                        transform.line((((x-1)*3,cam_displ_y-trsf_g[x-1]*3),(x*3,cam_displ_y-trsf_g[x]*3)),fill="green",width=2)
                        transform.line((((x-1)*3,cam_displ_y-trsf_b[x-1]*3),(x*3,cam_displ_y-trsf_b[x]*3)),fill="blue",width=2)
                    transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_TRSF == 1 and (flag_noir_blanc == 1 or flag_colour_camera == False) :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(2,256) :
                        transform.line((((x-1)*3,cam_displ_y-trsf_r[x-1]*3),(x*3,cam_displ_y-trsf_r[x]*3)),fill="green",width=2)
                    transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_TRGS == 1 or flag_TRCLL == 1:
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    if flag_TRGS == 1 :
                        for x in range(1,255) :
                            transform.line((((x-1)*3,cam_displ_y-Corr_GS[x-1]*512),(x*3,cam_displ_y-Corr_GS[x]*512)),fill="blue",width=4)
                    if flag_TRCLL == 1 :
                        for x in range(1,255) :
                            transform.line((((x-1)*3,cam_displ_y-Corr_CLL[x-1]*2),(x*3,cam_displ_y-Corr_CLL[x]*2)),fill="green",width=4)
                    transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)  
                cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
                cadre_image.create_image(cam_displ_x*fact_s/2,cam_displ_y*fact_s/2, image=cadre_image.photo)
                flag_image_video_loaded = False
            else :
                if flag_image_mode == False :
                    video.release()
                    flag_premier_demarrage = True
                    flag_cap_video = False
            if flag_image_mode == True :
                fenetre_principale.after(30, refresh)
            else :
                if flag_camera_ok == True :
                    fenetre_principale.after(2, refresh)
                else :
                    fenetre_principale.after(5, refresh)

        
def satellites_tracking ():
    global imggrey1,imggrey2,image_traitee,calque_satellites,calque_direction_satellites,flag_sat_exist,sat_x,sat_y,sat_s,sat_old_x,sat_old_y,sat_cntdwn,nb_sat,sat_id,compteur_sat, \
           flag_first_sat_pass,nb_trace_sat,sat_speed,max_sat

    if flag_noir_blanc == 0 and flag_colour_camera == True :
        imggrey2 = cv2.cvtColor(image_traitee, cv2.COLOR_BGR2GRAY)
        correspondance = np.zeros(1000,dtype=int)
        try :
            height,width,layers = image_traitee.shape
            imggrey2_int = np.int_(imggrey2)
            imggrey1_int = np.int_(imggrey1)
            diff_int = imggrey2_int - imggrey1_int
            diff_int = np.clip(diff_int,0,255)
            diff = np.uint8(diff_int)
            seuilb = np.percentile(diff, 99) + 20
            diff[0:50,0:width] = 0
            ret,thresh = cv2.threshold(diff, seuilb , 255, cv2.THRESH_BINARY)
            imggrey1 = imggrey2
            image_sat=cv2.merge((thresh,thresh,thresh))
            seuil_min_blob_sat = 30
            params_sat = cv2.SimpleBlobDetector_Params()    
            params_sat.minThreshold = seuil_min_blob_sat;     # Change thresholds
            params_sat.maxThreshold = 255;
            params_sat.thresholdStep = 10 # steps to go through
            params_sat.filterByColor = True    # Filter by color.    
            params_sat.blobColor = 255  # 0 for darkblobs - 255 for light blobs)
            params_sat.minDistBetweenBlobs = 2
            params_sat.filterByArea = True    # Filter by Area.
            params_sat.minArea = 4
            params_sat.maxArea = 2000
            params_sat.minRepeatability = 2
            params_sat.filterByCircularity = False
            params_sat.filterByConvexity = False
            params_sat.filterByInertia = False
            detector_sat = cv2.SimpleBlobDetector_create(params_sat)
            keypoints_sat = detector_sat.detect(image_sat)
            flag_sat = True
        except :
            flag_sat = False
            imggrey1 = imggrey2
            flag_first_sat_pass = False
        nb_sat = -1
        if flag_sat == True :
            for kp_sat in keypoints_sat:
                nb_sat=nb_sat+1
            if nb_sat >= 0 and nb_sat <= max_sat :
                nb_sat = -1
                for kp_sat in keypoints_sat:
                    nb_sat = nb_sat+1
                    sat_x[nb_sat] = int(kp_sat.pt[0])
                    sat_y[nb_sat] = int(kp_sat.pt[1])
                    sat_s[nb_sat] = int(kp_sat.size*2)
            if nb_sat >= max_sat :
                raz_tracking()
                nb_sat = -1

def remove_satellites ():
    global imggrey1,imggrey2,image_traitee,calque_satellites,calque_direction_satellites,flag_sat_exist,sat_x,sat_y,sat_s,sat_old_x,sat_old_y,sat_cntdwn,nb_sat,sat_id,compteur_sat, \
           flag_first_sat_pass,nb_trace_sat,sat_speed,max_sat

    satellites_tracking()
    for i in range(nb_sat+1):
        try :
            y1 = sat_y[i] - sat_s[i]
            y2 = sat_y[i] + sat_s[i]
            x1 = sat_x[i] - sat_s[i]
            x2 = sat_x[i] + sat_s[i]
            mask_sat = image_traitee[y1:y2,x1:x2]
            seuilb = abs(np.percentile(mask_sat[:,:,0], 70))
            seuilg = abs(np.percentile(mask_sat[:,:,1], 70))
            seuilr = abs(np.percentile(mask_sat[:,:,2], 70))
            axex = range (x1,x2)
            axey = range (y1,y2)
            for i in axex :
                for j in axey :
                    if image_traitee[j,i,0] > seuilb :
                        image_traitee[j,i,0] = abs(seuilb + random.randrange(0,40) - 30)
                    if image_traitee[j,i,1] > seuilg :
                        image_traitee[j,i,1] = abs(seuilg + random.randrange(0,40) - 30)
                    if image_traitee[j,i,2] > seuilr :
                        image_traitee[j,i,2] = abs(seuilr + random.randrange(0,40) - 30)
        except :
            pass
    
                        
def stars_detection(draw) :
    global image_traitee,calque_stars,stars_x,stars_y,stars_s,nb_stars

    if flag_noir_blanc == 0 and flag_colour_camera == True :
        calque_stars = np.zeros_like(image_traitee)
        seuilb = np.percentile(image_traitee[:,:,0], 95)
        seuilg = np.percentile(image_traitee[:,:,1], 95)
        seuilr = np.percentile(image_traitee[:,:,2], 95)
        seuil_min_blob = max(seuilb,seuilg,seuilr) + 15
        height,width,layers = image_traitee.shape
        image_stars = image_traitee.copy()
        image_stars[0:50,0:width] = 0
        if seuil_min_blob > 160 :
            seuil_min_blob = 160
        params = cv2.SimpleBlobDetector_Params()    
        params.minThreshold = seuil_min_blob;     # Change thresholds
        params.maxThreshold = 255;
        params.thresholdStep = 15 # steps to go through
        params.filterByColor = True    # Filter by color.    
        params.blobColor = 255  # 0 for darkblobs - 255 for light blobs)
        params.minDistBetweenBlobs = 3
        params.filterByArea = True    # Filter by Area.
        params.minArea = 2
        params.maxArea = 1000
        params.minRepeatability = 2
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image_stars)
        nb_stars = -1
        for kp in keypoints:
            nb_stars=nb_stars+1
            stars_x[nb_stars] = int(kp.pt[0])
            stars_y[nb_stars] = int(kp.pt[1])
            stars_s[nb_stars] = int(kp.size)
            if draw == True :
                cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size*2), (255,0,0), 1, cv2.LINE_AA)

def draw_satellite(x,y) :
    global image_reconstructed

    centercircle = (x,y)
    cv2.circle(image_reconstructed, centercircle, 7, (0,255,0), -1, cv2.LINE_AA)
    start = (x-10,y-10)
    stop =(x+10,y+10)
    cv2.line(image_reconstructed, start, stop, (0,255,0), 5, cv2.LINE_AA)

def draw_star(x,y,s) :
    global image_reconstructed,calque_reconstruct
    
    red = image_traitee[y,x,0]
    green = image_traitee[y,x,1]
    blue = image_traitee[y,x,2]
    rayon = 1
    s = int(s/1.7)
    for i in range (s) :
        centercircle = (x,y)
        red = int(red / (s/(s-(0.5*i))))
        green = int(green / (s/(s-(0.3*i))))
        blue = int(blue / (s/(s-(0.3*i))))
        cv2.circle(calque_reconstruct, centercircle, rayon, (red,green,blue), 1, cv2.LINE_AA)
        rayon = rayon + 1
    

def reconstruction_image() :
    global imggrey1,imggrey2,image_traitee,sat_x,sat_y,sat_s,stars_x,stars_y,stars_s,nb_sat, \
           flag_first_sat_pass,nb_trace_sat,sat_speed,max_sat,nb_stars,image_reconstructed,calque_reconstruct

    calque_reconstruct = np.zeros_like(image_traitee)
    image_reconstructed = cv2.GaussianBlur(image_traitee,(7,7),0)
    stars_detection(False)
    for i in range(nb_stars+1) :
        centercircle = (stars_x[i],stars_y[i])
        draw_star(stars_x[i],stars_y[i],stars_s[i])
    image_reconstructed = cv2.addWeighted(image_reconstructed, 1, calque_reconstruct, 1, 0)
    if flag_TRKSAT == 1 :
        satellites_tracking()
        for i in range(nb_sat+1):
            centercircle = (sat_x[i],sat_y[i])
            draw_satellite(sat_x[i],sat_y[i])
    
    
def cupy_RGBImage_2_cupy_separateRGB(cupyImageRGB):
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    return cupy_B,cupy_G,cupy_R

def numpy_RGBImage_2_numpy_separateRGB(numpyImageRGB):
    numpy_R = np.ascontiguousarray(numpyImageRGB[:,:,0], dtype=np.uint8)
    numpy_G = np.ascontiguousarray(numpyImageRGB[:,:,1], dtype=np.uint8)
    numpy_B = np.ascontiguousarray(numpyImageRGB[:,:,2], dtype=np.uint8)
    return numpy_R,numpy_G,numpy_B

def numpy_RGBImage_2_cupy_separateRGB(numpyImageRGB):
    cupyImageRGB = cp.asarray(numpyImageRGB)
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    return cupy_R,cupy_G,cupy_B

def cupy_RGBImage_2_numpy_separateRGB(cupyImageRGB):
    cupy_R = cp.ascontiguousarray(cupyImageRGB[:,:,0], dtype=cp.uint8)
    cupy_G = cp.ascontiguousarray(cupyImageRGB[:,:,1], dtype=cp.uint8)
    cupy_B = cp.ascontiguousarray(cupyImageRGB[:,:,2], dtype=cp.uint8)
    numpy_R = cupy_R.get()
    numpy_G = cupy_G.get()
    numpy_B = cupy_B.get()
    return numpy_R,numpy_G,numpy_B

def cupy_separateRGB_2_numpy_RGBimage(cupyR,cupyG,cupyB):
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    numpyRGB = cupyRGB.get()
    return numpyRGB

def cupy_separateRGB_2_cupy_RGBimage(cupyR,cupyG,cupyB):
    rgb = (cupyR[..., cp.newaxis], cupyG[..., cp.newaxis], cupyB[..., cp.newaxis])
    cupyRGB = cp.concatenate(rgb, axis=-1, dtype=cp.uint8)
    return cupyRGB

def numpy_separateRGB_2_numpy_RGBimage(npR,npG,npB):
    rgb = (npR[..., np.newaxis], npG[..., np.newaxis], np[..., np.newaxis])
    numpyRGB = np.concatenate(rgb, axis=-1, dtype=np.uint8)
    return numpyRGB


def gaussianblur_mono(image_mono,niveau_blur):
    image_gaussian_blur_mono = ndimage.gaussian_filter(image_mono, sigma = niveau_blur)
    return image_gaussian_blur_mono

def gaussianblur_colour(im_r,im_g,im_b,niveau_blur):
    im_GB_r = ndimage.gaussian_filter(im_r, sigma = niveau_blur)
    im_GB_g = ndimage.gaussian_filter(im_g, sigma = niveau_blur)
    im_GB_b = ndimage.gaussian_filter(im_b, sigma = niveau_blur)
    return im_GB_r,im_GB_g,im_GB_b

def image_negative_colour (red,green,blue):
    blue = cp.invert(blue,dtype=cp.uint8)
    green = cp.invert(green,dtype=cp.uint8)
    red = cp.invert(red,dtype=cp.uint8)
    return red,green,blue


def Template_tracking(image,dim) :
    global flag_STAB,flag_Template,Template,gsrc,gtmpl,gresult,matcher

    if flag_Template == False :
        rs = res_cam_y // 2 - res_cam_y // 6
        re = res_cam_y // 2 + res_cam_y // 6
        cs = res_cam_x // 2 - res_cam_x // 6
        ce = res_cam_x // 2 + res_cam_x // 6
        Template = image[rs:re,cs:ce]
        if dim == 3 :
            Template = cv2.cvtColor(Template, cv2.COLOR_BGR2GRAY)
        else :
            pass
        if flag_OpenCvCuda == True :
            gsrc = cv2.cuda_GpuMat()
            gtmpl = cv2.cuda_GpuMat()
            gresult = cv2.cuda_GpuMat()
            gtmpl.upload(Template)
            matcher = cv2.cuda.createTemplateMatching(cv2.CV_8UC1, cv2.TM_CCOEFF_NORMED)
        flag_Template = True
        new_image = image
    else :
        if flag_OpenCvCuda == False :
            if dim == 3 :
                imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else :
                imageGray = image
            result = cv2.matchTemplate(imageGray, Template,cv2.TM_CCOEFF_NORMED)
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        else :
            if dim == 3 :
                imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else :
                imageGray = image
            gsrc.upload(imageGray)
            gresult = matcher.match(gsrc, gtmpl)
            result = gresult.download()
            (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        if maxVal > 0.2 :
            width = int(image.shape[1] * 2)
            height = int(image.shape[0] * 2)
            if dim == 3 :
                tmp_image = np.zeros((height,width,3),np.uint8)
            else :
                tmp_image = np.zeros((height,width),np.uint8)
            (startX, startY) = maxLoc
            midX = startX + Template.shape[1]//2
            midY = startY + Template.shape[0]//2
            try :
                rs = int(res_cam_y / 4 - (midY - res_cam_y / 2)) # Y up
                re = int(rs + res_cam_y) # Y down
                cs = int(res_cam_x / 4 - (midX - res_cam_x / 2)) # X left
                ce = int(cs + res_cam_x) # X right
                tmp_image[rs:re,cs:ce] = image
                rs = res_cam_y // 4  # Y up
                re = res_cam_y // 4 + res_cam_y # Y down
                cs = res_cam_x // 4 # X left
                ce = res_cam_x // 4 + res_cam_x # X right
                new_image = tmp_image[rs:re,cs:ce]
            except :
                new_image = image
        else :
            new_image = image
    return new_image
        
    
def application_filtrage(res_b1,res_g1,res_r1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,compteur_images,numero_image,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm, g1_sm, g2_sm, g3_sm, g4_sm, g5_sm, r1_sm, r2_sm, r3_sm, r4_sm, r5_sm,\
           Im4OK,Im5OK,flag_hold_picture,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_camera,image_camera_old,image_brute_grey,s1, \
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,flag_front,Date_hour_image,image_brute,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_filtre_work,flag_AmpSoft,val_ampl,grad_vignet,compteur_AADF,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\
           Im1fsdnOK,Im2fsdnOK,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,val_reds,val_greens,val_blues,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\
           flag_dyn_AADP,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_g2,res_r2,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\
           imgb1,imgg1,imgr1,imgb2,imgg2,imgr2,imgb3,imgg3,imgr3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL

    start_time_test = cv2.getTickCount()

    if flag_HDR == False :
        if image_camera > numero_image :
            numero_image = numero_image + 1
            image_camera = numero_image
            image_camera_old = image_camera

    with s1 :     
        if flag_filtrage_ON == True :
            flag_filtre_work = True
            nb_ThreadsX = 32
            nb_ThreadsY = 32
            for x in range(1,256) :
                trsf_r[x] = x
                trsf_g[x] = x
                trsf_b[x] = x
            
            if flag_noir_blanc == 1 or flag_colour_camera == False :

                # traitement image monochrome

                if flag_DEMO == 1 :
                    image_base = res_b1.get()
                    
                height,width = res_b1.shape
                nb_pixels = height * width
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                
                r_gpu = res_b1
                               
                if flag_NB_estime == 1 :
                    
                    grey_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1,\
                        np.int_(width), np.int_(height)))

                    res_b1 = r_gpu

                # Image Negatif
                if ImageNeg == 1 :
                    res_b1 = cp.invert(res_b1,dtype=cp.uint8)
                    for x in range(1,256) :
                        trsf_r[x] = (int)(256-trsf_r[x])
                    trsf_r = np.clip(trsf_r,0,255)

                # 3 Frames Noise reduction
                if flag_3FNR == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_3FNR = compteur_3FNR + 1
                    if compteur_3FNR < 4 and FNR_First_Start == True:
                        if compteur_3FNR == 1 :
                            imgb1 = res_b1.copy()
                            img1_3FNROK = True
                        if compteur_3FNR == 2 :
                            imgb2 = res_b1.copy()
                            img2_3FNROK = True
                        if compteur_3FNR == 3 :
                            imgb3 = res_b1.copy()
                            img3_3FNROK = True
                            FNR_First_Start = True
                    else :
                        compteur_3FNR = 0
     
                    if img3_3FNROK == True :
                        if FNR_First_Start == False :
                            imgb3 = res_b1.copy()
                        
                        FNR_First_Start = False
                        b_gpu = res_b1
                
                        FNR_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb1, imgb2, imgb3, np.int_(width), np.int_(height)))
                                              
                        res_b1 = b_gpu
                        imgb1 = imgb2.copy()
                        imgb2 = b_gpu.copy()

                # ADaptative Basorber Noise Reduction
                if flag_AADF == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_AADF = compteur_AADF + 1
                    if compteur_AADF < 3 :
                        if compteur_AADF == 1 :
                            res_b2 = res_b1.copy()
                            Im1fsdnOK = True
                        if compteur_AADF == 2 :
                            Im2fsdnOK = True

                    b_gpu = res_b1
                
                    if Im2fsdnOK == True :
                        nb_images = 2
                        divise = 2.0

                        adaptative_absorber_denoise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, res_b2,\
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AADP),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))

                        res_b2 = res_b1.copy()
                        res_b1 = r_gpu

                # Reduce variation filter (turbulence management)
                if flag_reduce_variation == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_RV = compteur_RV + 1
                    if compteur_RV < 3 :
                        if compteur_RV == 1 :
                            res_b2 = res_b1.copy()
                            Im1rvOK = True
                        if compteur_RV == 2 :
                            Im2rvOK = True

                    b_gpu = res_b1
     
                    if Im2rvOK == True :
                        variation = int(255/100*val_reduce_variation)
                        
                        reduce_variation_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, res_b1, res_b2, np.int_(width), np.int_(height),np.int_(variation)))

                        res_b2 = res_b1.copy()
                        res_b1 = b_gpu

                # Denoise PAILLOU CUDA 1
                if flag_denoise_Paillou == 1 :
                    cell_size = 3
                    sqr_cell_size = cell_size * cell_size
                    r_gpu = res_b1
                    Denoise_Paillou_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.intc(width), np.intc(height), np.intc(cell_size), \
                                np.intc(sqr_cell_size)))

                    res_b1 = r_gpu

                # Denoise PAILLOU 2
                if flag_denoise_Paillou2 == 1 :
                    
                    r_gpu = res_b1
                    reduce_noise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.intc(width), np.intc(height)))                    
                    res_b1 = r_gpu

     
                # Denoise NLM2
                if flag_denoise_soft == 1 :
                    nb_ThreadsXs = 8
                    nb_ThreadsYs = 8
                    nb_blocksXs = (width // nb_ThreadsXs) + 1
                    nb_blocksYs = (height // nb_ThreadsYs) + 1
                    param=float(val_denoise)
                    Noise = 1.0/(param*param)
                    lerpC = 0.4
                    r_gpu = res_b1
                    NLM2_Mono_GPU((nb_blocksXs,nb_blocksYs),(nb_ThreadsXs,nb_ThreadsYs),(r_gpu, res_b1, np.intc(width),np.intc(height), np.float32(Noise), \
                         np.float32(lerpC)))

                    res_b1 = r_gpu
                    
                # Denoise KNN
                if flag_denoise_KNN == 1 :               
                    param=float(val_denoise_KNN)
                    Noise = 1.0/(param*param)
                    lerpC = 0.4
                    r_gpu = res_b1
                    KNN_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.intc(width),np.intc(height), np.float32(Noise), \
                         np.float32(lerpC)))                            

                    res_b1 = r_gpu

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) or (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) :
                    
                    r_gpu = res_b1

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                    # Histo equalize 2 CUDA
                    # Histo stretch CUDA
                    # Histo Phi Theta CUDA
                    
                    Histo_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), \
                       np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                       np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta)))

                    res_b1 = r_gpu

                    if flag_histogram_phitheta == 1 :
                        for x in range(1,256) :
                            trsf_r[x] = (int)(255.0/(1.0+math.exp(-1.0*val_phi*((trsf_r[x]-val_theta)/32.0))))
                        trsf_r = np.clip(trsf_r,0,255)

                    if flag_histogram_equalize2 == 1 :
                        for x in range(1,256) :
                            trsf_r[x] = (int)(255.0*math.pow(((trsf_r[x]) / 255.0),val_heq2))
                        trsf_r = np.clip(trsf_r,0,255)

                    if flag_histogram_stretch == 1 :
                        delta_histo = val_histo_max-val_histo_min
                        for x in range(1,256) :
                            trsf_r[x] = (int)((trsf_r[x]-val_histo_min)*(255.0/delta_histo))
                        trsf_r = np.clip(trsf_r,0,255)

                # Amplification image LInear or Gaussian
                if flag_AmpSoft == 1 :
                    if flag_lin_gauss == 1 or flag_lin_gauss == 2 :
                        correction = cp.asarray(Corr_GS)

                        Mono_ampsoft_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), \
                            np.float32(val_ampl), correction))

                        res_b1 = r_gpu

                    for x in range(1,256) :
                        trsf_r[x] = (int)(trsf_r[x] * val_ampl)
                    trsf_r = np.clip(trsf_r,0,255)


                # Amplification soft Stars Amplification
                if flag_AmpSoft == 1 and flag_lin_gauss == 3 :           
                    niveau_blur = 7
                    imagegreyblur=gaussianblur_mono(res_b1,niveau_blur)
                    correction = cp.asarray(Corr_GS)
                    r_gpu = res_b1
                                
                    Mono_staramp_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, imagegreyblur,np.int_(width), np.int_(height), \
                            np.float32(val_Mu),np.float32(val_Ro),np.float32(val_ampl), correction))

                    res_b1 = r_gpu

                # Gradient Removal or Vignetting reduction
                if flag_GR == True :
                    if grad_vignet == 1 :
                        seuilb = int(cp.percentile(res_b1, val_SGR))
                        th,img_b = cv2.threshold(res_b1.get(),seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        img_b = cv2.GaussianBlur(img_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)
                        att_b = cp.asarray(img_b) * ((100.0-val_AGR) / 100.0) 
                        resb = cp.subtract(cp.asarray(res_b1),att_b)
                        resb = cp.clip(resb,0,255)
                        res_b1 = cp.asarray(resb,dtype=cp.uint8)
                    else :
                        seuilb = int(cp.percentile(res_b1, val_SGR))
                        th,fd_b = cv2.threshold(res_b1.get(),seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        fd_b = cv2.GaussianBlur(fd_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)
                        pivot_b = int(cp.percentile(cp.asarray(res_b1), val_AGR))
                        corr_b = cp.asarray(res_b1).astype(cp.int16) - cp.asarray(fd_b).astype(cp.int16) + pivot_b
                        corr_b = cp.clip(corr_b,0,255)
                        res_b1 = cp.asarray(corr_b,dtype=cp.uint8)

                # Contrast CLAHE
                if flag_contrast_CLAHE ==1 :
                    if flag_OpenCvCuda == True :
                        clahe = cv2.cuda.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        srcb = cv2.cuda_GpuMat()
                        srcb.upload(res_b1.get())
                        resb = clahe.apply(srcb, cv2.cuda_Stream.Null())
                        resbb = resb.download()
                        res_b1 = cp.asarray(resbb)
                    else :        
                        clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        b = clahe.apply(res_b1.get())
                        res_b1 = cp.asarray(b)
                        
                # Contrast Low Light
                if flag_CLL == 1 :
                    correction_CLL = cp.asarray(Corr_CLL,dtype=cp.uint8)
                    r_gpu = res_b1
                    Contrast_Low_Light_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), correction_CLL))
                    res_b1 = r_gpu

                # Sum / Mean
                if val_FS > 1 and flag_hold_picture == 0 :
                    compteur_FS = compteur_FS+1
                    if compteur_FS > val_FS :
                        compteur_FS = 1
                    if compteur_FS == 1 :
                        b1_sm = cp.asarray(res_b1).astype(cp.int16)
                        Im1OK = True
                    if compteur_FS == 2 :
                        b2_sm = cp.asarray(res_b1).astype(cp.int16)
                        Im2OK = True
                    if compteur_FS == 3 :
                        b3_sm = cp.asarray(res_b1).astype(cp.int16)
                        Im3OK = True
                    if compteur_FS == 4 :
                        b4_sm = cp.asarray(res_b1).astype(cp.int16)
                        Im4OK = True
                    if compteur_FS == 5 :
                        b5_sm = cp.asarray(res_b1).astype(cp.int16)
                        Im5OK = True
                                            
                    if val_FS == 2 and Im2OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm) // 2
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                    
                    if val_FS == 3 and Im3OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm) // 3
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            
                    if val_FS == 4 and Im4OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm + b4_sm) // 4
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        
                    if val_FS == 5 and Im5OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm + b5_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm + b4_sm + b5_sm) // 5
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)

                # Image Sharpen 1
                if flag_sharpen_soft1 == 1 :

                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen * (tmp_b1 - res_b1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)

                # Image Sharpen 2
                if flag_sharpen_soft2 == 1 :
                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen2)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen2 * (tmp_b1 - res_b1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)

                image_traitee = res_b1.get()
                
                if flag_DEMO == 1 :
                    image_traitee[0:height,0:width//2] = image_base[0:height,0:width//2]             

            ######################################################################################################################
            ######################################################################################################################       
            ######################################################################################################################
                                
            if flag_noir_blanc == 0 and flag_colour_camera == True:

                # Colour image treatment
                
                if flag_DEMO == 1 :
                    if Dev_system == "Windows" :
                        image_base = cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
                    else :
                        image_base = cv2.merge((res_b1.get(),res_g1.get(),res_r1.get()))
                
                height,width = res_b1.shape
                                
                nb_pixels = height * width
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1

                # Image negative
                if ImageNeg == 1 :
                    res_b1,res_g1,res_r1 = image_negative_colour(res_b1,res_g1,res_r1)                        
                    for x in range(1,256) :
                        trsf_r[x] = (int)(256-trsf_r[x])
                        trsf_g[x] = (int)(256-trsf_g[x])
                        trsf_b[x] = (int)(256-trsf_b[x])
                    trsf_r = np.clip(trsf_r,0,255)
                    trsf_g = np.clip(trsf_g,0,255)
                    trsf_b = np.clip(trsf_b,0,255)

                # Adjust RGB channels soft
                if (val_reds != 100 or val_greens != 100 or val_blues != 100) :              
                    mod_red = val_reds / 100.0
                    mod_green = val_greens / 100.0
                    mod_blue = val_blues / 100.0

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                    
                    Set_RGB((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \
                            np.float32(mod_red), np.float32(mod_green), np.float32(mod_blue)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                    if val_reds != 0 :
                        for x in range(1,256) :
                            trsf_r[x] = (int)(trsf_r[x] * mod_red)
                        trsf_r = np.clip(trsf_r,0,255)
                    if val_greens != 0 :
                        for x in range(1,256) :
                            trsf_g[x] = (int)(trsf_g[x] * mod_green)
                        trsf_g = np.clip(trsf_g,0,255)
                    if val_blues != 0 :
                        for x in range(1,256) :
                            trsf_b[x] = (int)(trsf_b[x] * mod_blue)
                        trsf_b = np.clip(trsf_b,0,255)

                # 3 Frames Noise Reduction Filter
                if flag_3FNR == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_3FNR = compteur_3FNR + 1
                    if compteur_3FNR < 4 and FNR_First_Start == True:
                        if compteur_3FNR == 1 :
                            imgb1 = res_b1.copy()
                            imgg1 = res_g1.copy()
                            imgr1 = res_r1.copy()
                            img1_3FNROK = True
                        if compteur_3FNR == 2 :
                            imgb2 = res_b1.copy()
                            imgg2 = res_g1.copy()
                            imgr2 = res_r1.copy()
                            img2_3FNROK = True
                        if compteur_3FNR == 3 :
                            imgb3 = res_b1.copy()
                            imgg3 = res_g1.copy()
                            imgr3 = res_r1.copy()
                            img3_3FNROK = True
                            FNR_First_Start = True
                    else :
                        compteur_3FNR = 0
     
                    if img3_3FNROK == True :
                        if FNR_First_Start == False :
                            imgb3 = res_b1.copy()
                            imgg3 = res_g1.copy()
                            imgr3 = res_r1.copy()
                        
                        FNR_First_Start = False
                        b_gpu = res_b1
                        g_gpu = res_g1
                        r_gpu = res_r1
                
                        FNR_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr1, imgg1, imgb1, imgr2, imgg2, imgb2,\
                                             imgr3, imgg3, imgb3, np.int_(width), np.int_(height)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb1 = imgb2.copy()
                        imgg1 = imgg2.copy()
                        imgr1 = imgr2.copy()
                        imgr2 = r_gpu.copy()
                        imgg2 = g_gpu.copy()
                        imgb2 = b_gpu.copy()
                        
                                                     
                # Adaptative Absorber Denoise Filter
                if flag_AADF == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_AADF = compteur_AADF + 1
                    if compteur_AADF < 3 :
                        if compteur_AADF == 1 :
                            res_b2 = res_b1.copy()
                            res_g2 = res_g1.copy()
                            res_r2 = res_r1.copy()
                            Im1fsdnOK = True
                        if compteur_AADF == 2 :
                            Im2fsdnOK = True

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
     
                    if Im2fsdnOK == True :
                        
                        adaptative_absorber_denoise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AADP),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))


                        if flag_reduce_variation == False and flag_hold_picture == 0 and flag_image_mode == False :
                            res_b2 = res_b1.copy()
                            res_g2 = res_g1.copy()
                            res_r2 = res_r1.copy()
                            res_r1 = r_gpu
                            res_g1 = g_gpu
                            res_b1 = b_gpu
                        else :
                            res_r1 = r_gpu
                            res_g1 = g_gpu
                            res_b1 = b_gpu
                                

                # Reduce variation filter (turbulence management)
                if flag_reduce_variation == True and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_RV = compteur_RV + 1
                    if compteur_RV < 3 :
                        if compteur_RV == 1 :
                            res_b2 = res_b1.copy()
                            res_g2 = res_g1.copy()
                            res_r2 = res_r1.copy()
                            Im1rvOK = True
                        if compteur_RV == 2 :
                            Im2rvOK = True

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
     
                    if Im2rvOK == True :
                        variation = int(255/100*val_reduce_variation)
                        
                        reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                             np.int_(width), np.int_(height),np.int_(variation)))


                        res_b2 = res_b1.copy()
                        res_g2 = res_g1.copy()
                        res_r2 = res_r1.copy()
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                # Denoise image PAILLOU 1
                if flag_denoise_Paillou == 1 :
                    cell_size = 5
                    sqr_cell_size = cell_size * cell_size
                    
                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1

                    Denoise_Paillou_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.intc(cell_size), \
                                np.intc(sqr_cell_size)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                # Denoise image PAILLOU 2 
                if flag_denoise_Paillou2 == 1 :
                    
                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1

                    reduce_noise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height)))
                    
                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu


                # Denoise NLM2
                if flag_denoise_soft == 1 :
                    nb_ThreadsXs = 8
                    nb_ThreadsYs = 8
                    nb_blocksXs = (width // nb_ThreadsXs) + 1
                    nb_blocksYs = (height // nb_ThreadsYs) + 1
                    param=float(val_denoise)
                    Noise = 1.0/(param*param)
                    lerpC = 0.4

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                    
                    NLM2_Colour_GPU((nb_blocksXs,nb_blocksYs),(nb_ThreadsXs,nb_ThreadsYs),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.float32(Noise), \
                         np.float32(lerpC)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu
                
                # Denoise KNN
                if flag_denoise_KNN == 1 :
                    param=float(val_denoise_KNN)
                    Noise = 1.0/(param*param)
                    lerpC = 0.4

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                
                    KNN_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.float32(Noise), \
                         np.float32(lerpC)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) or\
                   (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) :
                   
                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                    # Histo equalize 2
                    # Histo stretch
                    # Histo Phi Theta
                
                    Histo_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \
                       np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                       np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                    if flag_histogram_phitheta == 1 :
                        for x in range(1,256) :
                            trsf_r[x] = (int)(255.0/(1.0+math.exp(-1.0*val_phi*((trsf_r[x]-val_theta)/32.0))))
                            trsf_g[x] = (int)(255.0/(1.0+math.exp(-1.0*val_phi*((trsf_g[x]-val_theta)/32.0))))
                            trsf_b[x] = (int)(255.0/(1.0+math.exp(-1.0*val_phi*((trsf_b[x]-val_theta)/32.0))))
                            trsf_r = np.clip(trsf_r,0,255)
                            trsf_g = np.clip(trsf_g,0,255)
                            trsf_b = np.clip(trsf_b,0,255)

                    if flag_histogram_equalize2 == 1 :
                        for x in range(1,256) :
                            trsf_r[x] = (int)(255.0*math.pow(((trsf_r[x]) / 255.0),val_heq2))
                            trsf_g[x] = (int)(255.0*math.pow(((trsf_g[x]) / 255.0),val_heq2))
                            trsf_b[x] = (int)(255.0*math.pow(((trsf_b[x]) / 255.0),val_heq2))
                            trsf_r = np.clip(trsf_r,0,255)
                            trsf_g = np.clip(trsf_g,0,255)
                            trsf_b = np.clip(trsf_b,0,255)

                    if flag_histogram_stretch == 1 :
                        delta_histo = val_histo_max-val_histo_min
                        for x in range(1,256) :
                            trsf_r[x] = (int)((trsf_r[x]-val_histo_min)*(255.0/delta_histo))
                            trsf_g[x] = (int)((trsf_g[x]-val_histo_min)*(255.0/delta_histo))
                            trsf_b[x] = (int)((trsf_b[x]-val_histo_min)*(255.0/delta_histo))
                            trsf_r = np.clip(trsf_r,0,255)
                            trsf_g = np.clip(trsf_g,0,255)
                            trsf_b = np.clip(trsf_b,0,255)
                    

                # Amplification soft Linear or Gaussian
                if flag_AmpSoft == 1 :
                    if flag_lin_gauss == 1 or flag_lin_gauss == 2 :
                        correction = cp.asarray(Corr_GS)

                        Colour_ampsoft_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \
                            np.float32(val_ampl), correction))

                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                    for x in range(1,256) :
                        trsf_r[x] = (int)(trsf_r[x] * val_ampl)
                        trsf_g[x] = (int)(trsf_g[x] * val_ampl)
                        trsf_b[x] = (int)(trsf_b[x] * val_ampl)
                    trsf_r = np.clip(trsf_r,0,255)
                    trsf_g = np.clip(trsf_g,0,255)
                    trsf_b = np.clip(trsf_b,0,255)

                # Amplification soft Stars Amplification
                if flag_AmpSoft == 1 and flag_lin_gauss == 3 :

                    r_gpu = res_r1
                    image_brute_grey = cv2.cvtColor(cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1), cv2.COLOR_RGB2GRAY)
                    imagegrey = cp.asarray(image_brute_grey)
                 
                    niveau_blur = 7
                    imagegreyblur=gaussianblur_mono(imagegrey,niveau_blur)             
                    correction = cp.asarray(Corr_GS)
                    r_gpu = res_r1
                    g_gpu = res_g1
                    b_gpu = res_b1

                    Colour_staramp_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, imagegrey, imagegreyblur,np.int_(width), np.int_(height), \
                            np.float32(val_Mu),np.float32(val_Ro),np.float32(val_ampl), correction))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                # Gradient Removal or Vignetting correction
                if flag_GR == True :
                    if grad_vignet == 1 :
                        seuilb = int(cp.percentile(res_b1, val_SGR))
                        seuilg = int(cp.percentile(res_g1, val_SGR))
                        seuilr = int(cp.percentile(res_r1, val_SGR))
                        th,img_b = cv2.threshold(res_b1.get(),seuilb,255,cv2.THRESH_TRUNC) 
                        th,img_g = cv2.threshold(res_g1.get(),seuilg,255,cv2.THRESH_TRUNC) 
                        th,img_r = cv2.threshold(res_r1.get(),seuilr,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        img_b,img_g,img_r = cv2.split(cv2.GaussianBlur(cv2.merge((img_b,img_g,img_r)),(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,))
                        att_b = cp.asarray(img_b) * ((100.0-val_AGR) / 100.0) 
                        att_g = cp.asarray(img_g) * ((100.0-val_AGR) / 100.0) 
                        att_r = cp.asarray(img_r) * ((100.0-val_AGR) / 100.0)
                        resb = cp.subtract(cp.asarray(res_b1),att_b)
                        resg = cp.subtract(cp.asarray(res_g1),att_g)
                        resr = cp.subtract(cp.asarray(res_r1),att_r)
                        resb = cp.clip(resb,0,255)
                        resg = cp.clip(resg,0,255)
                        resr = cp.clip(resr,0,255)      
                        res_b1 = cp.asarray(resb,dtype=cp.uint8)
                        res_g1 = cp.asarray(resg,dtype=cp.uint8)
                        res_r1 = cp.asarray(resr,dtype=cp.uint8)        
                    else :
                        seuilb = int(cp.percentile(res_b1, val_SGR))
                        seuilg = int(cp.percentile(res_g1, val_SGR))
                        seuilr = int(cp.percentile(res_r1, val_SGR))   
                        th,fd_b = cv2.threshold(res_b1.get(),seuilb,255,cv2.THRESH_TRUNC) 
                        th,fd_g = cv2.threshold(res_g1.get(),seuilg,255,cv2.THRESH_TRUNC) 
                        th,fd_r = cv2.threshold(res_r1.get(),seuilr,255,cv2.THRESH_TRUNC)
                        niveau_blur = val_NGB*2 + 3
                        fd_b,fd_g,fd_r = cv2.split(cv2.GaussianBlur(cv2.merge((fd_b,fd_g,fd_r)),(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,))
                        pivot_b = int(cp.percentile(cp.asarray(res_b1), val_AGR))
                        pivot_g = int(cp.percentile(cp.asarray(res_g1), val_AGR))
                        pivot_r = int(cp.percentile(cp.asarray(res_r1), val_AGR))               
                        corr_b = cp.asarray(res_b1).astype(cp.int16) - cp.asarray(fd_b).astype(cp.int16) + pivot_b
                        corr_g = cp.asarray(res_g1).astype(cp.int16) - cp.asarray(fd_g).astype(cp.int16) + pivot_g
                        corr_r = cp.asarray(res_r1).astype(cp.int16) - cp.asarray(fd_r).astype(cp.int16) + pivot_r           
                        corr_b = cp.clip(corr_b,0,255)
                        corr_g = cp.clip(corr_g,0,255)
                        corr_r = cp.clip(corr_r,0,255)         
                        res_b1 = cp.asarray(corr_b,dtype=cp.uint8)
                        res_g1 = cp.asarray(corr_g,dtype=cp.uint8)
                        res_r1 = cp.asarray(corr_r,dtype=cp.uint8)          
                                           
                # Contrast CLAHE
                if flag_contrast_CLAHE ==1 :
                    if flag_OpenCvCuda == True :
                        clahe = cv2.cuda.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        srcb = cv2.cuda_GpuMat()
                        srcb.upload(res_b1.get())
                        resb = clahe.apply(srcb, cv2.cuda_Stream.Null())
                        resbb = resb.download()

                        srcg = cv2.cuda_GpuMat()
                        srcg.upload(res_g1.get())
                        resg = clahe.apply(srcg, cv2.cuda_Stream.Null())
                        resgg = resg.download()

                        srcr = cv2.cuda_GpuMat()
                        srcr.upload(res_r1.get())
                        resr = clahe.apply(srcr, cv2.cuda_Stream.Null())
                        resrr = resr.download()

                        res_b1 = cp.asarray(resbb)
                        res_g1 = cp.asarray(resgg)
                        res_r1 = cp.asarray(resrr)
                    else :        
                        clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        b = clahe.apply(res_b1.get())
                        g = clahe.apply(res_g1.get())
                        r = clahe.apply(res_r1.get())
                        res_b1 = cp.asarray(b)
                        res_g1 = cp.asarray(g)
                        res_r1 = cp.asarray(r)

                # Contrast Low Light
                if flag_CLL == 1 :
                    correction_CLL = cp.asarray(Corr_CLL,dtype=cp.uint8)

                    r_gpu = res_r1
                    g_gpu = res_g1
                    b_gpu = res_b1

                    Contrast_Low_Light_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \
                        correction_CLL))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                if val_FS > 1 and flag_hold_picture == 0 and flag_image_mode == False :
                    compteur_FS = compteur_FS+1
                    if compteur_FS > val_FS :
                        compteur_FS = 1
                    if compteur_FS == 1 :
                        b1_sm = cp.asarray(res_b1).astype(cp.int16)
                        g1_sm = cp.asarray(res_g1).astype(cp.int16)
                        r1_sm = cp.asarray(res_r1).astype(cp.int16)
                        Im1OK = True
                    if compteur_FS == 2 :
                        b2_sm = cp.asarray(res_b1).astype(cp.int16)
                        g2_sm = cp.asarray(res_g1).astype(cp.int16)
                        r2_sm = cp.asarray(res_r1).astype(cp.int16)
                        Im2OK = True
                    if compteur_FS == 3 :
                        b3_sm = cp.asarray(res_b1).astype(cp.int16)
                        g3_sm = cp.asarray(res_g1).astype(cp.int16)
                        r3_sm = cp.asarray(res_r1).astype(cp.int16)
                        Im3OK = True
                    if compteur_FS == 4 :
                        b4_sm = cp.asarray(res_b1).astype(cp.int16)
                        g4_sm = cp.asarray(res_g1).astype(cp.int16)
                        r4_sm = cp.asarray(res_r1).astype(cp.int16)
                        Im4OK = True
                    if compteur_FS == 5 :
                        b5_sm = cp.asarray(res_b1).astype(cp.int16)
                        g5_sm = cp.asarray(res_g1).astype(cp.int16)
                        r5_sm = cp.asarray(res_r1).astype(cp.int16)
                        Im5OK = True
                                            
                    if val_FS == 2 and Im2OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm
                            sum_g = g1_sm + g2_sm
                            sum_r = r1_sm + r2_sm
                            sum_b = cp.clip(sum_b,0,255)
                            sum_g = cp.clip(sum_g,0,255)
                            sum_r = cp.clip(sum_r,0,255)         
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm) // 2
                            sum_g = (g1_sm + g2_sm) // 2
                            sum_r = (r1_sm + r2_sm) // 2
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)
                    
                    if val_FS == 3 and Im3OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm
                            sum_g = g1_sm + g2_sm + g3_sm
                            sum_r = r1_sm + r2_sm + r3_sm
                            sum_b = cp.clip(sum_b,0,255)
                            sum_g = cp.clip(sum_g,0,255)
                            sum_r = cp.clip(sum_r,0,255)         
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)          
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm) // 3
                            sum_g = (g1_sm + g2_sm + g3_sm) // 3
                            sum_r = (r1_sm + r2_sm + r3_sm) // 3
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)          
                            
                    if val_FS == 4 and Im4OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm
                            sum_g = g1_sm + g2_sm + g3_sm + g4_sm
                            sum_r = r1_sm + r2_sm + r3_sm + r4_sm
                            sum_b = cp.clip(sum_b,0,255)
                            sum_g = cp.clip(sum_g,0,255)
                            sum_r = cp.clip(sum_r,0,255)         
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)          
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm + b4_sm) // 4
                            sum_g = (g1_sm + g2_sm + g3_sm + g4_sm) // 4
                            sum_r = (r1_sm + r2_sm + r3_sm + r4_sm) // 4
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)          
                        
                    if val_FS == 5 and Im5OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm + b5_sm
                            sum_g = g1_sm + g2_sm + g3_sm + g4_sm + g5_sm
                            sum_r = r1_sm + r2_sm + r3_sm + r4_sm + r5_sm
                            sum_b = cp.clip(sum_b,0,255)
                            sum_g = cp.clip(sum_g,0,255)
                            sum_r = cp.clip(sum_r,0,255)         
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)
                        else :
                            sum_b = (b1_sm + b2_sm + b3_sm + b4_sm + b5_sm) // 5
                            sum_g = (g1_sm + g2_sm + g3_sm + g4_sm + g5_sm) // 5
                            sum_r = (r1_sm + r2_sm + r3_sm + r4_sm + r5_sm) // 5
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)

                # image saturation enhancement
                if flag_SAT == True :
                
                    r_gpu = res_r1.copy()
                    g_gpu = res_g1.copy()
                    b_gpu = res_b1.copy()

                    coul_gauss_r = res_r1.copy()
                    coul_gauss_g = res_g1.copy()
                    coul_gauss_b = res_b1.copy()

                    coul_gauss_r,coul_gauss_g,coul_gauss_b = gaussianblur_colour(coul_gauss_r,coul_gauss_g,coul_gauss_b,5)

                    if ImageNeg == 1 :
                        flag_neg_sat = 1
                    else :
                        flag_neg_sat = 0  

                    Saturation_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height),np.float32(val_SAT), np.int_(flag_neg_sat)))

                    coul_gauss_r = r_gpu.copy()
                    coul_gauss_g = g_gpu.copy()
                    coul_gauss_b = b_gpu.copy()
                        
                    coul_gauss_r,coul_gauss_g,coul_gauss_b = gaussianblur_colour(coul_gauss_r,coul_gauss_g,coul_gauss_b,7)

                    Saturation_Combine_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, coul_gauss_r, coul_gauss_g, coul_gauss_b, np.int_(width), np.int_(height)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu                        
                                            
                # image sharpen 1
                if flag_sharpen_soft1 == 1 :
                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen)
                    res_g1_blur = gaussianblur_mono(res_g1,val_sigma_sharpen)
                    res_r1_blur = gaussianblur_mono(res_r1,val_sigma_sharpen)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_g1 = cp.asarray(res_g1).astype(cp.int16)
                    tmp_r1 = cp.asarray(res_r1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen * (tmp_b1 - res_b1_blur)
                    tmp_g1 = tmp_g1 + val_sharpen * (tmp_g1 - res_g1_blur)
                    tmp_r1 = tmp_r1 + val_sharpen * (tmp_r1 - res_r1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    tmp_g1 = cp.clip(tmp_g1,0,255)
                    tmp_r1 = cp.clip(tmp_r1,0,255)
                    res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                    res_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                    res_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)

                # image sharpen 2
                if flag_sharpen_soft2 == 1 :
                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen2)
                    res_g1_blur = gaussianblur_mono(res_g1,val_sigma_sharpen2)
                    res_r1_blur = gaussianblur_mono(res_r1,val_sigma_sharpen2)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_g1 = cp.asarray(res_g1).astype(cp.int16)
                    tmp_r1 = cp.asarray(res_r1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen2 * (tmp_b1 - res_b1_blur)
                    tmp_g1 = tmp_g1 + val_sharpen2 * (tmp_g1 - res_g1_blur)
                    tmp_r1 = tmp_r1 + val_sharpen2 * (tmp_r1 - res_r1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    tmp_g1 = cp.clip(tmp_g1,0,255)
                    tmp_r1 = cp.clip(tmp_r1,0,255)
                    res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                    res_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                    res_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)

                if Dev_system == "Windows" :
                    if flag_reverse_RB == 0 :
                        image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
                    else :
                        image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_r1,res_g1,res_b1)
                else :
                    if flag_reverse_RB == 0 :
                        image_traitee = cv2.merge((res_b1.get(),res_g1.get(),res_r1.get()))
                    else :
                        image_traitee = cv2.merge((res_r1.get(),res_g1.get(),res_b1.get()))
                    

                if flag_DEMO == 1 :
                    image_traitee[0:height,0:width//2] = image_base[0:height,0:width//2]            
    
    stop_time_test = cv2.getTickCount()
    time_exec_test= int((stop_time_test-start_time_test)/cv2.getTickFrequency()*1000)

    TTQueue.append(time_exec_test)
    if len(TTQueue) > 10:
        TTQueue.pop(0)
    curTT = (sum(TTQueue)/len(TTQueue))

    flag_filtre_work = False 
             

def mount_info() :
    global azimut,hauteur,flag_mount_connect
    pass
                        
def pic_capture() :
    global start,nb_pic_cap,nb_acq_pic,labelInfo1,flag_cap_pic,nb_cap_pic,image_path,image_traitee,timer1,val_nb_captures
    if flag_camera_ok == True :
        if nb_cap_pic <= val_nb_captures :
            if time.time() - timer1 >= val_deltat :               
                if flag_HQ == 0 :
                    if flag_filter_wheel == True :
                        nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.jpg' # JPEG File format
                    else :
                        nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.jpg'  # JPEG File format
                else :
                    if flag_filter_wheel == True:
                        nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.tif' # TIF file format loseless format
                    else :
                        nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.tif'  # TIF file format loseless format
                if image_traitee.ndim == 3 :
                    if flag_HQ == 0 :
                        cv2.imwrite(os.path.join(image_path,nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
                    else :
                        cv2.imwrite(os.path.join(image_path, nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB)) # TIFF file format
                else :
                    if flag_HQ == 0 :
                        cv2.imwrite(os.path.join(image_path,nom_fichier), image_traitee, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
                    else :
                        cv2.imwrite(os.path.join(image_path, nom_fichier), image_traitee) # TIFF file format
                labelInfo1.config(text = "capture n° "+ nom_fichier)
                nb_cap_pic += 1
                timer1 = time.time()
        else :
            flag_cap_pic = False
            labelInfo1.config(text = "                                                                                                        ") 
            labelInfo1.config(text = " Capture pictures terminee")
    else :
        if flag_image_mode == True :
            val_nb_captures = 1
        if nb_cap_pic <= val_nb_captures :
            if flag_HQ == 0 :
                if flag_filter_wheel == True :
                    nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.jpg' # JPEG File format
                else :
                    nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.jpg'  # JPEG File format
            else :
                if flag_filter_wheel == True:
                    nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '_F' + "%01d" % fw_position + '.tif' # TIF file format loseless format
                else :
                    nom_fichier = start.strftime(
                        'PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.tif'  # TIF file format loseless format
            if image_traitee.ndim == 3 :
                if flag_HQ == 0 :
                    cv2.imwrite(os.path.join(image_path,nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
                else :
                    cv2.imwrite(os.path.join(image_path, nom_fichier), cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB)) # TIFF file format
            else :
                if flag_HQ == 0 :
                    cv2.imwrite(os.path.join(image_path,nom_fichier), image_traitee, [int(cv2.IMWRITE_JPEG_QUALITY), 95]) # JPEG file format
                else :
                    cv2.imwrite(os.path.join(image_path, nom_fichier), image_traitee) # TIFF file format
            labelInfo1.config(text = "capture n° "+ nom_fichier)
            nb_cap_pic += 1
        else :
            flag_cap_pic = False
            labelInfo1.config(text = "                                                                                                        ") 
            labelInfo1.config(text = " Capture pictures terminee")

def start_pic_capture() :
    global nb_acq_pic,flag_cap_pic,nb_cap_pic,start,timer1
    flag_cap_pic = True
    nb_cap_pic =1
    start = datetime.now()
    timer1 = time.time()
 
def stop_pic_capture() :
    global nb_cap_pic,val_nb_captures
    nb_cap_pic = val_nb_captures +1

def video_capture(image_sauve) :
    global image_traitee,start_video,nb_cap_video,nb_acq_video,labelInfo1,flag_cap_video,video_path,val_nb_capt_video,video,videoOut,echelle11,flag_filtrage_ON,\
           flag_pause_video,timer1,val_deltat,flag_DETECT_STARS,nb_sat,trig_count_down,flag_TRIGGER,flag_sat_detected,video_frame_number,video_frame_position
    if flag_camera_ok == True :
        if nb_cap_video == 1 :
            if flag_HQ == 0 and flag_filtrage_ON == True :
                if flag_filter_wheel == True:
                    if Dev_system == "Windows" :
                        nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.mp4'
                    else :
                        nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.mp4'
                else :
                    if Dev_system == "Windows" :
                       nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.mp4'
                    else :
                       nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.mp4'
            else :
                if flag_filter_wheel == True:
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.avi'
                else :
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
            path_video=os.path.join(video_path,nom_video)
            if flag_HQ == 1 or flag_filtrage_ON == False :
                fourcc = 0 # video RAW
                if image_sauve.ndim == 3 :
                    height,width,layers = image_sauve.shape
                    video = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor=True) # video RAW
                else :
                    height,width = image_sauve.shape
                    video = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor=False) # video RAW
            else :
                if image_sauve.ndim == 3 :
                    height,width,layers = image_sauve.shape
                    if Dev_system == "Windows" :
                        path_video_Windows = video_path + '/' + nom_video
                        gst_pipe = "appsrc ! video/x-raw,format=BGR,width=" + str(width) + ",height=" + str(height) + ",framerate=25/1 ! queue ! videoconvert ! nvh264enc ! video/x-h264,profile=high ! h264parse ! qtmux ! filesink location=" + path_video_Windows
                        video = cv2.VideoWriter(gst_pipe, cv2.CAP_GSTREAMER, 0, float(25), (width, height), isColor = (image_traitee.ndim > 1))
                    else :
                        gst_pipe = "appsrc ! video/x-raw,format=BGR,width=" + str(width) + ",height=" + str(height) + ",framerate=25/1 ! queue ! videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! nvv4l2h264enc qp-range=0,1:0,1:0,1 profile=3 control-rate=0 ! h264parse ! qtmux ! filesink location=" + path_video
                        video = cv2.VideoWriter(gst_pipe, cv2.CAP_GSTREAMER, 0, float(25), (width, height), isColor = (image_traitee.ndim > 1))
                else :
                    height,width = image_sauve.shape
                    if Dev_system == "Windows" :
                        fourcc = cv2.VideoWriter_fourcc(*'XVID') # video compressée
                        video = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor = False) # video compressée
                    else :
                        gst_pipe = "appsrc ! video/x-raw,format=GRAY8,width=" + str(width) + ",height=" + str(height) + ",framerate=25/1 ! queue ! videoconvert ! video/x-raw,format=GRAY8 ! nvv4l2h264enc ! h264parse ! qtmux ! filesink location=" + path_video
                        video = cv2.VideoWriter(gst_pipe, cv2.CAP_GSTREAMER, 0, float(25), (width, height), isColor = (image_traitee.ndim > 1))
                if  (not video.isOpened()):
                    fourcc = cv2.VideoWriter_fourcc(*'XVID') # video compressée
                    if image_sauve.ndim == 3 :
                        height,width,layers = image_sauve.shape
                        video = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor = True) # video compressée
                    else :
                        height,width = image_sauve.shape
                        video = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor = False) # video compressée
            labelInfo1.config(text = "                                                                                                        ") 
            labelInfo1.config(text = " Acquisition vidéo en cours")
            nb_cap_video = nb_cap_video+1
        if flag_pause_video == False :
            if time.time() - timer1 >= val_deltat :
                if flag_TRIGGER == 0 :
                    if nb_cap_video <= val_nb_capt_video :    
                        if image_sauve.ndim == 3 :
                            video.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                        else :
                            video.write(image_sauve)
                        if (nb_cap_video % 5) == 0 :
                            time_rec = int(nb_cap_video/2.5)/10
                            labelInfo1.config(text = " frame : " + str (nb_cap_video) + "    " + str (time_rec) + " sec                   ")
                        nb_cap_video += 1
                if flag_TRIGGER == 1 and flag_TRKSAT == 1 :
                    if nb_cap_video <= val_nb_capt_video :   
                        if flag_sat_detected == True :
                            trig_count_down = 10
                            texte = " Sat detected  "
                        else :
                            trig_count_down = trig_count_down - 1
                            texte = " No Sat  "
                        if trig_count_down > 0 :
                            nb_cap_video = nb_cap_video + 1
                            time_rec = int(nb_cap_video/2.5)/10
                            texte = texte + str (nb_cap_video) + "    " + str (time_rec) + " sec                   "
                            if image_traitee.ndim == 3 :
                                video.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                            else :
                                video.write(image_sauve)
                            labelInfo1.config(text = texte)                    
                        else :
                            trig_count_down = 0   
                if nb_cap_video > val_nb_capt_video :
                    video.release()
                    flag_cap_video = False
                    labelInfo1.config(text = " Acquisition vidéo terminee     ")
                timer1 = time.time()
        else :
            labelInfo1.config(text = " PAUSE VIDEO       ")
    else :
        if nb_cap_video == 1 :
            if flag_HQ == 0 :
                if flag_filter_wheel == True:
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.avi'
                else :
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
            else :
                if flag_filter_wheel == True:
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.avi'
                else :
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
                
            path_video=os.path.join(video_path,nom_video)
            if flag_HQ == 1 :
                fourcc = 0 # video RAW
                if image_sauve.ndim == 3 :
                    height,width,layers = image_sauve.shape
                    videoOut = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor=True) # video RAW
                else :
                    height,width = image_sauve.shape
                    videoOut = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor=False) # video RAW
            else :
                fourcc = cv2.VideoWriter_fourcc(*'XVID') # video compressée
                if image_sauve.ndim == 3 :
                    height,width,layers = image_sauve.shape
                    videoOut = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor = True) # video compressée
                else :
                    height,width = image_sauve.shape
                    videoOut = cv2.VideoWriter(path_video, cv2.CAP_FFMPEG, fourcc, 25, (width, height), isColor = False) # video compressée
            labelInfo1.config(text = "                                                                                                        ") 
            labelInfo1.config(text = " Acquisition vidéo en cours")
            nb_cap_video = nb_cap_video+1
        
        if flag_TRIGGER == 0 :
            if nb_cap_video > 1 and nb_cap_video < val_nb_capt_video + 1 :    
                if image_sauve.ndim == 3 :
                    videoOut.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                else :
                    videoOut.write(image_sauve)
                if (nb_cap_video % 5) == 0 :
                    time_rec = int(nb_cap_video/2.5)/10
                    labelInfo1.config(text = " frame : " + str (nb_cap_video) + "    " + str (time_rec) + " sec                   ")
                nb_cap_video += 1
                
        if flag_TRIGGER == 1 and flag_TRKSAT == 1 :
            if video_frame_position <  video_frame_number - 1 :    
                if flag_sat_detected == True :
                    trig_count_down = 10
                    texte = " Sat detected  "
                else :
                    trig_count_down = trig_count_down - 1
                    texte = " No Sat  "
                if trig_count_down > 0 :
                    nb_cap_video = nb_cap_video + 1
                    texte = texte + str (nb_cap_video) + "       "
                    if image_traitee.ndim == 3 :
                        videoOut.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                    else :
                        videoOut.write(image_sauve)
                    labelInfo1.config(text = texte)                    
                else :
                    trig_count_down = 0   
        if nb_cap_video > val_nb_capt_video or video_frame_position > video_frame_number - 2 :
            videoOut.release()
            flag_cap_video = False
            labelInfo1.config(text = " Acquisition vidéo terminee     ")

                
def start_video_capture() :
    global nb_cap_video,flag_cap_video,start_video,val_nb_capt_video,timer1,compteur_images,numero_image
    flag_cap_video = True
    nb_cap_video =1
    if val_nb_capt_video == 0 :
        val_nb_capt_video = 10000
    start_video = datetime.now()
    if flag_camera_ok == True :
        timer1 = time.time()
        compteur_images = 0
        numero_image = 0
    
 
def stop_video_capture() :
    global nb_cap_video,val_nb_capt_video,FDIF
    nb_cap_video = val_nb_capt_video +1
 
def pause_video_capture() :
    global flag_pause_video
    if flag_camera_ok == True :
        if flag_pause_video == True :
            flag_pause_video = False
        else :
            flag_pause_video = True

def load_video() :
    global flag_premier_demarrage,Video_Test,flag_image_mode,flag_image_video_loaded

    flag_image_video_loaded = False
    filetypes = (
            ('Movie files', '*.mp4 *.avi *.ser *.mov'),
            ('Image files', '*.tif *.tiff *.jpg *.jpeg'),
            ('ALL files', '*.*')
        )

    Video_name = fd.askopenfilename(title='Open a file',initialdir=video_path,filetypes=filetypes)
    if Video_name != "" :
        Video_Test = Video_name
        flag_premier_demarrage = True
        if Video_Test.lower().endswith(('.mp4', '.avi', '.ser', '.mov')) :
            flag_image_mode = False
        if Video_Test.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) :
            flag_image_mode = True
    flag_image_video_loaded = True

def choose_dir_vid() :
    global flag_premier_demarrage,video_path

    video_path = fd.askdirectory()

def choose_dir_pic() :
    global flag_premier_demarrage,image_path

    image_path = fd.askdirectory()

def mode_Lineaire() :
    global Corr_GS,flag_lin_gauss
    flag_lin_gauss = 1
    for x in range(0,255) :
        Corr_GS[x] = 1

def mode_Gauss() :
    global Corr_GS,flag_lin_gauss
    flag_lin_gauss = 2
    for x in range(0,255) :
        Corr_GS[x] = np.exp(-0.5*((x*0.0392157-5-val_Mu)/val_Ro)**2)

def mode_Stars() :
    global Corr_GS,flag_lin_gauss
    flag_lin_gauss = 3
    for x in range(0,255) :
        Corr_GS[x] = np.exp(-0.5*((x*0.0392157-5-val_Mu)/val_Ro)**2)
    

def mode_acq_rapide() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,timeoutexp,flag_read_speed,\
           frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_acq_rapide = "Fast"
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        flag_stop_acquisition=True
        time.sleep(1)
        exp_min=100 #µs
        exp_max=10000 #µs
        exp_delta=100 #µs
        exp_interval=2000 #µs
        val_exposition=exp_min
        echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
        echelle1.set (val_exposition)
        echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)
        exposition = val_exposition
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
        timeoutexp = (exposition / 1000) * 2 + 500
        camera.default_timeout = timeoutexp
        if flag_read_speed == "Slow" :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
        else :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        time.sleep(0.1)
        flag_stop_acquisition=False
 
def mode_acq_mediumF() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\
           val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_acq_rapide = "MedF"
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        flag_stop_acquisition=True
        time.sleep(1)
        exp_min=1 #ms
        exp_max=400 #ms
        exp_delta=1 #ms
        exp_interval=50
        val_exposition=exp_min
        echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
        echelle1.set (val_exposition)
        echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)
        exposition = val_exposition * 1000
        if flag_read_speed == "Slow" :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
        else :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
        timeoutexp = val_exposition * 2 + 500
        camera.default_timeout = timeoutexp
        time.sleep(0.1)
        flag_stop_acquisition=False

def mode_acq_mediumS() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\
           val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_acq_rapide = "MedS"
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        flag_stop_acquisition=True
        time.sleep(1)
        exp_min=1 #ms
        exp_max=1000 #ms
        exp_delta=1 #ms
        exp_interval=200
        val_exposition=exp_min
        echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
        echelle1.set (val_exposition)
        echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)
        exposition = val_exposition * 1000
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
        timeoutexp = val_exposition * 2 + 500
        camera.default_timeout = timeoutexp
        if flag_read_speed == "Slow" :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
        else :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        time.sleep(0.1)
        flag_stop_acquisition=False


def mode_acq_lente() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\
           val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_acq_rapide = "Slow"
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        flag_stop_acquisition=True
        time.sleep(1)
        exp_min=500 #ms
        exp_max=20000 #ms
        exp_delta=100 #ms
        exp_interval=5000
        val_exposition=exp_min
        echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
        echelle1.set (val_exposition)
        echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)
        exposition = val_exposition * 1000
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
        timeoutexp = val_exposition * 2 + 500
        camera.default_timeout = timeoutexp
        if flag_read_speed == "Slow" :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
        else :
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        time.sleep(0.1)
        flag_stop_acquisition=False

def valeur_exposition (event=None) :
    global timeoutexp,timeout_val,flag_acq_rapide,camera,val_exposition,echelle1,val_resolution,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        if flag_autoexposure_exposition == False :
            flag_stop_acquisition=True
            val_exposition = echelle1.get()
            if flag_acq_rapide == "Fast" :
                exposition = val_exposition
            else :
                exposition = val_exposition*1000
            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
            if flag_acq_rapide == "Fast" :
                timeoutexp = (exposition / 1000) * 2 + 500
            else :
                timeoutexp =  val_exposition * 2 + 500
            camera.default_timeout = timeoutexp
            time.sleep(0.05)
            flag_stop_acquisition=False


def valeur_gain (event=None) :
    global camera,val_gain,echelle2,flag_stop_acquisition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_sub_dark = False
        dispo_dark = 'Dark NO        '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        if flag_autoexposure_gain == False :
            val_gain = echelle2.get()
            camera.set_control_value(asi.ASI_GAIN, val_gain)


def choix_BIN1(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,mode_BIN,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video

    if flag_camera_ok == True :
        if flag_cap_video == False :
            reset_general_FS()
            reset_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_nouvelle_resolution = True
            flag_stop_acquisition=True
            stop_tracking()
            flag_sub_dark = False
            dispo_dark = 'Dark NO        '
            choix_sub_dark.set(0)
            labelInfoDark = Label (cadre, text = dispo_dark)
            labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
            time.sleep(0.5)
            echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
            val_resolution = 1
            echelle3.set (val_resolution)
            echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)
            time.sleep(0.1)
            mode_BIN = 1
            choix_resolution_camera()
            flag_stop_acquisition=False

def choix_BIN2(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,mode_BIN,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video

    if flag_camera_ok == True :
        if flag_cap_video == False :
            reset_general_FS()
            reset_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_nouvelle_resolution = True
            flag_stop_acquisition=True
            stop_tracking()
            flag_sub_dark = False
            dispo_dark = 'Dark NO        '
            choix_sub_dark.set(0)
            labelInfoDark = Label (cadre, text = dispo_dark)
            labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
            time.sleep(0.5)
            echelle3 = Scale (cadre, from_ = 1, to = 7, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
            val_resolution = 1
            echelle3.set (val_resolution)
            echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)
            mode_BIN = 2
            choix_resolution_camera()
            flag_stop_acquisition=False
    
def choix_resolution_camera(event=None) :
    global val_delai,val_FS,camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,echelle3,\
           flag_image_disponible,flag_acquisition_en_cours,flag_stop_acquisition,labelInfoDark,flag_sub_dark,dispo_dark,\
           choix_sub_dark,flag_nouvelle_resolution,tnr,inSize,backend,choix_TIP,flag_TIP,flag_cap_video
    
    if flag_camera_ok == True :
        if flag_cap_video == False :
            reset_FS()
            reset_general_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_stop_acquisition=True
            time.sleep(0.1)
            stop_tracking()
            flag_sub_dark = False
            dispo_dark = 'Dark NO        '
            choix_sub_dark.set(0)
            labelInfoDark = Label (cadre, text = dispo_dark)
            labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
            time.sleep(0.1)
            val_resolution = echelle3.get()
            if mode_BIN == 1 :
                res_cam_x = RES_X_BIN1[val_resolution-1]
                res_cam_y = RES_Y_BIN1[val_resolution-1]
            if mode_BIN == 2 :
                res_cam_x = RES_X_BIN2[val_resolution-1]
                res_cam_y = RES_Y_BIN2[val_resolution-1]
            inSize = (int(res_cam_x), int(res_cam_y))
            camera.stop_video_capture()
            time.sleep(0.1)
            camera.set_roi(None,None,res_cam_x, res_cam_y,mode_BIN,format_capture)
            time.sleep(0.1)
            flag_nouvelle_resolution = True
            camera.start_video_capture()
            print("resolution camera = ",res_cam_x," ",res_cam_y)
            flag_stop_acquisition=False

def choix_valeur_denoise(event=None) :
    global val_denoise
    val_denoise=echelle4.get()
    if val_denoise == 0 :
        val_denoise += 1

def choix_grid_CLAHE(event=None) :
    global val_grid_CLAHE
    val_grid_CLAHE=echelle109.get()

def commande_flipV() :
    global FlipV,val_FS,type_debayer,type_flip

    if flag_camera_ok == True :
        reset_FS()
        if choix_flipV.get() == 0 :
            FlipV = 0
        else :
            FlipV = 1
        if FlipV == 1 and FlipH == 1 :
            type_debayer = cv2.COLOR_BayerRG2RGB
            camera.set_control_value(asi.ASI_FLIP, 3)
        if FlipV == 0 and FlipH == 0 :
            type_debayer = cv2.COLOR_BayerBG2RGB
            camera.set_control_value(asi.ASI_FLIP, 0)
        if FlipV == 1 and FlipH == 0 :
            type_debayer = cv2.COLOR_BayerGR2RGB
            camera.set_control_value(asi.ASI_FLIP, 2)
        if FlipV == 0 and FlipH == 1 :
            type_debayer = cv2.COLOR_BayerGB2RGB
            camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipV.get() == 0 :
            FlipV = 0
        else :
            FlipV = 1
        if FlipV == 1 and FlipH == 1 :
            type_flip = "both"
        if FlipV == 0 and FlipH == 0 :
            type_flip = "none"
        if FlipV == 1 and FlipH == 0 :
            type_flip = "vertical"
        if FlipV == 0 and FlipH == 1 :
            type_flip = "horizontal"
        

def commande_flipH() :
    global FlipH,val_FS,type_debayer,type_flip

    if flag_camera_ok == True :
        reset_FS()
        if choix_flipH.get() == 0 :
            FlipH = 0
        else :
            FlipH = 1
        if FlipV == 1 and FlipH == 1 :
            type_debayer = cv2.COLOR_BayerRG2RGB
            camera.set_control_value(asi.ASI_FLIP, 3)
        if FlipV == 0 and FlipH == 0 :
            type_debayer = cv2.COLOR_BayerBG2RGB
            camera.set_control_value(asi.ASI_FLIP, 0)
        if FlipV == 1 and FlipH == 0 :
            type_debayer = cv2.COLOR_BayerGR2RGB
            camera.set_control_value(asi.ASI_FLIP, 2)
        if FlipV == 0 and FlipH == 1 :
            type_debayer = cv2.COLOR_BayerGB2RGB
            camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipH.get() == 0 :
            FlipH = 0
        else :
            FlipH = 1
        if FlipV == 1 and FlipH == 1 :
            type_flip = "both"
        if FlipV == 0 and FlipH == 0 :
            type_flip = "none"
        if FlipV == 1 and FlipH == 0 :
            type_flip = "vertical"
        if FlipV == 0 and FlipH == 1 :
            type_flip = "horizontal"

def commande_img_Neg() :
    global ImageNeg,trsf_r,trsf_g,trsf_b
    if choix_img_Neg.get() == 0 :
        ImageNeg = 0
    else :
        ImageNeg = 1
        
def commande_TIP() :
    global flag_TIP
    if choix_TIP.get() == 0 :
        flag_TIP = False
    else :
        flag_TIP = True

def commande_SAT() :
    global flag_SAT
    if choix_SAT.get() == 0 :
        flag_SAT = False
    else :
        flag_SAT = True

def commande_mount() :
    global flag_mountpos
    if choix_mount.get() == 0 :
        flag_mountpos = False
    else :
        flag_mountpos = True

def commande_cross() :
    global flag_cross
    if choix_cross.get() == 0 :
        flag_cross = False
    else :
        flag_cross = True

def commande_mode_full_res() :
    global flag_full_res
    if choix_mode_full_res.get() == 0 :
        flag_full_res = 0
    else :
        flag_full_res = 1

def choix_valeur_CLAHE(event=None) :
    global val_contrast_CLAHE,echelle9
    val_contrast_CLAHE=echelle9.get()
        
def commande_sharpen_soft1() :
    global flag_sharpen_soft1
    if choix_sharpen_soft1.get() == 0 :
        flag_sharpen_soft1 = 0
    else :
        flag_sharpen_soft1 = 1

def commande_sharpen_soft2() :
    global flag_sharpen_soft2
    if choix_sharpen_soft2.get() == 0 :
        flag_sharpen_soft2 = 0
    else :
        flag_sharpen_soft2 = 1
        
def commande_denoise_soft() :
    global flag_denoise_soft
    if choix_denoise_soft.get() == 0 :
        flag_denoise_soft = 0
    else :
        flag_denoise_soft = 1

def commande_denoise_Paillou() :
    global flag_denoise_Paillou
    if choix_denoise_Paillou.get() == 0 :
        flag_denoise_Paillou = 0
    else :
        flag_denoise_Paillou = 1

def commande_denoise_Paillou2() :
    global flag_denoise_Paillou2
    if choix_denoise_Paillou2.get() == 0 :
        flag_denoise_Paillou2 = 0
    else :
        flag_denoise_Paillou2 = 1

def commande_HST() :
    global flag_HST
    if choix_HST.get() == 0 :
        flag_HST = 0
    else :
        flag_HST = 1

def commande_TRSF() :
    global flag_TRSF
    if choix_TRSF.get() == 0 :
        flag_TRSF = 0
    else :
        flag_TRSF = 1

def commande_TRGS() :
    global flag_TRGS
    if choix_TRGS.get() == 0 :
        flag_TRGS = 0
    else :
        flag_TRGS = 1

def commande_TRCLL() :
    global flag_TRCLL
    if choix_TRCLL.get() == 0 :
        flag_TRCLL = 0
    else :
        flag_TRCLL = 1

def commande_DEMO() :
    global flag_DEMO
    if choix_DEMO.get() == 0 :
        flag_DEMO = 0
    else :
        flag_DEMO = 1

def commande_STAB() :
    global flag_STAB, flag_Template
    if choix_STAB.get() == 0 :
        flag_STAB = False
    else :
        flag_STAB = True
    flag_Template = False

def commande_DETECT_STARS() :
    global flag_DETECT_STARS,flag_nouvelle_resolution
    if choix_DETECT_STARS.get() == 0 :
        flag_DETECT_STARS = 0
    else :
        flag_DETECT_STARS = 1
        flag_nouvelle_resolution = True
        
def commande_denoise_stacking_Paillou() :
    global flag_AADF,val_FSDN,compteur_AADF,Im1fsdnOK,Im2fsdnOK
    if choix_denoise_stacking_Paillou.get() == 0 :
        flag_AADF = False
    else :
        flag_AADF = True
        val_FSDN = 1
        compteur_AADF = 0
        Im1fsdnOK = False
        Im2fsdnOK = False

def commande_3FNR() :
    global flag_3FNR,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start
    if choix_3FNR.get() == 0 :
        flag_3FNR = False
    else :
        flag_3FNR = True
        compteur_3FNR = 0
        img1_3FNROK = False
        img2_3FNROK = False
        img3_3FNROK = False
        FNR_First_Start = True


def commande_ghost_reducer() :
    global choix_ghost_reducer, flag_ghost_reducer
    if choix_ghost_reducer.get() == 0 :
        flag_ghost_reducer = 0
    else :
        flag_ghost_reducer = 1

def choix_KNN() :
    global flag_denoise_KNN
    if choix_denoise_KNN.get() == 0 :
        flag_denoise_KNN = 0
    else :
        flag_denoise_KNN = 1

def commande_reduce_variation() :
    global flag_reduce_variation,val_RV,compteur_RV,Im1rvOK,Im2rvOK
    if choix_reduce_variation.get() == 0 :
        flag_reduce_variation = False
    else :
        flag_reduce_variation = True
        val_RV = 1
        compteur_RV = 0
        Im1rvOK = False
        Im2rvOK = False

def commande_DEF() :
    global flag_DEF
    if choix_DEF.get() == 0 :
        flag_DEF = 0
    else :
        flag_DEF = 1

def commande_GR() :
    global flag_GR
    if choix_GR.get() == 0 :
        flag_GR = 0
    else :
        flag_GR = 1

def commande_HDR() :
    global flag_HDR
    if choix_HDR.get() == 0 :
        flag_HDR = False
    else :
        flag_HDR = True

def choix_val_ghost_reducer(event=None) :
    global val_ghost_reducer,echelle130
    val_ghost_reducer=echelle130.get()

def commande_histogram_equalize2() :
    global flag_histogram_equalize2
    if choix_histogram_equalize2.get() == 0 :
        flag_histogram_equalize2 = 0
    else :
        flag_histogram_equalize2 = 1

def choix_histo_min(event=None) :
    global camera,val_histo_min,echelle5
    val_histo_min=echelle5.get()
    
def choix_phi(event=None) :
    global val_phi,echelle12
    val_phi=echelle12.get()

def choix_theta(event=None) :
    global val_theta,echelle13
    val_theta=echelle13.get()
    
def choix_histo_max(event=None) :
    global camera,val_histo_max,echelle6
    val_histo_max=echelle6.get()
      
def commande_histogram_stretch() :
    global flag_histogram_stretch
    if choix_histogram_stretch.get() == 0 :
        flag_histogram_stretch = 0
    else :
        flag_histogram_stretch = 1
    
def commande_histogram_phitheta() :
    global flag_histogram_phitheta
    if choix_histogram_phitheta.get() == 0 :
        flag_histogram_phitheta = 0
    else :
        flag_histogram_phitheta = 1
      
def commande_contrast_CLAHE() :
    global flag_contrast_CLAHE
    if choix_contrast_CLAHE.get() == 0 :
        flag_contrast_CLAHE= 0
    else :
        flag_contrast_CLAHE = 1

def commande_CLL() :
    global flag_CLL
    if choix_CLL.get() == 0 :
        flag_CLL = 0
    else :
        flag_CLL = 1

def commande_filtrage_ON() :
    global flag_filtrage_ON
    reset_general_FS()
    if choix_filtrage_ON.get() == 0 :
        flag_filtrage_ON= 0
    else :
        flag_filtrage_ON = 1

def commande_HQ_capt() :
    global flag_HQ
    if choix_HQ_capt.get() == 0 :
        flag_HQ = 0
    else :
        flag_HQ = 1

def commande_AmpSoft() :
    global flag_AmpSoft
    if choix_AmpSoft.get() == 0 :
        flag_AmpSoft = 0
    else :
        flag_AmpSoft = 1

def commande_hard_bin() :
    global camera

    if flag_camera_ok == True :
        if choix_hard_bin.get() == 0 :
            camera.set_control_value(asi.ASI_HARDWARE_BIN,0)
        else :
            camera.set_control_value(asi.ASI_HARDWARE_BIN,1)
        time.sleep(0.1)

def commande_hold_picture() :
    global flag_hold_picture
    if choix_hold_picture.get() == 0 :
        flag_hold_picture = 0
    else :
        flag_hold_picture = 1

def choix_mean_stacking(event=None):
    global flag_stacking
    flag_stacking = "Mean"
    reset_FS()

def choix_sum_stacking(event=None):
    global flag_stacking
    flag_stacking = "Sum"
    reset_FS()
    
def choix_front(event=None):
    global flag_front
    flag_front = True
    print("Front")

def choix_back(event=None):
    global flag_front
    flag_front = False
    print("Back")

def choix_dyn_high(event=None):
    global flag_dyn_AADP
    flag_dyn_AADP = 1
    

def choix_dyn_low(event=None):
    global flag_dyn_AADP
    flag_dyn_AADP = 0    


def commande_autoexposure() :
    global flag_autoexposure_exposition,flag_autoexposure_gain,camera,controls,echelle1,val_exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        flag_sub_dark = False
        dispo_dark = 'Dark NON dispo '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        if choix_autoexposure.get() == 0 :
            flag_autoexposure_exposition = False
            camera.set_control_value(asi.ASI_EXPOSURE,controls['Exposure']['DefaultValue'],auto=False)
            val_exposition = echelle1.get()
            exposition = val_exposition * 1000
            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
            timeout = (camera.get_control_value(asi.ASI_EXPOSURE)[0] / 1000) * 2 + 500
            camera.default_timeout = timeout
            camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        else :
            flag_autoexposure_exposition = True
            mode_acq_mediumS()
            camera.set_control_value(asi.ASI_EXPOSURE,controls['Exposure']['DefaultValue'],auto=True)
            camera.set_control_value(controls['AutoExpMaxExpMS']['ControlType'], 500)

def commande_autogain() :
    global flag_autoexposure_exposition,flag_autoexposure_gain,camera,controls,val_gain,labelInfoDark,flag_sub_dark,dispo_dark

    if flag_camera_ok == True :
        flag_sub_dark = False
        dispo_dark = 'Dark NON dispo '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        if choix_autogain.get() == 0 :
            flag_autoexposure_gain = False
            camera.set_control_value(asi.ASI_GAIN,controls['Gain']['DefaultValue'],auto=False)
            camera.set_control_value(asi.ASI_GAIN, val_gain)
        else :
            flag_autoexposure_gain = True
            camera.set_control_value(asi.ASI_GAIN,controls['Gain']['DefaultValue'],auto=True)
            camera.set_control_value(controls['AutoExpMaxGain']['ControlType'], controls['AutoExpMaxGain']['MaxValue'])
        
def commande_noir_blanc() :
    global val_FS,flag_noir_blanc,flag_NB_estime,format_capture,flag_stop_acquisition,flag_filtrage_ON,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    if flag_camera_ok == True :
        reset_FS()
        reset_general_FS()
        flag_stop_acquisition=True
        dispo_dark = 'Dark NON dispo '
        choix_sub_dark.set(0)
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark)
        flag_restore_filtrage = False
        if flag_filtrage_ON == True :
            flag_filtrage_ON == False
            flag_restore_filtrage = True
            time.sleep(0.3)
        if choix_noir_blanc.get() == 0 and flag_colour_camera == True :
            flag_noir_blanc = 0
            camera.stop_video_capture()
            time.sleep(0.1)
            format_capture = asi.ASI_IMG_RAW8
            camera.set_image_type(format_capture)
            time.sleep(0.2)
            camera.start_video_capture()
        else :
            flag_noir_blanc = 1
            if choix_noir_blanc_estime.get() == 0 :
                flag_NB_estime = 0
            else :
                flag_NB_estime = 1
            if flag_colour_camera == True :
                camera.stop_video_capture()
                time.sleep(0.5)
                if flag_NB_estime == 1 :
                    format_capture = asi.ASI_IMG_RAW8
                else :
                    format_capture = asi.ASI_IMG_Y8
                camera.set_image_type(format_capture)
                time.sleep(0.3)
                camera.start_video_capture()
            else :
                time.sleep(0.1)
                format_capture = asi.ASI_IMG_RAW8
                camera.set_image_type(format_capture)
                time.sleep(0.1)
                camera.start_video_capture()
        flag_stop_acquisition = False
        time.sleep(0.4)
        if flag_restore_filtrage == True :
            flag_filtrage_ON = True

def commande_reverse_RB() :
    global flag_reverse_RB
    if choix_reverse_RB.get() == 0 :
        flag_reverse_RB = 0
    else :
        flag_reverse_RB = 1

def choix_nb_captures(event=None) :
    global val_nb_captures
    val_nb_captures=echelle8.get()

def choix_deltat(event=None) :
    global val_deltat
    val_deltat = echelle65.get()

def choix_nb_video(event=None) :
    global val_nb_capt_video
    val_nb_capt_video=echelle11.get()


def mode_gradient() :
    global grad_vignet
    grad_vignet = 1

def mode_vignetting() :
    global grad_vignet
    grad_vignet = 2
      
def choix_w_red(event=None) :
    global val_red, echelle14

    if flag_camera_ok == True :
        val_red=echelle14.get()
        camera.set_control_value(asi.ASI_WB_R, val_red)
    
def choix_w_blue(event=None) :
    global val_blue, echelle15

    if flag_camera_ok == True :
        val_blue=echelle15.get()
        camera.set_control_value(asi.ASI_WB_B, val_blue)

def choix_heq2(event=None) :
    global val_heq2, echelle16
    val_heq2=echelle16.get()

def choix_val_KNN(event=None) :
    global val_denoise_KNN, echelle30
    val_denoise_KNN=echelle30.get()

def choix_amplif(event=None) :
    global val_ampl, echelle80
    val_ampl = echelle80.get()

def choix_SGR(event=None) :
    global val_SGR, echelle60
    val_SGR=echelle60.get()

def choix_val_reduce_variation(event=None) :
    global val_reduce_variation, echelle270
    val_reduce_variation=echelle270.get()

def choix_AGR(event=None) :
    global val_AGR, echelle61
    val_AGR=echelle61.get()

def choix_USB(event=None) :
    global val_USB, echelle50,camera

    if flag_camera_ok == True :
        val_USB=echelle50.get()
        camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, val_USB)

def choix_ASI_GAMMA(event=None) :
    global ASIGAMMA, echelle200,camera

    if flag_camera_ok == True :
        ASIGAMMA=echelle204.get()
        camera.set_control_value(asi.ASI_GAMMA, ASIGAMMA)

def choix_sensor_ratio_4_3(event=None) :
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\
           RES_X_BIN1_4_3,RES_Y_BIN1_4_3,RES_X_BIN2_4_3,RES_Y_BIN2_4_3

    if flag_camera_ok == True :
        sensor_factor = "4_3"
        cam_displ_x = int(1350*fact_s)
        cam_displ_y = int(1012*fact_s)
        RES_X_BIN1 = RES_X_BIN1_4_3
        RES_Y_BIN1 = RES_Y_BIN1_4_3
        RES_X_BIN2 = RES_X_BIN2_4_3
        RES_Y_BIN2 = RES_Y_BIN2_4_3
        choix_resolution_camera()
        

def choix_sensor_ratio_16_9(event=None) :
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\
           RES_X_BIN1_16_9,RES_Y_BIN1_16_9,RES_X_BIN2_16_9,RES_Y_BIN2_16_9

    if flag_camera_ok == True :
        sensor_factor = "16_9"
        cam_displ_x = int(1350*fact_s)
        cam_displ_y = int(760*fact_s)
        RES_X_BIN1 = RES_X_BIN1_16_9
        RES_Y_BIN1 = RES_Y_BIN1_16_9
        RES_X_BIN2 = RES_X_BIN2_16_9
        RES_Y_BIN2 = RES_Y_BIN2_16_9
        choix_resolution_camera()

def choix_sensor_ratio_1_1(event=None) :
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\
           RES_X_BIN1_1_1,RES_Y_BIN1_1_1,RES_X_BIN2_1_1,RES_Y_BIN2_1_1

    if flag_camera_ok == True :
        v = "1_1"
        cam_displ_x = int(1012*fact_s)
        cam_displ_y = int(1012*fact_s)
        RES_X_BIN1 = RES_X_BIN1_1_1
        RES_Y_BIN1 = RES_Y_BIN1_1_1
        RES_X_BIN2 = RES_X_BIN2_1_1
        RES_Y_BIN2 = RES_Y_BIN2_1_1
        choix_resolution_camera()

def choix_read_speed_fast(event=None) :
    global cam_read_speed,camera,flag_read_speed

    if flag_camera_ok == True :
        camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
        flag_read_speed = "Fast"

def choix_read_speed_slow(event=None) :
    global cam_read_speed,camera,flag_read_speed

    if flag_camera_ok == True :
        camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
        flag_read_speed = "Slow"

def choix_position_EFW0(event=None) :
    global fw_position

    if flag_camera_ok == True :
        if flag_filter_wheel == True :
            fw_position = 0
            filter_wheel.set_position(fw_position)

def choix_position_EFW1(event=None) :
    global fw_position

    if flag_camera_ok == True :
        if flag_filter_wheel == True :
            fw_position = 1
            filter_wheel.set_position(fw_position)

def choix_position_EFW2(event=None) :
    global fw_position

    if flag_camera_ok == True :
        if flag_filter_wheel == True :
            fw_position = 2
            filter_wheel.set_position(fw_position)

def choix_position_EFW3(event=None) :
    global fw_position

    if flag_camera_ok == True :
        if flag_filter_wheel == True :
            fw_position = 3
            filter_wheel.set_position(fw_position)

def choix_position_EFW4(event=None) :
    global fw_position

    if flag_camera_ok == True :
        if flag_filter_wheel == True :
            fw_position = 4
            filter_wheel.set_position(fw_position)

def choix_bayer_RAW(event=None) :
    global video_debayer
    video_debayer = 0 # RAW

def choix_bayer_BG(event=None) :
    global video_debayer
    video_debayer = cv2.COLOR_BayerBG2RGB

def choix_bayer_GR(event=None) :
    global video_debayer
    video_debayer = cv2.COLOR_BayerGR2RGB
    
def choix_bayer_GB(event=None) :
    global video_debayer
    video_debayer = cv2.COLOR_BayerGB2RGB

def choix_bayer_RG(event=None) :
    global video_debayer
    video_debayer = cv2.COLOR_BayerRG2RGB


def choix_FS(event=None) :
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK,stack_div
    val_FS=echelle20.get()
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    if flag_stacking == "Mean":
        stack_div = val_FS
    else :
        stack_div = 1

def reset_FS(event=None) :
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    echelle20.set(val_FS)

def reset_general_FS():
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK,stack_div,echelle20,\
           flag_AADF,val_FSDN,compteur_AADF,Im1fsdnOK,Im2fsdnOK,\
           flag_STAB,flag_Template,choix_STAB,compteur_RV,Im1rvOK,Im2rvOK,flag_reduce_variation,choix_reduce_variation,\
           flag_3FNR,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    echelle20.set(val_FS)
    choix_denoise_stacking_Paillou.set(0)
    flag_AADF = False
    compteur_AADF = 0
    Im1fsdnOK = False
    Im2fsdnOK = False
    flag_STAB = False
    flag_Template = False
    choix_STAB.set(0)
    compteur_RV = 0
    Im1rvOK = False
    Im2rvOK = False
    flag_reduce_variation = False
    choix_reduce_variation.set(0)
    choix_3FNR.set(0)
    flag_3FNR = False
    compteur_3FNR = 0
    img1_3FNROK = False
    img2_3FNROK = False
    img3_3FNROK = False
    FNR_First_Start = True
   

def choix_val_sharpen(event=None) :
    global val_sharpen, echelle152
    val_sharpen = echelle152.get()

def choix_val_sharpen2(event=None) :
    global val_sharpen2, echelle154
    val_sharpen2 = echelle154.get()

def choix_val_sigma_sharpen(event=None) :
    global val_sigma_sharpen, echelle153
    val_sigma_sharpen = echelle153.get()

def choix_val_sigma_sharpen2(event=None) :
    global val_sigma_sharpen2, echelle155
    val_sigma_sharpen2 = echelle155.get()

def choix_val_SAT(event=None) :
    global val_SAT, echelle70
    val_SAT = echelle70.get()

def commande_FW() :
    time.sleep(0.01)

def choix_w_reds(event=None) :
    global val_reds, echelle100
    val_reds = echelle100.get()

def choix_w_greens(event=None) :
    global val_greens, echelle101
    val_greens = echelle101.get()
    
def choix_w_blues(event=None) :
    global val_blues, echelle102
    val_blues = echelle102.get()

def choix_Mu(event=None) :
    global val_Mu, echelle182,Corr_GS,flag_lin_gauss
    val_Mu = echelle82.get()
    if flag_lin_gauss == 2 or flag_lin_gauss == 3 :
        for x in range(0,255) :
            Corr_GS[x] = np.exp(-0.5*((x*0.0392157-5-val_Mu)/val_Ro)**2)

def choix_Ro(event=None) :
    global val_Ro, echelle184,Corr_GS,flag_lin_gauss
    val_Ro = echelle84.get()
    if flag_lin_gauss == 2 or flag_lin_gauss == 3 :
        for x in range(0,255) :
            Corr_GS[x] = np.exp(-0.5*((x*0.0392157-5-val_Mu)/val_Ro)**2)


def choix_Var_CLL(event=None) :
    global val_MuCLL, val_RoCLL, val_AmpCLL, echelle200, echelle201, echelle202,Corr_CLL
    val_MuCLL = echelle200.get()
    val_RoCLL = echelle201.get()
    val_AmpCLL = echelle202.get()
    for x in range(0,256) :
        Corr = np.exp(-0.5*((x*0.0392157-val_MuCLL)/val_RoCLL)**2)
        Corr_CLL[x] = int(x * (1/(1 + val_AmpCLL*Corr)))
        if x> 0 :
            if Corr_CLL[x] <= Corr_CLL[x-1] :
                Corr_CLL[x] = Corr_CLL[x-1]
        
def commande_TRKSAT() :
    global flag_TRKSAT,flag_nouvelle_resolution
    if choix_TRKSAT.get() == 0 :
        flag_TRKSAT = 0
    else :
        flag_TRKSAT = 1
        flag_nouvelle_resolution = True

def commande_CONST() :
    global flag_CONST,flag_nouvelle_resolution
    if choix_CONST.get() == 0 :
        flag_CONST = 0
    else :
        flag_CONST = 1
        flag_nouvelle_resolution = True

def commande_REMSAT() :
    global flag_REMSAT,flag_nouvelle_resolution
    if choix_REMSAT.get() == 0 :
        flag_REMSAT = 0
    else :
        flag_REMSAT = 1
        flag_nouvelle_resolution = True

def commande_TRIGGER() :
    global flag_TRIGGER
    if choix_TRIGGER.get() == 0 :
        flag_TRIGGER = 0
    else :
        flag_TRIGGER = 1

def choix_nb_darks(event=None) :
    global val_nb_darks, echelle10
    val_nb_darks=echelle10.get()

def commande_sub_dark() :
    global flag_sub_dark,dispo_dark
    if choix_sub_dark.get() == 0 :
        flag_sub_dark = False
    else :
        flag_sub_dark = True

def raz_framecount() :
    global compteur_images,numero_image,image_camera,image_camera_old
    compteur_images = 0
    numero_image = 0
    image_camera = 0
    image_camera_old = 0

def start_cap_dark() :
    global timeoutexp,val_FS,camera,flag_sub_dark,dispo_dark,labelInfoDark,flag_stop_acquisition,flag_acquisition_en_cours,labelInfo1,val_nb_darks,text_info1,xLI1,yLI1,Master_Dark

    if flag_camera_ok == True :
        reset_FS()
        if askyesno("Cover the Lens", "Dark acquisition continue ?") :
            flag_stop_acquisition=True
            text_info1 = "Initialisation capture DARK"
            labelInfo1 = Label (cadre, text = text_info1)
            labelInfo1.place(anchor="w", x=xLI1+delta_s,y=yLI1)
            while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
                time.sleep(0.1)
            dispo_dark = 'Dark non disponible'
            print("debut acquisition darks")
            try :
                num_dark = 1
                while num_dark <= val_nb_darks :
                    time.sleep(0.1)
                    dark_tmp=camera.capture_video_frame(filename=None,timeout=timeoutexp)
                    if num_dark == 1 :
                        dark_tempo=np.uint32(dark_tmp)
                    else :
                        dark_tempo= dark_tempo + dark_tmp
                    print(num_dark)
                    num_dark += 1
                time.sleep(0.5)
                dark_tempo = dark_tempo // val_nb_darks
                dark_tempo[dark_tempo > 255] = 255
                dark_tempo[dark_tempo < 0] = 0
                Mean_dark = np.uint8(dark_tempo)
                dispo_dark = 'Dark disponible'
                if flag_colour_camera == True :
                    if flag_noir_blanc == 0 :
                        Master_Dark = Mean_dark.copy()
                        Master_Dark = cv2.GaussianBlur(Master_Dark,(27,27),cv2.BORDER_DEFAULT,) * 3
                        print('master dark ok colour')
                    else :
                        Master_Dark = cv2.cvtColor(Mean_dark, cv2.COLOR_BGR2GRAY)
                        Master_Dark = gaussianblur_mono(Master_Dark,27) * 3
                        print('master dark ok gray')
                else :
                    Master_Dark = cv2.GaussianBlur(Master_Dark,(27,27),cv2.BORDER_DEFAULT,) * 3
                    print('master dark ok mono raw8')    
            except :
                flag_sub_dark = False
                dispo_dark = 'Dark non disponible'
                print("erreur creation Dark")
                time.sleep(0.05)
            camera.stop_video_capture()
            time.sleep(0.1)
            labelInfoDark = Label (cadre, text = dispo_dark)
            labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark-5)
            flag_stop_acquisition=False
            camera.start_video_capture()
            time.sleep(0.1)
        else :
            print("we stop")
            flag_sub_dark = False
            labelInfoDark = Label (cadre, text = "Dark OK")
            labelInfoDark.place(anchor="w", x=xdark+delta_s,y=ydark-5)

def raz_tracking() :
    global flag_nouvelle_resolution
    flag_nouvelle_resolution = True

def stop_tracking() :
    global flag_TRKSAT,flag_DETECT_STARS,flag_nouvelle_resolution,choix_DETECT_STARS,choix_TRKSAT
    flag_TRKSAT = 0
    flag_DETECT_STARS = 0    
    flag_nouvelle_resolution = True
    choix_DETECT_STARS.set(0)
    choix_TRKSAT.set(0)


init_camera()


# initialisation des boites scrolbar, buttonradio et checkbutton

xS3=1700
yS3=80

xS1=1490
yS1=270


###########################
#     VARIOUS WIDGETS     #
###########################

# Bandwidth USB
labelParam50 = Label (cadre, text = "USB")
labelParam50.place(anchor="w", x=15,y=180)
echelle50 = Scale (cadre, from_ = 0, to = 100, command= choix_USB, orient=VERTICAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle50.set(val_USB)
echelle50.place(anchor="c", x=20,y=245)

if flag_camera_ok == True :
    # FW
    labelSR = Label (cadre, text = "Ratio :")
    labelSR.place (anchor="w",x=5, y=50)
    RBSR1 = Radiobutton(cadre,text="4/3", variable=sensor_ratio,command=choix_sensor_ratio_4_3,value=0)
    RBSR1.place(anchor="w",x=5, y=80)
    RBSR2 = Radiobutton(cadre,text="16/9", variable=sensor_ratio,command=choix_sensor_ratio_16_9,value=1)
    RBSR2.place(anchor="w",x=5, y=110)
    RBSR3 = Radiobutton(cadre,text="1/1", variable=sensor_ratio,command=choix_sensor_ratio_1_1,value=2)
    RBSR3.place(anchor="w",x=5, y=140)


if flag_camera_ok == True :
    # FW
    labelFW = Label (cadre, text = "FW :")
    labelFW.place (anchor="w",x=5, y=405)
    RBEFW1 = Radiobutton(cadre,text="#1", variable=fw_position_,command=choix_position_EFW0,value=0)
    RBEFW1.place(anchor="w",x=5, y=430)
    RBEFW2 = Radiobutton(cadre,text="#2", variable=fw_position_,command=choix_position_EFW1,value=1)
    RBEFW2.place(anchor="w",x=5, y=450)
    RBEFW3 = Radiobutton(cadre,text="#3", variable=fw_position_,command=choix_position_EFW2,value=2)
    RBEFW3.place(anchor="w",x=5, y=470)
    RBEFW4 = Radiobutton(cadre,text="#4", variable=fw_position_,command=choix_position_EFW3,value=3)
    RBEFW4.place(anchor="w",x=5, y=490)
    RBEFW5 = Radiobutton(cadre,text="#5", variable=fw_position_,command=choix_position_EFW4,value=4)
    RBEFW5.place(anchor="w",x=5, y=510)
else :
    labelFW = Label (cadre, text = "Debayer")
    labelFW.place (anchor="w",x=5, y=405)
    RBEFW1 = Radiobutton(cadre,text="RAW", variable=bayer_sensor,command=choix_bayer_RAW,value=1)
    RBEFW1.place(anchor="w",x=5, y=430)
    RBEFW2 = Radiobutton(cadre,text="BG", variable=bayer_sensor,command=choix_bayer_BG,value=2)
    RBEFW2.place(anchor="w",x=5, y=450)
    RBEFW3 = Radiobutton(cadre,text="GR", variable=bayer_sensor,command=choix_bayer_GR,value=3)
    RBEFW3.place(anchor="w",x=5, y=470)
    RBEFW4 = Radiobutton(cadre,text="GB", variable=bayer_sensor,command=choix_bayer_GB,value=4)
    RBEFW4.place(anchor="w",x=5, y=490)
    RBEFW5 = Radiobutton(cadre,text="RG", variable=bayer_sensor,command=choix_bayer_RG,value=5)
    RBEFW5.place(anchor="w",x=5, y=510)

labelTRK = Label (cadre, text = "Tracking")
labelTRK.place (anchor="w",x=0, y=550)

# Find Stars
CBDTCSTARS = Checkbutton(cadre,text="Stars", variable=choix_DETECT_STARS,command=commande_DETECT_STARS,onvalue = 1, offvalue = 0)
CBDTCSTARS.place(anchor="w",x=0, y=570)

# Track Satellites
CBTRKSAT = Checkbutton(cadre,text="Sat Detect", variable=choix_TRKSAT,command=commande_TRKSAT,onvalue = 1, offvalue = 0)
CBTRKSAT.place(anchor="w",x=0, y=590)

# Remove Satellites
CBREMSAT = Checkbutton(cadre,text="Sat Remov", variable=choix_REMSAT,command=commande_REMSAT,onvalue = 1, offvalue = 0)
CBREMSAT.place(anchor="w",x=0, y=610)

# Track Meteor
CBTRIGGER = Checkbutton(cadre,text="Trigger", variable=choix_TRIGGER,command=commande_TRIGGER,onvalue = 1, offvalue = 0)
CBTRIGGER.place(anchor="w",x=0, y=630)

# RAZ Tracking traces
Button7 = Button (cadre,text = "RAZ Trk", command = raz_tracking,padx=10,pady=0)
Button7.place(anchor="w", x=0,y=655)

# Track Meteor
CBCONST = Checkbutton(cadre,text="Reconst", variable=choix_CONST,command=commande_CONST,onvalue = 1, offvalue = 0)
CBCONST.place(anchor="w",x=0, y=675)


# Flip Vertical
CBFV = Checkbutton(cadre,text="Flip V", variable=choix_flipV,command=commande_flipV,onvalue = 1, offvalue = 0)
CBFV.place(anchor="w",x=1840+delta_s, y=30)

# Flip Horizontal
CBFH = Checkbutton(cadre,text="Flip H", variable=choix_flipH,command=commande_flipH,onvalue = 1, offvalue = 0)
CBFH.place(anchor="w",x=1840+delta_s, y=60)

# Set Gamma value ZWO camera
labelParam204 = Label (cadre, text = "Gamma Cam")
labelParam204.place(anchor="w", x=1840+delta_s,y=90)
echelle204 = Scale (cadre, from_ = 0, to = 100, command= choix_ASI_GAMMA, orient=VERTICAL, length = 130, width = 7, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle204.set(ASIGAMMA)
echelle204.place(anchor="c", x=1840+delta_s,y=170)

# Set camera read speed
labelCamSpeed = Label (cadre, text = "Cam Speed")
labelCamSpeed.place (anchor="w",x=1840+delta_s, y=260)
RBCS1 = Radiobutton(cadre,text="Fast", variable=cam_read_speed,command=choix_read_speed_fast,value=0)
RBCS1.place(anchor="w",x=1840+delta_s, y=280)
RBCS2 = Radiobutton(cadre,text="Slow", variable=cam_read_speed,command=choix_read_speed_slow,value=1)
RBCS2.place(anchor="w",x=1840+delta_s, y=300)

# Text in picture
CBTIP = Checkbutton(cadre,text="TIP", variable=choix_TIP,command=commande_TIP,onvalue = 1, offvalue = 0)
CBTIP.place(anchor="w",x=1840+delta_s, y=95+240)

# Mount
CBMNT = Checkbutton(cadre,text="AZ/H", variable=choix_mount,command=commande_mount,onvalue = 1, offvalue = 0)
CBMNT.place(anchor="w",x=1840+delta_s, y=115+240)

# Cross
CBCR = Checkbutton(cadre,text="Cr", variable=choix_cross,command=commande_cross,onvalue = 1, offvalue = 0)
CBCR.place(anchor="w",x=1840+delta_s, y=135+240)


# Histogram
CBHST = Checkbutton(cadre,text="Hst", variable=choix_HST,command=commande_HST,onvalue = 1, offvalue = 0)
CBHST.place(anchor="w",x=1840+delta_s, y=165+240)

# Histogram
CBTRSF = Checkbutton(cadre,text="Trsf", variable=choix_TRSF,command=commande_TRSF,onvalue = 1, offvalue = 0)
CBTRSF.place(anchor="w",x=1840+delta_s, y=190+240)

# Affichage fonction de transfert amplification soft
CBTRGS = Checkbutton(cadre,text="TrGS", variable=choix_TRGS,command=commande_TRGS,onvalue = 1, offvalue = 0)
CBTRGS.place(anchor="w",x=1840+delta_s, y=210+240)

# Affichage fonction de transfert Contrast Low Light
CBTRCLL = Checkbutton(cadre,text="TrCLL", variable=choix_TRCLL,command=commande_TRCLL,onvalue = 1, offvalue = 0)
CBTRCLL.place(anchor="w",x=1840+delta_s, y=230+240)

# Saturation
CBSAT = Checkbutton(cadre,text="SAT", variable=choix_SAT,command=commande_SAT,onvalue = 1, offvalue = 0)
CBSAT.place(anchor="w",x=1840+delta_s,y=260+240)
echelle70 = Scale (cadre, from_ = 0, to = 40, command= choix_val_SAT, orient=VERTICAL, length = 240, width = 7, resolution = 0.01, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle70.set(val_SAT)
echelle70.place(anchor="c", x=1855+delta_s,y=400+240)

# DEMO
CBDEMO = Checkbutton(cadre,text="Demo", variable=choix_DEMO,command=commande_DEMO,onvalue = 1, offvalue = 0)
CBDEMO.place(anchor="w",x=1840+delta_s, y=560+240)

# Choix filtrage ON
CBF = Checkbutton(cadre,text="Filters ON", variable=choix_filtrage_ON,command=commande_filtrage_ON,onvalue = 1, offvalue = 0)
CBF.place(anchor="w",x=1450+delta_s, y=50)

# Fulres displaying
CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=1525+delta_s, y=50)

# Choix forcage N&B
CBFNB = Checkbutton(cadre,text="Set B&W", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=1590+delta_s, y=50)

# Choix forcage N&B Estimate
CBFNBE = Checkbutton(cadre,text="B&W Est", variable=choix_noir_blanc_estime,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNBE.place(anchor="w",x=1660+delta_s, y=50)

# Choix forcage N&B Estimate
CBRRB = Checkbutton(cadre,text="Reverse R-B", variable=choix_reverse_RB,command=commande_reverse_RB,onvalue = 1, offvalue = 0)
CBRRB.place(anchor="w",x=1735+delta_s, y=50)


# Stacking Mode
RBSM1 = Radiobutton(cadre,text="MEAN", variable=choix_stacking,command=choix_mean_stacking,value=1)
RBSM1.place(anchor="w",x=1610+delta_s, y=20)
RBSM2 = Radiobutton(cadre,text="SUM", variable=choix_stacking,command=choix_sum_stacking,value=2)
RBSM2.place(anchor="w",x=1660+delta_s, y=20)
# Number frames stacked
echelle20 = Scale (cadre, from_ = 1, to = 5, command= choix_FS, orient=HORIZONTAL, length = 80, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle20.set(val_FS)
echelle20.place(anchor="w", x=1720+delta_s,y=20)


###########################
#   EXPOSITION SETTINGS   #
###########################

# Speed mode acquisition
labelMode_Acq = Label (cadre, text = "Speed Mode")
labelMode_Acq.place (anchor="w",x=1450+delta_s, y=240)
RBMA1 = Radiobutton(cadre,text="Fast", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=1510+delta_s, y=240)
RBMA2 = Radiobutton(cadre,text="Med_F", variable=mode_acq,command=mode_acq_mediumF,value=2)
RBMA2.place(anchor="w",x=1560+delta_s, y=240)
RBMA3 = Radiobutton(cadre,text="Med_S", variable=mode_acq,command=mode_acq_mediumS,value=3)
RBMA3.place(anchor="w",x=1610+delta_s, y=240)
RBMA4 = Radiobutton(cadre,text="Slow", variable=mode_acq,command=mode_acq_lente,value=4)
RBMA4.place(anchor="w",x=1660+delta_s, y=240)

# Automatic exposition time
CBOE = Checkbutton(cadre,text="AE", variable=choix_autoexposure,command=commande_autoexposure,onvalue = 1, offvalue = 0)
CBOE.place(anchor="w",x=1440+delta_s, y=yS1)

# Exposition setting
echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
echelle1.set(val_exposition)
echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)

# Choix du mode BINNING - 1, 2 ou 3
labelBIN = Label (cadre, text = "BIN : ")
labelBIN.place (anchor="w",x=1450+delta_s, y=80)
RBB1 = Radiobutton(cadre,text="BIN1", variable=choix_bin,command=choix_BIN1,value=1)
RBB1.place(anchor="w",x=1475+delta_s, y=80)
RBB2 = Radiobutton(cadre,text="BIN2", variable=choix_bin,command=choix_BIN2,value=2)
RBB2.place(anchor="w",x=1520+delta_s, y=80)

# Choix Hardware Bin
CBHB = Checkbutton(cadre,text="HB", variable=choix_hard_bin,command=commande_hard_bin,onvalue = 1, offvalue = 0)
CBHB.place(anchor="w",x=1565+delta_s, y=80)

# Choix HDR
CBHDR = Checkbutton(cadre,text="HDR", variable=choix_HDR,command=commande_HDR,onvalue = 1, offvalue = 0)
CBHDR.place(anchor="w",x=1615+delta_s, y=80)


# Resolution setting
labelParam3 = Label (cadre, text = "RES : ")
labelParam3.place(anchor="w", x=1670+delta_s,y=80)
echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle3.set (val_resolution)
echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)

# choix hold picture
CBOP = Checkbutton(cadre,text="Hold Picture", variable=choix_hold_picture,command=commande_hold_picture,onvalue = 1, offvalue = 0)
CBOP.place(anchor="w",x=1730+delta_s, y=240)
    
# Automatic gain
CBOG = Checkbutton(cadre,text="Auto Gain", variable=choix_autogain,command=commande_autogain,onvalue = 1, offvalue = 0)
CBOG.place(anchor="w",x=1440+delta_s, y=120)

echelle2 = Scale (cadre, from_ = 0, to = val_maxgain , command= valeur_gain, orient=HORIZONTAL, length = 320, width = 7, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle2.set(val_gain)
echelle2.place(anchor="w", x=1500+delta_s,y=120)

# Signal amplification soft
CBAS = Checkbutton(cadre,text="Amplif Soft", variable=choix_AmpSoft,command=commande_AmpSoft,onvalue = 1, offvalue = 0)
CBAS.place(anchor="w",x=1450+delta_s, y=160)
echelle80 = Scale (cadre, from_ = 0, to = 10.0, command= choix_amplif, orient=HORIZONTAL, length = 280, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle80.set(val_ampl)
echelle80.place(anchor="w", x=1540+delta_s,y=160)

RBMuRo1 = Radiobutton(cadre,text="Lin", variable=mode_Lin_Gauss,command=mode_Lineaire,value=1)
RBMuRo1.place(anchor="w",x=1440+delta_s, y=200)
RBMuRo2 = Radiobutton(cadre,text="Gauss", variable=mode_Lin_Gauss,command=mode_Gauss,value=2)
RBMuRo2.place(anchor="w",x=1480+delta_s, y=200)
RBMuRo3 = Radiobutton(cadre,text="Stars", variable=mode_Lin_Gauss,command=mode_Stars,value=3)
RBMuRo3.place(anchor="w",x=1525+delta_s, y=200)

labelParam82 = Label (cadre, text = "µX") # choix Mu X
labelParam82.place(anchor="w", x=1573+delta_s,y=200)
echelle82 = Scale (cadre, from_ = -5.0, to = 5.0, command= choix_Mu, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle82.set(val_Mu)
echelle82.place(anchor="w", x=1593+delta_s,y=200)

labelParam84 = Label (cadre, text = "Ro") # choix Mu X
labelParam84.place(anchor="w", x=1705+delta_s,y=200)
echelle84 = Scale (cadre, from_ = 0.2, to = 5.0, command= choix_Ro, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle84.set(val_Ro)
echelle84.place(anchor="w", x=1720+delta_s,y=200)


# Camera Red balance
labelParam14 = Label (cadre, text = "CRed") # choix balance rouge
labelParam14.place(anchor="w", x=1450+delta_s,y=310)
echelle14 = Scale (cadre, from_ = 1, to = 99, command= choix_w_red, orient=HORIZONTAL, length = 140, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle14.set(val_red)
echelle14.place(anchor="w", x=1485+delta_s,y=310)

# Camera Blue balance
labelParam15 = Label (cadre, text = "CBlue") # choix balance bleue
labelParam15.place(anchor="w", x=1645+delta_s,y=310)
echelle15 = Scale (cadre, from_ = 1, to = 99, command= choix_w_blue, orient=HORIZONTAL, length = 140, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle15.set(val_blue)
echelle15.place(anchor="w", x=1680+delta_s,y=310)

# Software Red balance
labelParam100 = Label (cadre, text = "R") # choix balance rouge
labelParam100.place(anchor="w", x=1440+delta_s,y=350)
echelle100 = Scale (cadre, from_ = 70, to = 130, command= choix_w_reds, orient=HORIZONTAL, length = 160, width = 7, resolution = 0.5, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle100.set(val_reds)
echelle100.place(anchor="w", x=1440+delta_s+15,y=350)

# Software Green balance
labelParam101 = Label (cadre, text = "G") # choix balance rouge
labelParam101.place(anchor="w", x=1640+delta_s,y=350)
echelle101 = Scale (cadre, from_ = 70, to = 130, command= choix_w_greens, orient=HORIZONTAL, length = 160, width = 7, resolution = 0.5, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle101.set(val_greens)
echelle101.place(anchor="w", x=1640+delta_s+15,y=350)

# Software Blue balance
labelParam102 = Label (cadre, text = "B") # choix balance bleue
labelParam102.place(anchor="w", x=1440+delta_s,y=380)
echelle102 = Scale (cadre, from_ = 70, to = 130, command= choix_w_blues, orient=HORIZONTAL, length = 160, width = 7, resolution = 0.5, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle102.set(val_blues)
echelle102.place(anchor="w", x=1440+delta_s+15,y=380)


###########################
# SHARPEN DENOISE WIDGETS #
###########################

# Choix Sharpen 1 & 2
CBSS1 = Checkbutton(cadre,text="Sharpen 1 Val/Sigma", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=1450+delta_s, y=420)
echelle152 = Scale (cadre, from_ = 0, to = 6, command= choix_val_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle152.set(val_sharpen)
echelle152.place(anchor="w", x=1560+delta_s,y=420)
echelle153 = Scale (cadre, from_ = 1, to = 11, command= choix_val_sigma_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 2, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle153.set(val_sigma_sharpen)
echelle153.place(anchor="w", x=1690+delta_s,y=420)

CBSS2 = Checkbutton(cadre,text="Sharpen 2 Val/Sigma", variable=choix_sharpen_soft2,command=commande_sharpen_soft2,onvalue = 1, offvalue = 0)
CBSS2.place(anchor="w",x=1450+delta_s, y=450)
echelle154 = Scale (cadre, from_ = 0, to = 6, command= choix_val_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle154.set(val_sharpen2)
echelle154.place(anchor="w", x=1560+delta_s,y=450)
echelle155 = Scale (cadre, from_ = 1, to = 11, command= choix_val_sigma_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 2, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle155.set(val_sigma_sharpen2)
echelle155.place(anchor="w", x=1690+delta_s,y=450)

# Choix filtre Denoise Paillou image
CBEPF = Checkbutton(cadre,text="NR P1", variable=choix_denoise_Paillou,command=commande_denoise_Paillou,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1450+delta_s, y=490)

# Choix filtre Denoise Paillou 2 image
CBEPF2 = Checkbutton(cadre,text="NR P2", variable=choix_denoise_Paillou2,command=commande_denoise_Paillou2,onvalue = 1, offvalue = 0)
CBEPF2.place(anchor="w",x=1500+delta_s, y=490)

# Choix filtre Denoise KNN
CBEPF = Checkbutton(cadre,text="KNN", variable=choix_denoise_KNN,command=choix_KNN,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1550+delta_s, y=490)
echelle30 = Scale (cadre, from_ = 0.05, to = 1.2, command= choix_val_KNN, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle30.set(val_denoise_KNN)
echelle30.place(anchor="w", x=1600+delta_s,y=490)

# Choix filtre Denoise NLM2
CBDS = Checkbutton(cadre,text="NLM2", variable=choix_denoise_soft,command=commande_denoise_soft,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=1690+delta_s, y=490)
echelle4 = Scale (cadre, from_ = 0.1, to = 1.2, command= choix_valeur_denoise, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle4.set(val_denoise)
echelle4.place(anchor="w", x=1740+delta_s,y=490)

# Stabilization
CBSTAB = Checkbutton(cadre,text="STAB", variable=choix_STAB,command=commande_STAB,onvalue = 1, offvalue = 0)
CBSTAB.place(anchor="w",x=1450+delta_s, y=530)

# Choix filtre Denoise Paillou 3FNR
CB3FNR = Checkbutton(cadre,text="NR 3FNR", variable=choix_3FNR,command=commande_3FNR,onvalue = 1, offvalue = 0)
CB3FNR.place(anchor="w",x=1510+delta_s, y=530)

# Choix filtre Denoise Paillou AADF
CBEPFS = Checkbutton(cadre,text="AADF", variable=choix_denoise_stacking_Paillou,command=commande_denoise_stacking_Paillou,onvalue = 1, offvalue = 0)
CBEPFS.place(anchor="w",x=1575+delta_s, y=530)

# AADF Mode
RBAADP1 = Radiobutton(cadre,text="H", variable=choix_dyn_AADP,command=choix_dyn_high,value=1)
RBAADP1.place(anchor="w",x=1620+delta_s, y=530)
RBAADP2 = Radiobutton(cadre,text="L", variable=choix_dyn_AADP,command=choix_dyn_low,value=2)
RBAADP2.place(anchor="w",x=1650+delta_s, y=530)

# AADF ghost reducer
CBGR = Checkbutton(cadre,text="GR", variable=choix_ghost_reducer,command=commande_ghost_reducer,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1686+delta_s, y=530)
echelle130 = Scale (cadre, from_ = 20, to = 70, command= choix_val_ghost_reducer, orient=HORIZONTAL, length = 100, width = 7, resolution = 2, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle130.set(val_ghost_reducer)
echelle130.place(anchor="w", x=1720+delta_s,y=523)


# Choix Turbulence Reduction Filter
CBRV = Checkbutton(cadre,text="TRF", variable=choix_reduce_variation,command=commande_reduce_variation,onvalue = 1, offvalue = 0)
CBRV.place(anchor="w",x=1450+delta_s, y=565)

echelle270 = Scale (cadre, from_ = 0.5, to = 10, command= choix_val_reduce_variation, orient=HORIZONTAL, length = 250, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=10,sliderlength=20)
echelle270.set(val_reduce_variation)
echelle270.place(anchor="w", x=1490+delta_s,y=565)



#####################
# HISTOGRAM WIDGETS #
#####################

# Choix filtre Gradient Removal
CBGR = Checkbutton(cadre,text="Grad/Vignet", variable=choix_GR,command=commande_GR,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1450+delta_s, y=605)

# Choix du mode gradient ou vignetting
RBGV1 = Radiobutton(cadre,text="Gr", variable=gradient_vignetting,command=mode_gradient,value=1)
RBGV1.place(anchor="w",x=1530+delta_s, y=605)
RBGV2 = Radiobutton(cadre,text="Vig", variable=gradient_vignetting,command=mode_vignetting,value=2)
RBGV2.place(anchor="w",x=1560+delta_s, y=605)

# Choix Parametre Seuil Gradient Removal
echelle60 = Scale (cadre, from_ = 0, to = 100, command= choix_SGR, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle60.set(val_SGR)
echelle60.place(anchor="w", x=1600+delta_s,y=605)

# Choix Parametre Atenuation Gradient Removal
labelParam61 = Label (cadre, text = "At")
labelParam61.place(anchor="e", x=1735+delta_s,y=605)
echelle61 = Scale (cadre, from_ = 0, to = 100, command= choix_AGR, orient=HORIZONTAL, length = 80, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle61.set(val_AGR)
echelle61.place(anchor="w", x=1740+delta_s,y=605)

# Choix du mode image en négatif
CBIN = Checkbutton(cadre,text="Img Neg", variable=choix_img_Neg,command=commande_img_Neg,onvalue = 1, offvalue = 0)
CBIN.place(anchor="w",x=1450+delta_s, y=645)

# Histogram equalize
CBHE2 = Checkbutton(cadre,text="Gamma", variable=choix_histogram_equalize2,command=commande_histogram_equalize2,onvalue = 1, offvalue = 0)
CBHE2.place(anchor="w",x=1520+delta_s, y=645)
echelle16 = Scale (cadre, from_ = 0.1, to = 4, command= choix_heq2, orient=HORIZONTAL, length = 240, width = 7, resolution = 0.05, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle16.set(val_heq2)
echelle16.place(anchor="w", x=1580+delta_s,y=645)

# Choix histogramme stretch
CBHS = Checkbutton(cadre,text="Histo Stretch", variable=choix_histogram_stretch,command=commande_histogram_stretch,onvalue = 1, offvalue = 0)
CBHS.place(anchor="w",x=1450+delta_s, y=685)

labelParam5 = Label (cadre, text = "Min") # choix valeur histogramme strech minimum
labelParam5.place(anchor="w", x=1555+delta_s,y=685)
echelle5 = Scale (cadre, from_ = 0, to = 150, command= choix_histo_min, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle5.set(val_histo_min)
echelle5.place(anchor="w", x=1580+delta_s,y=685)

labelParam6 = Label (cadre, text = "Max") # choix valeur histogramme strech maximum
labelParam6.place(anchor="w", x=1700+delta_s,y=685)
echelle6 = Scale (cadre, from_ = 155, to = 255, command= choix_histo_max, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle6.set(val_histo_max)
echelle6.place(anchor="w", x=1720+delta_s,y=685)

#Choix histogramme Sigmoide
CBHPT = Checkbutton(cadre,text="Histo Sigmoide", variable=choix_histogram_phitheta,command=commande_histogram_phitheta,onvalue = 1, offvalue = 0)
CBHPT.place(anchor="w",x=1450+delta_s, y=725)

labelParam12 = Label (cadre, text = "Pnt") # choix valeur histogramme Signoide param 1
labelParam12.place(anchor="w", x=1555+delta_s,y=725)
echelle12 = Scale (cadre, from_ = 0.5, to = 3, command= choix_phi, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle12.set(val_phi)
echelle12.place(anchor="w", x=1580+delta_s,y=725)

labelParam13 = Label (cadre, text = "Dec") # choix valeur histogramme Signoide param 2
labelParam13.place(anchor="w", x=1700+delta_s,y=725)
echelle13 = Scale (cadre, from_ = 50, to = 200, command= choix_theta, orient=HORIZONTAL, length = 100, width = 7, resolution = 2, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle13.set(val_theta)
echelle13.place(anchor="w", x=1720+delta_s,y=725)

# Choix contrast CLAHE
CBCC = Checkbutton(cadre,text="Contrast CLAHE", variable=choix_contrast_CLAHE,command=commande_contrast_CLAHE,onvalue = 1, offvalue = 0)
CBCC.place(anchor="w",x=1450+delta_s, y=765)

labelParam9 = Label (cadre, text = "Clip") # choix valeur contrate CLAHE
labelParam9.place(anchor="w", x=1555+delta_s,y=765)
echelle9 = Scale (cadre, from_ = 0.1, to = 4, command= choix_valeur_CLAHE, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle9.set(val_contrast_CLAHE)
echelle9.place(anchor="w", x=1580+delta_s,y=765)
labelParam109 = Label (cadre, text = "Grid") # choix valeur contrate CLAHE
labelParam109.place(anchor="w", x=1700+delta_s,y=765)
echelle109 = Scale (cadre, from_ = 4, to = 24, command= choix_grid_CLAHE, orient=HORIZONTAL, length = 100, width = 7, resolution = 4, label="",showvalue=1,tickinterval=8,sliderlength=20)
echelle109.set(val_grid_CLAHE)
echelle109.place(anchor="w", x=1720+delta_s,y=765)

# Choix Contrast Low Light
CBCLL = Checkbutton(cadre,text="CLL", variable=choix_CLL,command=commande_CLL,onvalue = 1, offvalue = 0)
CBCLL.place(anchor="w",x=1450+delta_s, y=805)

labelParam200 = Label (cadre, text = "µ") # choix Mu CLL
labelParam200.place(anchor="w", x=1500+delta_s,y=805)
echelle200 = Scale (cadre, from_ = 0, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle200.set(val_MuCLL)
echelle200.place(anchor="w", x=1510+delta_s,y=805)

labelParam201 = Label (cadre, text = "Ro") # choix Ro CLL
labelParam201.place(anchor="w", x=1605+delta_s,y=805)
echelle201 = Scale (cadre, from_ = 0.5, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle201.set(val_RoCLL)
echelle201.place(anchor="w", x=1625+delta_s,y=805)

labelParam202 = Label (cadre, text = "amp") # choix Amplification CLL
labelParam202.place(anchor="w", x=1715+delta_s,y=805)
echelle202 = Scale (cadre, from_ = 0.5, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle202.set(val_AmpCLL)
echelle202.place(anchor="w", x=1740+delta_s,y=805)


####################
# CAPTURES WIDGETS #
####################

# Choix HQ Capture
CBHQC = Checkbutton(cadre,text="RAW Capture", variable=choix_HQ_capt,command=commande_HQ_capt,onvalue = 1, offvalue = 0)
CBHQC.place(anchor="w",x=1450+delta_s, y=1010)

# Number of pictures to capture
echelle8 = Scale (cadre, from_ = 1, to = 501, command= choix_nb_captures, orient=HORIZONTAL, length = 250, width = 7, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle8.set(val_nb_captures)
echelle8.place(anchor="w", x=1570+delta_s,y=883)

# Frames number Video
echelle11 = Scale (cadre, from_ = 0, to = 4000, command= choix_nb_video, orient=HORIZONTAL, length = 250, width = 7, resolution = 20, label="",showvalue=1,tickinterval=500,sliderlength=20)
echelle11.set(val_nb_capt_video)
echelle11.place(anchor="w", x=1570+delta_s,y=930)

labelParam65 = Label (cadre, text = "Delta T") # choix valeur contrate CLAHE
labelParam65.place(anchor="w", x=1535+delta_s,y=975)
echelle65 = Scale (cadre, from_ = 0, to = 60, command= choix_deltat, orient=HORIZONTAL, length = 250, width = 7, resolution = 1, label="",showvalue=1,tickinterval=10,sliderlength=20)
echelle65.set(val_deltat)
echelle65.place(anchor="w",x=1570+delta_s,y=975)

labelInfo1 = Label (cadre, text = text_info1) # label info n°1
labelInfo1.place(anchor="w", x=1550+delta_s,y=1010)

labelParam100 = Label (cadre, text = "Treatment time : ")
labelParam100.place(anchor="w", x=1450+delta_s,y=20)
labelInfo2 = Label (cadre, text = "") # label info n°1
labelInfo2.place(anchor="w", x=1520+delta_s,y=20)

labelInfoDark = Label (cadre, text = dispo_dark) # label info Dark
labelInfoDark.place(anchor="w", x=1750+delta_s,y=843)

# Choix appliquer dark
CBAD = Checkbutton(cadre,text="Sub Dark", variable=choix_sub_dark,command=commande_sub_dark,onvalue = 1, offvalue = 0)
CBAD.place(anchor="w",x=1685+delta_s, y=843)

echelle10 = Scale (cadre, from_ = 5, to = 30, command= choix_nb_darks, orient=HORIZONTAL, length = 100, width = 7, resolution =1, label="",showvalue=1,tickinterval=5,sliderlength=20)
echelle10.set(val_nb_darks)
echelle10.place(anchor="w", x=1575+delta_s,y=843)


####################
#      BUTTONS     #
####################

Button1 = Button (cadre,text = "Start CAP", command = start_pic_capture,padx=10,pady=0)
Button1.place(anchor="w", x=1460+delta_s,y=875)

Button2 = Button (cadre,text = "Stop CAP", command = stop_pic_capture,padx=10,pady=0)
Button2.place(anchor="w", x=1460+delta_s,y=900)

Button3 = Button (cadre,text = "Start REC", command = start_video_capture,padx=10,pady=0)
Button3.place(anchor="w", x=1460+delta_s,y=935)

Button4 = Button (cadre,text = "Stop REC", command = stop_video_capture,padx=10,pady=0)
Button4.place(anchor="w", x=1460+delta_s,y=960)

Button5 = Button (cadre,text = "Pause REC", command = pause_video_capture,padx=10,pady=0)
Button5.place(anchor="w", x=1460+delta_s,y=985)

Button6 = Button (cadre,text = "Cap Dark", command = start_cap_dark,padx=10,pady=0)
Button6.place(anchor="w", x=1460+delta_s,y=840)

# RAZ frame counter
Button10 = Button (cadre,text = "RZ Fr Cnt", command = raz_framecount,padx=10,pady=0)
Button10.place(anchor="w", x=5,y=370)

if flag_camera_ok == False :
    Button12 = Button (cadre,text = "Load V/I", command = load_video,padx=10,pady=0)
    Button12.place(anchor="w", x=5,y=840)

Button15 = Button (cadre,text = "Dir Vid", command = choose_dir_vid,padx=10,pady=0)
Button15.place(anchor="w", x=5,y=870)

Button16 = Button (cadre,text = "Dir Pic", command = choose_dir_pic,padx=10,pady=0)
Button16.place(anchor="w", x=5,y=900)


Button (fenetre_principale, text = "Quit", command = quitter,padx=10,pady=5).place(x=1700+delta_s,y=1030, anchor="w")

if flag_camera_ok == True :
    choix_BIN1()
    mode_acq_mediumF()
    if my_os == "win32" :
        init_efw()

cadre_image = Canvas (cadre, width = int(1350 * fact_s), height = int(1012 * fact_s), bg = "dark grey")
cadre_image.place(anchor="w", x=70,y=int(1012 * fact_s)/2+5)

if flag_camera_ok == False :
    filetypes = (
            ('Movie files', '*.mp4 *.avi *.ser *.mov'),
            ('Image files', '*.tif *.tiff *.jpg *.jpeg'),
            ('ALL files', '*.*')
        )

    Video_Test = fd.askopenfilename(title='Open a file',initialdir=video_path,filetypes=filetypes)
    if Video_Test.lower().endswith(('.mp4', '.avi', '.ser', '.mov')) :
        flag_image_mode = False
    if Video_Test.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) :
        flag_image_mode = True
    flag_image_video_loaded = True
    

fenetre_principale.after(500, refresh)
fenetre_principale.mainloop()
fenetre_principale.destroy()
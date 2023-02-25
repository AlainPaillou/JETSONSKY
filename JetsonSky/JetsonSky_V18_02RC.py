
############################################################################################################
#                                                                                                          #
#                      Images/videos acquisition & treatment software for ZWO cameras                      #
#                                                                                                          #
#                                                                                                          #
#                                   Copyright Alain Paillou 2018-2022                                      #
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
# - ASI678MC
# - ASI1600MC, ASI1600MM

############################################################################################################


# Libraries import
import os
import time
import numpy as np

try :
    import cupy as cp
    flag_cupy = True
except :
    flag_cupy = False

if flag_cupy == True :
    from cupyx.scipy import ndimage

import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw
from PIL import ImageEnhance
from tkinter import *
from tkinter.messagebox import askyesno
from tkinter.font import nametofont
import cv2
from threading import Thread
from datetime import datetime
import argparse
import sys
import math
import zwoasi as asi
try :
    import zwoefw as efw
except :
    pass
import pycuda.driver as drv
import pycuda.tools as tools
import pycuda.autoprimaryctx
from pycuda.compiler import SourceModule
import pycuda.cumath

try :
    import vpi
    flag_vpi = True
except :
    flag_vpi = False
    
if flag_vpi == True :
    print("VPI library OK")

if flag_cupy == True :
    print("Cupy libray OK")

############################################
#                Main program              #
############################################

JetsonSky_version = "V18.02RC"

# Choose the size of the fonts in the Main Window - It depends of your system - can be set from 5 to 7
MainWindowFontSize = 6

my_os = sys.platform
if my_os == "win32" :
    Dev_system = "PC Windows"
    print("Windows system")
    titre = "JetsonSky " +  JetsonSky_version + " Windows release - CUDA / OPENCV / CUPY / VPI - Copyright Alain PAILLOU 2018-2022"

if my_os == "linux" :
    Dev_system = "Jetson Linux"
    print("Linux system")
    titre = "JetsonSky " +  JetsonSky_version + " Linux release - CUDA / OPENCV / CUPY / VPI - Copyright Alain PAILLOU 2018-2022" 

if Dev_system == "Jetson Linux" :
    # Choose your directories for images and videos
    image_path = '/home/JetsonSky/Images'
    video_path= '/home/JetsonSky/Videos'

    # Path to librairies ZWO Jetson sbc
    sys.path.insert(0, "/home/JetsonSky/Lib")
    env_filename_camera = '/home/JetsonSky/Lib/libASICamera2.so'
    env_filename_efw = '/home/JetsonSky/Lib/libEFWFilter.so'

    USB178 = 90
    USB485 = 90
    
else :
    # Choose your directories for images and videos
    image_path = 'C:\Soft\JetsonSky\Images'
    video_path= 'C:\Soft\JetsonSky\Videos'

    # Path to librairies ZWO Windows
    sys.path.insert(0, "C:\Soft\JetsonSky\Lib")
    env_filename_camera = "C:\Soft\JetsonSky\Lib\ASICamera2.dll"
    env_filename_efw = "C:\Soft\JetsonSky\Lib\EFW_filter.dll"

    USB178 = 90
    USB485 = 90
    

# Setting the Images & videos quality
flag_HQ = 0 # si 1 : sauvegarde HQ non compressée    si 0 : sauvegarde LQ compressée

# Initialisation des constantes d'exposition mode rapide
exp_min=100 #µs
exp_max=10000 #µs
exp_delta=100 #µs
exp_interval=2000

#  Initialisation des paramètres fixés par les boites scalebar
format_capture = asi.ASI_IMG_RAW8
val_resolution = 1
mode_BIN=1
res_x_max = 3096
res_y_max = 2080
res_cam_x = 3096
res_cam_y = 2080
cam_displ_x = 1350
cam_displ_y = 1015

RES_X_BIN1 = [3096,2560,1920,1600,1280,1024,800,640,320]
RES_Y_BIN1 = [2080,1920,1440,1200,960,768,600,480,240]
DELAY_BIN1 = [0.02,0.02,0.02,0.01,0.01,0.01,0.01,0.0005,0.005]

RES_X_BIN2 = [1544,1280,960,800,640,512,400]
RES_Y_BIN2 = [1040,960,720,600,480,384,300]
DELAY_BIN2 = [0.02,0.02,0.01,0.01,0.005,0.005,0.005]

val_exposition = 1000 #  temps exposition en µs
timeoutexp = 1 + 500
exposition = 0
val_gain = 100
val_maxgain = 600
val_denoise = 0.4
val_histo_min = 0
val_histo_max = 255
val_contrast_CLAHE = 1.5
val_phi = 1.0
val_theta = 100
val_heq2 = 1.0
text_info1 = "Test information"
val_nb_captures = 1
nb_cap_video =0
val_nb_capt_video = 100
compteur_images = 0
numero_image = 0
val_nb_darks = 5
dispo_dark = 'Dark NO       '
FlipV = 0
FlipH = 0
ImageNeg = 0
val_red = 63
val_blue = 74
val_FS = 1
compteur_FS = 0
compteur_FSDN = 0
val_denoise_KNN = 0.2
val_USB = 90
val_SGR = 95
val_AGR = 50
val_NGB = 13
val_SAT = 1.0
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
FDIF = 0
xdark=1750
ydark=830
xLI1=1475
yLI1=1000
Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')
hauteur = 0.0
azimut = 0.0
val_ghost_reducer = 50
compute_library = "opencv"

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

stars_x = np.zeros(10000,dtype=int)
stars_y = np.zeros(10000,dtype=int)
stars_s = np.zeros(10000,dtype=int)

sat_x = np.zeros(10000,dtype=int)
sat_y = np.zeros(10000,dtype=int)
sat_s = np.zeros(10000,dtype=int)
sat_old_x = np.zeros(10000,dtype=int)
sat_old_y = np.zeros(10000,dtype=int)
sat_id  = np.zeros(10000,dtype=int)
sat_old_id  = np.zeros(10000,dtype=int)

correspondance = np.zeros(10000,dtype=int)
sat_speed = np.zeros(10000,dtype=int)
compteur_sat = 0
old_sat = 0
flag_first_sat_pass = True
nb_trace_sat = -1
    

def quitter() :
    global camera,flag_autorise_acquisition,thread1,fenetre_principale,flag_image_disponible,flag_quitter
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


# définition fenetre principale
fenetre_principale = Tk ()
#w, h = fenetre_principale.winfo_screenwidth(), fenetre_principale.winfo_screenheight()-20
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
choix_autoexposure.set(0) # Initialisation openGL

choix_autogain = IntVar ()
choix_autogain.set(0) # Initialisation openGL

choix_stacking = IntVar ()
choix_stacking.set(1) # Initialisation mode stacking Mean

type_library = IntVar ()
type_library.set(1) # Initialisation library OpenCV

choix_type_clahe = IntVar ()
choix_type_clahe.set(1) # Initialisation mode CLAHE Daylight

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

choix_bilateral = IntVar()
choix_bilateral.set(0) # Initialisation filtre Median inactif

choix_mode_full_res = IntVar()
choix_mode_full_res.set(0) # Initialisation mode full resolution inactif

choix_sharpen_soft1 = IntVar()
choix_sharpen_soft1.set(0) # initialisation mode sharpen software 1 inactif

choix_unsharp_mask = IntVar()
choix_unsharp_mask.set(0) # initialisation mode unsharp mask inactif

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

choix_noir_blanc = IntVar()
choix_noir_blanc.set(0) # initialisation mode noir et blanc inactif

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

choix_denoise_stacking_Paillou = IntVar()
choix_denoise_stacking_Paillou.set(0) # Initialisation Filtrage Denoise Paillou stacking

choix_sub_dark = IntVar()
choix_sub_dark.set(0) # Initialisation sub dark inactif

choix_GR = IntVar()
choix_GR.set(0) # Initialisation Filtre Gradient Removal

presence_FW = IntVar()
presence_FW.set(0) # Initialisation absence FW

fw_position = IntVar()
fw_position.set(0) # Position filter wheel

choix_AmpSoft = IntVar()
choix_AmpSoft.set(0) # Initialisation amplification software OFF

choix_HST = IntVar()
choix_HST.set(0) # Initialisation histogram OFF

choix_HDR = IntVar()
choix_HDR.set(0) # Initialisation histogram OFF


choix_TRSF = IntVar()
choix_TRSF.set(0) # Initialisation histogram OFF

choix_TRGS = IntVar()
choix_TRGS.set(0) # Initialisation histogram transformation gain soft

choix_DEMO = IntVar()
choix_DEMO.set(0) # Initialisation Demo OFF

choix_FDIF = IntVar()
choix_FDIF.set(0) # Initialisation Demo OFF

choix_dyn_AADP = IntVar ()
choix_dyn_AADP.set(1) # Initialisation mode dynamique High AADP

choix_ghost_reducer = IntVar ()
choix_ghost_reducer.set(0) # Initialisation ghost reducer

choix_TRKSAT = IntVar()
choix_TRKSAT.set(0) # Initialisation Remove background OFF

choix_TRKMET = IntVar()
choix_TRKMET.set(0) # Initialisation Remove background OFF

choix_BGSUB = IntVar()
choix_BGSUB.set(0) # Initialisation Background subtraction OFF

# Initialisation des filtres soft
flag_2DConv = 0
flag_bilateral = 0
flag_full_res = 0
flag_sharpen_soft1 = 0
flag_unsharp_mask = 0
flag_denoise_soft = 0
flag_histogram_equalize2 = 0
flag_histogram_stretch = 0
flag_histogram_phitheta = 0
flag_contrast_CLAHE = 0
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
flag_acq_rapide = True
flag_colour_camera = True
flag_filter_wheel = False
flag_seuillage_PB = False
Im1OK = False
Im2OK = False
Im3OK = False
filter_on = False
flag_filtrage_ON = True
flag_filtre_work = False
flag_denoise_KNN = False
flag_denoise_Paillou = False
flag_denoise_stacking_Paillou = False
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
flag_FDIF = 0
flag_dyn_AADP = 1
flag_front = True
flag_TRGS = 0
flag_lin_gauss = 1
flag_BCap = 0
flag_TRKSAT = 0
flag_TRKMET = 0
flag_NB_estime = 0
flag_nouvelle_resolution = False
flag_sat_exist = False
flag_mountpos = True
flag_mount_connect = False
flag_mountpos = False
flag_type_clahe = 1 # Daylight
flag_supported_camera = False
flag_ghost_reducer = 0
flag_BG_Ok = False
flag_sub_BG = False
flag_HDR = False
flag_new_image = False

mod = SourceModule("""
__global__ void grey_estimate_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
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
//    img_r[index1] = dest_r[index1];
  }
}
""")

grey_estimate_Mono = mod.get_function("grey_estimate_Mono")


mod = SourceModule("""
__global__ void Mono_ampsoft_GPU(unsigned char *dest_r, unsigned char *img_r,
unsigned char *grey_gpu, unsigned char *grey_blur_gpu,long int width, long int height, float val_Mu, float val_Ro, float val_ampl, int flag_lin_gauss, float *Corr_GS)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int delta;
  unsigned char index_grey;
  float mul_factor;
  int vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (flag_lin_gauss == 3) {
          delta =(int)(min(max(int(grey_gpu[index] - grey_blur_gpu[index]), 0), 255));
          index_grey = grey_gpu[index];
          mul_factor = Corr_GS[index_grey];
          dest_r[index] = (int)(min(max(int(img_r[index] + delta*mul_factor*val_ampl), 0), 255));
          }
      else {
          vr = img_r[index];  
          dest_r[index] = (int)(min(max(int(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr])), 0), 255));
          }
  }
}
""")
Mono_ampsoft_GPU = mod.get_function("Mono_ampsoft_GPU")

mod = SourceModule("""
__global__ void Colour_ampsoft_GPU(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
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
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
}
""")

Colour_ampsoft_GPU = mod.get_function("Colour_ampsoft_GPU")


mod = SourceModule("""
__global__ void Colour_clahe_GPU(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, unsigned char *img_grey,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int vgrey;
  float factor;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vgrey = (int)(int(img_r[index] * 0.299 + img_g[index] * 0.587 + img_b[index] * 0.114));
      factor = img_grey[index]/vgrey;
      dest_r[index] = (int)(min(max(int(img_r[index]*factor), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index]*factor), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index]*factor), 0), 255));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
}
""")

Colour_clahe_GPU = mod.get_function("Colour_clahe_GPU")


mod = SourceModule("""
__global__ void Saturation_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
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
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
}
""")

Saturation_Color = mod.get_function("Saturation_Color")


mod = SourceModule("""
__global__ void Saturation_Combine(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,  unsigned char *img_grey,long int width, long int height)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float X;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      X = img_grey[index] / (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
      dest_r[index] = (int)(min(max(int(img_r[index]*X), 0), 255));
      dest_g[index] = (int)(min(max(int(img_g[index]*X), 0), 255));
      dest_b[index] = (int)(min(max(int(img_b[index]*X), 0), 255));
      img_r[index] = dest_r[index];
      img_g[index] = dest_g[index];
      img_b[index] = dest_b[index];
    } 
}
""")

Saturation_Combine = mod.get_function("Saturation_Combine")



mod = SourceModule("""
__global__ void Colour_staramp_GPU(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b,
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
""")
Colour_staramp_GPU = mod.get_function("Colour_staramp_GPU")


mod = SourceModule("""
__global__ void Smooth_Mono_high(unsigned char *dest_r, unsigned char *img_r, long int width, long int height)
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
    img_r[index] = dest_r[index];
  }
}
""")

Smooth_Mono_high = mod.get_function("Smooth_Mono_high")


mod = SourceModule("""
__global__ void Stacking_mean_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r1, unsigned char *img_g1, unsigned char *img_b1, unsigned char *img_r2, unsigned char *img_g2, unsigned char *img_b2,
unsigned char *img_r3, unsigned char *img_g3, unsigned char *img_b3, unsigned char *img_r4, unsigned char *img_g4, unsigned char *img_b4,
unsigned char *img_r5, unsigned char *img_g5, unsigned char *img_b5,long int width, long int height, float divise,int nb_images)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (divise == 1) {
          if (nb_images == 2.0 ) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index])),0), 255));
          dest_g[index] = (int)(min(max(int((img_g1[index] + img_g2[index])),0), 255));
          dest_b[index] = (int)(min(max(int((img_b1[index] + img_b2[index])),0), 255));
          }
          if (nb_images == 3.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index])),0), 255));
          dest_g[index] = (int)(min(max(int((img_g1[index] + img_g2[index] + img_g3[index])),0), 255));
          dest_b[index] = (int)(min(max(int((img_b1[index] + img_b2[index] + img_b3[index])),0), 255));
          }
          if (nb_images == 4.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index])),0), 255));
          dest_g[index] = (int)(min(max(int((img_g1[index] + img_g2[index] + img_g3[index]+ img_g4[index])),0), 255));
          dest_b[index] = (int)(min(max(int((img_b1[index] + img_b2[index] + img_b3[index]+ img_b4[index])),0), 255));
          }    
          if (nb_images == 5.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]+ img_r5[index])),0), 255));
          dest_g[index] = (int)(min(max(int((img_g1[index] + img_g2[index] + img_g3[index]+ img_g4[index]+ img_g5[index])),0), 255));
          dest_b[index] = (int)(min(max(int((img_b1[index] + img_b2[index] + img_b3[index]+ img_b4[index]+ img_b5[index])),0), 255));
          }
        }
      if (divise > 1) {
          if (nb_images == 2.0 ) { 
          dest_r[index] = (int)((img_r1[index] + img_r2[index]) / 2.0);
          dest_g[index] = (int)((img_g1[index] + img_g2[index]) / 2.0);
          dest_b[index] = (int)((img_b1[index] + img_b2[index]) / 2.0);
          }
          if (nb_images == 3.0) {
          dest_r[index] = (int)((img_r1[index] + img_r2[index] + img_r3[index]) / 3.0);
          dest_g[index] = (int)((img_g1[index] + img_g2[index] + img_g3[index]) / 3.0);
          dest_b[index] = (int)((img_b1[index] + img_b2[index] + img_b3[index]) / 3.0);
          }
          if (nb_images == 4.0) {
          dest_r[index] = (int)((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]) / 4.0);
          dest_g[index] = (int)((img_g1[index] + img_g2[index] + img_g3[index]+ img_g4[index]) / 4.0);
          dest_b[index] = (int)((img_b1[index] + img_b2[index] + img_b3[index]+ img_b4[index]) / 4.0);
          }    
          if (nb_images == 5.0) {
          dest_r[index] = (int)((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]+ img_r5[index]) / 5.0);
          dest_g[index] = (int)((img_g1[index] + img_g2[index] + img_g3[index]+ img_g4[index]+ img_g5[index]) / 5.0);
          dest_b[index] = (int)((img_b1[index] + img_b2[index] + img_b3[index]+ img_b4[index]+ img_b5[index]) / 5.0);
          }
        }
    } 
}
""")

Stacking_mean_Color = mod.get_function("Stacking_mean_Color")

mod = SourceModule("""
__global__ void Stacking_mean_Mono(unsigned char *dest_r, unsigned char *img_r1, unsigned char *img_r2,
unsigned char *img_r3, unsigned char *img_r4, unsigned char *img_r5, long int width, long int height, float divise,int nb_images)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      if (divise == 1) {
          if (nb_images == 2.0 ) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index])),0), 255));
          }
          if (nb_images == 3.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index])),0), 255));
          }
          if (nb_images == 4.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index])),0), 255));
          }    
          if (nb_images == 5.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]+ img_r5[index])),0), 255));
          }
        }
      if (divise > 1) {
          if (nb_images == 2.0 ) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index]) / 2.0),0), 255));
          }
          if (nb_images == 3.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]) / 3.0),0), 255));
          }
          if (nb_images == 4.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]) / 4.0),0), 255));
          }    
          if (nb_images == 5.0) {
          dest_r[index] = (int)(min(max(int((img_r1[index] + img_r2[index] + img_r3[index]+ img_r4[index]+ img_r5[index]) / 5.0),0), 255));
          }
        }
    } 
}
""")

Stacking_mean_Mono = mod.get_function("Stacking_mean_Mono")


mod = SourceModule("""
__global__ void adaptative_absorber_denoise_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
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
      if (img_r[index] > 210) {
          img_r[index] = (int)((img_r[index-1] + img_r[index+1])/2.0);
          }
      if (img_g[index] > 210) {
          img_g[index] = (int)((img_g[index-1] + img_g[index+1])/2.0);
          }
      if (img_b[index] > 210) {
          img_b[index] = (int)((img_b[index-1] + img_b[index+1])/2.0);
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
""")
adaptative_absorber_denoise_Color = mod.get_function("adaptative_absorber_denoise_Color")



mod = SourceModule("""
__global__ void adaptative_absorber_denoise_Mono(unsigned char *dest_r, unsigned char *img_r, unsigned char *old_r, long int width, long int height, int flag_dyn_AADP,
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
""")
adaptative_absorber_denoise_Mono = mod.get_function("adaptative_absorber_denoise_Mono")




mod = SourceModule("""
__global__ void Denoise_Paillou_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
            img_r[index1] = dest_r[index1];
            img_g[index1] = dest_g[index1];
            img_b[index1] = dest_b[index1];
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
            dest_g[index1] = int((img_g[ix - 1 + iy * imageW] + img_g[ix + 1 + iy * imageW] + img_g[ix + (iy-1) * imageW] + img_g[ix + (iy+1) * imageW])/4.0);
            dest_b[index1] = int((img_b[ix - 1 + iy * imageW] + img_b[ix + 1 + iy * imageW] + img_b[ix + (iy-1) * imageW] + img_b[ix + (iy+1) * imageW])/4.0);
            img_r[index1] = dest_r[index1];
            img_g[index1] = dest_g[index1];
            img_b[index1] = dest_b[index1];
        }
    }
}
""")

Denoise_Paillou_Colour = mod.get_function("Denoise_Paillou_Colour")

mod = SourceModule("""
__global__ void Denoise_Paillou_Mono(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, int cell_size, int sqr_cell_size)
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
            img_r[index1] = dest_r[index1];
            }
        else {
            dest_r[index1] = int((img_r[ix - 1 + iy * imageW] + img_r[ix + 1 + iy * imageW] + img_r[ix + (iy-1) * imageW] + img_r[ix + (iy+1) * imageW])/4.0);
            img_r[index1] = dest_r[index1];
        }
    }
}
""")

Denoise_Paillou_Mono = mod.get_function("Denoise_Paillou_Mono")



mod = SourceModule("""
__global__ void Histo_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
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
""")

Histo_Mono = mod.get_function("Histo_Mono")


mod = SourceModule("""
__global__ void Sharp_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
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
    img_r[index] = dest_r[index];
    }
  }
}
""")

Sharp_Mono = mod.get_function("Sharp_Mono")


mod = SourceModule("""
__global__ void Smooth_Mono(unsigned char *dest_r, unsigned char *img_r, long int width, long int height,
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
    img_r[index] = dest_r[index];
    }
  }
}
""")

Smooth_Mono = mod.get_function("Smooth_Mono")



mod = SourceModule("""
__global__ void Histo_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
""")


Histo_Color = mod.get_function("Histo_Color")

mod = SourceModule("""
__global__ void Set_RGB(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
          img_r[index] = dest_r[index];
          }
      if (mod_green != 1.0) {        
          dest_g[index] = (int)(min(max(int(img_g[index] * mod_green), 0), 255));
          img_g[index] = dest_g[index];
          }
      if (mod_red != 1.0) {  
          dest_b[index] = (int)(min(max(int(img_b[index] * mod_red), 0), 255));
          img_b[index] = dest_b[index];
          }
    } 
}
""")


Set_RGB = mod.get_function("Set_RGB")

mod = SourceModule("""
__global__ void Smooth_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
    img_r[index] = dest_r[index];
    img_g[index] = dest_g[index];
    img_b[index] = dest_b[index];
    }

  }
}
""")

Smooth_Color = mod.get_function("Smooth_Color")


mod = SourceModule("""
__global__ void Sharp_Color(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
""")

Sharp_Color = mod.get_function("Sharp_Color")



mod = SourceModule("""
__global__ void NLM2_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
        img_r[index5] = dest_r[index5];
        img_g[index5] = dest_g[index5];
        img_b[index5] = dest_b[index5];
    }
}
""")

NLM2_Colour_GPU = mod.get_function("NLM2_Colour")


mod = SourceModule("""
__global__ void NLM2_Mono(unsigned char *dest_r, unsigned char *img_r,
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
        img_r[index5] = dest_r[index5];
    }
}
""")

NLM2_Mono_GPU = mod.get_function("NLM2_Mono")



mod = SourceModule("""
__global__ void KNN_Colour(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g,
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
        img_r[index5] = dest_r[index5];
        img_g[index5] = dest_g[index5];
        img_b[index5] = dest_b[index5];
    }
}
""")

KNN_Colour_GPU = mod.get_function("KNN_Colour")


mod = SourceModule("""
__global__ void KNN_Mono(unsigned char *dest_r, unsigned char *img_r, int imageW, int imageH, float Noise, float lerpC)
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
        img_r[index5] = dest_r[index5];
    }
}
""")

KNN_Mono_GPU = mod.get_function("KNN_Mono")

cv2.setUseOptimized(True)

def init_camera() :
    global camera,flag_colour_camera,format_capture,controls,res_x_max,res_y_max,res_cam_x,res_cam_y, \
           cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,RES_X_BIN3,RES_Y_BIN3, \
           DELAY_BIN1,DELAY_BIN2,DELAY_BIN3,val_delai,val_blue,val_red,val_USB,val_maxgain, \
           flag_supported_camera

    if env_filename_camera:
        asi.init(env_filename_camera)
        time.sleep(1)
    else:
        print('The filename of the SDK library for camera is required')
        sys.exit(1)

    num_cameras = asi.get_num_cameras()
    if num_cameras == 0:
        print('No cameras found')
        sys.exit(0)

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
        res_x_max = 4656
        res_y_max = 3520
        res_cam_x = 4656
        res_cam_y = 3520
        cam_displ_x = 1350
        cam_displ_y = 1015
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [4656,3600,3000,2400,2000,1600,1280,1024,800]
        RES_Y_BIN1 = [3520,2700,2250,1800,1500,1200,960,770,600]
        DELAY_BIN1 = [0.05,0.05,0.04,0.03,0.02,0.02,0.01,0.01,0.005]

        RES_X_BIN2 = [2328,1800,1504,1200,1000,640,400]
        RES_Y_BIN2 = [1760,1350,1130,900,750,480,300]
        DELAY_BIN2 = [0.03,0.02,0.02,0.01,0.01,0.01,0.01]

        val_delai = DELAY_BIN1[0]
    
    if cameras_found[0] == "ZWO ASI178MC" or cameras_found[0] == "ZWO ASI178MM" or cameras_found[0] == "ZWO ASI178MM Pro" :
        res_x_max = 3096
        res_y_max = 2080
        res_cam_x = 3096
        res_cam_y = 2080
        cam_displ_x = 1350
        cam_displ_y = 1015
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [3096,2560,1920,1600,1280,1024,800,640,320]
        RES_Y_BIN1 = [2080,1920,1440,1200,960,768,600,480,240]
        DELAY_BIN1 = [0.02,0.02,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [1544,1280,960,800,640,512,400]
        RES_Y_BIN2 = [1040,960,720,600,480,384,300]
        DELAY_BIN2 = [0.02,0.01,0.01,0.01,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]

    if cameras_found[0] == "ZWO ASI294MC" or cameras_found[0] == "ZWO ASI294MC Pro" or cameras_found[0] == "ZWO ASI294MM" or cameras_found[0] == "ZWO ASI294MM Pro" :
        res_x_max = 4144
        res_y_max = 2822
        res_cam_x = 4144
        res_cam_y = 2822
        cam_displ_x = 1350
        cam_displ_y = 900
        val_USB = USB485
        flag_supported_camera = True

        RES_X_BIN1 = [4144,3240,2880,2400,1800,1536,1200,960,600]
        RES_Y_BIN1 = [2822,2160,1920,1600,1200,1024,800,640,400]
        DELAY_BIN1 = [0.04,0.03,0.02,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [2072,1600,1440,1200,904,768,600]
        RES_Y_BIN2 = [1410,1080,960,800,600,512,400]
        DELAY_BIN2 = [0.03,0.02,0.01,0.01,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]


    if cameras_found[0] == "ZWO ASI485MC" or cameras_found[0] == "ZWO ASI585MC" or cameras_found[0] == "ZWO ASI678MC" :
        res_x_max = 3840
        res_y_max = 2160
        res_cam_x = 3840
        res_cam_y = 2160
        cam_displ_x = 1350
        cam_displ_y = 900
        val_USB = USB485
        flag_supported_camera = True

        RES_X_BIN1 = [3840,3240,2880,2400,1800,1536,1200,960,600]
        RES_Y_BIN1 = [2160,2160,1920,1600,1200,1024,800,640,400]
        DELAY_BIN1 = [0.02,0.02,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [1920,1600,1440,1200,904,768,600]
        RES_Y_BIN2 = [1080,1080,960,800,600,512,400]
        DELAY_BIN2 = [0.02,0.02,0.01,0.01,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]

    if cameras_found[0] == "ZWO ASI290MC" or cameras_found[0] == "ZWO ASI290MM" or cameras_found[0] == "ZWO ASI462MC" or cameras_found[0] == "ZWO ASI385MC" :
        res_x_max = 1936
        res_y_max = 1096
        res_cam_x = 1936
        res_cam_y = 1096
        cam_displ_x = 1350
        cam_displ_y = 900
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [1936,1600,1280,1024,800,640,320,320,320]
        RES_Y_BIN1 = [1096,900,720,580,460,360,180,180,180]
        DELAY_BIN1 = [0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [968,800,640,512,400,320,160]
        RES_Y_BIN2 = [548,450,360,290,230,180,90]
        DELAY_BIN2 = [0.02,0.01,0.005,0.005,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]

    if cameras_found[0] == "ZWO ASI662MC" or cameras_found[0] == "ZWO ASI482MC" :
        res_x_max = 1920
        res_y_max = 1080
        res_cam_x = 1920
        res_cam_y = 1080
        cam_displ_x = 1350
        cam_displ_y = 900
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [1920,1600,1280,1024,800,640,320,320,320]
        RES_Y_BIN1 = [1080,900,720,580,460,360,180,180,180]
        DELAY_BIN1 = [0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [960,800,640,512,400,320,160]
        RES_Y_BIN2 = [540,450,360,290,230,180,90]
        DELAY_BIN2 = [0.02,0.01,0.005,0.005,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]


    if cameras_found[0] == "ZWO ASI224MC" :
        res_x_max = 1304
        res_y_max = 976
        res_cam_x = 1304
        res_cam_y = 976
        cam_displ_x = 1350
        cam_displ_y = 1015
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [1304,1280,1024,800,640,320,320,320,320]
        RES_Y_BIN1 = [976,960,768,600,480,240,240,240,240]
        DELAY_BIN1 = [0.01,0.01,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [640,640,512,400,320,160,160]
        RES_Y_BIN2 = [480,480,384,300,240,120,120]
        DELAY_BIN2 = [0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]

    if cameras_found[0] == "ZWO ASI533MC" or cameras_found[0] == "ZWO ASI533MM" or cameras_found[0] == "ZWO ASI533MC Pro" or cameras_found[0] == "ZWO ASI533MM Pro" :
        res_x_max = 3008
        res_y_max = 2260
        res_cam_x = 3008
        res_cam_y = 2260
        cam_displ_x = 1350
        cam_displ_y = 1015
        val_USB = USB178
        flag_supported_camera = True

        RES_X_BIN1 = [3008,2560,1920,1600,1280,1024,800,640,320]
        RES_Y_BIN1 = [2260,1920,1440,1200,960,768,600,480,240]
        DELAY_BIN1 = [0.02,0.02,0.01,0.01,0.005,0.005,0.005,0.005,0.005]

        RES_X_BIN2 = [1504,1280,960,800,640,512,400]
        RES_Y_BIN2 = [1130,960,720,600,480,384,300]
        DELAY_BIN2 = [0.02,0.01,0.01,0.01,0.005,0.005,0.005]

        val_delai = DELAY_BIN1[0]


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

        camera.set_roi(None,None,res_cam_x, res_cam_y,mode_BIN,format_capture)
        camera.set_control_value(asi.ASI_GAIN, val_gain)
        camera.set_control_value(asi.ASI_EXPOSURE, val_exposition)
        camera.set_control_value(asi.ASI_WB_B, val_blue)
        camera.set_control_value(asi.ASI_WB_R, val_red)
        camera.set_control_value(asi.ASI_FLIP, 0)
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
        print("Exit JetsonSky")
        quitter()

def init_efw():
    global filter_wheel, fw_position, flag_filter_wheel,presence_FW

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
    else :
        labelFW.config(text = "FW ON")
        print("EFW OK")
        flag_filter_wheel = True
        efw_id = 0
        filter_wheel = efw.EFW(efw_id)
        filter_wheel.open()
        fw_position = 0
        filter_wheel.set_position(fw_position)
        presence_FW.set(1)  # Initialisation présence FW

    
def start_acquisition() :
    global flag_autorise_acquisition,thread_1,flag_stop_acquisition
    flag_autorise_acquisition = True
    flag_stop_acquisition = False
    thread_1 = acquisition("1")
    thread_1.start()
             
  
class acquisition(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global type_debayer,camera,nb_erreur,image_brute,flag_acquisition_en_cours,flag_autorise_acquisition,Date_hour_image,\
               flag_image_disponible,frame,flag_noir_blanc,flag_hold_picture,val_delai,flag_BCap,image_BG,flag_sub_dark,Master_Dark,\
               flag_nouvelle_resolution,calque_stars,calque_satellites,calque_TIP,calque_direction_satellites,res_bb1,res_gg1,res_rr1,\
               flag_sub_dark,choix_BGSUB,flag_BG_OK,flag_sub_BG,image_BG,numero_image,echelle1,flag_acq_rapide,timeoutexp,flag_new_image

        while flag_autorise_acquisition == True :
            if flag_stop_acquisition == False and flag_filtre_work == False :
                if flag_hold_picture == 0 and flag_HDR == False :
                    flag_acquisition_en_cours = True
                    try :
                        if flag_noir_blanc == 0 and flag_colour_camera == True :
                            image_tmp=camera.capture_video_frame_RAW8(filename=None,timeout=timeoutexp)
                            image_brute=cv2.cvtColor(image_tmp, type_debayer)
                        else :
                            image_brute=camera.capture_video_frame_RAW8(filename=None,timeout=timeoutexp)
                        if flag_nouvelle_resolution == True :
                            flag_sub_dark = False
                            calque_stars = np.zeros_like(image_brute)
                            calque_satellites = np.zeros_like(image_brute)
                            calque_direction_satellites = np.zeros_like(image_brute)
                            calque_TIP = np.zeros_like(image_brute)
                            flag_nouvelle_resolution = False
                        if flag_sub_dark == True and dispo_dark == 'Dark disponible' :
                            image_brute = cv2.subtract(image_brute,Master_Dark)
                        if image_brute.ndim == 3 :
                            res_bb1,res_gg1,res_rr1=cv2.split(image_brute) 
                        else :
                            res_bb1 = image_brute
                            res_gg1 = res_bb1
                            res_rr1 = res_bb1
                        flag_image_disponible = True
                        flag_new_image = True
                        flag_acquisition_en_cours = False
                    except :
                        nb_erreur += 1
                        print("erreur acquisition : ",nb_erreur)
                        time.sleep(0.005)
                else :
                    time.sleep(0.1)
            flag_acquisition_en_cours = False
            time.sleep(0.005)

    def stop(self) :
        global flag_start_acquisition
        flag_start_acquisition = False
        
    def reprise(self) :
        global flag_start_acquisition
        flag_start_acquisition = True
        

def refresh() :
    global camera,traitement, img_cam,cadre_image,rawCapture,image_affichee,image_brute,flag_image_disponible,flag_quitter,timeoutexp,flag_new_image,\
           thread_1,flag_acquisition_en_cours,flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2,image_traitee,val_SAT,exposition,\
           val_gain, echelle2,val_exposition,echelle1,imggrey1,imggrey2,flag_FDIF,FDIF,flag_TRKSAT,correspondance,labelInfo2,Date_hour_image,\
           calque_stars,calque_satellites,calque_TIP,flag_nouvelle_resolution,nb_sat,sat_x,sat_y,sat_old_x,sat_old_y,res_bb1,res_gg1,res_rr1,\
           calque_direction_satellites,correspondance,vitesse,sat_id,sat_old_id,compteur_sat,old_sat,sat_speed,compteur_images,numero_image,nb_erreur

    if flag_premier_demarrage == True :
        flag_premier_demarrage = False
        start_acquisition()
    if flag_autoexposure_gain == True :
        val_gain = (int)(camera.get_control_value(asi.ASI_GAIN)[0])
        echelle2.set(val_gain)        
    if flag_autoexposure_exposition == True :
        val_exposition = (int)(camera.get_control_value(asi.ASI_EXPOSURE)[0]) // 1000
        echelle1.set(val_exposition)        
    if (flag_image_disponible == True and flag_new_image == True) or flag_HDR == True:
        if flag_HDR == True :
            try :
                image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutexp)
                image_brute1=cv2.cvtColor(image_tmp, type_debayer)
                val_exposition = echelle1.get()
                if flag_acq_rapide == True :
                    exposition = val_exposition // 10
                else :
                    exposition = val_exposition*1000 // 10
                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                timeoutHDR = timeoutexp // 10
                time.sleep(0.1)
                image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                image_brute2=cv2.cvtColor(image_tmp, type_debayer)
                if flag_acq_rapide == True :
                    exposition = val_exposition // 20
                else :
                    exposition = val_exposition*1000 // 20
                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                timeoutHDR = timeoutexp // 20
                time.sleep(0.1)
                image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                image_brute3=cv2.cvtColor(image_tmp, type_debayer)
                if flag_acq_rapide == True :
                    exposition = val_exposition // 40
                else :
                    exposition = val_exposition*1000 // 40
                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                timeoutHDR = timeoutexp // 40
                time.sleep(0.1)
                image_tmp=camera.capture_video_frame(filename=None,timeout=timeoutHDR)
                image_brute4=cv2.cvtColor(image_tmp, type_debayer)
                val_exposition = echelle1.get()
                if flag_acq_rapide == True :
                    exposition = val_exposition
                else :
                    exposition = val_exposition*1000
                camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                time.sleep(0.1)
                compteur_images = compteur_images + 1
                Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                img_list = [image_brute1,image_brute2,image_brute3,image_brute4]
                alignMTB = cv2.createAlignMTB()
                alignMTB.process(img_list, img_list)
                merge_mertens = cv2.createMergeMertens()
                res_mertens = merge_mertens.process(img_list)
                image_brute = np.clip(res_mertens*255, 0, 255).astype('uint8')
                if image_brute.ndim == 3 :
                    res_bb1,res_gg1,res_rr1=cv2.split(image_brute) 
                else :
                    res_bb1 = image_brute
                    res_gg1 = res_bb1
                    res_rr1 = res_bb1
                flag_image_disponible = True
                flag_acquisition_en_cours = False
            except :
                nb_erreur += 1
                print("erreur acquisition : ",nb_erreur)
                time.sleep(0.01)
        if flag_stop_acquisition == False and flag_filtre_work == False :
            if flag_nouvelle_resolution == True :
                calque_stars = np.zeros_like(image_brute)
                calque_satellites = np.zeros_like(image_brute)
                calque_direction_satellites = np.zeros_like(image_brute)
                flag_nouvelle_resolution = False
            Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
            compteur_images = compteur_images + 1
            application_filtrage(res_bb1,res_gg1,res_rr1)
            flag_new_image = False
            if flag_TRKSAT == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                satellites_tracking()
            if flag_FDIF == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                stars_detection()
            if flag_TRKSAT == 1 and nb_sat < 10 and flag_noir_blanc == 0 and flag_colour_camera == True :
                font = cv2.FONT_HERSHEY_SIMPLEX
                size = 0.5
                for i in range(nb_sat+1):
                    if correspondance[i] >=0:
                        centercircle = (sat_x[i],sat_y[i])
                        center_texte = (sat_x[i]+10,sat_y[i]+10)
                        texte = "Satellite No " + str(sat_id[i])
                        cv2.circle(image_traitee, centercircle, 5, (0,255,0), 2, cv2.LINE_AA)
                        cv2.putText(image_traitee, texte,center_texte, font, size, (0, 255, 0), 1, cv2.LINE_AA)
                        center_texte = (sat_x[i]+10,sat_y[i]+25)
                        texte = "Relative Speed " + str(sat_speed[i])
                        cv2.putText(image_traitee, texte,center_texte, font, size, (255, 255, 0), 1, cv2.LINE_AA)                    
                image_traitee = cv2.addWeighted(image_traitee, 1, calque_direction_satellites, 1, 0)     
                image_traitee = cv2.addWeighted(image_traitee, 1, calque_satellites, 1, 0)       
            if flag_FDIF == 1 and flag_noir_blanc == 0 and flag_colour_camera == True :
                image_traitee = cv2.addWeighted(image_traitee, 1, calque_stars, 1, 0)
            if flag_TIP == 1 or flag_mountpos :
                image_traitee = cv2.addWeighted(image_traitee, 1, calque_TIP, 1, 0)
            if flag_cap_pic == True:
                pic_capture()
            if flag_cap_video == True :
                video_capture()
            if res_cam_x > cam_displ_x and flag_full_res == 0 :
                image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_AREA)
                cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
            else :
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
            if flag_TRGS == 1:
                transform = PIL.ImageDraw.Draw(cadre_image.im)
                for x in range(1,255) :
                    transform.line((((x-1)*3,cam_displ_y-Corr_GS[x-1]*512),(x*3,cam_displ_y-Corr_GS[x]*512)),fill="blue",width=4)
                transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                
            cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
            cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
        else :
            application_filtrage()
            if res_cam_x > cam_displ_x and flag_full_res == 0 :
                image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_AREA)
                cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
            else :
                cadre_image.im=PIL.Image.fromarray(image_traitee)
            cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
            cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)    
    if flag_quitter == False:
        if flag_HDR == False :
            fenetre_principale.after(4, refresh)
        else :
            fenetre_principale.after(20, refresh)

def satellites_tracking ():
    global imggrey1,imggrey2,image_traitee,calque_satellites,flag_sat_exist,sat_x,sat_y,sat_s,sat_old_x,sat_old_y,sat_cntdwn,nb_sat,sat_id,compteur_sat, \
           flag_first_sat_pass,nb_trace_sat,sat_speed

    if flag_noir_blanc == 0 and flag_colour_camera == True :
        imggrey2 = cv2.cvtColor(image_traitee, cv2.COLOR_BGR2GRAY)
        correspondance = np.zeros(1000,dtype=int)
        try :
            calque_direction_satellites[:,:,0] = 0
            calque_direction_satellites[:,:,1] = 0
            calque_direction_satellites[:,:,2] = 0
            imggrey2_int = np.int_(imggrey2)
            imggrey1_int = np.int_(imggrey1)
            diff_int = imggrey2_int - imggrey1_int
            diff_int = np.clip(diff_int,0,255)
            diff = np.uint8(diff_int)
            seuilb = np.percentile(diff, 99) + 15
            diff[0:50,0:1000] = 0
            ret,thresh = cv2.threshold(diff, seuilb , 255, cv2.THRESH_BINARY)
            imggrey1 = imggrey2
            image_sat=cv2.merge((thresh,thresh,thresh))
            seuil_min_blob_sat = 150
            params_sat = cv2.SimpleBlobDetector_Params()    
            params_sat.minThreshold = seuil_min_blob_sat;     # Change thresholds
            params_sat.maxThreshold = 255;
            params_sat.thresholdStep = 15 # steps to go through
            params_sat.filterByColor = True    # Filter by color.    
            params_sat.blobColor = 255  # 0 for darkblobs - 255 for light blobs)
            params_sat.minDistBetweenBlobs = 10
            params_sat.filterByArea = True    # Filter by Area.
            params_sat.minArea = 3
            params_sat.maxArea = 2000
            params_sat.minRepeatability = 3
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
        nb_sat = 0
        if flag_sat == True :
            if flag_first_sat_pass == False :
                for kp_sat in keypoints_sat:
                    nb_sat=nb_sat+1
                if nb_sat >= 0 and nb_sat < 10 :
                    nb_sat = -1
                    for kp_sat in keypoints_sat:
                        nb_sat = nb_sat+1
                        sat_x[nb_sat] = int(kp_sat.pt[0])
                        sat_y[nb_sat] = int(kp_sat.pt[1])
                    for i in range(nb_sat+1) :
                        dist_min = 100000
                        correspondance[i] = -1
                        for j in range(nb_trace_sat+1) :
                            if sat_old_x[j] > 0 :
                                distance = (int)(math.sqrt((sat_x[i]-sat_old_x[j])*(sat_x[i]-sat_old_x[j])+(sat_y[i]-sat_old_y[j])*(sat_y[i]-sat_old_y[j])))
                            else :
                                distance = -1
                            if distance > 0 and distance < dist_min :
                                dist_min = distance
                                correspondance[i] = j
                                sat_id[i] = sat_old_id[correspondance[i]]
                        if dist_min > 40 :
                            correspondance[i] = -1
                            nb_trace_sat = nb_trace_sat + 1
                            sat_id[i] = nb_trace_sat
                            sat_old_x[nb_trace_sat]=sat_x[i]
                            sat_old_y[nb_trace_sat]=sat_y[i]
                            sat_old_id[nb_trace_sat]=nb_trace_sat
                    for j in range(nb_trace_sat+1) :
                        flag_active_trace = False
                        for i in range(nb_sat+1):
                            if sat_old_id[j] == sat_id[i] :
                                flag_active_trace = True
                        if flag_active_trace == False :
                            sat_old_x[j]= -1
                            sat_old_y[j]= -1
                            sat_old_id[j]= -1
                    for i in range(nb_sat+1) :
                        if correspondance[i] >=0 and sat_old_x[correspondance[i]] > 0 :
                            start_point = (sat_old_x[correspondance[i]],sat_old_y[correspondance[i]])
                            end_point = (sat_x[i],sat_y[i])
                            cv2.line(calque_satellites,start_point,end_point,(0,255,0),1)           
                            delta_x = (sat_x[i] - sat_old_x[correspondance[i]]) * 10
                            delta_y = (sat_y[i] - sat_old_y[correspondance[i]]) * 10
                            sat_speed[i] = math.sqrt(delta_x*delta_x+delta_y*delta_y)
                            direction = (sat_x[i] + delta_x,sat_y[i]+delta_y)
                            cv2.line(calque_satellites,start_point,end_point,(0,255,0),1)           
                            cv2.line(calque_direction_satellites,end_point,direction,(255,255,0),1)           
                            sat_old_x[correspondance[i]]=sat_x[i]
                            sat_old_y[correspondance[i]]=sat_y[i]
                            sat_old_id[correspondance[i]]=sat_id[i]                        
                        else :
                            sat_old_x[correspondance[i]]= -1
                            sat_old_y[correspondance[i]]= -1
                            sat_old_id[correspondance[i]]= -1
                            
                if nb_sat >= 10 :
                    raz_tracking()
                    nb_sat = 0
                        
            if flag_first_sat_pass == True :
                for kp_sat in keypoints_sat:
                    nb_sat=nb_sat+1
                if nb_sat >= 0 :
                    nb_sat = -1
                    for kp_sat in keypoints_sat:
                        nb_sat = nb_sat+1
                        sat_x[nb_sat] = int(kp_sat.pt[0])
                        sat_y[nb_sat] = int(kp_sat.pt[1])
                        sat_id[nb_sat] = nb_sat
                        sat_old_x[nb_sat] = sat_x[nb_sat]
                        sat_old_y[nb_sat] = sat_y[nb_sat]
                        sat_old_id[nb_sat] = nb_sat
                    nb_trace_sat = nb_sat
                    flag_first_sat_pass = False

def stars_detection() :
    global image_traitee,calque_stars,sat_x,sat_y,sat_s,nb_sat

    if flag_noir_blanc == 0 and flag_colour_camera == True :
        calque_stars[:,:,0] = 0
        calque_stars[:,:,1] = 0
        calque_stars[:,:,2] = 0                            
        seuilb = np.percentile(image_traitee[:,:,0], 95)
        seuilg = np.percentile(image_traitee[:,:,1], 95)
        seuilr = np.percentile(image_traitee[:,:,2], 95)
        seuil_min_blob = max(seuilb,seuilg,seuilr) + 25
        if seuil_min_blob > 220 :
            seuil_min_blob = 220
        params = cv2.SimpleBlobDetector_Params()    
        params.minThreshold = seuil_min_blob;     # Change thresholds
        params.maxThreshold = 255;
        params.thresholdStep = 10 # steps to go through
        params.filterByColor = False    # Filter by color.    
        params.blobColor = 255  # 0 for darkblobs - 255 for light blobs)
        params.minDistBetweenBlobs = 4
        params.filterByArea = True    # Filter by Area.
        params.minArea = 2
        params.maxArea = 500
        params.minRepeatability = 4
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(image_traitee)
        for kp in keypoints:
            if int(kp.pt[0]) < 750 and int(kp.pt[1]) < 50 :
                pass
            else:
                affiche = True
                for i in range(nb_sat+1) :
                    affiche = True
                    if int(kp.pt[0]) == sat_x[i] and int(kp.pt[1]) == sat_y[i] :
                        affiche = False
                if affiche == True :
                    cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (255,0,0), 1, cv2.LINE_AA)


def gaussianblur_mono(image_mono,niveau_blur):
    global compute_library

    if compute_library == "cupy" :
        img_gpu = cp.asarray(image_mono)
        blur_mono = ndimage.gaussian_filter(img_gpu, sigma = niveau_blur)
        image_gaussian_blur_mono = cp.asnumpy(blur_mono)
    elif compute_library == "vpi" :
        if niveau_blur > 11 :
            niveau_blur = 11
        backend = vpi.Backend.CUDA
        image_vpi = vpi.asimage(image_mono)
        with backend:
            gaussian = image_vpi.gaussian_filter(niveau_blur, 1.7, border=vpi.Border.ZERO)
        image_gaussian_blur_mono = gaussian.cpu()
    else :
        image_gaussian_blur_mono = cv2.GaussianBlur(image_mono,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)

    return image_gaussian_blur_mono

def gaussianblur_colour(im_b,im_g,im_r,niveau_blur):
    global compute_library

    if compute_library == "cupy" :
        img_gpu_r = cp.asarray(im_r)
        img_gpu_g = cp.asarray(im_g)
        img_gpu_b = cp.asarray(im_b)
        blur_r = ndimage.gaussian_filter(img_gpu_r, sigma = niveau_blur)
        blur_g = ndimage.gaussian_filter(img_gpu_g, sigma = niveau_blur)
        blur_b = ndimage.gaussian_filter(img_gpu_b, sigma = niveau_blur)
        im_GB_r = cp.asnumpy(blur_r)
        im_GB_g = cp.asnumpy(blur_g)
        im_GB_b = cp.asnumpy(blur_b)
    elif compute_library == "vpi" :
        if niveau_blur > 11 :
            niveau_blur = 11
        backend = vpi.Backend.CUDA
        image_vpi_b = vpi.asimage(im_b)
        image_vpi_g = vpi.asimage(im_g)
        image_vpi_r = vpi.asimage(im_r)
        with backend:
            gaussian_b = image_vpi_b.gaussian_filter(niveau_blur, 1.7, border=vpi.Border.ZERO)
            gaussian_g = image_vpi_g.gaussian_filter(niveau_blur, 1.7, border=vpi.Border.ZERO)
            gaussian_r = image_vpi_r.gaussian_filter(niveau_blur, 1.7, border=vpi.Border.ZERO)
        im_GB_b = gaussian_b.cpu()
        im_GB_g = gaussian_g.cpu()
        im_GB_r = gaussian_r.cpu()
    else :
        im_GB_r = cv2.GaussianBlur(im_r,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)
        im_GB_g = cv2.GaussianBlur(im_g,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)
        im_GB_b = cv2.GaussianBlur(im_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT)

    return im_GB_b,im_GB_g,im_GB_r

def image_negative_colour (red,green,blue):
    global compute_library

    if compute_library == "cupy" :
        img_gpu_r = cp.asarray(red)
        img_gpu_g = cp.asarray(green)
        img_gpu_b = cp.asarray(blue)
        img_gpu_b = cp.invert(img_gpu_b,dtype=np.uint8)
        img_gpu_g = cp.invert(img_gpu_g,dtype=np.uint8)
        img_gpu_r = cp.invert(img_gpu_r,dtype=np.uint8)
        blue = cp.asnumpy(img_gpu_b)
        green = cp.asnumpy(img_gpu_g)
        red = cp.asnumpy(img_gpu_r)
    else :
        blue = np.invert(blue,dtype=np.uint8)
        green = np.invert(green,dtype=np.uint8)
        red = np.invert(red,dtype=np.uint8)

    return blue,green,blue


def application_filtrage(res_b1,res_g1,res_r1) :
    global image_traitee1,image_traitee2,image_traitee3,image_traitee4,image_traitee5,compteur_FS,Im1OK,Im2OK,Im3OK,compteur_images,numero_image,\
           Im4OK,Im5OK,flag_hold_picture,image_brute,img_brut_tmp,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max, \
           flag_cap_pic,flag_traitement,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,flag_front,Date_hour_image,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_filtre_work,flag_AmpSoft,val_ampl,grad_vignet,compteur_FSDN,flag_SAT,val_SAT,flag_NB_estime,\
           Im1fsdnOK,Im2fsdnOK,image_traiteefsdn1,image_traiteefsdn2,old_image,val_reds,val_greens,val_blues,trsf_r,trsf_g,trsf_b,\
           flag_dyn_AADP,Corr_GS,calque_TIP,azimut,hauteur,flag_type_clahe,val_ghost_reducer,res_b2,res_g2,res_r2

    start_time_test = cv2.getTickCount()

    numero_image = numero_image + 1

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

            imag_brute = res_b1.copy()
            height,width = imag_brute.shape
            nb_pixels = height * width
            nb_blocksX = (width // nb_ThreadsX) + 1
            nb_blocksY = (height // nb_ThreadsY) + 1
                           
            # bilateral openCV
            if flag_bilateral == 1 :
                imag_brute=cv2.bilateralFilter(imag_brute,5,125,125) # Application filtre bilateral

            if flag_NB_estime == 1 :
                r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                drv.memcpy_htod(r_gpu, imag_brute)
                drv.memcpy_htod(img_r_gpu, imag_brute)

                grey_estimate_Mono(r_gpu, img_r_gpu,\
                    np.int_(width), np.int_(height),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(imag_brute, r_gpu)
                r_gpu.free()
                img_r_gpu.free()

            if flag_denoise_stacking_Paillou == True and flag_hold_picture == 0 :
                compteur_FSDN = compteur_FSDN + 1
                if compteur_FSDN < 3 :
                    if compteur_FSDN == 1 :
                        old_image = imag_brute.copy()
                        Im1fsdnOK = True
                    if compteur_FSDN == 2 :
                        Im2fsdnOK = True

                res_r1 = imag_brute
            
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)

                if Im2fsdnOK == True :
                    res_r1 = imag_brute
                    res_r2 = old_image
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_r_gpu2, res_r2)

                    adaptative_absorber_denoise_Mono(r_gpu, img_r_gpu1, img_r_gpu2,\
                                         np.int_(width), np.int_(height),np.intc(flag_dyn_AADP),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer),\
                                         block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(imag_brute, r_gpu)
                r_gpu.free()
                img_r_gpu1.free()
                img_r_gpu2.free()
                old_image = imag_brute.copy()

            # Amplification soft image
            if flag_AmpSoft == 1 :                
                niveau_blur = 7
                imagegreyblur=gaussianblur_mono(imag_brute,niveau_blur)
                            
                r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)

                drv.memcpy_htod(img_r_gpu, imag_brute)

                grey_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                grey_blur_gpu = drv.mem_alloc(imagegreyblur.size * imagegreyblur.dtype.itemsize)
                drv.memcpy_htod(grey_gpu, imag_brute)
                drv.memcpy_htod(grey_blur_gpu, imagegreyblur)
                
                correction = drv.mem_alloc(Corr_GS.size * Corr_GS.dtype.itemsize)
                drv.memcpy_htod(correction, Corr_GS)

                Mono_ampsoft_GPU(r_gpu, img_r_gpu, grey_gpu, grey_blur_gpu,np.int_(width), np.int_(height), \
                        np.float32(val_Mu),np.float32(val_Ro),np.float32(val_ampl), np.intc(flag_lin_gauss), correction, \
			block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(imag_brute, r_gpu)
                r_gpu.free()
                img_r_gpu.free()
                grey_gpu.free()
                grey_blur_gpu.free()
                correction.free()

                if flag_lin_gauss == 1 or flag_lin_gauss == 2 :
                    for x in range(1,256) :
                        trsf_r[x] = (int)(trsf_r[x] * val_ampl)
                    trsf_r = np.clip(trsf_r,0,255)

            if flag_2DConv == 1 or ((flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1)) or\
               flag_denoise_Paillou == 1 or flag_denoise_soft == 1 or flag_denoise_KNN == 1 :
                
                r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                drv.memcpy_htod(r_gpu, imag_brute)
                drv.memcpy_htod(img_r_gpu, imag_brute)

            # 2DConv = Blur CUDA
            if flag_2DConv == 1 :
                Smooth_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_2DConv), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

            # Histo equalize 2 CUDA
            # Histo stretch CUDA
            # Histo Phi Theta CUDA           
            if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                
                Histo_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                   np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta), \
                   block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

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

            # Denoise PAILLOU CUDA
            if flag_denoise_Paillou == 1 :
                cell_size = 3
                sqr_cell_size = cell_size * cell_size
                Denoise_Paillou_Mono(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.intc(cell_size), \
                            np.intc(sqr_cell_size),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
 
            # Denoise NLM2 CUDA
            if flag_denoise_soft == 1 :
                nb_ThreadsXs = 8
                nb_ThreadsYs = 8
                nb_blocksXs = (width // nb_ThreadsXs) + 1
                nb_blocksYs = (height // nb_ThreadsYs) + 1
                param=float(val_denoise)
                Noise = 1.0/(param*param)
                lerpC = 0.4
                NLM2_Mono_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsXs,nb_ThreadsYs,1), grid=(nb_blocksXs,nb_blocksYs))
                
            # Denoise KNN CUDA
            if flag_denoise_KNN == 1 :               
                param=float(val_denoise_KNN)
                Noise = 1.0/(param*param)
                lerpC = 0.4
                KNN_Mono_GPU(r_gpu, img_r_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))                               
               
            if flag_2DConv == 1 or ((flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) and flag_front == True) or \
                flag_denoise_Paillou == 1 or flag_denoise_soft == 1 or flag_denoise_KNN == 1 :

                drv.memcpy_dtoh(imag_brute, r_gpu)
                r_gpu.free()
                img_r_gpu.free()

            # Gradient Removal openCV
            if flag_GR == True :
                if grad_vignet == 1 :
                    if compute_library == "opencv" :
                        seuilb = np.percentile(imag_brute, val_SGR)
                        th,img_b = cv2.threshold(imag_brute,seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        img_b = cv2.GaussianBlur(imag_brute,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,)
                        factor = (100.0-val_AGR) / 100.0
                        att_b = img_b * factor
                        gradient = np.uint8(att_b)
                        imag_brute = cv2.subtract(imag_brute,gradient)
                    if compute_library == "cupy" :
                        seuilb = int(cp.percentile(cp.asarray(imag_brute), val_SGR))
                        th,img_b = cv2.threshold(imag_brute,seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        img_b = cv2.GaussianBlur(imag_brute,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,)
                        factor = (100.0-val_AGR) / 100.0
                        att_b = cp.asarray(img_b) * factor
                        resb = cp.subtract(cp.asarray(imag_brute),att_b)
                        resb = cp.clip(resb,0,255)
                        imag_brute = np.uint8(cp.asnumpy(resb))
                else :
                    if compute_library == "opencv" :
                        seuilb = np.percentile(imag_brute, val_SGR)
                        th,fd_b = cv2.threshold(imag_brute,seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        fd_b = cv2.GaussianBlur(fd_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,)
                        pivot_b = np.int16(np.percentile(imag_brute, val_AGR))
                        corr_b = np.int16(np.int16(imag_brute)) - np.int16(fd_b) + pivot_b
                        imag_brute = np.uint8(np.clip(corr_b,0,255))
                    if compute_library == "cupy" :
                        seuilb = int(cp.percentile(cp.asarray(imag_brute), val_SGR))
                        th,fd_b = cv2.threshold(imag_brute,seuilb,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        fd_b = cv2.GaussianBlur(fd_b,(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,)
                        pivot_b = int(cp.percentile(cp.asarray(imag_brute), val_AGR))
                        corr_b = cp.asarray(imag_brute).astype(cp.int16) - cp.asarray(fd_b).astype(cp.int16) + pivot_b
                        corr_b = cp.clip(corr_b,0,255)
                        imag_brute = np.uint8(cp.asnumpy(corr_b))

            # Image Negatif
            if ImageNeg == 1 :
                imag_brute = cv2.bitwise_not(imag_brute) # Test fonction cv2.bitwise_not
                for x in range(1,256) :
                    trsf_r[x] = (int)(256-trsf_r[x])
                trsf_r = np.clip(trsf_r,0,255)

            # Contrast CLAHE openCV
            if flag_contrast_CLAHE ==1 :
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                imag_brute = clahe.apply(imag_brute)
                    
            if (val_FS > 1 and flag_hold_picture == 0) and Dev_system == "PC Windows" :
                compteur_FS = compteur_FS+1
                if compteur_FS > val_FS :
                    compteur_FS = 1
                if compteur_FS == 1 :
                    image_traitee1 = imag_brute.copy()
                    Im1OK = True
                if compteur_FS == 2 :
                    image_traitee2 = imag_brute.copy()
                    Im2OK = True
                if compteur_FS == 3 :
                    image_traitee3 = imag_brute.copy()
                    Im3OK = True
                if compteur_FS == 4 :
                    image_traitee4 = imag_brute.copy()
                    Im4OK = True
                if compteur_FS == 5 :
                    image_traitee5 = imag_brute.copy()
                    Im5OK = True
                    
                if val_FS == 1 :
                    image_traitee = imag_brute.copy()
                    
                if val_FS == 2 and Im2OK == True :
                    nb_images = 2
                    res_r1 = image_traitee1.copy()
                    res_r2 = image_traitee2.copy()
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_r_gpu2, res_r2)

                    Stacking_mean_Mono(r_gpu, img_r_gpu1, img_r_gpu2, \
                                        img_r_gpu3, img_r_gpu4, img_r_gpu5, \
                                        np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                        block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                  
                    drv.memcpy_dtoh(res_r1, r_gpu)
                    r_gpu.free()
                    img_r_gpu1.free()
                    img_r_gpu2.free()
                    img_r_gpu3.free()
                    img_r_gpu4.free()
                    img_r_gpu5.free()

                    imag_brute=res_r1

                if val_FS == 3 and Im3OK == True :
                    nb_images = 3
                    res_r1 = image_traitee1.copy()
                    res_r2 = image_traitee2.copy()
                    res_r3 = image_traitee3.copy()
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_r_gpu3, res_r3)

                    Stacking_mean_Mono(r_gpu, img_r_gpu1, img_r_gpu2, \
                                        img_r_gpu3, img_r_gpu4, img_r_gpu5, \
                                        np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                        block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    r_gpu.free()
                    img_r_gpu1.free()
                    img_r_gpu2.free()
                    img_r_gpu3.free()
                    img_r_gpu4.free()
                    img_r_gpu5.free()

                    imag_brute=res_r1
                        
                if val_FS == 4 and Im4OK == True :
                    nb_images = 4
                    res_r1 = image_traitee1.copy()
                    res_r2 = image_traitee2.copy()
                    res_r3 = image_traitee3.copy()
                    res_r4 = image_traitee4.copy()
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_r_gpu3, res_r3)
                    drv.memcpy_htod(img_r_gpu4, res_r4)

                    Stacking_mean_Mono(r_gpu, img_r_gpu1, img_r_gpu2, \
                                        img_r_gpu3, img_r_gpu4, img_r_gpu5, \
                                        np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                        block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    r_gpu.free()
                    img_r_gpu1.free()
                    img_r_gpu2.free()
                    img_r_gpu3.free()
                    img_r_gpu4.free()
                    img_r_gpu5.free()

                    imag_brute=res_r1
                    
                if val_FS == 5 and Im5OK == True :
                    nb_images = 5
                    res_r1 = image_traitee1.copy()
                    res_r2 = image_traitee2.copy()
                    res_r3 = image_traitee3.copy()
                    res_r4 = image_traitee4.copy()
                    res_r5 = image_traitee5.copy()
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_r_gpu3, res_r3)
                    drv.memcpy_htod(img_r_gpu4, res_r4)
                    drv.memcpy_htod(img_r_gpu5, res_r5)

                    Stacking_mean_Mono(r_gpu, img_r_gpu1, img_r_gpu2, \
                                        img_r_gpu3, img_r_gpu4, img_r_gpu5, \
                                        np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                        block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    r_gpu.free()
                    img_r_gpu1.free()
                    img_r_gpu2.free()
                    img_r_gpu3.free()
                    img_r_gpu4.free()
                    img_r_gpu5.free()

                    imag_brute=res_r1
                    
            if (val_FS > 1 and flag_hold_picture == 0) and Dev_system == "Jetson Linux" :
                compteur_FS = compteur_FS+1
                if compteur_FS > val_FS :
                    compteur_FS = 1
                if compteur_FS == 1 :
                    image_traitee1 = imag_brute.copy()
                    Im1OK = True
                if compteur_FS == 2 :
                    image_traitee2 = imag_brute.copy()
                    Im2OK = True
                if compteur_FS == 3 :
                    image_traitee3 = imag_brute.copy()
                    Im3OK = True
                if compteur_FS == 4 :
                    image_traitee4 = imag_brute.copy()
                    Im4OK = True
                if compteur_FS == 5 :
                    image_traitee5 = imag_brute.copy()
                    Im5OK = True
                    
                if val_FS == 1 :
                    image_traitee = imag_brute.copy()
                    
                if val_FS == 2 and Im2OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                
                if val_FS == 3 and Im3OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                        
                if val_FS == 4 and Im4OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee4, 1, 0)
                    
                if val_FS == 5 and Im5OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee4, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee5, 1, 0)

            if (val_FS > 1 and flag_hold_picture == 0) and Dev_system == "Jetson Linux" :
                pass
            else :
                image_traitee = imag_brute

            # SharpenSoft1 = Sharpen 1 CUDA
            # UnsharpMask CUDA
            if flag_sharpen_soft1 == 1 or flag_unsharp_mask == 1 :
                r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(imag_brute.size * imag_brute.dtype.itemsize)
                drv.memcpy_htod(r_gpu, image_traitee)
                drv.memcpy_htod(img_r_gpu, image_traitee)
                res_r = np.empty_like(image_traitee)
                
                Sharp_Mono(r_gpu, img_r_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_sharpen_soft1), np.intc(flag_unsharp_mask), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                
                drv.memcpy_dtoh(image_traitee, r_gpu)
                r_gpu.free()
                img_r_gpu.free()


            if flag_DEMO == 1 :
                image_traitee[0:height,0:width//2] = image_brute[0:height,0:width//2]             

        ######################################################################################################################
        ######################################################################################################################       
        ######################################################################################################################
                            
        if flag_noir_blanc == 0 and flag_colour_camera == True:

            # Colour image treatment
            
            height,width = res_b1.shape
                            
            nb_pixels = height * width
            nb_blocksX = (width // nb_ThreadsX) + 1
            nb_blocksY = (height // nb_ThreadsY) + 1
                        
            if flag_denoise_stacking_Paillou == True and flag_hold_picture == 0 :
                compteur_FSDN = compteur_FSDN + 1
                if compteur_FSDN < 3 :
                    if compteur_FSDN == 1 :
                        res_b2 = res_b1.copy()
                        res_g2 = res_g1.copy()
                        res_r2 = res_r1.copy()
                        Im1fsdnOK = True
                    if compteur_FSDN == 2 :
                        Im2fsdnOK = True
                
                b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
            
                g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
            
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)


                if Im2fsdnOK == True :
                    nb_images = 2
                    divise = 2.0
                    drv.memcpy_htod(img_b_gpu1, res_b1)  
                    drv.memcpy_htod(img_g_gpu1, res_g1)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_b_gpu2, res_b2)  
                    drv.memcpy_htod(img_g_gpu2, res_g2)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    
                    adaptative_absorber_denoise_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2,\
                                         np.int_(width), np.int_(height),np.intc(flag_dyn_AADP),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    drv.memcpy_dtoh(res_g1, g_gpu)
                    drv.memcpy_dtoh(res_b1, b_gpu)
                    r_gpu.free()
                    g_gpu.free()
                    b_gpu.free()
                    img_r_gpu1.free()
                    img_g_gpu1.free()
                    img_b_gpu1.free()
                    img_r_gpu2.free()
                    img_g_gpu2.free()
                    img_b_gpu2.free()
                    res_b2 = res_b1.copy()
                    res_g2 = res_g1.copy()
                    res_r2 = res_r1.copy()

            # bilateral openCV
            if flag_bilateral == 1 :
                if flag_vpi == False :
                    res_b1,res_g1,res_r1=cv2.split(cv2.bilateralFilter(cv2.merge((res_b1,res_g1,res_r1)),5,125,125)) # Application filtre bilateral
                else :
                    colour_image = cv2.merge((res_b1,res_g1,res_r1))
                    with vpi.Backend.CUDA:
                        inSize = (int(width), int(height))
                        frame = vpi.asimage(colour_image).convert(vpi.Format.NV12_ER)
                        tnr = vpi.TemporalNoiseReduction(inSize, vpi.Format.NV12_ER)
                        denoised = tnr(frame, preset=vpi.TNRPreset.INDOOR_LOW_LIGHT, strength=1)
                        denoised = denoised.convert(vpi.Format.RGB8)
                    colour_image = denoised.cpu()
                    res_b1,res_g1,res_r1=cv2.split(colour_image)
                                
            if flag_AmpSoft == 1 and flag_lin_gauss == 3 :
                imagegrey = cv2.cvtColor(cv2.merge((res_b1,res_g1,res_r1)), cv2.COLOR_BGR2GRAY)
                imagegreyblur = imagegrey
                
                niveau_blur = 7
                imagegreyblur=gaussianblur_mono(imagegrey,niveau_blur)
                
                nb_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                nimg_b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
            
                ng_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                nimg_g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
            
                nr_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                nimg_r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)

                drv.memcpy_htod(nimg_b_gpu, res_b1)  
                drv.memcpy_htod(nimg_g_gpu, res_g1)
                drv.memcpy_htod(nimg_r_gpu, res_r1)

                grey_gpu = drv.mem_alloc(imagegrey.size * imagegrey.dtype.itemsize)
                grey_blur_gpu = drv.mem_alloc(imagegreyblur.size * imagegreyblur.dtype.itemsize)
                drv.memcpy_htod(grey_gpu, imagegrey)
                drv.memcpy_htod(grey_blur_gpu, imagegreyblur)
                
                correction = drv.mem_alloc(Corr_GS.size * Corr_GS.dtype.itemsize)
                drv.memcpy_htod(correction, Corr_GS)

                Colour_staramp_GPU(nr_gpu, ng_gpu, nb_gpu, nimg_r_gpu, nimg_g_gpu, nimg_b_gpu, grey_gpu, grey_blur_gpu,np.int_(width), np.int_(height), \
                        np.float32(val_Mu),np.float32(val_Ro),np.float32(val_ampl), correction, block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(res_r1, nr_gpu)
                drv.memcpy_dtoh(res_g1, ng_gpu)
                drv.memcpy_dtoh(res_b1, nb_gpu)
                nr_gpu.free()
                ng_gpu.free()
                nb_gpu.free()
                nimg_r_gpu.free()
                nimg_g_gpu.free()
                nimg_b_gpu.free()
                grey_gpu.free()
                grey_blur_gpu.free()
                correction.free()
            
            if flag_2DConv == 1 or ((flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1)) or\
               (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) or flag_denoise_Paillou == 1 or flag_denoise_soft == 1 or flag_denoise_KNN == 1 or\
               val_reds != 100 or val_greens != 100 or val_blues != 100 :
               

                b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
            
                g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
            
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)

                drv.memcpy_htod(img_b_gpu, res_b1)  
                drv.memcpy_htod(img_g_gpu, res_g1)
                drv.memcpy_htod(img_r_gpu, res_r1)

            if flag_2DConv == 1 :
                # 2DConv = Blur CUDA               
                Smooth_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_2DConv), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                

            if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                # Histo equalize 2 CUDA
                # Histo stretch CUDA
                # Histo Phi Theta CUDA
            
                Histo_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \
                   np.intc(flag_histogram_equalize2), np.float32(val_heq2), np.intc(flag_histogram_phitheta), np.float32(val_phi), np.float32(val_theta), \
                   block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))


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
                        
            # Denoise PAILLOU CUDA
            if flag_denoise_Paillou == 1 :
                cell_size = 5
                sqr_cell_size = cell_size * cell_size

                Denoise_Paillou_Colour(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.intc(cell_size), \
                            np.intc(sqr_cell_size),block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

            # Denoise NLM2 CUDA
            if flag_denoise_soft == 1 :
                nb_ThreadsXs = 8
                nb_ThreadsYs = 8
                nb_blocksXs = (width // nb_ThreadsXs) + 1
                nb_blocksYs = (height // nb_ThreadsYs) + 1
                param=float(val_denoise)
                Noise = 1.0/(param*param)
                lerpC = 0.4
                
                NLM2_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsXs,nb_ThreadsYs,1), grid=(nb_blocksXs,nb_blocksYs))
            
            # Denoise KNN CUDA
            if flag_denoise_KNN == 1 :
                param=float(val_denoise_KNN)
                Noise = 1.0/(param*param)
                lerpC = 0.4
            
                KNN_Colour_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu,np.intc(width),np.intc(height), np.float32(Noise), \
                     np.float32(lerpC), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

            # Amplification soft image
            if flag_AmpSoft == 1 :
                if flag_lin_gauss == 1 or flag_lin_gauss == 2 :
                    correction = drv.mem_alloc(Corr_GS.size * Corr_GS.dtype.itemsize)
                    drv.memcpy_htod(correction, Corr_GS)

                    Colour_ampsoft_GPU(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                        np.float32(val_ampl), correction, block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    correction.free()

                for x in range(1,256) :
                    trsf_r[x] = (int)(trsf_r[x] * val_ampl)
                    trsf_g[x] = (int)(trsf_g[x] * val_ampl)
                    trsf_b[x] = (int)(trsf_b[x] * val_ampl)
                trsf_r = np.clip(trsf_r,0,255)
                trsf_g = np.clip(trsf_g,0,255)
                trsf_b = np.clip(trsf_b,0,255)

             # Adjust RGB channels soft
            if (val_reds != 100 or val_greens != 100 or val_blues != 100) :              
                mod_red = val_reds / 100.0
                mod_green = val_greens / 100.0
                mod_blue = val_blues / 100.0
                
                Set_RGB(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                        np.float32(mod_red), np.float32(mod_green), np.float32(mod_blue), \
                        block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

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

            if flag_2DConv == 1 or ((flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) and flag_front == True) or\
               (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) or flag_denoise_Paillou == 1 or flag_denoise_soft == 1 or flag_denoise_KNN == 1 or\
               val_reds != 100 or val_greens != 100 or val_blues != 100 :

                drv.memcpy_dtoh(res_r1, r_gpu)
                drv.memcpy_dtoh(res_g1, g_gpu)
                drv.memcpy_dtoh(res_b1, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
       
            # Gradient Removal openCV
            if flag_GR == True :
                if grad_vignet == 1 :
                    if compute_library == "opencv" :
                        seuilb = np.percentile(res_b1, val_SGR)
                        seuilg = np.percentile(res_g1, val_SGR)
                        seuilr = np.percentile(res_r1, val_SGR)
                        th,img_b = cv2.threshold(res_b1,seuilb,255,cv2.THRESH_TRUNC) 
                        th,img_g = cv2.threshold(res_g1,seuilg,255,cv2.THRESH_TRUNC) 
                        th,img_r = cv2.threshold(res_r1,seuilr,255,cv2.THRESH_TRUNC) 
                        niveau_blur = val_NGB*2 + 3
                        img_b,img_g,img_r = cv2.split(cv2.GaussianBlur(cv2.merge((img_b,img_g,img_r)),(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,))
                        factor = (100.0-val_AGR) / 100.0
                        att_b = img_b * factor
                        att_g = img_g * factor
                        att_r = img_r * factor
                        img_b = np.uint8(att_b)
                        img_g = np.uint8(att_g)
                        img_r = np.uint8(att_r)
                        res_b1 = cv2.subtract(res_b1,img_b)
                        res_g1 = cv2.subtract(res_g1,img_g)
                        res_r1 = cv2.subtract(res_r1,img_r)
                    if compute_library == "cupy" :
                        seuilb = int(cp.percentile(cp.asarray(res_b1), val_SGR))
                        seuilg = int(cp.percentile(cp.asarray(res_g1), val_SGR))
                        seuilr = int(cp.percentile(cp.asarray(res_r1), val_SGR))
                        th,img_b = cv2.threshold(res_b1,seuilb,255,cv2.THRESH_TRUNC) 
                        th,img_g = cv2.threshold(res_g1,seuilg,255,cv2.THRESH_TRUNC) 
                        th,img_r = cv2.threshold(res_r1,seuilr,255,cv2.THRESH_TRUNC) 
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
                        res_b1 = np.uint8(cp.asnumpy(resb))
                        res_g1 = np.uint8(cp.asnumpy(resg))
                        res_r1 = np.uint8(cp.asnumpy(resr))
                else :
                    if compute_library == "opencv" :
                        seuilb = np.percentile(res_b1, val_SGR)
                        seuilg = np.percentile(res_g1, val_SGR)
                        seuilr = np.percentile(res_r1, val_SGR)      
                        th,fd_b = cv2.threshold(res_b1,seuilb,255,cv2.THRESH_TRUNC) 
                        th,fd_g = cv2.threshold(res_g1,seuilg,255,cv2.THRESH_TRUNC) 
                        th,fd_r = cv2.threshold(res_r1,seuilr,255,cv2.THRESH_TRUNC)
                        niveau_blur = val_NGB*2 + 3
                        fd_b,fd_g,fd_r = cv2.split(cv2.GaussianBlur(cv2.merge((fd_b,fd_g,fd_r)),(niveau_blur,niveau_blur),cv2.BORDER_DEFAULT,))
                        pivot_b = np.int16(np.percentile(res_b1, val_AGR))
                        pivot_g = np.int16(np.percentile(res_g1, val_AGR))
                        pivot_r = np.int16(np.percentile(res_r1, val_AGR))                
                        corr_b = np.int16(np.int16(res_b1) - np.int16(fd_b) + pivot_b)
                        corr_g = np.int16(np.int16(res_g1) - np.int16(fd_g) + pivot_g)
                        corr_r = np.int16(np.int16(res_r1) - np.int16(fd_r) + pivot_r)
                        res_b1 = np.uint8(np.clip(corr_b,0,255))
                        res_g1 = np.uint8(np.clip(corr_g,0,255))
                        res_r1 = np.uint8(np.clip(corr_r,0,255))                
                    if compute_library == "cupy" :
                        seuilb = int(cp.percentile(cp.asarray(res_b1), val_SGR))
                        seuilg = int(cp.percentile(cp.asarray(res_g1), val_SGR))
                        seuilr = int(cp.percentile(cp.asarray(res_r1), val_SGR))   
                        th,fd_b = cv2.threshold(res_b1,seuilb,255,cv2.THRESH_TRUNC) 
                        th,fd_g = cv2.threshold(res_g1,seuilg,255,cv2.THRESH_TRUNC) 
                        th,fd_r = cv2.threshold(res_r1,seuilr,255,cv2.THRESH_TRUNC)
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
                        res_b1 = np.uint8(cp.asnumpy(corr_b))
                        res_g1 = np.uint8(cp.asnumpy(corr_g))
                        res_r1 = np.uint8(cp.asnumpy(corr_r))                

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
                                
            # Contrast CLAHE openCV
            if flag_contrast_CLAHE ==1 :
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                res_b1 = clahe.apply(res_b1)
                res_g1 = clahe.apply(res_g1)
                res_r1 = clahe.apply(res_r1)

            if (val_FS > 1 and flag_hold_picture == 0) and Dev_system == "PC Windows" :
                compteur_FS = compteur_FS+1
                if compteur_FS > val_FS :
                    compteur_FS = 1
                if compteur_FS == 1 :
                    image_traitee1 = cv2.merge((res_b1,res_g1,res_r1))
                    Im1OK = True
                if compteur_FS == 2 :
                    image_traitee2 = cv2.merge((res_b1,res_g1,res_r1))
                    Im2OK = True
                if compteur_FS == 3 :
                    image_traitee3 = cv2.merge((res_b1,res_g1,res_r1))
                    Im3OK = True
                if compteur_FS == 4 :
                    image_traitee4 = cv2.merge((res_b1,res_g1,res_r1))
                    Im4OK = True
                if compteur_FS == 5 :
                    image_traitee5 = cv2.merge((res_b1,res_g1,res_r1))
                    Im5OK = True
                    
                if val_FS == 1 :
                    image_traitee = cv2.merge((res_b1,res_g1,res_r1))
                    
                if val_FS == 2 and Im2OK == True :
                    nb_images = 2
                    res_b1,res_g1,res_r1 = cv2.split(image_traitee1)
                    res_b2,res_g2,res_r2 = cv2.split(image_traitee2)
                    b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu3 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu4 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu5 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                
                    g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu3 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu4 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu5 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_b_gpu1, res_b1)  
                    drv.memcpy_htod(img_g_gpu1, res_g1)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_b_gpu2, res_b2)  
                    drv.memcpy_htod(img_g_gpu2, res_g2)
                    drv.memcpy_htod(img_r_gpu2, res_r2)

                    Stacking_mean_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2, \
                                            img_r_gpu3, img_g_gpu3, img_b_gpu3, img_r_gpu4, img_g_gpu4, img_b_gpu4, img_r_gpu5, img_g_gpu5, img_b_gpu5, \
                                            np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                            block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))
                    
                    drv.memcpy_dtoh(res_r1, r_gpu)
                    drv.memcpy_dtoh(res_g1, g_gpu)
                    drv.memcpy_dtoh(res_b1, b_gpu)
                    r_gpu.free()
                    g_gpu.free()
                    b_gpu.free()
                    img_r_gpu1.free()
                    img_g_gpu1.free()
                    img_b_gpu1.free()
                    img_r_gpu2.free()
                    img_g_gpu2.free()
                    img_b_gpu2.free()
                    img_r_gpu3.free()
                    img_g_gpu3.free()
                    img_b_gpu3.free()
                    img_r_gpu4.free()
                    img_g_gpu4.free()
                    img_b_gpu4.free()
                    img_r_gpu5.free()
                    img_g_gpu5.free()
                    img_b_gpu5.free()
                
                if val_FS == 3 and Im3OK == True :
                    nb_images = 3
                    res_b1,res_g1,res_r1 = cv2.split(image_traitee1)
                    res_b2,res_g2,res_r2 = cv2.split(image_traitee2)
                    res_b3,res_g3,res_r3 = cv2.split(image_traitee3)
                    b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu3 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu4 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu5 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                
                    g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu3 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu4 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu5 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_b_gpu1, res_b1)  
                    drv.memcpy_htod(img_g_gpu1, res_g1)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_b_gpu2, res_b2)  
                    drv.memcpy_htod(img_g_gpu2, res_g2)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_b_gpu3, res_b3)  
                    drv.memcpy_htod(img_g_gpu3, res_g3)
                    drv.memcpy_htod(img_r_gpu3, res_r3)

                    Stacking_mean_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2, \
                                            img_r_gpu3, img_g_gpu3, img_b_gpu3, img_r_gpu4, img_g_gpu4, img_b_gpu4, img_r_gpu5, img_g_gpu5, img_b_gpu5, \
                                            np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                            block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    drv.memcpy_dtoh(res_g1, g_gpu)
                    drv.memcpy_dtoh(res_b1, b_gpu)
                    r_gpu.free()
                    g_gpu.free()
                    b_gpu.free()
                    img_r_gpu1.free()
                    img_g_gpu1.free()
                    img_b_gpu1.free()
                    img_r_gpu2.free()
                    img_g_gpu2.free()
                    img_b_gpu2.free()
                    img_r_gpu3.free()
                    img_g_gpu3.free()
                    img_b_gpu3.free()
                    img_r_gpu4.free()
                    img_g_gpu4.free()
                    img_b_gpu4.free()
                    img_r_gpu5.free()
                    img_g_gpu5.free()
                    img_b_gpu5.free()
                        
                if val_FS == 4 and Im4OK == True :
                    nb_images = 4
                    res_b1,res_g1,res_r1 = cv2.split(image_traitee1)
                    res_b2,res_g2,res_r2 = cv2.split(image_traitee2)
                    res_b3,res_g3,res_r3 = cv2.split(image_traitee3)
                    res_b4,res_g4,res_r4 = cv2.split(image_traitee4)
                    b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu3 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu4 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu5 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                
                    g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu3 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu4 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu5 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_b_gpu1, res_b1)  
                    drv.memcpy_htod(img_g_gpu1, res_g1)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_b_gpu2, res_b2)  
                    drv.memcpy_htod(img_g_gpu2, res_g2)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_b_gpu3, res_b3)  
                    drv.memcpy_htod(img_g_gpu3, res_g3)
                    drv.memcpy_htod(img_r_gpu3, res_r3)
                    drv.memcpy_htod(img_b_gpu4, res_b4)  
                    drv.memcpy_htod(img_g_gpu4, res_g4)
                    drv.memcpy_htod(img_r_gpu4, res_r4)

                    Stacking_mean_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2, \
                                            img_r_gpu3, img_g_gpu3, img_b_gpu3, img_r_gpu4, img_g_gpu4, img_b_gpu4, img_r_gpu5, img_g_gpu5, img_b_gpu5, \
                                            np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                            block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    drv.memcpy_dtoh(res_g1, g_gpu)
                    drv.memcpy_dtoh(res_b1, b_gpu)
                    r_gpu.free()
                    g_gpu.free()
                    b_gpu.free()
                    img_r_gpu1.free()
                    img_g_gpu1.free()
                    img_b_gpu1.free()
                    img_r_gpu2.free()
                    img_g_gpu2.free()
                    img_b_gpu2.free()
                    img_r_gpu3.free()
                    img_g_gpu3.free()
                    img_b_gpu3.free()
                    img_r_gpu4.free()
                    img_g_gpu4.free()
                    img_b_gpu4.free()
                    img_r_gpu5.free()
                    img_g_gpu5.free()
                    img_b_gpu5.free()
                    
                if val_FS == 5 and Im5OK == True :
                    nb_images = 5
                    res_b1,res_g1,res_r1 = cv2.split(image_traitee1)
                    res_b2,res_g2,res_r2 = cv2.split(image_traitee2)
                    res_b3,res_g3,res_r3 = cv2.split(image_traitee3)
                    res_b4,res_g4,res_r4 = cv2.split(image_traitee4)
                    res_b5,res_g5,res_r5 = cv2.split(image_traitee5)
                    b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu1 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu2 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu3 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu4 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                    img_b_gpu5 = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                
                    g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu1 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu2 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu3 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu4 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                    img_g_gpu5 = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                
                    r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu1= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu2= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu3= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu4= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    img_r_gpu5= drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                    drv.memcpy_htod(img_b_gpu1, res_b1)  
                    drv.memcpy_htod(img_g_gpu1, res_g1)
                    drv.memcpy_htod(img_r_gpu1, res_r1)
                    drv.memcpy_htod(img_b_gpu2, res_b2)  
                    drv.memcpy_htod(img_g_gpu2, res_g2)
                    drv.memcpy_htod(img_r_gpu2, res_r2)
                    drv.memcpy_htod(img_b_gpu3, res_b3)  
                    drv.memcpy_htod(img_g_gpu3, res_g3)
                    drv.memcpy_htod(img_r_gpu3, res_r3)
                    drv.memcpy_htod(img_b_gpu4, res_b4)  
                    drv.memcpy_htod(img_g_gpu4, res_g4)
                    drv.memcpy_htod(img_r_gpu4, res_r4)
                    drv.memcpy_htod(img_b_gpu5, res_b5)  
                    drv.memcpy_htod(img_g_gpu5, res_g5)
                    drv.memcpy_htod(img_r_gpu5, res_r5)

                    Stacking_mean_Color(r_gpu, g_gpu, b_gpu, img_r_gpu1, img_g_gpu1, img_b_gpu1, img_r_gpu2, img_g_gpu2, img_b_gpu2, \
                                            img_r_gpu3, img_g_gpu3, img_b_gpu3, img_r_gpu4, img_g_gpu4, img_b_gpu4, img_r_gpu5, img_g_gpu5, img_b_gpu5, \
                                            np.int_(width), np.int_(height),np.float32(stack_div), np.int_(nb_images),\
                                            block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                    drv.memcpy_dtoh(res_r1, r_gpu)
                    drv.memcpy_dtoh(res_g1, g_gpu)
                    drv.memcpy_dtoh(res_b1, b_gpu)
                    r_gpu.free()
                    g_gpu.free()
                    b_gpu.free()
                    img_r_gpu1.free()
                    img_g_gpu1.free()
                    img_b_gpu1.free()
                    img_r_gpu2.free()
                    img_g_gpu2.free()
                    img_b_gpu2.free()
                    img_r_gpu3.free()
                    img_g_gpu3.free()
                    img_b_gpu3.free()
                    img_r_gpu4.free()
                    img_g_gpu4.free()
                    img_b_gpu4.free()
                    img_r_gpu5.free()
                    img_g_gpu5.free()
                    img_b_gpu5.free()

            if (val_FS > 1 and flag_hold_picture == 0) and Dev_system == "Jetson Linux":
                compteur_FS = compteur_FS+1
                if compteur_FS > val_FS :
                    compteur_FS = 1
                if compteur_FS == 1 :
                    image_traitee1 = cv2.merge((res_b1,res_g1,res_r1))
                    Im1OK = True
                if compteur_FS == 2 :
                    image_traitee2 = cv2.merge((res_b1,res_g1,res_r1))
                    Im2OK = True
                if compteur_FS == 3 :
                    image_traitee3 = cv2.merge((res_b1,res_g1,res_r1))
                    Im3OK = True
                if compteur_FS == 4 :
                    image_traitee4 = cv2.merge((res_b1,res_g1,res_r1))
                    Im4OK = True
                if compteur_FS == 5 :
                    image_traitee5 = cv2.merge((res_b1,res_g1,res_r1))
                    Im5OK = True
                                        
                if val_FS == 2 and Im2OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    res_b1,res_g1,res_r1=cv2.split(image_traitee)          
                
                if val_FS == 3 and Im3OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                    res_b1,res_g1,res_r1=cv2.split(image_traitee)          
                        
                if val_FS == 4 and Im4OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee4, 1, 0)
                    res_b1,res_g1,res_r1=cv2.split(image_traitee)          
                    
                if val_FS == 5 and Im5OK == True :
                    image_traitee = cv2.addWeighted(image_traitee1, 1, image_traitee2, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee3, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee4, 1, 0)
                    image_traitee = cv2.addWeighted(image_traitee, 1, image_traitee5, 1, 0)
                    res_b1,res_g1,res_r1=cv2.split(image_traitee)          

            if flag_sharpen_soft1 == 1 or flag_unsharp_mask == 1 :
                # SharpenSoft1 = Sharpen 1 CUDA
                # UnsharpMask CUDA
                b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
            
                g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
            
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)

                drv.memcpy_htod(img_b_gpu, res_b1)  
                drv.memcpy_htod(img_g_gpu, res_g1)
                drv.memcpy_htod(img_r_gpu, res_r1)
            
                Sharp_Color(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, np.int_(width), np.int_(height), \
                   np.intc(flag_sharpen_soft1), np.intc(flag_unsharp_mask), block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(res_r1, r_gpu)
                drv.memcpy_dtoh(res_g1, g_gpu)
                drv.memcpy_dtoh(res_b1, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()

            # image saturation enhancement
            if flag_SAT == True :
                colour_base_image = cv2.merge((res_b1,res_g1,res_r1))
                imggrey = cv2.cvtColor(colour_base_image, cv2.COLOR_BGR2GRAY)
                img_base=PIL.Image.fromarray(colour_base_image)
                converter = ImageEnhance.Color(img_base)
                img_sat = converter.enhance(val_SAT)
                res_b1,res_g1,res_r1 = cv2.split(np.array(img_sat))
                res_b1,res_g1,res_r1 = gaussianblur_colour(res_b1,res_g1,res_r1,11)                              
                img_grey_gpu = drv.mem_alloc(imggrey.size * imggrey.dtype.itemsize)
                drv.memcpy_htod(img_grey_gpu, imggrey)                  
                b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)
                img_b_gpu = drv.mem_alloc(res_b1.size * res_b1.dtype.itemsize)            
                g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)
                img_g_gpu = drv.mem_alloc(res_g1.size * res_g1.dtype.itemsize)            
                r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                img_r_gpu = drv.mem_alloc(res_r1.size * res_r1.dtype.itemsize)
                drv.memcpy_htod(img_b_gpu, res_b1)  
                drv.memcpy_htod(img_g_gpu, res_g1)
                drv.memcpy_htod(img_r_gpu, res_r1)

                Saturation_Combine(r_gpu, g_gpu, b_gpu, img_r_gpu, img_g_gpu, img_b_gpu, img_grey_gpu, np.int_(width), np.int_(height), \
                    block=(nb_ThreadsX,nb_ThreadsY,1), grid=(nb_blocksX,nb_blocksY))

                drv.memcpy_dtoh(res_r1, r_gpu)
                drv.memcpy_dtoh(res_g1, g_gpu)
                drv.memcpy_dtoh(res_b1, b_gpu)
                r_gpu.free()
                g_gpu.free()
                b_gpu.free()
                img_r_gpu.free()
                img_g_gpu.free()
                img_b_gpu.free()
                img_grey_gpu.free()
                
            image_traitee = cv2.merge((res_b1,res_g1,res_r1))

            if flag_DEMO == 1 :
                image_traitee[0:height,0:width//2] = image_brute[0:height,0:width//2]               
    else :
        if flag_noir_blanc == 0 and flag_colour_camera == True:        
            image_traitee = cv2.merge((res_b1,res_g1,res_r1))
        else :
            image_traitee = res_b1
        
    stop_time_test = cv2.getTickCount()
    time_exec_test= int((stop_time_test-start_time_test)/cv2.getTickFrequency()*1000)
    labelInfo2.config(text = str(time_exec_test) + " ms       ")
        
    if flag_TIP == True :
        if flag_noir_blanc == 0 and flag_colour_camera == True:
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
        calque_TIP =np.zeros_like(image_traitee)
        texte_TIP = Date_hour_image + "  Frame nbr : " + str(numero_image)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(calque_TIP, texte_TIP, (pos,pos), font, size, (255, 255, 255), 1, cv2.LINE_AA)

    if flag_mountpos == True :
        if flag_noir_blanc == 0 and flag_colour_camera == True:
            height,width,layers = image_traitee.shape
        else :
            height,width = image_traitee.shape 
        if width > 2000 :
            size = 1
        else :
            size = 0.5
            pos = 20
        if width < 600 :
            size = 0.3
            pos = 15
        if flag_TIP == False :
            calque_TIP =np.zeros_like(image_traitee)
        font = cv2.FONT_HERSHEY_SIMPLEX
#        mount_info()
        posX = width // 2 -100
        posY = 30
        cv2.putText(calque_TIP, str(azimut), (posX,posY), font, size, (255, 0, 0), 1, cv2.LINE_AA)
        posX = width - 100
        posY = height // 2
        cv2.putText(calque_TIP, str(hauteur), (posX,posY), font, size, (255, 0, 0), 1, cv2.LINE_AA)
    flag_filtre_work = False 
             

def mount_info() :
    global azimut,hauteur,flag_mount_connect
    pass
                        
def pic_capture() :
    global start,nb_pic_cap,nb_acq_pic,labelInfo1,flag_cap_pic,nb_cap_pic,image_path,image_traitee,timer1
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
    
def start_pic_capture() :
    global nb_acq_pic,flag_cap_pic,nb_cap_pic,start,timer1
    flag_cap_pic = True
    nb_cap_pic =1
    start = datetime.now()
    timer1 = time.time()
 
def stop_pic_capture() :
    global nb_cap_pic
    nb_cap_pic = val_nb_captures +1

def video_capture() :
    global image_traitee,start_video,nb_cap_video,nb_acq_video,labelInfo1,flag_cap_video,video_path,val_nb_capt_video,video,echelle11,flag_pause_video,timer1,val_deltat,FDIF,flag_FDIF
    if nb_cap_video == 1 :
        if flag_HQ == 0:
            fourcc = cv2.VideoWriter_fourcc(*'XVID') # video compressée
        else :
            fourcc = 0 # video RAW
            #fourcc = cv2.VideoWriter_fourcc(*'DIB ') # video non compressee
        if flag_filter_wheel == True:
            nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.avi'
        else :
            nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
        if image_traitee.ndim == 3 :
            height,width,layers = image_traitee.shape
            if flag_HQ == 0 :
                video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor = True) # video compressée
            else :
                video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor=True) # video RAW
        else :
            height,width = image_traitee.shape
            if flag_HQ == 0 :
                video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor = False) # video compressée
            else :
                video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor=False) # video RAW
        labelInfo1.config(text = "                                                                                                        ") 
        labelInfo1.config(text = " Acquisition vidéo en cours")
    if flag_pause_video == False :
        if time.time() - timer1 >= val_deltat :       
            if nb_cap_video <= val_nb_capt_video :
                if (flag_FDIF == 1 and FDIF > 0) or flag_FDIF == 0 :
                    FDIF = FDIF - 1
                    #print(FDIF)
                    if image_traitee.ndim == 3 :
                        video.write(cv2.cvtColor(image_traitee, cv2.COLOR_BGR2RGB))
                    else :
                        video.write(image_traitee)
                    if (nb_cap_video % 10) == 0 :    
                        labelInfo1.config(text = " frame : " + str (nb_cap_video) + "                            ")
                    nb_cap_video += 1
            else :
                FDIF = 0
                video.release()
                flag_cap_video = False
                labelInfo1.config(text = " Acquisition vidéo terminee     ")
            timer1 = time.time()
    else :
        labelInfo1.config(text = " PAUSE VIDEO       ")
        
def start_video_capture() :
    global nb_cap_video,flag_cap_video,start_video,val_nb_capt_video,timer1,compteur_images,numero_image
    flag_cap_video = True
    nb_cap_video =1
    if val_nb_capt_video == 0 :
        val_nb_capt_video = 10000
    start_video = datetime.now()
    timer1 = time.time()
    compteur_images = 0
    numero_image = 0
    
 
def stop_video_capture() :
    global nb_cap_video,val_nb_capt_video,FDIF
    FDIF = 0
    nb_cap_video = val_nb_capt_video +1
 
def pause_video_capture() :
    global flag_pause_video
    if flag_pause_video == True :
        flag_pause_video = False
    else :
        flag_pause_video = True

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
    

# Fonctions récupération des paramètres grace aux scalebars
def mode_acq_rapide() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,timeoutexp,\
           frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark

    flag_acq_rapide = True
    flag_sub_dark = False
    dispo_dark = 'Dark NO        '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    flag_stop_acquisition=True
    time.sleep(1)
    exp_min=100 #µs
    exp_max=10000 #µs
    exp_delta=100 #µs
    exp_interval=2000 #µs
    val_exposition=exp_min
    labelParam1.config(text = "Exposition µs")
    echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 310, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
    echelle1.set (val_exposition)
    echelle1.place(anchor="w", x=xS1,y=yS1)
    exposition = val_exposition
    camera.set_control_value(asi.ASI_EXPOSURE, exposition)
    timeoutexp = (exposition / 1000) * 2 + 500
    camera.default_timeout = timeoutexp
    camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,1)
    time.sleep(0.1)
    flag_stop_acquisition=False
 
def mode_acq_medium() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,\
           val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark
    flag_acq_rapide = False
    flag_sub_dark = False
    dispo_dark = 'Dark NO        '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    flag_stop_acquisition=True
    time.sleep(1)
    exp_min=1 #ms
    exp_max=500 #ms
    exp_delta=1 #ms
    exp_interval=50
    val_exposition=exp_min
    labelParam1.config(text = "Exposition ms")
    echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 310, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
    echelle1.set (val_exposition)
    echelle1.place(anchor="w", x=xS1,y=yS1)
    exposition = val_exposition * 1000
    camera.set_control_value(asi.ASI_EXPOSURE, exposition)
    timeoutexp = val_exposition * 2 + 500
    camera.default_timeout = timeoutexp
    camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
    time.sleep(0.1)
    flag_stop_acquisition=False
 
def mode_acq_lente() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,\
           val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark
    flag_acq_rapide = False
    flag_sub_dark = False
    dispo_dark = 'Dark NO        '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    flag_stop_acquisition=True
    time.sleep(1)
    exp_min=500 #ms
    exp_max=20000 #ms
    exp_delta=500 #ms
    exp_interval=5000
    val_exposition=exp_min
    labelParam1.config(text = "Exposition ms")
    echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 310, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
    echelle1.set (val_exposition)
    echelle1.place(anchor="w", x=xS1,y=yS1)
    exposition = val_exposition * 1000
    camera.set_control_value(asi.ASI_EXPOSURE, exposition)
    timeoutexp = val_exposition * 2 + 500
    camera.default_timeout = timeoutexp
    camera.set_control_value(asi.ASI_HIGH_SPEED_MODE,0)
    time.sleep(0.1)
    flag_stop_acquisition=False

def valeur_exposition (event=None) :
    global timeoutexp,timeout_val,flag_acq_rapide,camera,val_exposition,echelle1,val_resolution,flag_stop_acquisition,exposition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark
    flag_sub_dark = False
    dispo_dark = 'Dark NO        '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    if flag_autoexposure_exposition == False :
        flag_stop_acquisition=True
        val_exposition = echelle1.get()
        if flag_acq_rapide == True :
            exposition = val_exposition
        else :
            exposition = val_exposition*1000
        camera.set_control_value(asi.ASI_EXPOSURE, exposition)
        if flag_acq_rapide == True :
            timeoutexp = (exposition / 1000) * 2 + 500
        else :
            timeoutexp =  val_exposition * 2 + 500
        camera.default_timeout = timeoutexp
        time.sleep(0.05)
        flag_stop_acquisition=False


def valeur_gain (event=None) :
    global camera,val_gain,echelle2,flag_stop_acquisition,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark
    flag_sub_dark = False
    dispo_dark = 'Dark NO        '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    if flag_autoexposure_gain == False :
        val_gain = echelle2.get()
        camera.set_control_value(asi.ASI_GAIN, val_gain)


def choix_BIN1(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,mode_BIN,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video

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
        labelInfoDark.place(anchor="w", x=xdark,y=ydark)
        time.sleep(0.5)
        echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
        val_resolution = 1
        echelle3.set (val_resolution)
        echelle3.place(anchor="w", x=xS3,y=yS3)
        time.sleep(0.1)
        mode_BIN = 1
        choix_resolution_camera()
        flag_stop_acquisition=False

def choix_BIN2(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,mode_BIN,labelInfoDark,flag_sub_dark,dispo_dark,choix_sub_dark,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video

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
        labelInfoDark.place(anchor="w", x=xdark,y=ydark)
        time.sleep(0.5)
        echelle3 = Scale (cadre, from_ = 1, to = 7, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
        val_resolution = 1
        echelle3.set (val_resolution)
        echelle3.place(anchor="w", x=xS3,y=yS3)
        mode_BIN = 2
        choix_resolution_camera()
        flag_stop_acquisition=False
    
def choix_resolution_camera(event=None) :
    global val_delai,val_FS,camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,res_x_max,res_y_max,echelle3,\
           flag_image_disponible,flag_acquisition_en_cours,flag_stop_acquisition,labelInfoDark,flag_sub_dark,dispo_dark,\
           choix_sub_dark,flag_nouvelle_resolution,tnr,inSize,backend,choix_TIP,flag_TIP,flag_cap_video
    
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
        labelInfoDark.place(anchor="w", x=xdark,y=ydark)
        time.sleep(0.1)
        val_resolution = echelle3.get()
        if mode_BIN == 1 :
            res_cam_x = RES_X_BIN1[val_resolution-1]
            res_cam_y = RES_Y_BIN1[val_resolution-1]
            val_delai = DELAY_BIN1[val_resolution-1]
        if mode_BIN == 2 :
            res_cam_x = RES_X_BIN2[val_resolution-1]
            res_cam_y = RES_Y_BIN2[val_resolution-1]
            val_delai = DELAY_BIN2[val_resolution-1]
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

def commande_flipV() :
    global FlipV,val_FS,type_debayer
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

def commande_flipH() :
    global FlipH,val_FS,type_debayer
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

def commande_img_Neg() :
    global ImageNeg,trsf_r,trsf_g,trsf_b
    if choix_img_Neg.get() == 0 :
        ImageNeg = 0
    else :
        ImageNeg = 1
        
def commande_2DConvol() :
    global flag_2DConv
    if choix_2DConv.get() == 0 :
        flag_2DConv = 0
    else :
        flag_2DConv = 1
                
def commande_bilateral() :
    global flag_bilateral
    if choix_bilateral.get() == 0 :
        flag_bilateral = 0
    else :
        flag_bilateral = 1

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
        
def commande_unsharp_mask() :
    global flag_unsharp_mask
    if choix_unsharp_mask.get() == 0 :
        flag_unsharp_mask = 0
    else :
        flag_unsharp_mask = 1

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

def commande_DEMO() :
    global flag_DEMO
    if choix_DEMO.get() == 0 :
        flag_DEMO = 0
    else :
        flag_DEMO = 1

def commande_FDIF() :
    global flag_FDIF,flag_nouvelle_resolution
    if choix_FDIF.get() == 0 :
        flag_FDIF = 0
    else :
        flag_FDIF = 1
        flag_nouvelle_resolution = True
        
def commande_denoise_stacking_Paillou() :
    global flag_denoise_stacking_Paillou,val_FSDN,compteur_FSDN,Im1fsdnOK,Im2fsdnOK
    if choix_denoise_stacking_Paillou.get() == 0 :
        flag_denoise_stacking_Paillou = False
    else :
        flag_denoise_stacking_Paillou = True
        val_FSDN = 1
        compteur_FSDN = 0
        Im1fsdnOK = False
        Im2fsdnOK = False

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

def choix_library_opencv(event=None):
    global compute_library
    compute_library = "opencv"
    print("OPENCV")


def choix_library_cupy(event=None):
    global compute_library,type_library
    if flag_cupy == True :
        compute_library = "cupy"
        print("CUPY")
    else :
        compute_library = "opencv"
        type_library.set(1)
        print("OPENCV")

def choix_library_vpi(event=None):
    global compute_library,type_library
    if flag_vpi == True :
        compute_library = "vpi"
        print("VPI")
    else :
        compute_library = "opencv"
        type_library.set(1)
        print("OPENCV")
    
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
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
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
        mode_acq_medium()
        camera.set_control_value(asi.ASI_EXPOSURE,controls['Exposure']['DefaultValue'],auto=True)
        camera.set_control_value(controls['AutoExpMaxExpMS']['ControlType'], 500)

def commande_autogain() :
    global flag_autoexposure_exposition,flag_autoexposure_gain,camera,controls,val_gain,labelInfoDark,flag_sub_dark,dispo_dark
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
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
    reset_FS()
    reset_general_FS()
    flag_stop_acquisition=True
    dispo_dark = 'Dark NON dispo '
    choix_sub_dark.set(0)
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark,y=ydark)
    flag_restore_filtrage = False
    if flag_filtrage_ON == True :
        flag_filtrage_ON == False
        flag_restore_filtrage = True
        time.sleep(1)
    if choix_noir_blanc.get() == 0 and flag_colour_camera == True :
        flag_noir_blanc = 0
        camera.stop_video_capture()
        time.sleep(0.1)
        format_capture = asi.ASI_IMG_RAW8
        camera.set_image_type(format_capture)
        time.sleep(0.1)
        camera.start_video_capture()
    else :
        flag_noir_blanc = 1
        if choix_noir_blanc_estime.get() == 0 :
            flag_NB_estime = 0
        else :
            flag_NB_estime = 1
        if flag_colour_camera == True :
            camera.stop_video_capture()
            time.sleep(0.1)
            if flag_NB_estime == 1 :
                format_capture = asi.ASI_IMG_RAW8
            else :
                format_capture = asi.ASI_IMG_Y8
            camera.set_image_type(format_capture)
            time.sleep(0.1)
            camera.start_video_capture()
        else :
            time.sleep(0.1)
            format_capture = asi.ASI_IMG_RAW8
            camera.set_image_type(format_capture)
            time.sleep(0.1)
            camera.start_video_capture()
    flag_stop_acquisition = False
    time.sleep(1)
    if flag_restore_filtrage == True :
        flag_filtrage_ON = True


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
    val_red=echelle14.get()
    camera.set_control_value(asi.ASI_WB_R, val_red)
    
def choix_w_blue(event=None) :
    global val_blue, echelle15
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

def choix_AGR(event=None) :
    global val_AGR, echelle61
    val_AGR=echelle61.get()

def choix_USB(event=None) :
    global val_USB, echelle50,camera
    val_USB=echelle50.get()
    camera.set_control_value(asi.ASI_BANDWIDTHOVERLOAD, val_USB)

def choix_position_EFW0(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 0
        filter_wheel.set_position(fw_position)

def choix_position_EFW1(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 1
        filter_wheel.set_position(fw_position)

def choix_position_EFW2(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 2
        filter_wheel.set_position(fw_position)

def choix_position_EFW3(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 3
        filter_wheel.set_position(fw_position)

def choix_position_EFW4(event=None) :
    global fw_position
    if flag_filter_wheel == True :
        fw_position = 4
        filter_wheel.set_position(fw_position)

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
           flag_denoise_stacking_Paillou,val_FSDN,compteur_FSDN,Im1fsdnOK,Im2fsdnOK
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    echelle20.set(val_FS)
    choix_denoise_stacking_Paillou.set(0)
    flag_denoise_stacking_Paillou = False
    compteur_FSDN = 0
    Im1fsdnOK = False
    Im2fsdnOK = False

    
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

        
def commande_TRKSAT() :
    global flag_TRKSAT,flag_nouvelle_resolution
    if choix_TRKSAT.get() == 0 :
        flag_TRKSAT = 0
    else :
        flag_TRKSAT = 1
        flag_nouvelle_resolution = True

def commande_TRKMET() :
    global flag_TRKMET,flag_nouvelle_resolution
    if choix_TRKMET.get() == 0 :
        flag_TRKMET = 0
    else :
        flag_TRKMET = 1
        flag_nouvelle_resolution = True

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
    global compteur_images,numero_image
    compteur_images = 0
    numero_image = 0

def start_cap_dark() :
    global timeoutexp,val_FS,camera,flag_sub_dark,dispo_dark,labelInfoDark,flag_stop_acquisition,flag_acquisition_en_cours,labelInfo1,val_nb_darks,text_info1,xLI1,yLI1,Master_Dark
    reset_FS()
    if askyesno("Cover the Lens", "Dark acquisition continue ?") :
        flag_stop_acquisition=True
        text_info1 = "Initialisation capture DARK"
        labelInfo1 = Label (cadre, text = text_info1)
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
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
#                    dark_tempo= dark_tempo + np.uint32(dark_tmp)
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
                    Master_Dark = cv2.cvtColor(Mean_dark, cv2.COLOR_BGR2RGB)
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
        labelInfoDark.place(anchor="w", x=xdark,y=ydark-5)
        flag_stop_acquisition=False
        camera.start_video_capture()
        time.sleep(0.1)
    else :
        print("on arrete")
        flag_sub_dark = False
        labelInfoDark = Label (cadre, text = "Dark OK")
        labelInfoDark.place(anchor="w", x=xdark,y=ydark-5)

def raz_tracking() :
    global flag_nouvelle_resolution
    flag_nouvelle_resolution = True

def stop_tracking() :
    global flag_TRKSAT,flag_FDIF,flag_nouvelle_resolution,choix_FDIF,choix_TRKSAT,choix_TRKMET
    flag_TRKSAT = 0
    flag_FDIF = 0    
    flag_nouvelle_resolution = True
    choix_FDIF.set(0)
    choix_TRKSAT.set(0)
    choix_TRKMET.set(0)

def choix_CLAHE_DAYLLIGHT(event=None):
    global flag_type_clahe
    flag_type_clahe = 1

def choix_CLAHE_STARS(event=None):
    global flag_type_clahe
    flag_type_clahe = 2


init_camera()


# initialisation des boites scrolbar, buttonradio et checkbutton

xS3=1700 # Choix résolution
yS3=95

xS1=1510
yS1=325


###########################
#     VARIOUS WIDGETS     #
###########################


# Flip Vertical
CBFV = Checkbutton(cadre,text="Flip V", variable=choix_flipV,command=commande_flipV,onvalue = 1, offvalue = 0)
CBFV.place(anchor="w",x=5, y=30)

# Flip Horizontal
CBFH = Checkbutton(cadre,text="Flip H", variable=choix_flipH,command=commande_flipH,onvalue = 1, offvalue = 0)
CBFH.place(anchor="w",x=5, y=60)

# Text in picture
CBTIP = Checkbutton(cadre,text="TIP", variable=choix_TIP,command=commande_TIP,onvalue = 1, offvalue = 0)
CBTIP.place(anchor="w",x=5, y=95)

# Mount
CBMNT = Checkbutton(cadre,text="AZ/H", variable=choix_mount,command=commande_mount,onvalue = 1, offvalue = 0)
CBMNT.place(anchor="w",x=5, y=115)

# Cross
CBCR = Checkbutton(cadre,text="Cr", variable=choix_cross,command=commande_cross,onvalue = 1, offvalue = 0)
CBCR.place(anchor="w",x=5, y=135)


# Histogram
CBHST = Checkbutton(cadre,text="Hst", variable=choix_HST,command=commande_HST,onvalue = 1, offvalue = 0)
CBHST.place(anchor="w",x=5, y=165)

# Histogram
CBTRSF = Checkbutton(cadre,text="Trsf", variable=choix_TRSF,command=commande_TRSF,onvalue = 1, offvalue = 0)
CBTRSF.place(anchor="w",x=5, y=195)

# Affichage fonction de transfert amplification soft
CBTRGS = Checkbutton(cadre,text="TrGS", variable=choix_TRGS,command=commande_TRGS,onvalue = 1, offvalue = 0)
CBTRGS.place(anchor="w",x=5, y=220)

# DEMO
CBDEMO = Checkbutton(cadre,text="Demo", variable=choix_DEMO,command=commande_DEMO,onvalue = 1, offvalue = 0)
CBDEMO.place(anchor="w",x=5, y=890)

# Libraries OpenCV, PyTorch or VPI
labellibraries = Label (cadre, text = "Libraries")
labellibraries.place (anchor="w",x=5, y=920)
RBTL1 = Radiobutton(cadre,text="OpenCV", variable=type_library,command=choix_library_opencv,value=1)
RBTL1.place(anchor="w",x=5, y=940)
RBTL2 = Radiobutton(cadre,text="Cupy", variable=type_library,command=choix_library_cupy,value=2)
RBTL2.place(anchor="w",x=5, y=960)
RBTL3 = Radiobutton(cadre,text="VPI", variable=type_library,command=choix_library_vpi,value=3)
RBTL3.place(anchor="w",x=5, y=980)


labelTRK = Label (cadre, text = "Tracking")
labelTRK.place (anchor="w",x=0, y=700)

# Find Stars
CBFDIF = Checkbutton(cadre,text="Stars", variable=choix_FDIF,command=commande_FDIF,onvalue = 1, offvalue = 0)
CBFDIF.place(anchor="w",x=0, y=720)

# Track Satellites
CBTRKSAT = Checkbutton(cadre,text="Satellit", variable=choix_TRKSAT,command=commande_TRKSAT,onvalue = 1, offvalue = 0)
CBTRKSAT.place(anchor="w",x=0, y=740)

# Track Meteor
CBTRKMET = Checkbutton(cadre,text="Meteor", variable=choix_TRKMET,command=commande_TRKMET,onvalue = 1, offvalue = 0)
CBTRKMET.place(anchor="w",x=0, y=760)

Button7 = Button (cadre,text = "RAZ Trk", command = raz_tracking,padx=10,pady=0)
Button7.place(anchor="w", x=0,y=790)

# RAZ frame counter
Button10 = Button (cadre,text = "RZ Fr Cnt", command = raz_framecount,padx=10,pady=0)
Button10.place(anchor="w", x=5,y=250)


# Saturation
CBSAT = Checkbutton(cadre,text="SAT", variable=choix_SAT,command=commande_SAT,onvalue = 1, offvalue = 0)
CBSAT.place(anchor="w",x=5,y=420)
echelle70 = Scale (cadre, from_ = 0, to = 30, command= choix_val_SAT, orient=VERTICAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle70.set(val_SAT)
echelle70.place(anchor="c", x=15,y=485)

# Choix filtrage ON
CBF = Checkbutton(cadre,text="Filters ON", variable=choix_filtrage_ON,command=commande_filtrage_ON,onvalue = 1, offvalue = 0)
CBF.place(anchor="w",x=1450, y=50)

# Fulres displaying
CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=1525, y=50)

# Choix forcage N&B
CBFNB = Checkbutton(cadre,text="Set B&W", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=1590, y=50)

# Choix forcage N&B Estimate
CBFNBE = Checkbutton(cadre,text="B&W Est", variable=choix_noir_blanc_estime,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNBE.place(anchor="w",x=1590, y=65)


# Stacking Mode
RBSM1 = Radiobutton(cadre,text="MEAN", variable=choix_stacking,command=choix_mean_stacking,value=1)
RBSM1.place(anchor="w",x=1695, y=20)
RBSM2 = Radiobutton(cadre,text="SUM", variable=choix_stacking,command=choix_sum_stacking,value=2)
RBSM2.place(anchor="w",x=1760, y=20)
# Number frames stacked
labelParam20 = Label (cadre, text = "# FS")
labelParam20.place(anchor="w", x=1675,y=50)
echelle20 = Scale (cadre, from_ = 1, to = 5, command= choix_FS, orient=HORIZONTAL, length = 120, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle20.set(val_FS)
echelle20.place(anchor="w", x=1700,y=50)


###########################
#   EXPOSITION SETTINGS   #
###########################

# Speed mode acquisition
labelMode_Acq = Label (cadre, text = "Speed Mode")
labelMode_Acq.place (anchor="w",x=1450, y=280)
RBMA1 = Radiobutton(cadre,text="Fast", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=1510, y=280)
RBMA2 = Radiobutton(cadre,text="Medium", variable=mode_acq,command=mode_acq_medium,value=2)
RBMA2.place(anchor="w",x=1560, y=280)
RBMA3 = Radiobutton(cadre,text="Slow", variable=mode_acq,command=mode_acq_lente,value=3)
RBMA3.place(anchor="w",x=1620, y=280)

# Choix du mode BINNING - 1, 2 ou 3
labelBIN = Label (cadre, text = "BIN : ")
labelBIN.place (anchor="w",x=1450, y=95)
RBB1 = Radiobutton(cadre,text="BIN1", variable=choix_bin,command=choix_BIN1,value=1)
RBB1.place(anchor="w",x=1475, y=95)
RBB2 = Radiobutton(cadre,text="BIN2", variable=choix_bin,command=choix_BIN2,value=2)
RBB2.place(anchor="w",x=1520, y=95)

# Choix Hardware Bin
CBHB = Checkbutton(cadre,text="HB", variable=choix_hard_bin,command=commande_hard_bin,onvalue = 1, offvalue = 0)
CBHB.place(anchor="w",x=1565, y=95)

# Choix HDR
CBHDR = Checkbutton(cadre,text="HDR", variable=choix_HDR,command=commande_HDR,onvalue = 1, offvalue = 0)
CBHDR.place(anchor="w",x=1620, y=95)


# Resolution setting
labelParam3 = Label (cadre, text = "RES : ")
labelParam3.place(anchor="w", x=1665,y=95)
echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 120, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle3.set (val_resolution)
echelle3.place(anchor="w", x=xS3,y=yS3)

# Automatic exposition time
CBOG = Checkbutton(cadre,text="Auto", variable=choix_autogain,command=commande_autogain,onvalue = 1, offvalue = 0)
CBOG.place(anchor="w",x=1440, y=120)


# choix hold picture
CBOP = Checkbutton(cadre,text="Hold Picture", variable=choix_hold_picture,command=commande_hold_picture,onvalue = 1, offvalue = 0)
CBOP.place(anchor="w",x=1700, y=280)

# Automatic exposition time
CBOE = Checkbutton(cadre,text="Auto", variable=choix_autoexposure,command=commande_autoexposure,onvalue = 1, offvalue = 0)
CBOE.place(anchor="w",x=1440, y=305)

# Exposition setting
labelParam1 = Label (cadre, text = "Exposition ms")
labelParam1.place(anchor="w", x=1450,y=325)
echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 310, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
echelle1.set(val_exposition)
echelle1.place(anchor="w", x=1510,y=325)


labelFW = Label (cadre, text = "FW :")
labelFW.place (anchor="w",x=5, y=555)


# FW
RBEFW1 = Radiobutton(cadre,text="#1", variable=fw_position,command=choix_position_EFW0,value=1)
RBEFW1.place(anchor="w",x=5, y=580)
RBEFW2 = Radiobutton(cadre,text="#2", variable=fw_position,command=choix_position_EFW1,value=2)
RBEFW2.place(anchor="w",x=5, y=600)
RBEFW3 = Radiobutton(cadre,text="#3", variable=fw_position,command=choix_position_EFW2,value=3)
RBEFW3.place(anchor="w",x=5, y=620)
RBEFW4 = Radiobutton(cadre,text="#4", variable=fw_position,command=choix_position_EFW3,value=4)
RBEFW4.place(anchor="w",x=5, y=640)
RBEFW5 = Radiobutton(cadre,text="#5", variable=fw_position,command=choix_position_EFW4,value=5)
RBEFW5.place(anchor="w",x=5, y=660)


labelParam2 = Label (cadre, text = "Gain")
labelParam2.place(anchor="w", x=1450,y=140)
echelle2 = Scale (cadre, from_ = 0, to = val_maxgain , command= valeur_gain, orient=HORIZONTAL, length = 330, width = 10, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle2.set(val_gain)
echelle2.place(anchor="w", x=1490,y=140)

# Bandwidth USB
labelParam50 = Label (cadre, text = "USB")
labelParam50.place(anchor="w", x=15,y=290)
echelle50 = Scale (cadre, from_ = 0, to = 100, command= choix_USB, orient=VERTICAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle50.set(val_USB)
echelle50.place(anchor="c", x=15,y=350)


# Signal amplification soft
CBAS = Checkbutton(cadre,text="Amplif Soft", variable=choix_AmpSoft,command=commande_AmpSoft,onvalue = 1, offvalue = 0)
CBAS.place(anchor="w",x=1450, y=185)
echelle80 = Scale (cadre, from_ = 0, to = 20.0, command= choix_amplif, orient=HORIZONTAL, length = 280, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle80.set(val_ampl)
echelle80.place(anchor="w", x=1540,y=185)

RBMuRo1 = Radiobutton(cadre,text="Lin", variable=mode_Lin_Gauss,command=mode_Lineaire,value=1)
RBMuRo1.place(anchor="w",x=1440, y=230)
RBMuRo2 = Radiobutton(cadre,text="Gauss", variable=mode_Lin_Gauss,command=mode_Gauss,value=2)
RBMuRo2.place(anchor="w",x=1480, y=230)
RBMuRo3 = Radiobutton(cadre,text="Stars", variable=mode_Lin_Gauss,command=mode_Stars,value=3)
RBMuRo3.place(anchor="w",x=1440, y=250)

labelParam82 = Label (cadre, text = "µX") # choix Mu X
labelParam82.place(anchor="w", x=1533,y=230)
echelle82 = Scale (cadre, from_ = -5.0, to = 5.0, command= choix_Mu, orient=HORIZONTAL, length = 120, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle82.set(val_Mu)
echelle82.place(anchor="w", x=1553,y=230)

labelParam84 = Label (cadre, text = "Ro") # choix Mu X
labelParam84.place(anchor="w", x=1685,y=230)
echelle84 = Scale (cadre, from_ = 0.5, to = 5.0, command= choix_Ro, orient=HORIZONTAL, length = 120, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle84.set(val_Ro)
echelle84.place(anchor="w", x=1700,y=230)


# Camera Red balance
labelParam14 = Label (cadre, text = "CRed") # choix balance rouge
labelParam14.place(anchor="w", x=1450,y=370)
echelle14 = Scale (cadre, from_ = 1, to = 99, command= choix_w_red, orient=HORIZONTAL, length = 140, width = 10, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle14.set(val_red)
echelle14.place(anchor="w", x=1485,y=370)

# Camera Blue balance
labelParam15 = Label (cadre, text = "CBlue") # choix balance bleue
labelParam15.place(anchor="w", x=1645,y=370)
echelle15 = Scale (cadre, from_ = 1, to = 99, command= choix_w_blue, orient=HORIZONTAL, length = 140, width = 10, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle15.set(val_blue)
echelle15.place(anchor="w", x=1680,y=370)

# Software Red balance
labelParam100 = Label (cadre, text = "R") # choix balance rouge
labelParam100.place(anchor="w", x=1440,y=415)
echelle100 = Scale (cadre, from_ = 60, to = 140, command= choix_w_reds, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle100.set(val_reds)
echelle100.place(anchor="w", x=1440+15,y=415)

# Software Green balance
labelParam101 = Label (cadre, text = "G") # choix balance rouge
labelParam101.place(anchor="w", x=1570,y=415)
echelle101 = Scale (cadre, from_ = 60, to = 140, command= choix_w_greens, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle101.set(val_greens)
echelle101.place(anchor="w", x=1570+15,y=415)

# Software Blue balance
labelParam102 = Label (cadre, text = "B") # choix balance bleue
labelParam102.place(anchor="w", x=1700,y=415)
echelle102 = Scale (cadre, from_ = 60, to = 140, command= choix_w_blues, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=40,sliderlength=20)
echelle102.set(val_blues)
echelle102.place(anchor="w", x=1700+15,y=415)


###########################
# SHARPEN DENOISE WIDGETS #
###########################

# Choix Bilateral filter
CBBL = Checkbutton(cadre,text="Bilat OpenCV / VPI Noise remove", variable=choix_bilateral,command=commande_bilateral,onvalue = 1, offvalue = 0)
CBBL.place(anchor="w",x=1610, y=455)

# Choix 2D convolution filter
CB2DC = Checkbutton(cadre,text="2D convol", variable=choix_2DConv,command=commande_2DConvol,onvalue = 1, offvalue = 0)
CB2DC.place(anchor="w",x=1760, y=455)

CBSS1 = Checkbutton(cadre,text="Sharpen", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=1450, y=455)

CBUSM = Checkbutton(cadre,text="Unsharp Mask", variable=choix_unsharp_mask,command=commande_unsharp_mask,onvalue = 1, offvalue = 0)
CBUSM.place(anchor="w",x=1520, y=455)

# Choix filtre Denoise Paillou
CBEPF = Checkbutton(cadre,text="Dn Paillou", variable=choix_denoise_Paillou,command=commande_denoise_Paillou,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1450, y=485)

# Choix filtre Denoise Paillou AADF
CBEPFS = Checkbutton(cadre,text="AADF", variable=choix_denoise_stacking_Paillou,command=commande_denoise_stacking_Paillou,onvalue = 1, offvalue = 0)
CBEPFS.place(anchor="w",x=1520, y=485)
# AADF Mode
RBAADP1 = Radiobutton(cadre,text="High", variable=choix_dyn_AADP,command=choix_dyn_high,value=1)
RBAADP1.place(anchor="w",x=1565, y=485)
RBAADP2 = Radiobutton(cadre,text="Low", variable=choix_dyn_AADP,command=choix_dyn_low,value=2)
RBAADP2.place(anchor="w",x=1610, y=485)
# AADF ghost reducer
CBGR = Checkbutton(cadre,text="GR", variable=choix_ghost_reducer,command=commande_ghost_reducer,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1650, y=485)
echelle130 = Scale (cadre, from_ = 20, to = 70, command= choix_val_ghost_reducer, orient=HORIZONTAL, length = 130, width = 10, resolution = 2, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle130.set(val_ghost_reducer)
echelle130.place(anchor="w", x=1690,y=475)


# Choix filtre Denoise KNN
CBEPF = Checkbutton(cadre,text="Dn KNN", variable=choix_denoise_KNN,command=choix_KNN,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1450, y=520)
echelle30 = Scale (cadre, from_ = 0.05, to = 1.2, command= choix_val_KNN, orient=HORIZONTAL, length = 110, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle30.set(val_denoise_KNN)
echelle30.place(anchor="w", x=1510,y=520)

# Choix filtre Denoise NLM2
CBDS = Checkbutton(cadre,text="Dn NLM2", variable=choix_denoise_soft,command=commande_denoise_soft,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=1645, y=520)
echelle4 = Scale (cadre, from_ = 0.1, to = 1.2, command= choix_valeur_denoise, orient=HORIZONTAL, length = 110, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle4.set(val_denoise)
echelle4.place(anchor="w", x=1710,y=520)


#####################
# HISTOGRAM WIDGETS #
#####################

# Choix filtre Gradient Removal
CBGR = Checkbutton(cadre,text="Grad/Vignet", variable=choix_GR,command=commande_GR,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1450, y=560)

# Choix du mode gradient ou vignetting
RBGV1 = Radiobutton(cadre,text="Gradient", variable=gradient_vignetting,command=mode_gradient,value=1)
RBGV1.place(anchor="w",x=1450, y=590)
RBGV2 = Radiobutton(cadre,text="Vignetting", variable=gradient_vignetting,command=mode_vignetting,value=2)
RBGV2.place(anchor="w",x=1510, y=590)

# Choix Parametre Seuil Gradient Removal
echelle60 = Scale (cadre, from_ = 0, to = 100, command= choix_SGR, orient=HORIZONTAL, length = 160, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle60.set(val_SGR)
echelle60.place(anchor="w", x=1530,y=560)

# Choix Parametre Atenuation Gradient Removal
labelParam61 = Label (cadre, text = "At")
labelParam61.place(anchor="e", x=1720,y=560)
echelle61 = Scale (cadre, from_ = 0, to = 100, command= choix_AGR, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle61.set(val_AGR)
echelle61.place(anchor="w", x=1720,y=560)

# Choix du mode image en négatif
CBIN = Checkbutton(cadre,text="Img Neg", variable=choix_img_Neg,command=commande_img_Neg,onvalue = 1, offvalue = 0)
CBIN.place(anchor="w",x=1450, y=625)

# Histogram equalize
CBHE2 = Checkbutton(cadre,text="Histo Eq", variable=choix_histogram_equalize2,command=commande_histogram_equalize2,onvalue = 1, offvalue = 0)
CBHE2.place(anchor="w",x=1520, y=625)
echelle16 = Scale (cadre, from_ = 0.3, to = 4, command= choix_heq2, orient=HORIZONTAL, length = 240, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle16.set(val_heq2)
echelle16.place(anchor="w", x=1580,y=625)

# Choix histogramme stretch
CBHS = Checkbutton(cadre,text="Histo Stretch", variable=choix_histogram_stretch,command=commande_histogram_stretch,onvalue = 1, offvalue = 0)
CBHS.place(anchor="w",x=1450, y=670)

labelParam5 = Label (cadre, text = "Min") # choix valeur histogramme strech minimum
labelParam5.place(anchor="w", x=1555,y=670)
echelle5 = Scale (cadre, from_ = 0, to = 150, command= choix_histo_min, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle5.set(val_histo_min)
echelle5.place(anchor="w", x=1580,y=670)

labelParam6 = Label (cadre, text = "Max") # choix valeur histogramme strech maximum
labelParam6.place(anchor="w", x=1700,y=670)
echelle6 = Scale (cadre, from_ = 155, to = 255, command= choix_histo_max, orient=HORIZONTAL, length = 100, width = 10, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle6.set(val_histo_max)
echelle6.place(anchor="w", x=1720,y=670)

#Choix histogramme Sigmoide
CBHPT = Checkbutton(cadre,text="Histo Sigmoide", variable=choix_histogram_phitheta,command=commande_histogram_phitheta,onvalue = 1, offvalue = 0)
CBHPT.place(anchor="w",x=1450, y=715)

labelParam12 = Label (cadre, text = "Pnt") # choix valeur histogramme Signoide param 1
labelParam12.place(anchor="w", x=1555,y=715)
echelle12 = Scale (cadre, from_ = 0.5, to = 3, command= choix_phi, orient=HORIZONTAL, length = 100, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle12.set(val_phi)
echelle12.place(anchor="w", x=1580,y=715)

labelParam13 = Label (cadre, text = "Dec") # choix valeur histogramme Signoide param 2
labelParam13.place(anchor="w", x=1700,y=715)
echelle13 = Scale (cadre, from_ = 50, to = 200, command= choix_theta, orient=HORIZONTAL, length = 100, width = 10, resolution = 2, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle13.set(val_theta)
echelle13.place(anchor="w", x=1720,y=715)

# Choix contrast CLAHE
CBCC = Checkbutton(cadre,text="Contrast CLAHE", variable=choix_contrast_CLAHE,command=commande_contrast_CLAHE,onvalue = 1, offvalue = 0)
CBCC.place(anchor="w",x=1450, y=760)

labelParam9 = Label (cadre, text = "Clip") # choix valeur contrate CLAHE
labelParam9.place(anchor="w", x=1555,y=760)
echelle9 = Scale (cadre, from_ = 0.1, to = 6, command= choix_valeur_CLAHE, orient=HORIZONTAL, length = 240, width = 10, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle9.set(val_contrast_CLAHE)
echelle9.place(anchor="w", x=1580,y=760)
# Choix du mode CLAHE - Daylight or Night
RBClahe1 = Radiobutton(cadre,text="DayLight", variable=choix_type_clahe,command=choix_CLAHE_DAYLLIGHT,value=1)
RBClahe1.place(anchor="w",x=1450, y=785)
RBClahe2 = Radiobutton(cadre,text="Stars", variable=choix_type_clahe,command=choix_CLAHE_STARS,value=2)
RBClahe2.place(anchor="w",x=1510, y=785)


####################
# CAPTURES WIDGETS #
####################

# Choix HQ Capture
CBHQC = Checkbutton(cadre,text="RAW Capture", variable=choix_HQ_capt,command=commande_HQ_capt,onvalue = 1, offvalue = 0)
CBHQC.place(anchor="w",x=1450, y=990)

# Number of pictures to capture
echelle8 = Scale (cadre, from_ = 1, to = 501, command= choix_nb_captures, orient=HORIZONTAL, length = 250, width = 10, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle8.set(val_nb_captures)
echelle8.place(anchor="w", x=1570,y=860)

# Frames number Video
echelle11 = Scale (cadre, from_ = 0, to = 4000, command= choix_nb_video, orient=HORIZONTAL, length = 250, width = 10, resolution = 20, label="",showvalue=1,tickinterval=500,sliderlength=20)
echelle11.set(val_nb_capt_video)
echelle11.place(anchor="w", x=1570,y=910)

labelParam65 = Label (cadre, text = "Delta T") # choix valeur contrate CLAHE
labelParam65.place(anchor="w", x=1535,y=955)
echelle65 = Scale (cadre, from_ = 0, to = 60, command= choix_deltat, orient=HORIZONTAL, length = 250, width = 10, resolution = 1, label="",showvalue=1,tickinterval=10,sliderlength=20)
echelle65.set(val_deltat)
echelle65.place(anchor="w",x=1570,y=955)

labelInfo1 = Label (cadre, text = text_info1) # label info n°1
labelInfo1.place(anchor="w", x=1550,y=990)

labelParam100 = Label (cadre, text = "Treatment time : ")
labelParam100.place(anchor="w", x=1450,y=20)
labelInfo2 = Label (cadre, text = "") # label info n°1
labelInfo2.place(anchor="w", x=1540,y=20)

labelInfoDark = Label (cadre, text = dispo_dark) # label info Dark
labelInfoDark.place(anchor="w", x=1750,y=820)

# Choix appliquer dark
CBAD = Checkbutton(cadre,text="Sub Dark", variable=choix_sub_dark,command=commande_sub_dark,onvalue = 1, offvalue = 0)
CBAD.place(anchor="w",x=1685, y=820)

echelle10 = Scale (cadre, from_ = 5, to = 30, command= choix_nb_darks, orient=HORIZONTAL, length = 100, width = 10, resolution =1, label="",showvalue=1,tickinterval=5,sliderlength=20)
echelle10.set(val_nb_darks)
echelle10.place(anchor="w", x=1575,y=820)


####################
#      BUTTONS     #
####################

Button1 = Button (cadre,text = "Start CAP", command = start_pic_capture,padx=10,pady=0)
Button1.place(anchor="w", x=1460,y=855)

Button2 = Button (cadre,text = "Stop CAP", command = stop_pic_capture,padx=10,pady=0)
Button2.place(anchor="w", x=1460,y=880)

Button3 = Button (cadre,text = "Start REC", command = start_video_capture,padx=10,pady=0)
Button3.place(anchor="w", x=1460,y=915)

Button4 = Button (cadre,text = "Stop REC", command = stop_video_capture,padx=10,pady=0)
Button4.place(anchor="w", x=1460,y=940)

Button5 = Button (cadre,text = "Pause REC", command = pause_video_capture,padx=10,pady=0)
Button5.place(anchor="w", x=1460,y=965)

Button6 = Button (cadre,text = "Cap Dark", command = start_cap_dark,padx=10,pady=0)
Button6.place(anchor="w", x=1460,y=820)


Button (fenetre_principale, text = "Quit", command = quitter,padx=10,pady=5).place(x=1800,y=990, anchor="e")

choix_BIN1()
mode_acq_medium()

if my_os == "win32" :
    init_efw()

cadre_image = Canvas (cadre, width = cam_displ_x, height = cam_displ_y, bg = "grey")
cadre_image.place(anchor="w", x=70,y=cam_displ_y/2+5)

fenetre_principale.after(500, refresh)
fenetre_principale.mainloop()
fenetre_principale.destroy()

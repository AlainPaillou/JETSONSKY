
############################################################################################################
#                                                                                                          #
#                                                JetsonSky                                                 #
#                                                                                                          #
#                      Images/videos acquisition & treatment software for ZWO cameras                      #
#                                                                                                          #
#                                   Images/videos treatment software                                       #
#                                                                                                          #
#                                Using NVidia GPU computing - CUDA & CUPY                                  #
#                                                                                                          #
#                                   Copyright Alain Paillou 2018-2025                                      #
#                                                                                                          #
#                  Free of use for personal and non commercial or professional use                         #
#                                                                                                          #
#       This software or part of this software is NOT free of use for commercial or professional use       #
#                                                                                                          #
############################################################################################################



JetsonSky_version = "V52_03RC"



# Supported ZWO cameras :  
#
# ASI120MC, ASI120MM
# ASI178MC, ASI178MM, ASI178MM Pro
# ASI224MC
# ASI290MC, ASI290MM
# ASI294MM, ASI294MM Pro
# ASI294MC, ASI294MC Pro
# ASI385MC
# ASI462MC
# ASI482MC
# ASI485MC, ASI585MC, ASI585MM
# ASI533MC, ASI533MM, ASI533MC Pro, ASI533MM Pro
# ASI662MC
# ASI676MC
# ASI678MC, ASI678MM
# ASI715MC
# ASI1600MC, ASI1600MM

############################################################################################################


# Filters order :  
                                                                                                       
# RGB software adjustment                                                                                                         
# Image Negative                                                                                                        
# Luminance estimate if mono sensor was used instead of colour sensor                                                                                                        
# 2 to 5 images SUM or MEAN                                                                                                        
# Reduce consecutive images variation considering previous or best frame                                                                                                        
# 3 frames noise removal front apply                                                                                                        
# adaptive absorber noise removal filter front apply - High or low dynamic                                                                                                       
# Noise removal filter Paillou 1                                                                                                        
# Noise removal filter Paillou 2                                                                                                       
# NLM2 noise removal filter                                                                                                        
# KNN noise removal filter                                                                                                        
# Luminance adjust                                                                                                        
# Image software amplification Linear or Gaussian                                                                                                         
# Star amplification                                                                                                        
# Gradient or Vignetting management                                                                                                        
# Contrast LOW Light                                                                                                        
# Contrast CLAHE                                                                                                        
# Color saturation enhancement                                                                                                        
# 3 frames noise removal back apply                                                                                                        
# adaptive absorber noise removal filter front apply - High dynamic only                                                                                                         
# Image sharpen 1                                                                                                        
# Image sharpen 2                                                                                                         


############################################################################################################

# YOLOv8 models :  
                                                                                                      
# Moon craters model :
# - Small crater
# - Crater
# - Large crater

# Satellites model :
# - Satellite
# - Shooting star
# - Plane


############################################################################################################

# Support 16 bits SER file 
                                                                                                      
############################################################################################################



# Choose your keyboard layout
keyboard_layout = "AZERTY"
#keyboard_layout = "QWERTY"


import os
import sys
import random
import math
import time
from datetime import datetime
from threading import Thread
import Serfile as Serfile
from collections import defaultdict
import numpy as np
import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw
from PIL import ImageEnhance
from tkinter import *
from tkinter.messagebox import askyesno
from tkinter import filedialog as fd
from tkinter.font import nametofont

my_os = sys.platform

if my_os == "win32" :
    import keyboard
    flag_pynput = False
    Dev_system = "Windows"
    nb_ThreadsX = 16
    nb_ThreadsY = 16
    print("Windows system")
    if keyboard_layout == "AZERTY" :
        print("Keyboard layout : AZERTY")
        titre = "JetsonSky " +  JetsonSky_version + " Windows release - CUDA / CUPY / OPENCV - ZSQD Stab - Arrows Zoom - ShiftR TGHFV - ShiftB OLMK; - Copyright Alain PAILLOU 2018-2025"
    else :
        print("Keyboard layout : QWERTY")
        titre = "JetsonSky " +  JetsonSky_version + " Windows release - CUDA / CUPY / OPENCV - WSAD Stab - Arrows Zoom - ShiftR TGHFV - ShiftB OL:K, - Copyright Alain PAILLOU 2018-2025"

if my_os == "linux" :
    from pynput import keyboard
    from pynput.keyboard import Key
    Dev_system = "Linux"
    nb_ThreadsX = 16
    nb_ThreadsY = 16
    print("Linux system")
    if keyboard_layout == "AZERTY" :
        print("Keyboard layout : AZERTY")
        titre = "JetsonSky " +  JetsonSky_version + " Linux release - CUDA / CUPY / OPENCV - ZSQD Stab - Arrows Zoom - ShiftR TGHFV - ShiftB OLMK; - Copyright Alain PAILLOU 2018-2025"
    else :
        print("Keyboard layout : QWERTY")
        titre = "JetsonSky " +  JetsonSky_version + " Linux release - CUDA / CUPY / OPENCV - WSAD Stab - Arrows Zoom - ShiftR TGHFV - ShiftB OL:K, - Copyright Alain PAILLOU 2018-2025"


Moon_crater_model = "./AI_models/AI_craters_model6_8s_3c_180e.pt"
Satellites_model = "./AI_models/AI_Sat_model1_8n_3c_300e.pt"
Custom_satellites_model_tracker = "./AI_models/sattelite_custom_tracker.yaml"

# Libraries import
flag_torch_OK = False
try :
    import torch
    print("Pytorch loaded")
    flag_torch_OK = True
except :
    print("No Pytorch available")
    flag_torch_OK = False


# Set your GSTREAMER directories in the following lines
if Dev_system == "Windows" :
    try :
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\bin")
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\lib")
        os.add_dll_directory("D:\\gstreamer\\1.0\\x86_64\\lib\\gstreamer-1.0")
        print("GSTREAMER directory founded")
    except Exception as error :
        print("No GSTREAMER directory find")

    
try :
    import cupy as cp
    from cupyx.scipy import ndimage
    print("Cupy libray Loaded")
    flag_cupy = True
except :
    flag_cupy = False
    print("Cupy libray not loaded")
    print("JetsonSky can't work without CUPY library")
    sys.exit()

cupy_context = cp.cuda.Stream(non_blocking=True) # CUPY context


if Dev_system == "Windows" :
    try :
        import psutil
        flag_psutil = True
        process = psutil.Process(os.getpid())
        process.nice(psutil.REALTIME_PRIORITY_CLASS)
        print("Python priority set to HIGHEST")
    except :
        print("No Python priority set")
else :
    print("No Python priority set")


import cv2
try :
    CudaEnable = cv2.cuda.getCudaEnabledDeviceCount()
    if CudaEnable == 1 :
        flag_OpenCvCuda = True
        print("OpenCV loaded with CUDA")
    else :
        flag_OpenCvCuda = False
        print("OpenCV loaded")
except :
        flag_OpenCvCuda = False
        print("OpenCV loaded")


flag_YOLO_OK = False

if flag_torch_OK == True :
    try :
        from ultralytics import YOLO
        print("YOLOv8 loaded")
        flag_YOLO_OK = True
    except :
        print("YOLOv8 NOT loaded")
        flag_YOLO_OK = False

if flag_YOLO_OK == True :
    try :
        model_craters_predict = YOLO(Moon_crater_model, task="predict")
        model_craters_track = YOLO(Moon_crater_model, task="track")
        flag_crater_model_loaded = True
        print("Craters model loaded")
    except :
        flag_crater_model_loaded = False
        print("Craters model NOT loaded")
else :
    flag_crater_model_loaded = False
    print("Craters model NOT loaded")

if flag_YOLO_OK == True :
    try :
        model_satellites_predict = YOLO(Satellites_model, task="predict")
        model_satellites_track = YOLO(Satellites_model, task="track")
        flag_satellites_model_loaded = True
        print("Satellites model loaded")
    except :
        flag_satellites_model_loaded = False
        print("Satellites model NOT loaded")
else :
    flag_satellites_model_loaded = False
    print("Satellites model NOT loaded")

    
try :
    import zwoasi_cupy as asi
except :
    print("ASI camera Python binding missing")

    
try :
    import zwoefw as efw
except :
    print("Filter Wheel Python binding missing")


try :
    import synscan
except :
    print("Synscan Python binding missing")


############################################
#                Main program              #
############################################


# Choose the size of the fonts in the Main Window - It depends of your system - can be set from 5 to 7
MainWindowFontSize = 6

if Dev_system == "Linux" :
    # Choose your directories for images and videos
    image_path = os.path.join(os.getcwd(), 'Images')
    video_path = os.path.join(os.getcwd(), 'Videos')

    # Path to librairies ZWO Jetson sbc
    env_filename_camera = os.path.join(os.getcwd(), 'Lib','libASICamera2.so')
    env_filename_efw = os.path.join(os.getcwd(), 'Lib','libEFWFilter.so')

    USBCam = 70
    
else :
    # Choose your directories for images and videos
    image_path = os.path.join(os.getcwd(), 'Images')
    video_path = os.path.join(os.getcwd(), 'Videos')
    
    # Path to librairies ZWO Windows
    env_filename_camera = os.path.join(os.getcwd(), 'Lib','ASICamera2.dll')
    env_filename_efw = os.path.join(os.getcwd(), 'Lib','EFW_filter.dll')

    USBCam = 95

    
# Variables initialization
flag_HQ = 0
frame_limit = 18
frame_skip = 3
exp_min=100 #µs
exp_max=10000 #µs
exp_delta=100 #µs
exp_interval=2000
val_resolution = 1
format_capture = asi.ASI_IMG_RAW16
mode_BIN=1
sensor_bits_depth = 14
delta_s = 0
fact_s = 1.0
res_cam_x = 3096
res_cam_y = 2080
cam_displ_x = int(1350 * fact_s)
cam_displ_y = int(1012 * fact_s)
sensor_factor = "4/3"
TH_16B = 16
threshold_16bits = 2 ** TH_16B - 1

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

Camera_Bayer = "RAW"
Video_Bayer = "RAW"
GPU_BAYER = 0 #  0 : RAW   1 : RGGB   2 : BGGR    3 : GRBG     4 : GBRG
type_debayer = 0

val_exposition = 1000
timeoutexp = 1 + 500
exposition = 0
val_gain = 100
val_maxgain = 600
val_denoise = 0.4
val_histo_min = 0
val_histo_max = 255
val_contrast_CLAHE = 1.0
val_reduce_variation = 1
val_phi = 1.0
val_theta = 100
val_heq2 = 1.0
text_info1 = "Test information"
text_info10 = ""
val_nb_captures = 1
nb_cap_video =0
val_nb_capt_video = 100
compteur_images = 0
numero_image = 0
image_camera = 0
image_camera_old = 0
FlipV = 0
FlipH = 0
ImageNeg = 0
val_red = 63
val_blue = 74
val_FS = 1
compteur_FS = 0
compteur_AADF = 0
compteur_AADFB = 0
compteur_RV = 0
val_denoise_KNN = 0.2
val_USB = 90
val_SGR = 95
val_AGR = 50
val_NGB = 13
val_SAT = 1.0
ASIGAMMA = 50
ASIAUTOMAXBRIGHTNESS = 50
nb_erreur = 0
text_TIP = ""
val_ampl = 1.0
val_deltat = 0
timer1 = 0.0
grad_vignet = 1
stack_div = 1
val_reds = 1.0
val_blues = 1.0
val_greens = 1.0
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
azimut_monture = 0.0
hauteur_monture = 0.0
val_ghost_reducer = 50
val_grid_CLAHE = 8
total_start = 0.0
total_stop = 0.0
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
val_sigma_sharpen2 = 2.0
trig_count_down = 0
video_frame_number = 0
video_frame_position = 0
compteur_3FNR = 0
compteur_3FNRB = 0
delta_zx = 0
delta_zy = 0
delta_tx = 0
delta_ty = 0
mean_quality = 0
max_quality = 0
max_qual = 0
min_qual = 10000
val_BFR = 50
SFN = 0 # Skip frame number
frame_number = 0
max_qual_PT = 0
delta_RX = 0
delta_RY = 0
delta_BX = 0
delta_BY = 0
key_pressed = ""
DSW = 0
SER_depth = 8
previous_frame_number = -1
frame_position = 0
mode_HDR = "Mertens"

IQ_Method = "Sobel"
laplacianksize = 7
SobelSize = 5

beta = 1.0
alpha = 5
Corr_cont = np.zeros(255,dtype=np.single)
for x in range(0,255) :
    Corr_cont[x] = (1-math.exp(-1*x/alpha)+beta)/(1+beta)

quality = np.zeros(258,dtype=float)
tmp_qual = np.zeros(258,dtype=float)
for x in range(1,258) :
    quality[x] = 0
max_quality = 1
quality_pos = 1

    
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

stars_x = np.zeros(1000000,dtype=int)
stars_y = np.zeros(1000000,dtype=int)
stars_s = np.zeros(1000000,dtype=int)

sat_frame_target = 5
sat_frame_count = 0
sat_frame_target_AI = 5
sat_frame_count_AI = 0
sat_x = np.zeros(100000,dtype=int)
sat_y = np.zeros(100000,dtype=int)
sat_s = np.zeros(100000,dtype=int)
sat_old_x = np.zeros(100000,dtype=int)
sat_old_y = np.zeros(100000,dtype=int)
sat_old_dx = np.zeros(100000,dtype=int)
sat_old_dy = np.zeros(100000,dtype=int)
sat_id  = np.zeros(100000,dtype=int)
sat_old_id  = np.zeros(100000,dtype=int)
correspondance = np.zeros(100000,dtype=int)
sat_speed = np.zeros(100000,dtype=int)
compteur_sat = 0
old_sat = 0
flag_first_sat_pass = True
nb_trace_sat = -1
nb_sat = 0
max_sat = 20

curFPS = 0.0    
fpsQueue = []

curTT = 0.0    
TTQueue = []

CONFIDENCE_THRESHOLD_LIMIT_CRATERS = 0.1
track_crater_history = defaultdict(lambda: [])
track_satellite_history = defaultdict(lambda: [])

Pi = math.pi
conv_rad = Pi / 180
conv_deg = 180 /Pi

lat_obs = 48.0175
long_obs = -4.0340
alt_obs = 0
Polaris_AD = 2.507
Polaris_DEC = 89.25
zone = 2
jours_obs = 0
mois_obs = 0
annee_obs = 0

heure_obs = 0
min_obs = 0
second_obs = 0
azimut_cible = 0.0
hauteur_cible = 0.0
delta_azimut = 0.0
delta_hauteur = 0.0


cv2.setUseOptimized(True)


def quitter() :
    global camera,flag_autorise_acquisition,thread1,fenetre_principale,flag_image_disponible,flag_quitter,flag_acquisition_mount,thread_1,thread_2,thread_3,flag_keyboard_management

    if flag_camera_ok == True :
        thread_2.stop()
        flag_autorise_acquisition = False
        flag_image_disponible = False
        flag_quitter = True
        flag_acquisition_mount = False
        time.sleep(1)
        if Dev_system == "Windows" :
            thread_3.stop()
        time.sleep(1)
        if flag_filter_wheel == True :
            filter_wheel.close()
        try :
            camera.stop_video_capture()
            time.sleep(0.5)
            camera.close()
        except :
            print("Close camera error")
        fenetre_principale.quit()
    else :
        flag_autorise_acquisition = False
        flag_image_disponible = False
        flag_keyboard_management = False
        if Dev_system == "Windows" :
            thread_3.stop()
        time.sleep(1)
        fenetre_principale.quit()


# Main Window
fenetre_principale = Tk ()
screen_width = fenetre_principale.winfo_screenwidth()
screen_height = fenetre_principale.winfo_screenheight()

image_JetsonSky = cv2.imread('JetsonSky_Logo.jpg',cv2.IMREAD_COLOR)
cv2.imshow(titre, image_JetsonSky)
cv2.waitKey()
cv2.destroyAllWindows()

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

cadre = Frame (fenetre_principale, width = w , heigh = h)
cadre.pack ()


gradient_vignetting = IntVar()
gradient_vignetting.set(1) # Initialisation du mode gradient ou vignetting a gradient

mode_acq = IntVar()
mode_acq.set(2) # Initialisation du mode d'acquisition a Moyen

mode_HDR_select = IntVar()
mode_HDR_select.set(1) # Initialisation du mode HDR Mertens

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

choix_mode_full_res = IntVar()
choix_mode_full_res.set(0) # Initialisation mode full resolution inactif

choix_sharpen_soft1 = IntVar()
choix_sharpen_soft1.set(0) # initialisation mode sharpen software 1 inactif

choix_sharpen_soft2 = IntVar()
choix_sharpen_soft2.set(0) # initialisation mode sharpen software 2 inactif

choix_NLM2 = IntVar()
choix_NLM2.set(0) # initialisation mode denoise software inactif

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

choix_AADF = IntVar()
choix_AADF.set(0) # Initialisation Filtrage AADF Front

choix_AADFB = IntVar()
choix_AADFB.set(0) # Initialisation Filtrage AADF Back

choix_3FNR = IntVar()
choix_3FNR.set(0) # Initialisation Filtrage 3 frames noise removal Front

choix_3FNRB = IntVar()
choix_3FNRB.set(0) # Initialisation Filtrage 3 frames noise removal Back

choix_reduce_variation = IntVar()
choix_reduce_variation.set(0) # Initialisation Filtrage reduce variation pre treatment

choix_reduce_variation_post_treatment = IntVar()
choix_reduce_variation_post_treatment.set(0) # Initialisation Filtrage reduce variation pre treatment

choix_GR = IntVar()
choix_GR.set(0) # Initialisation Filtre Gradient Removal

Sat_Vid_Img = IntVar()
Sat_Vid_Img.set(0) # Initialisation mode saturation Video

presence_FW = IntVar()
presence_FW.set(0) # Initialisation absence FW

fw_position_ = IntVar()
fw_position_.set(0) # Position filter wheel

cam_read_speed = IntVar()
cam_read_speed.set(1) # Set cam read speed to slow

demo_side = IntVar()
demo_side.set(0) # Set Left side demo

sensor_ratio = IntVar()
sensor_ratio.set(0) # Set sensor ratio

choix_AmpSoft = IntVar()
choix_AmpSoft.set(0) # Initialisation amplification software OFF

bayer_sensor = IntVar()
bayer_sensor.set(1) # Bayer sensor pattern

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

choix_HOTPIX = IntVar()
choix_HOTPIX.set(0) # Initialisation Hot Pixels removal

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

choix_IMQE = IntVar()
choix_IMQE.set(0) # Initialisation Image Quality Estimation OFF

choix_BFR = IntVar()
choix_BFR.set(0) # Initialisation Bad Frame Remove

choix_BFReference = IntVar()
choix_BFReference.set(1) # Initialisation Best frame reference for variation reduction filter

choix_false_colours = IntVar()
choix_false_colours.set(0) # Initialisation false colours mode

choix_AI_Craters = IntVar()
choix_AI_Craters.set(0) # Initialisation AI craters off

choix_AI_Satellites = IntVar()
choix_AI_Satellites.set(0) # Initialisation AI satellites off

choix_AI_Trace = IntVar()
choix_AI_Trace.set(0) # Initialisation AI trajectory trace off

choix_16bLL = IntVar()
choix_16bLL.set(0) # Initialisation 16 bits Low Light

choix_sub_img_ref = IntVar()
choix_sub_img_ref.set(0) # Initialisation Imge reference subtract


flag_16b = False
flag_full_res = 0
flag_sharpen_soft1 = 0
flag_sharpen_soft2 = 0
flag_NLM2 = 0
flag_histogram_equalize2 = 0
flag_histogram_stretch = 0
flag_histogram_phitheta = 0
flag_contrast_CLAHE = 0
flag_CLL = 0
flag_noir_blanc = 0
flag_AmpSoft = 0
flag_autorise_acquisition = False
flag_image_disponible = False
flag_premier_demarrage = True
flag_BIN2 = False
flag_cap_pic = False
flag_cap_video = False
flag_acq_rapide = "MedF"
flag_colour_camera = True
flag_filter_wheel = False
Im1OK = False
Im2OK = False
Im3OK = False
Im4OK = False
Im5OK = False
Im1fsdnOK = False
Im2fsdnOK = False
Im1fsdnOKB = False
Im2fsdnOKB = False
Im1rvOK = False
Im2rvOK = False
filter_on = False
flag_filtrage_ON = True
flag_filtre_work = False
flag_denoise_KNN = False
flag_denoise_Paillou = False
flag_denoise_Paillou2 = False
flag_AADF = False
flag_AADFB = False
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
flag_dyn_AADF = 1
flag_TRGS = 0
flag_lin_gauss = 1
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
flag_HDR = False
flag_new_image = False
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
FNRB_First_Start = False
img1_3FNROKB  = False
img2_3FNROKB  = False
img3_3FNROKB = False
flag_3FNRB = False
flag_TRCLL = 0
flag_hot_pixels = False
flag_demo_side = "Left"
flag_iscolor = True
flag_new_stab_window = False
flag_SAT_Image = False
flag_IQE = False
flag_IsColor = True
flag_BFR = False
flag_BFREF = False
flag_BFREF_image = False
flag_BFReference = "BestFrame"
flag_reduce_variation_post_treatment = False
flag_BFREFPT = False
flag_BFREF_image = False
flag_acquisition_mount = True
flag_false_colours = False
flag_AI_Craters = False
flag_AI_Satellites = False
flag_AI_Trace = False
flag_keyboard_management = False
flag_new_frame_position = False
flag_HB = False
flag_img_sat_buf1 = False
flag_img_sat_buf2 = False
flag_img_sat_buf3 = False
flag_img_sat_buf4 = False
flag_img_sat_buf5 = False
flag_img_sat_buf1_AI = False
flag_img_sat_buf2_AI = False
flag_img_sat_buf3_AI = False
flag_img_sat_buf4_AI = False
flag_img_sat_buf5_AI = False
flag_capture_image_reference = False
flag_image_reference_OK = False
flag_image_ref_sub = False
flag_SER_file = False



##########################################
#           CUDA & CUPY Kernels          #
##########################################


BIN_Color_GPU = cp.RawKernel(r'''
extern "C" __global__
void BIN_Color_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height, int BIN_mode)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,i1,i2,i3,i4;
  int tmp_r,tmp_g,tmp_b;

  index = i * width + j;
  
  i1 = (i * 2) * (width * 2) + (j * 2);
  i2 = (i * 2) * (width * 2) + (j * 2 + 1);
  i3 = (i * 2 + 1) * (width * 2) + (j * 2);
  i4 = (i * 2 + 1) * (width * 2) + (j * 2 + 1);
  
  if (i < height && i > 0 && j < width && j > 0) {
      if (BIN_mode == 0) {
          tmp_r = img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4];  
          tmp_g = img_g[i1] + img_g[i2] + img_g[i3] + img_g[i4];  
          tmp_b = img_b[i1] + img_b[i2] + img_b[i3] + img_b[i4];

          dest_r[index] = (int)(min(max(tmp_r, 0), 255));
          dest_g[index] = (int)(min(max(tmp_g, 0), 255));
          dest_b[index] = (int)(min(max(tmp_b, 0), 255));
          }
      else {
          dest_r[index] = (img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4]) / 4;  
          dest_g[index] = (img_g[i1] + img_g[i2] + img_g[i3] + img_g[i4]) / 4;  
          dest_b[index] = (img_b[i1] + img_b[i2] + img_b[i3] + img_b[i4]) / 4;
          }
    }
}
''', 'BIN_Color_GPU_C')

BIN_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void BIN_Mono_GPU_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, int BIN_mode)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,i1,i2,i3,i4;
  int tmp_r;

  index = i * width + j;
  
  i1 = (i * 2) * (width * 2) + (j * 2);
  i2 = (i * 2) * (width * 2) + (j * 2 + 1);
  i3 = (i * 2 + 1) * (width * 2) + (j * 2);
  i4 = (i * 2 + 1) * (width * 2) + (j * 2 + 1);
  
  if (i < height && i > 0 && j < width && j > 0) {
      if (BIN_mode == 0) {
          tmp_r = img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4];  

          dest_r[index] = (int)(min(max(tmp_r, 0), 255));
          }
      else {
          dest_r[index] = (img_r[i1] + img_r[i2] + img_r[i3] + img_r[i4]) / 4;  
          }
    }
}
''', 'BIN_Mono_GPU_C')


RGB_Align_GPU = cp.RawKernel(r'''
extern "C" __global__
void RGB_Align_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height,
long int delta_RX, long int delta_RY, long int delta_BX, long int delta_BY)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index,indexR,indexB,iR,jR,iB,jB;
  
  index = i * width + j;
  indexR = (i + delta_RY) * width + j + delta_RX;
  indexB = (i + delta_BY) * width + j + delta_BX;
  iR = i + delta_RY;
  jR = j + delta_RX;
  iB = i + delta_BY;
  jB = j + delta_BX;

  if (i < height && j < width) {
      if (iR > 0 && iR< height && jR > 0 && jR < width && iB > 0 && iB< height && jB > 0 && jB < width) {
        dest_r[index] = img_r[indexR];
        dest_g[index] = img_g[index];
        dest_b[index] = img_b[indexB];
        }
      else {
        dest_r[index] = 0;
        dest_g[index] = 0;
        dest_b[index] = 0;
        }
    } 
}
''', 'RGB_Align_GPU_C')

Image_Debayer_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Image_Debayer_Mono_GPU_C(unsigned char *dest_r, unsigned char *img, long int width, long int height, int GPU_BAYER)
{

  long int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  long int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  int r1,r2,r3,r4,g1,g2,g3,g4,b1,b2,b3,b4;
  float att;
  
  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;
  i5 = (i-1) * width + j-1;
  i6 = (i-1) * width + j;
  i7 = (i-1) * width + j+1;
  i8 = (i-1) * width + j+2;
  i9 = i * width + j+2;
  i10 = (i+1) * width + j+2;
  i11 = (i+2) * width + j+2;
  i12 = (i+2) * width + j+1;
  i13 = (i+2) * width + j;
  i14 = (i+2) * width + j-1;
  i15 = (i+1) * width + j-1;
  i16 = i * width + j-1;
  att = 1 / 4.0;
  
  if (i < (height-1) && i > 0 && j < (width-1) && j > 0) {
      if (GPU_BAYER == 1) {
// RGGB
          r1=img[i1];  
          g1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          b1=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          r2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          g2=img[i2];
          b2=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          r3=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          g3=img[i3];
          b3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          r4=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          g4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          b4=img[i4];
          }
// BGGR
      if (GPU_BAYER == 2) {
          b1=img[i1];  
          g1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          r1=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          b2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          g2=img[i2];
          r2=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          b3=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          g3=img[i3];
          r3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          b4=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          g4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          r4=img[i4];
          }
// GBRG
      if (GPU_BAYER == 3) {
          r1=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          g1=img[i1];
          b1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          r2=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          g2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          b2=img[i2];

          r3=img[i3];
          g3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          b3=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          r4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          g4=img[i4];
          b4=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
// GRBG
      if (GPU_BAYER == 4) {
          b1=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          g1=img[i1];
          r1=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          b2=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          g2=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          r2=img[i2];

          b3=img[i3];
          g3=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          r3=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          b4=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          g4=img[i4];
          r4=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
      dest_r[i1] = (int)(min(max(int(0.299*r1 + 0.587*g1 + 0.114*b1), 0), 255));
      dest_r[i2] = (int)(min(max(int(0.299*r2 + 0.587*g2 + 0.114*b2), 0), 255));
      dest_r[i3] = (int)(min(max(int(0.299*r3 + 0.587*g3 + 0.114*b3), 0), 255));
      dest_r[i4] = (int)(min(max(int(0.299*r4 + 0.587*g4 + 0.114*b4), 0), 255));
    }
}
''', 'Image_Debayer_Mono_GPU_C')

Image_Debayer_GPU = cp.RawKernel(r'''
extern "C" __global__
void Image_Debayer_GPU_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b,
unsigned char *img, long int width, long int height, int GPU_BAYER)
{

  long int j = (threadIdx.x + blockIdx.x * blockDim.x) * 2;
  long int i = (threadIdx.y + blockIdx.y * blockDim.y) * 2;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  float att;
  
  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;
  i5 = (i-1) * width + j-1;
  i6 = (i-1) * width + j;
  i7 = (i-1) * width + j+1;
  i8 = (i-1) * width + j+2;
  i9 = i * width + j+2;
  i10 = (i+1) * width + j+2;
  i11 = (i+2) * width + j+2;
  i12 = (i+2) * width + j+1;
  i13 = (i+2) * width + j;
  i14 = (i+2) * width + j-1;
  i15 = (i+1) * width + j-1;
  i16 = i * width + j-1;
  att = 1 / 4.0;
  
  if (i < (height-1) && i > 0 && j < (width-1) && j > 0) {
// RGGB
      if (GPU_BAYER == 1) {
          dest_r[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_b[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_r[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_b[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_r[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_b[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_r[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_b[i4]=img[i4];
          }
// BGGR
      if (GPU_BAYER == 2) {
          dest_b[i1]=img[i1];  
          dest_g[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
          dest_r[i1]=min(img[i4],img[i5])+int(att*abs(img[i4]-img[i5]));
      
          dest_b[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_g[i2]=img[i2];
          dest_r[i2]=min(img[i4],img[i7])+int(att*abs(img[i4]-img[i7]));

          dest_b[i3]=min(img[i1],img[i13])+int(att*abs(img[i1]-img[i13]));
          dest_g[i3]=img[i3];
          dest_r[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));

          dest_b[i4]=min(img[i1],img[i11])+int(att*abs(img[i1]-img[i11]));
          dest_g[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_r[i4]=img[i4];
          }
// GBRG
      if (GPU_BAYER == 3) {
          dest_r[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_b[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_r[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_b[i2]=img[i2];

          dest_r[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_b[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_r[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_b[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
// GRBG
      if (GPU_BAYER == 4) {
          dest_b[i1]=min(img[i3],img[i6])+int(att*abs(img[i3]-img[i6]));
          dest_g[i1]=img[i1];
          dest_r[i1]=min(img[i2],img[i16])+int(att*abs(img[i2]-img[i16]));
      
          dest_b[i2]=min(img[i3],img[i8])+int(att*abs(img[i3]-img[i8]));
          dest_g[i2]=min(img[i1],img[i9])+int(att*abs(img[i1]-img[i9]));
          dest_r[i2]=img[i2];

          dest_b[i3]=img[i3];
          dest_g[i3]=min(img[i4],img[i15])+int(att*abs(img[i4]-img[i15]));
          dest_r[i3]=min(img[i14],img[i2])+int(att*abs(img[i14]-img[i2]));

          dest_b[i4]=min(img[i3],img[i10])+int(att*abs(img[i3]-img[i10]));
          dest_g[i4]=img[i4];
          dest_r[i4]=min(img[i12],img[i2])+int(att*abs(img[i12]-img[i2]));
          }
    }
}
''', 'Image_Debayer_GPU_C')

Dead_Pixels_Remove_Colour_GPU = cp.RawKernel(r'''
extern "C" __global__
void Dead_Pixels_Remove_Colour_C(unsigned char *dest, unsigned char *img, long int width, long int height, unsigned char Threshold, int GPU_BAYER)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11,i12,i13,i14,i15,i16;
  int Delta1r, Delta2r;
  int Delta1g1, Delta2g1;
  int Delta1g2, Delta2g2;
  int Delta1b, Delta2b;

  i1 = i * width + j;
  i2 = i * width + j+1;
  i3 = (i+1) * width + j;
  i4 = (i+1) * width + j+1;

  if (i < (height-2) && i > 1 && j < (width-2) && j > 1) {
    if (GPU_BAYER == 1) {
        Delta1r = abs(img[i1] - img[i1-2]);  
        Delta2r = abs(img[i1] - img[i1+2]);
        Delta1g1 = abs(img[i2] - img[i2-2]);  
        Delta2g1 = abs(img[i2] - img[i2+2]);
        Delta1g2 = abs(img[i3] - img[i3-2]);  
        Delta2g2 = abs(img[i3] - img[i3+2]);
        Delta1b = abs(img[i4] - img[i4-2]);  
        Delta2b = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 2) {
        Delta1b = abs(img[i1] - img[i1-2]);  
        Delta2b = abs(img[i1] - img[i1+2]);
        Delta1g1 = abs(img[i2] - img[i2-2]);  
        Delta2g1 = abs(img[i2] - img[i2+2]);
        Delta1g2 = abs(img[i3] - img[i3-2]);  
        Delta2g2 = abs(img[i3] - img[i3+2]);
        Delta1r = abs(img[i4] - img[i4-2]);  
        Delta2r = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 3) {
        Delta1g1 = abs(img[i1] - img[i1-2]);  
        Delta2g1 = abs(img[i1] - img[i1+2]);
        Delta1b = abs(img[i2] - img[i2-2]);  
        Delta2b = abs(img[i2] - img[i2+2]);
        Delta1r = abs(img[i3] - img[i3-2]);  
        Delta2r = abs(img[i3] - img[i3+2]);
        Delta1g2 = abs(img[i4] - img[i4-2]);  
        Delta2g2 = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER == 4) {
        Delta1g1 = abs(img[i1] - img[i1-2]);  
        Delta2g1 = abs(img[i1] - img[i1+2]);
        Delta1r = abs(img[i2] - img[i2-2]);  
        Delta2r = abs(img[i2] - img[i2+2]);
        Delta1b = abs(img[i3] - img[i3-2]);  
        Delta2b = abs(img[i3] - img[i3+2]);
        Delta1g2 = abs(img[i4] - img[i4-2]);  
        Delta2g2 = abs(img[i4] - img[i4+2]);
        }
    if (GPU_BAYER > 0) {
        if ((Delta1r > Threshold) && (Delta2r > Threshold)) { 
            dest[i1] = int((img[i1-2] + img[i1+2]) / 2.0);
        }
        else {
            dest[i1] = img[i1];
        }

        if ((Delta1g1 > Threshold) && (Delta2g1 > Threshold)) { 
            dest[i2] = int((img[i2-2] + img[i2+2]) / 2.0);
        }
        else {
            dest[i2] = img[i2];
        }

        if ((Delta1g2 > Threshold) && (Delta2g2 > Threshold)) { 
            dest[i3] = int((img[i3-2] + img[i3+2]) / 2.0);
        }
        else {
            dest[i3] = img[i3];
        }

        if ((Delta1b > Threshold) && (Delta2b > Threshold)) { 
            dest[i4] = int((img[i4-2] + img[i4+2]) / 2.0);
        }
        else {
            dest[i4] = img[i4];
        }
      }
    else {
        dest[i1] = img[i1];
        dest[i2] = img[i2];
        dest[i3] = img[i3];
        dest[i4] = img[i4];
        }
    }
}
''', 'Dead_Pixels_Remove_Colour_C')

Dead_Pixels_Remove_Mono_GPU = cp.RawKernel(r'''
extern "C" __global__
void Dead_Pixels_Remove_Mono_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, unsigned char Threshold)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int Delta1r, Delta2r;
  
  index = i * width + j;
  
  if (i < height && i > 1 && j < width && j >1) {
      Delta1r = abs(img_r[index] - img_r[index-1]);  
      Delta2r = abs(img_r[index] - img_r[index+1]);
      
      if ((Delta1r > Threshold) && (Delta2r > Threshold)) { 
      dest_r[index] = int((img_r[index-1] + img_r[index+1]) / 2.0);
      }
      else {
      dest_r[index] = img_r[index];
      }      
    } 
}
''', 'Dead_Pixels_Remove_Mono_C')

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

color_estimate_Mono = cp.RawKernel(r'''
extern "C" __global__
void color_estimate_Mono_C(unsigned char *dest_r, unsigned char *dest_g, unsigned char *dest_b, unsigned char *img_r, unsigned char *img_g, unsigned char *img_b, long int width, long int height)
{
  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  float Rcalc,Gcalc,Bcalc;
  float Lum_grey,Lum_color,Lum_factor;
  
  index = i * width + j;

  if (i < height && j < width) {
    Lum_grey = (0.299*img_r[index] + 0.587*img_g[index] + 0.114*img_b[index]);
    Lum_color = img_r[index] + img_g[index] + img_b[index];
    Lum_factor = Lum_color / Lum_grey;
    Rcalc = img_r[index] * Lum_factor;
    Gcalc = img_g[index] * Lum_factor;
    Bcalc = img_b[index] * Lum_factor;
    dest_r[index] = (int)(min(max(int(Rcalc), 0), 255));
    dest_g[index] = (int)(min(max(int(Gcalc), 0), 255));
    dest_b[index] = (int)(min(max(int(Bcalc), 0), 255));
  }
}
''', 'color_estimate_Mono_C')

Mono_ampsoft_GPU = cp.RawKernel(r'''
extern "C" __global__
void Mono_ampsoft_GPU_C(unsigned char *dest_r, unsigned char *img_r, long int width, long int height, float val_ampl, float *Corr_GS)
{

  long int j = threadIdx.x + blockIdx.x * blockDim.x;
  long int i = threadIdx.y + blockIdx.y * blockDim.y;
  long int index;
  int cor,vr;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];
      cor = (int)(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr]));
      dest_r[index] = min(max(cor, 0), 255);
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
  int cor_r,cor_g,cor_b;
  int vr,vg,vb;
  
  index = i * width + j;
  
  if (i < height && j < width) {
      vr = img_r[index];  
      vg = img_g[index];  
      vb = img_b[index];
      cor_r = (int)(img_r[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vr]));
      cor_g = (int)(img_g[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vg]));
      cor_b = (int)(img_b[index] * (1.0 + (val_ampl-1.0)*Corr_GS[vb]));
      dest_r[index] = min(max(cor_r, 0), 255);
      dest_g[index] = min(max(cor_g, 0), 255);
      dest_b[index] = min(max(cor_b, 0), 255);
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
long int width, long int height, int flag_dyn_AADF, int flag_ghost_reducer, int val_ghost_reducer)
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
      delta_r = old_r[index] - img_r[index];
      delta_g = old_g[index] - img_g[index];
      delta_b = old_b[index] - img_b[index];
      if (flag_dyn_AADF == 1) {
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
          if (delta_r > 0 && flag_dyn_AADF == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AADF == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          if (delta_g > 0 && flag_dyn_AADF == 1 && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.025995987)*1.2669433195)));
          }
          if ((delta_g < 0 || flag_dyn_AADF == 0) && flag_g == 0) {
              dest_g[index] = (int)((old_g[index] - delta_g / (__powf(abs(delta_g),-0.54405)*20.8425))); 
          }
          if (delta_b > 0 && flag_dyn_AADF == 1 && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.025995987)*1.2669433195)));
          }
          if ((delta_b < 0 || flag_dyn_AADF == 0) && flag_b == 0) {
              dest_b[index] = (int)((old_b[index] - delta_b / (__powf(abs(delta_b),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AADF == 1) {
              coef_r = __powf(abs(delta_r),-0.025995987)*1.2669433195;
          }
          else {
              coef_r = __powf(abs(delta_r),-0.54405)*20.8425; 
          }
          if (delta_g > 0 && flag_dyn_AADF == 1) {
              coef_g = __powf(abs(delta_g),-0.025995987)*1.2669433195;
          }
          else {
              coef_g = __powf(abs(delta_g),-0.54405)*20.8425; 
          }
          if (delta_b > 0 && flag_dyn_AADF == 1) {
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
void adaptative_absorber_denoise_Mono_C(unsigned char *dest_r, unsigned char *img_r, unsigned char *old_r, long int width, long int height, int flag_dyn_AADF,
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
      delta_r = old_r[index] - img_r[index];
      if (flag_dyn_AADF == 1) {
          flag_ghost_reducer = 0;
      }
      if (flag_ghost_reducer == 1) {
          if (abs(delta_r) > val_ghost_reducer) {
              flag_r = 1;
              dest_r[index] = img_r[index];
          }
          if (delta_r > 0 && flag_dyn_AADF == 1 && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.025995987)*1.2669433195)));
          }
          if ((delta_r < 0 || flag_dyn_AADF == 0) && flag_r == 0) {
              dest_r[index] = (int)((old_r[index] - delta_r / (__powf(abs(delta_r),-0.54405)*20.8425))); 
          }
          }
      if (flag_ghost_reducer == 0) {
          if (delta_r > 0 && flag_dyn_AADF == 1) {
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
              dest_r[index] = min(max(old_r[index] - variation, 0), 255);
          }
          else {
              dest_r[index] = min(max(old_r[index] + variation, 0), 255);          
          }
      }
      else {
          dest_r[index] = img_r[index];
      }
      
      if (abs(delta_g) > variation) {
          if (delta_g >= 0) {
              dest_g[index] = min(max(old_g[index] - variation, 0), 255);
          }
          else {
              dest_g[index] = min(max(old_g[index] + variation, 0), 255);          
          }
      }
      else {
          dest_g[index] = img_g[index];
      }

      if (abs(delta_b) > variation) {
          if (delta_b >= 0) {
              dest_b[index] = min(max(old_b[index] - variation, 0), 255);
          }
          else {
              dest_b[index] = min(max(old_b[index] + variation, 0), 255);          
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
              dest_r[index] = min(max(old_r[index] - variation, 0), 255);
          }
          else {
              dest_r[index] = min(max(old_r[index] + variation, 0), 255);          
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
      if (mod_blue != 1.0) {
          dest_r[index] = (int)(min(max(int(img_r[index] * mod_blue), 0), 255));
          }
      else {
          dest_r[index] = img_r[index];
          }
      if (mod_green != 1.0) {        
          dest_g[index] = (int)(min(max(int(img_g[index] * mod_green), 0), 255));
          }
      else {
          dest_g[index] = img_g[index];
          }
      if (mod_red != 1.0) {  
          dest_b[index] = (int)(min(max(int(img_b[index] * mod_red), 0), 255));
          }
      else {
          dest_b[index] = img_b[index];
          }
    } 
}
''', 'Set_RGB_C')


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



# Keyboard configuration setup, depending of your keyboard layout and country
if Dev_system == "Windows" :
    if keyboard_layout == "AZERTY" :
        # Red channel shifting
        red_up = 't'
        red_down = 'g'
        red_right = 'h'
        red_left = 'f'
        red_reset = 'v'

        # Blue channel shifting
        blue_up = 'o'
        blue_down = 'l'
        blue_right = 'm'
        blue_left = 'k'
        blue_reset = ';'

        # Zoom displacement
        zoom_up = 'haut' # UP ARROW
        zoom_down = 'bas' # DOWN ARROW
        zoom_right = 'droite' # RIGHT ARROW
        zoom_left = 'gauche' # LEFT ARROW
        zoom_reset = 'espace' # SPACE KEY

        # hock stabilization window displacement
        stab_up = 'z'
        stab_down = 's'
        stab_right = 'd'
        stab_left = 'q'
        stab_zone_more = '+'
        stab_zone_less = '-'
    else :
        # Red channel shifting
        red_up = 't'
        red_down = 'g'
        red_right = 'h'
        red_left = 'f'
        red_reset = 'v'

        # Blue channel shifting
        blue_up = 'o'
        blue_down = 'l'
        blue_right = ':'
        blue_left = 'k'
        blue_reset = ','

        # Zoom displacement
        zoom_up = 'up' # UP ARROW
        zoom_down = 'down' # DOWN ARROW
        zoom_right = 'right' # RIGHT ARROW
        zoom_left = 'left' # LEFT ARROW
        zoom_reset = 'space' # SPACE KEY

        # hock stabilization window displacement
        stab_up = 'w'
        stab_down = 's'
        stab_right = 'd'
        stab_left = 'a'
        stab_zone_more = '+'
        stab_zone_less = '-'

if Dev_system == "Linux" :
    if keyboard_layout == "AZERTY" :
        # Red channel shifting
        red_up = ('t')
        red_down = ('g')
        red_right = ('h')
        red_left = ('f')
        red_reset = ('v')

        # Blue channel shifting
        blue_up = ('o')
        blue_down = ('l')
        blue_right = ('m')
        blue_left = ('k')
        blue_reset = (';')

        # Zoom displacement
        zoom_up = Key.up # UP ARROW
        zoom_down = Key.down # DOWN ARROW
        zoom_right = Key.right # RIGHT ARROW
        zoom_left = Key.left # LEFT ARROW
        zoom_reset = Key.space # SPACE KEY

        # hock stabilization window displacement
        stab_up = ('z')
        stab_down = ('s')
        stab_right = ('d')
        stab_left = ('q')
        stab_zone_more = ('+')
        stab_zone_less = ('-')
    else :
        # Red channel shifting
        red_up = ('t')
        red_down = ('g')
        red_right = ('h')
        red_left = ('f')
        red_reset = ('v')

        # Blue channel shifting
        blue_up = ('o')
        blue_down = ('l')
        blue_right = (':')
        blue_left = ('k')
        blue_reset = (',')

        # Zoom displacement
        zoom_up = Key.up # UP ARROW
        zoom_down = Key.down # DOWN ARROW
        zoom_right = Key.right # RIGHT ARROW
        zoom_left = Key.left # LEFT ARROW
        zoom_reset = Key.space # SPACE KEY

        # hock stabilization window displacement
        stab_up = ('w')
        stab_down = ('s')
        stab_right = ('d')
        stab_left = ('a')
        stab_zone_more = ('+')
        stab_zone_less = ('-')





def init_efw():
    global filter_wheel,fw_position,flag_filter_wheel,presence_FW,fw_position_

    try :
        if env_filename_efw:
            efw.init(env_filename_efw)
            num_efw = efw.get_num_efw()
    except :
        print('The SDK library for EFW is required')
        num_efw = 0

    if num_efw == 0:
        labelFW.config(text = "FW OFF")
        flag_filter_wheel = False
        print("No filter wheel find")
        fw_position = 0
        presence_FW.set(0)  # Initialisation présence FW
        fw_position_.set(0)
    else :
        labelFW.config(text = "FW ON")
        print("Filter Wheel Detected")
        flag_filter_wheel = True
        efw_id = 0
        filter_wheel = efw.EFW(efw_id)
        filter_wheel.open()
        fw_position = 0
        filter_wheel.set_position(fw_position)
        fw_position_.set(0)
        presence_FW.set(1)  # Initialisation présence FW


def angle2degminsec(angle) :

    deg = int(angle)
    minute = int((angle - int(angle)) * 60)
    sec = int((angle - (deg + minute / 60)) * 3600)
    result = str(deg) + "d "+str(abs(minute))+"' " + str(abs(sec)) + "''"
    return result

def on_press(key):
    global key_pressed
    
    if Dev_system == "Linux" :
        key_pressed = ""
        try :
            if key.char == red_up :
                key_pressed = "RED_UP"
            if key.char == red_down :
                key_pressed = "RED_DOWN"
            if key.char == red_right :
                key_pressed = "RED_RIGHT"
            if key.char == red_left :
                key_pressed = "RED_LEFT"
            if key.char == red_reset :
                key_pressed = "RED_RESET"
            if key.char == blue_up :
                key_pressed = "BLUE_UP"
            if key.char == blue_down :
                key_pressed = "BLUE_DOWN"
            if key.char == blue_right :
                key_pressed = "BLUE_RIGHT"
            if key.char == blue_left :
                key_pressed = "BLUE_LEFT"
            if key.char == blue_reset :
                key_pressed = "BLUE_RESET"
            if key.char == stab_up :
                key_pressed = "STAB_UP"
            if key.char == stab_down :
                key_pressed = "STAB_DOWN"
            if key.char == stab_right :
                key_pressed = "STAB_RIGHT"
            if key.char == stab_left :
                key_pressed = "STAB_LEFT"
            if keyboard.is_pressed(stab_zone_more) :
                key_pressed = "STAB_ZONE_MORE"
            if keyboard.is_pressed(stab_zone_less) :
                key_pressed = "STAB_ZONE_LESS"
        except :       
            if key == zoom_up :
                key_pressed = "ZOOM_UP"
            if key == zoom_down :
                key_pressed = "ZOOM_DOWN"
            if key == zoom_right :
                key_pressed = "ZOOM_RIGHT"
            if key == zoom_left :
                key_pressed = "ZOOM_LEFT"
            if key == zoom_reset :
                key_pressed = "ZOOM_RESET"

def on_release(key):
    pass

def start_keyboard() :
    global flag_keyboard_management,thread_3
    
    flag_keyboard_management = True
    thread_3 = keyboard_management("3")
    thread_3.start()

class keyboard_management(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global key_pressed,flag_keyboard_management

        while flag_keyboard_management == True :
            key_pressed = ""
            if Dev_system == "Windows" :
                if keyboard.is_pressed(red_up) :
                    key_pressed = "RED_UP"
                if keyboard.is_pressed(red_down) :
                    key_pressed = "RED_DOWN"
                if keyboard.is_pressed(red_right) :
                    key_pressed = "RED_RIGHT"
                if keyboard.is_pressed(red_left) :
                    key_pressed = "RED_LEFT"
                if keyboard.is_pressed(red_reset) :
                    key_pressed = "RED_RESET"
                if keyboard.is_pressed(blue_up) :
                    key_pressed = "BLUE_UP"
                if keyboard.is_pressed(blue_down) :
                    key_pressed = "BLUE_DOWN"
                if keyboard.is_pressed(blue_right) :
                    key_pressed = "BLUE_RIGHT"
                if keyboard.is_pressed(blue_left) :
                    key_pressed = "BLUE_LEFT"
                if keyboard.is_pressed(blue_reset) :
                    key_pressed = "BLUE_RESET"
                if keyboard.is_pressed(zoom_up) :
                    key_pressed = "ZOOM_UP"
                if keyboard.is_pressed(zoom_down) :
                    key_pressed = "ZOOM_DOWN"
                if keyboard.is_pressed(zoom_right) :
                    key_pressed = "ZOOM_RIGHT"
                if keyboard.is_pressed(zoom_left) :
                    key_pressed = "ZOOM_LEFT"
                if keyboard.is_pressed(zoom_reset) :
                    key_pressed = "ZOOM_RESET"
                if keyboard.is_pressed(stab_up) :
                    key_pressed = "STAB_UP"
                if keyboard.is_pressed(stab_down) :
                    key_pressed = "STAB_DOWN"
                if keyboard.is_pressed(stab_right) :
                    key_pressed = "STAB_RIGHT"
                if keyboard.is_pressed(stab_left) :
                    key_pressed = "STAB_LEFT"
                if keyboard.is_pressed(stab_zone_more) :
                    key_pressed = "STAB_ZONE_MORE"
                if keyboard.is_pressed(stab_zone_less) :
                    key_pressed = "STAB_ZONE_LESS"
            time.sleep(0.02)

    def stop(self) :
        global flag_keyboard_management
        
        flag_keyboard_management = False
        
    def reprise(self) :
        global flag_keyboard_management
        
        flag_keyboard_management = True

def start_mount() :
    global flag_acquisition_mount,thread_2
    
    flag_autorise_acquisition = True
    flag_stop_acquisition = False
    thread_2 = acquisition_mount("2")
    thread_2.start()

class acquisition_mount(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global smc,azimut,hauteur,flag_mount_connect,flag_mountpos,flag_acquisition_mount,delta_azimut,delta_hauteur,azimut_monture,hauteur_monture

        while flag_acquisition_mount == True :
            if flag_mountpos == True :
                if flag_mount_connect == False :
                    try :
                        smc=synscan.motors()
                        flag_mount_connect = True
                    except Exception as error :
                        print("Mount error : ", error)
                        flag_mount_connect = False
                if flag_mount_connect == True :
                    try :
                        azimut_monture = smc.axis_get_pos(1)
                        hauteur_monture = smc.axis_get_pos(2)
                        azimut = azimut_monture + delta_azimut
                        hauteur = hauteur_monture + delta_hauteur
                        if azimut < 0 :
                            azimut = 360 + azimut
                        if azimut > 360 :
                            azimut = azimut - 360
                    except :
                        time.sleep(0.2)
            time.sleep(0.2)

    def stop(self) :
        global flag_acquisition_mount
        
        flag_acquisition_mount = False
        
    def reprise(self) :
        global flag_acquisition_mount
        
        flag_acquisition_mount = True

        
def camera_acquisition() :
    global type_debayer,camera,nb_erreur,image_brute,flag_autorise_acquisition,image_camera,key_pressed,flag_stop_acquisition,\
            flag_image_disponible,flag_noir_blanc,flag_hold_picture,image_camera_old,cupy_context,delta_RX,delta_RY,delta_BX,delta_BY,mode_BIN,flag_HB,\
            numero_image,timeoutexp,flag_new_image,flag_filtrage_ON,flag_filtre_work,flag_IsColor,labelInfo10,\
            threshold_16bits,choix_hard_bin,flag_capture_image_reference,Image_Reference,flag_image_reference_OK
        
    with cupy_context :
        try :
            ret = False
            if flag_16b == True :
                image_camera_tmp=camera.capture_video_frame_RAW16_CUPY(filename=None,timeout=timeoutexp)
                image_brute_cam16 = cp.asarray(image_camera_tmp,dtype=cp.uint16)
                image_brute_cam16[image_brute_cam16 > threshold_16bits] = threshold_16bits
                if mode_BIN == 2 :
                    if flag_HB == False :
                        image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0) * 4.0
                    else :
                        image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0)
                else :
                    image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0)
                image_brute_cam8 = cp.clip(image_brute_cam8,0,255)                                    
                image_brute_cam = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                if flag_capture_image_reference == True :
                    Image_Reference = np.asarray(image_brute_cam.get(),dtype=np.uint8)
                    flag_capture_image_reference = False
                    flag_image_reference_OK = True
                if flag_image_ref_sub == True and flag_image_reference_OK == True :
                    try :
                        image_2_subtract = np.asarray(image_brute_cam.get(),dtype=np.uint8)
                        image_subracted_np = cv2.subtract(image_2_subtract,Image_Reference)
                        image_brute_cam = cp.asarray(image_subracted_np,dtype=cp.uint8)
                    except :
                        pass
            else :
                image_brute_cam = camera.capture_video_frame_RAW8_CUPY(filename=None,timeout=timeoutexp)
                if flag_capture_image_reference == True :
                    Image_Reference = np.asarray(image_brute_cam.get(),dtype=np.uint8)
                    flag_capture_image_reference = False
                    flag_image_reference_OK = True
                if flag_image_ref_sub == True and flag_image_reference_OK == True :
                    try :
                        image_2_subtract = np.asarray(image_brute_cam.get(),dtype=np.uint8)
                        image_subracted_np = cv2.subtract(image_2_subtract,Image_Reference)
                        image_brute_cam = cp.asarray(image_subracted_np,dtype=cp.uint8)
                    except :
                        pass
            if flag_filtrage_ON == False :
                res_bb = image_brute_cam
                res_gg = image_brute_cam
                res_rr = image_brute_cam
            else :
                if flag_noir_blanc == 0 and flag_colour_camera == True :
                    res_rr = cp.zeros_like(image_brute_cam,dtype=cp.uint8)
                    res_gg = cp.zeros_like(image_brute_cam,dtype=cp.uint8)
                    res_bb = cp.zeros_like(image_brute_cam,dtype=cp.uint8)
                    img = image_brute_cam
                    height,width = image_brute_cam.shape
                    if  flag_hot_pixels == True :
                        Pixel_threshold = 80
                        nb_blocksX = (width // nb_ThreadsX) + 1
                        nb_blocksY = (height // nb_ThreadsY) + 1
                        Dead_Pixels_Remove_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_rr, img,  np.intc(width), np.intc(height), np.intc(Pixel_threshold), np.intc(GPU_BAYER)))
                        img = res_rr1.copy()
                    nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                    nb_blocksY = ((height // 2) // nb_ThreadsY) + 1
                    Image_Debayer_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_rr, res_gg, res_bb, img, np.intc(width), np.intc(height), np.intc(GPU_BAYER)))
                    if key_pressed != "" :
                        if key_pressed == "RED_UP" :
                            delta_RY = delta_RY + 1
                            key_pressed = ""
                        if key_pressed == "RED_DOWN" :
                            delta_RY = delta_RY - 1
                            key_pressed = ""
                        if key_pressed == "RED_RIGHT" :
                            delta_RX = delta_RX - 1
                            key_pressed = ""
                        if key_pressed == "RED_LEFT" :
                            delta_RX = delta_RX + 1
                            key_pressed = ""
                        if key_pressed == "RED_RESET" :
                            delta_RX = 0
                            delta_RY = 0
                            key_pressed = ""
                        if key_pressed == "BLUE_UP" :
                            delta_BY = delta_BY + 1
                            key_pressed = ""
                        if key_pressed == "BLUE_DOWN" :
                            delta_BY = delta_BY - 1
                            key_pressed = ""
                        if key_pressed == "BLUE_RIGHT" :
                            delta_BX = delta_BX - 1
                            key_pressed = ""
                        if key_pressed == "BLUE_LEFT" :
                            delta_BX = delta_BX + 1
                            key_pressed = ""
                        if key_pressed == "BLUE_RESET" :
                            delta_BX = 0
                            delta_BY = 0
                            key_pressed = ""
                    if delta_RX !=0 or delta_RY !=0 or delta_BX !=0 or delta_BY != 0 :
                        texte = "Shift Red : " + str(delta_RX) + " : "+str(delta_RY) + "    Shift Blue : " + str(delta_BX) + " : " + str(delta_BY) + "             "
                        labelInfo10.config(text = texte)
                        img_r = res_r.copy()
                        img_g = res_g.copy()
                        img_b = res_b.copy()
                        res_r = cp.zeros_like(img_r,dtype=cp.uint8)
                        res_g = cp.zeros_like(img_g,dtype=cp.uint8)
                        res_b = cp.zeros_like(img_b,dtype=cp.uint8)
                        nb_blocksX = (width // nb_ThreadsX) + 1
                        nb_blocksY = (height // nb_ThreadsY) + 1
                        RGB_Align_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_rr, res_gg, res_bb, img_r, img_g, img_b, np.intc(width), np.intc(height), np.intc(delta_RX), np.intc(delta_RY), np.intc(delta_BX), np.intc(delta_BY)))
                    Dim = 3
                    flag_IsColor = True
                else :
                    Dim = 1
                    if flag_colour_camera == False :
                        if  flag_hot_pixels == True :
                            img = cp.asarray(image_brute_cam,dtype=cp.uint8)
                            Pixel_threshold = 80
                            height,width = image_brute_cam.shape
                            nb_blocksX = (width // nb_ThreadsX) + 1
                            nb_blocksY = (height // nb_ThreadsY) + 1
                            Dead_Pixels_Remove_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, img, np.intc(width), np.intc(height), np.intc(Pixel_threshold)))
                            res_bb1 = res_r.copy()
                        else :
                            res_bb = image_brute_cam
                        flag_IsColor = False
                    else :
                        res_r = cp.zeros_like(image_brute_cam,dtype=cp.uint8)
                        img = image_brute_cam
                        height,width = image_brute_cam.shape
                        if  flag_hot_pixels == True :
                            Pixel_threshold = 80
                            nb_blocksX = (width // nb_ThreadsX) + 1
                            nb_blocksY = (height // nb_ThreadsY) + 1
                            Dead_Pixels_Remove_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, img, np.intc(width), np.intc(height), np.intc(Pixel_threshold), np.intc(GPU_BAYER)))
                            img = res_r.copy()
                        nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                        nb_blocksY = ((height // 2) // nb_ThreadsY) + 1
                        Image_Debayer_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, img, np.intc(width), np.intc(height), np.intc(GPU_BAYER)))
                        res_bb = res_r.copy()
                        flag_IsColor = False
                if flag_STAB == True and flag_HDR == False and flag_filtrage_ON == True :
                    if Dim == 3 :
                        image_brute = cupy_separateRGB_2_numpy_RGBimage(res_rr,res_gg,res_bb)
                        image_brute = Template_tracking(image_brute,Dim)
                        res_rr,res_gg,res_bb = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                    else :
                        image_brute = res_bb.get()
                        image_brute = Template_tracking(image_brute,Dim)
                        res_bb = cp.asarray(image_brute,dtype=cp.uint8)
            if flag_IsColor == False :
                res_rr = res_bb.copy()
                res_gg = res_bb.copy()
            image_camera_old = image_camera   
            image_camera = image_camera + 1
            flag_new_image = True
            ret = True
        except Exception as error :
            camera.stop_video_capture()
            camera.stop_exposure()
            time.sleep(0.2)
            camera.start_video_capture()
            nb_erreur += 1
            print("An error occurred : ", error)
            print("Capture error : ",nb_erreur)
            ret = False
            res_rr = 0
            res_gg = 0
            res_bb = 0
    return ret,res_rr,res_gg,res_bb
     

def refresh() :
    global video,flag_cap_video,camera,traitement,cadre_image,image_brute,flag_image_disponible,flag_quitter,timeoutexp,flag_new_image,curFPS,fpsQueue,image_brut_read,delta_zx,delta_zy,flag_new_stab_window,\
           flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2,image_traitee,val_SAT,exposition,total_start,total_stop,flag_image_video_loaded,flag_BFReference,\
           val_gain, echelle2,val_exposition,echelle1,imggrey1,imggrey2,flag_DETECT_STARS,flag_TRKSAT,flag_REMSAT,flag_CONST,labelInfo2,Date_hour_image,image_camera,val_nb_capt_video,flag_image_mode,\
           calque_stars,calque_satellites,calque_TIP,flag_nouvelle_resolution,nb_sat,sat_x,sat_y,sat_s,res_bb1,res_gg1,res_rr1,image_camera_old,cupy_context,flag_TRIGGER,flag_filtrage_ON,flag_filtre_work,\
           compteur_images,numero_image,nb_erreur,image_brute_grey,flag_sat_detected,start_time_video,stop_time_video,time_exec_test,total_time,font,image_camera_old,flag_colour_camera,flag_iscolor,\
           image_camera,res_cam_x,res_cam_y,flag_premier_demarrage,Video_Test,TTQueue,curTT,max_sat,video_frame_number,video_frame_position,image_reconstructed,stars_x,strs_y,stars_s,nb_stars,\
           quality,max_quality,quality_pos,tmp_qual,flag_IsColor,mean_quality,SFN,min_qual,max_qual,val_BFR,SFN,frame_number,delta_tx,delta_ty,labelInfo10,flag_GO,BFREF_image,flag_BFREF_image,\
           delta_RX,delta_RY,delta_BX,delta_BY,track,track_crater_history,track_sat_history,track_sat,track_crater,key_pressed,DSW,echelle210,frame_position,h,flag_new_frame_position,\
           curFPS,SER_depth,flag_SER_file,res_cam_x_base,res_cam_y_base,flag_capture_image_reference,Image_Reference,flag_image_reference_OK,previous_frame_number

    with cupy_context :
    
        if flag_camera_ok == True :
        
            if flag_premier_demarrage == True :
                flag_premier_demarrage = False
                start_mount()
                if Dev_system == "Windows" :
                    start_keyboard()
            if flag_autoexposure_gain == True :
                val_gain = (int)(camera.get_control_value(asi.ASI_GAIN)[0])
                echelle2.set(val_gain)        
            if flag_autoexposure_exposition == True :
                val_exposition = (int)(camera.get_control_value(asi.ASI_EXPOSURE)[0]) // 1000
                echelle1.set(val_exposition)        
            if flag_stop_acquisition == False :
                if flag_HDR == True and flag_filtrage_ON == True :
                    try :
                        if flag_16b == False :
                            image_brute1=camera.capture_video_frame_RAW8_NUMPY(filename=None,timeout=timeoutexp)
                            val_exposition = echelle1.get()
                            if flag_acq_rapide == "Fast" :
                                exposition = val_exposition // 2
                            else :
                                exposition = val_exposition*1000 // 2
                            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                            timeoutHDR = timeoutexp // 2
                            time.sleep(0.1)
                            camera.start_video_capture()
                            image_brute2=camera.capture_video_frame_RAW8_NUMPY(filename=None,timeout=timeoutexp)
                            if flag_acq_rapide == "Fast" :
                                exposition = val_exposition // 4
                            else :
                                exposition = val_exposition*1000 // 4
                            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                            timeoutHDR = timeoutexp // 4
                            time.sleep(0.1)
                            camera.start_video_capture()
                            image_brute3=camera.capture_video_frame_RAW8_NUMPY(filename=None,timeout=timeoutexp)
                            if flag_acq_rapide == "Fast" :
                                exposition = val_exposition // 6
                            else :
                                exposition = val_exposition*1000 // 6
                            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                            timeoutHDR = timeoutexp // 6
                            time.sleep(0.1)
                            camera.start_video_capture()
                            image_brute4=camera.capture_video_frame_RAW8_NUMPY(filename=None,timeout=timeoutexp)
                            val_exposition = echelle1.get()
                            if flag_acq_rapide == "Fast" :
                                exposition = val_exposition
                            else :
                                exposition = val_exposition*1000
                            camera.set_control_value(asi.ASI_EXPOSURE, exposition)
                        else :
                            if (16 - TH_16B) <= 5 :
                                delta_th = (16 - TH_16B) / 3.0
                            else :
                                delta_th = 5.0 / 3.0
                            
                            thres4 = 2 ** TH_16B - 1
                            thres3 = 2 ** (TH_16B + delta_th) - 1
                            thres2 = 2 ** (TH_16B + delta_th * 2) - 1
                            thres1 = 2 ** (TH_16B + delta_th * 3) - 1
                            
                            image_camera_base = camera.capture_video_frame_RAW16_CUPY(filename=None,timeout=timeoutexp)
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres1] = thres1
                            if mode_BIN == 2 :
                                if flag_HB == False :
                                    image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0) * 4.0
                                else :
                                    image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
                            else :
                                image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute1 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres2] = thres2
                            if mode_BIN == 2 :
                                if flag_HB == False :
                                    image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0) * 4.0
                                else :
                                    image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
                            else :
                                image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute2 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres3] = thres3
                            if mode_BIN == 2 :
                                if flag_HB == False :
                                    image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0) * 4.0
                                else :
                                    image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
                            else :
                                image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute3 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres4] = thres4
                            if mode_BIN == 2 :
                                if flag_HB == False :
                                    image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0) * 4.0
                                else :
                                    image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
                            else :
                                image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute4 = image_brute_cam_tmp.get()

                        if mode_HDR == "Mertens" :
                            img_list = [image_brute1,image_brute2,image_brute3,image_brute4]                       
                            merge_mertens = cv2.createMergeMertens()
                            res_mertens = merge_mertens.process(img_list)
                            res_mertens_cp = cp.asarray(res_mertens,dtype=cp.float32)
                            image_brute_cp = cp.clip(res_mertens_cp*255, 0, 255).astype('uint8')
                            image_brute = image_brute_cp.get()
                            image_brute = cv2.cvtColor(image_brute, type_debayer)
                        else :
                            img_list = [image_brute1,image_brute2,image_brute3,image_brute4]
                            img_list = cp.asarray(img_list)
                            tempo_hdr = cp.mean(img_list,axis=0)
                            tempo_hdr_cp = cp.asarray(tempo_hdr,dtype=cp.uint8)
                            image_brute = tempo_hdr_cp.get()
                            image_brute = cv2.cvtColor(image_brute, type_debayer)
                            
                        compteur_images = compteur_images + 1
                        Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                        
                        if flag_STAB == True and flag_HDR == True and flag_filtrage_ON == True :
                            if image_brute.ndim == 3 :
                                flag_IsColor = True
                                Dim = 3
                            else :
                                flag_IsColor = False
                                Dim = 1
                            image_brute = Template_tracking(image_brute,Dim)
                        if image_brute.ndim == 3 :
                            flag_IsColor = True
                            res_bb1,res_gg1,res_rr1 = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                        else :
                            flag_IsColor = False
                            res_bb1 = cp.asarray(image_brute)
                        flag_image_disponible = True
                        flag_new_image = True
                        image_camera_old = image_camera   
                        image_camera = image_camera + 1
                        numero_image = image_camera - 1
                        ret_img = True
                    except Exception as error :
                        nb_erreur += 1
#                        print("An error occurred : ", error)
                        print("Capture error : ",nb_erreur)
                        time.sleep(0.01)
                        ret_img = False
                else :
                    ret_img,res_rr1,res_gg1,res_bb1 = camera_acquisition()
                Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                if ret_img == True :
                    compteur_images = compteur_images + 1
                    frame_number = frame_number + 1
                    flag_GO = True
                    if flag_BFR == True and flag_image_mode == False :
                        if flag_IsColor == True :
                            rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                            re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                            cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                            ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                            img_numpy_crop = cupy_separateRGB_2_numpy_RGBimage(res_rr1,res_gg1,res_bb1)
                            crop_im_grey = cv2.cvtColor(img_numpy_crop, cv2.COLOR_BGR2GRAY)
                            img_qual = Image_Quality(crop_im_grey,IQ_Method)
                            if img_qual > max_qual :
                                max_qual = img_qual
                            if img_qual < min_qual :
                                min_qual = img_qual
                            quality_threshold = min_qual + (max_qual - min_qual) * (val_BFR / 100)
                            if img_qual < quality_threshold :
                                flag_GO = False
                                SFN = SFN + 1
                            ratio = int((SFN / frame_number) * 1000) / 10
                            texte = "SFN : " + str(SFN) + "   B/T : "+str(ratio) + "  Thres : " + str(int(quality_threshold*10)/10) + "  Qual : " + str(int(img_qual*10)/10) + "             "
                            labelInfo10.config(text = texte)
                        else :
                            rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                            re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                            cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                            ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                            img_numpy_crop = res_bb1.get()
                            crop_Im = img_numpy_crop[rs:re,cs:ce]
                            img_qual = Image_Quality(crop_Im,IQ_Method)
                            if img_qual > max_qual :
                                max_qual = img_qual
                            if img_qual < min_qual :
                                 min_qual = img_qual
                            quality_threshold = min_qual + (max_qual - min_qual) * (val_BFR / 100)
                            if img_qual < quality_threshold :
                                flag_GO = False
                                SFN = SFN + 1
                            ratio = int((SFN / frame_number) * 1000) / 10 
                            texte = "SFN : " + str(SFN) + "   B/T : "+str(ratio) + "  Thres : " + str(int(quality_threshold*10)/10) + "  Qual : " + str(int(img_qual*10)/10) + "             "
                            labelInfo10.config(text = texte)

                    if flag_BFR == False and flag_BFREF == True and flag_BFReference == "BestFrame" and flag_image_mode == False :
                        if flag_IsColor == True :
                            rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                            re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                            cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                            ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                            img_numpy_crop = cupy_separateRGB_2_numpy_RGBimage(res_rr1,res_gg1,res_bb1)
                            crop_Im = img_numpy_crop[rs:re,cs:ce]
                            crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                            img_qual = Image_Quality(crop_im_grey,IQ_Method)
                            if img_qual > max_qual :
                                max_qual = img_qual
                                BFREF_image = cupy_separateRGB_2_numpy_RGBimage(res_bb1,res_gg1,res_rr1)
                                flag_BFREF_image = True
                        else :
                            rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                            re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                            cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                            ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                            img_numpy_crop = res_bb1.get()
                            crop_Im = img_numpy_crop[rs:re,cs:ce]
                            img_qual = Image_Quality(crop_Im,IQ_Method)
                            if img_qual > max_qual :
                                max_qual = img_qual
                                BFREF_image = res_bb1.get()
                                flag_BFREF_image = True
                                
                    if flag_filtrage_ON == True :
                        if flag_IsColor == True :
                            application_filtrage_color(res_rr1,res_gg1,res_bb1)
                        else :
                            application_filtrage_mono(res_bb1)
                    else :
                        image_traitee = res_bb1.get()
                        curTT = 0             
                    if flag_IQE == True :
                        if flag_IsColor == True :
                            rs = res_cam_y // 2 - res_cam_y // 8
                            re = res_cam_y // 2 + res_cam_y // 8
                            cs = res_cam_x // 2 - res_cam_x // 8
                            ce = res_cam_x // 2 + res_cam_x // 8
                            crop_Im = image_traitee[rs:re,cs:ce]
                            crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                            quality[quality_pos] = Image_Quality(crop_im_grey,IQ_Method)
                            if quality[quality_pos] > max_quality :
                                max_quality = quality[quality_pos]
                            quality_pos = quality_pos + 1
                            if quality_pos > 255 :
                                quality_pos = 1
                        else :
                            rs = res_cam_y // 2 - res_cam_y // 8
                            re = res_cam_y // 2 + res_cam_y // 8
                            cs = res_cam_x // 2 - res_cam_x // 8
                            ce = res_cam_x // 2 + res_cam_x // 8
                            crop_Im = image_traitee[rs:re,cs:ce]
                            quality[quality_pos] = Image_Quality(crop_Im,IQ_Method)
                            if quality[quality_pos] > max_quality :
                                max_quality = quality[quality_pos]
                            quality_pos = quality_pos + 1
                            if quality_pos > 255 :
                                quality_pos = 1
                    flag_new_image = False
                    if flag_AI_Craters == True and flag_crater_model_loaded == True:
                        if flag_IsColor == True :
                            image_model = image_traitee
                        else :
                            image_model = cv2.merge((image_traitee,image_traitee,image_traitee))
                        if flag_AI_Trace == True :
                            result_craters = model_craters_track.track(image_model, device = 0, half=True, conf = 0.05, persist = True, verbose=False)
                            result_craters2 = model_craters_track(image_model, conf = 0.05)[0]
                        else :
                            result_craters = model_craters_predict.predict(image_model, device = 0,max_det = 100, half=True, verbose=False)
                            result_craters2 = model_craters_predict(image_model, conf = 0.05)[0]
                        boxes_crater = result_craters2.boxes.xywh.cpu()
                        bboxes_crater = np.array(result_craters2.boxes.xyxy.cpu(), dtype="int")
                        classes_crater = np.array(result_craters2.boxes.cls.cpu(), dtype="int")
                        confidence_crater = result_craters2.boxes.conf.cpu()
                        if flag_AI_Trace == True :
                            track_crater_ids = (result_craters2.boxes.id.int().cpu().tolist() if result_craters2.boxes.id is not None else None)
                        else :
                            track_crater_ids = None
                        if track_crater_ids :
                            for cls, box, track_crater_id in zip(classes_crater, boxes_crater, track_crater_ids):
                                x, y, w1, h1 = box
                                object_name = model_craters_track.names[cls]
                                if object_name == "Small crater":
                                    BOX_COLOUR = (0, 255, 255)
                                if object_name == "Crater":
                                    BOX_COLOUR = (0, 255, 0)
                                if object_name == "Large crater":
                                    BOX_COLOUR = (255, 150, 30)
                                if flag_IsColor == False :
                                    BOX_COLOUR = (255, 255, 255)
                                track_crater = track_crater_history[track_crater_id]
                                track_crater.append((float(x), float(y)))  # x, y center point
                                if len(track_crater) > 30:  # retain 90 tracks for 90 frames
                                    track_crater.pop(0)
                                points = np.hstack(track_crater).astype(np.int32).reshape((-1, 1, 2))
                                if flag_AI_Trace == True :
                                    cv2.polylines(image_traitee, [points], isClosed=False, color=BOX_COLOUR, thickness=1)
                        for cls, bbox in zip(classes_crater, bboxes_crater):
                            (x, y, x2, y2) = bbox
                            if flag_AI_Trace == True :
                                object_name = model_craters_track.names[cls]
                            else :
                                object_name = model_craters_predict.names[cls]
                            if object_name == "Small crater":
                                BOX_COLOUR = (0, 255, 255)
                            if object_name == "Crater":
                                BOX_COLOUR = (0, 255, 0)
                            if object_name == "Large crater":
                                BOX_COLOUR = (255, 150, 30)
                            if flag_IsColor == False :
                                BOX_COLOUR = (255, 255, 255)
                            cv2.rectangle(image_traitee, (x, y), (x2, y2), BOX_COLOUR, 1)
                            cv2.putText(image_traitee, f"{object_name}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, 1)
#                            cv2.putText(image_traitee, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, BOX_COLOUR, 2)
                    if flag_AI_Satellites == True and flag_satellites_model_loaded == True :
                        flag_sat_OK,image_model = satellites_tracking_AI()
                        calque_satellites_AI = np.zeros_like(image_traitee)
                        if flag_sat_OK == True :
                            try :
                                if flag_AI_Trace == True :
                                    result_sat = model_satellites_track.track(image_model, tracker = Custom_satellites_model_tracker, device = 0, half=False, conf = 0.01, persist = True, verbose=False)
                                    result_sat2 = model_satellites_track(image_model, conf = 0.01)[0]
                                else :
                                    result_sat = model_satellites_predict.predict(image_model, device = 0,max_det = 100, half=True, verbose=False)
                                    result_sat2 = model_satellites_predict(image_model, conf = 0.1)[0]
                                model_OK = True
                            except :
                                model_OK = False
                            if model_OK == True :
                                boxes_sat = result_sat2.boxes.xywh.cpu()
                                bboxes_sat = np.array(result_sat2.boxes.xyxy.cpu(), dtype="int")
                                classes_sat = np.array(result_sat2.boxes.cls.cpu(), dtype="int")
                                confidence_sat = result_sat2.boxes.conf.cpu()
                                if flag_AI_Trace == True :
                                    track_sat_ids = (result_sat2.boxes.id.int().cpu().tolist() if result_sat2.boxes.id is not None else None)
                                else :
                                    track_sat_ids = None
                                if track_sat_ids :
                                    for cls, box, track_sat_id in zip(classes_sat, boxes_sat, track_sat_ids):
                                        x, y, w1, h1 = box
                                        object_name = model_satellites_track.names[cls]
                                        if object_name == "Shooting star":
                                            BOX_COLOUR = (255, 0, 0)
                                        if object_name == "Plane":
                                            BOX_COLOUR = (255, 255, 0)
                                            ep = 2
                                        if object_name == "Satellite":
                                            BOX_COLOUR = (0, 255, 0)
                                            ep = 1
                                        if flag_IsColor == False :
                                            BOX_COLOUR = (255, 255, 255)
                                        track_sat = track_satellite_history[track_sat_id]
                                        track_sat.append((float(x), float(y)))  # x, y center point
                                        if len(track_sat) > 30:  # retain 90 tracks for 90 frames
                                            track_sat.pop(0)
                                        points = np.hstack(track_sat).astype(np.int32).reshape((-1, 1, 2))
                                        if flag_AI_Trace == True :
                                            cv2.polylines(calque_satellites_AI, [points], isClosed=False, color=BOX_COLOUR, thickness=1)
                                for cls, bbox,conf in zip(classes_sat, bboxes_sat, confidence_sat):
                                    (x, y, x2, y2) = bbox
                                    if flag_AI_Trace == True :
                                        object_name = model_satellites_track.names[cls]
                                    else :
                                        object_name = model_satellites_predict.names[cls]
                                    if object_name == "Shooting star":
                                        bbox_text = "Sht Star"
                                        ep = 2
                                        BOX_COLOUR = (255, 0, 0)
                                    if object_name == "Plane":
                                        box_text = "Plane"
                                        ep = 2
                                        BOX_COLOUR = (255, 255, 0)
                                    if object_name == "Satellite":
                                        box_text = "Sat"
                                        ep = 1
                                        BOX_COLOUR = (0, 255, 0)
                                    if flag_IsColor == False :
                                        ep = 1
                                        BOX_COLOUR = (255, 255, 255)
                                    cv2.rectangle(calque_satellites_AI, (x, y), (x2, y2), BOX_COLOUR, 1)
                                    cv2.putText(calque_satellites_AI, f"{box_text}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, ep)                         
#                                    cv2.putText(calque_satellites_AI, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, ep)
                    if flag_filtrage_ON == True :
                        if flag_CONST == 0 :
                            if (flag_TRKSAT == 1 and flag_image_mode == False) and flag_REMSAT == 0 :
                                satellites_tracking()
                            if (flag_REMSAT == 1 and flag_image_mode == False) :
                                remove_satellites()
                            flag_sat_detected = False
                            if flag_DETECT_STARS == 1 and flag_image_mode == False :
                                stars_detection(True)
                            if flag_false_colours == True :
                                if flag_IsColor == True :
                                    tmp_grey = cv2.cvtColor(image_traitee, cv2.COLOR_BGR2GRAY)
                                    image_traitee = cv2.applyColorMap(tmp_grey, cv2.COLORMAP_RAINBOW)
                                else:
                                    image_traitee = cv2.applyColorMap(image_traitee, cv2.COLORMAP_RAINBOW)
                            if (flag_TRKSAT == 1  and nb_sat >= 0 and nb_sat < max_sat and flag_image_mode == False) and flag_REMSAT == 0 :
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                size = 0.5
                                for i in range(nb_sat+1):
                                    if correspondance[i] >=0:
                                        centercircle = (sat_x[i],sat_y[i])
                                        center_texte = (sat_x[i]+10,sat_y[i]+10)
                                        texte = "Sat"
                                        if flag_IsColor == True :
                                            cv2.circle(calque_direction_satellites, centercircle, 7, (0,255,0), 1, cv2.LINE_AA)
                                            cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (0, 255, 0), 1, cv2.LINE_AA)
                                            center_texte = (sat_x[i]+10,sat_y[i]+25)
                                            texte = "Rel Speed " + str(sat_speed[i])
                                            cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 0), 1, cv2.LINE_AA)
                                        else :
                                            cv2.circle(calque_direction_satellites, centercircle, 7, (255,255,255), 1, cv2.LINE_AA)
                                            cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 255), 1, cv2.LINE_AA)
                                            center_texte = (sat_x[i]+10,sat_y[i]+25)
                                            texte = "Rel Speed " + str(sat_speed[i])
                                            cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 255), 1, cv2.LINE_AA)
                                image_traitee = cv2.addWeighted(image_traitee, 1, calque_direction_satellites, 1, 0)     
                            if flag_DETECT_STARS == 1 and flag_image_mode == False :
                                image_traitee = cv2.addWeighted(image_traitee, 1, calque_stars, 1, 0)
                            if flag_AI_Satellites and flag_satellites_model_loaded == True :
                                image_traitee = cv2.addWeighted(image_traitee, 1, calque_satellites_AI, 1, 0)
                        else :
                            reconstruction_image()
                            image_traitee = image_reconstructed
                    total_stop = cv2.getTickCount()
                    total_time= int((total_stop-total_start)/cv2.getTickFrequency()*1000)
                    fpsQueue.append(1000/total_time)
                    if len(fpsQueue) > 5:
                        fpsQueue.pop(0)
                    curFPS = (sum(fpsQueue)/len(fpsQueue))
                    texte_TIP1 = Date_hour_image + "  Frame nbr : " + str(numero_image) + "  FPS : " + str(int(curFPS*10)/10)
                    if flag_16b == False :
                        texte_TIP2 = "Exp = " + str(int(exposition / 1000)) + "ms | Gain = " + str(val_gain) + " | 8 bits "
                    else :
                        texte_TIP2 = "Exp = " + str(int(exposition / 1000)) + "ms | Gain = " + str(val_gain) + " | 16 bits " + ": 1 to " + str(TH_16B) + " bits -> 8 bits display"
                    if flag_TIP == True :
                        if flag_IsColor == True and flag_filtrage_ON == True :
                            height,width,layers = image_traitee.shape
                        else :
                            height,width = image_traitee.shape 
                        pos1 = 30
                        pos2 = 70
                        if width > 2000 :
                            size = 1
                        else :
                            size = 0.5
                            pos1 = 20
                            pos2 = 50
                        if width < 600 :
                            size = 0.3
                            pos1 = 15
                            pos2 = 40
                        cv2.putText(image_traitee, texte_TIP1, (pos1,pos1), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(image_traitee, texte_TIP2, (pos1,pos2), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                    if flag_mountpos == True :
                        if flag_IsColor == True and flag_filtrage_ON == True :
                            height,width,layers = image_traitee.shape
                        else :
                            height,width = image_traitee.shape 
                        if width > 2000 :
                            size = 1
                            posAX = width // 2 - 120
                            posAY = height - 30
                            posHX = width - 220
                            posHY = height // 2 - 10
                        else :
                            size = 0.5
                            posAX = width // 2 - 100
                            posAY = height - 20
                            posHX = width - 150
                            posHY = height // 2 - 5
                        if width < 600 :
                            size = 0.3
                            posAX = width // 2 - 80
                            posAY = height - 15
                            posHX = width - 80
                            posHY = height // 2
                        texte_mount = angle2degminsec(azimut)
                        cv2.putText(image_traitee, texte_mount, (posAX,posAY), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                        texte_mount = angle2degminsec(hauteur)
                        cv2.putText(image_traitee, texte_mount, (posHX,posHY), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                    labelInfo2.config(text = str(curTT) + " ms   FPS : " + str(int(curFPS*10)/10) + "    ")
                    total_start = cv2.getTickCount()
                    if flag_cap_pic == True:
                        pic_capture()
                    if flag_cap_video == True :
                        video_capture(image_traitee)
                    if flag_new_stab_window == True :
                        cv2.rectangle(image_traitee,start_point,end_point, (255,0,0), 2, cv2.LINE_AA)
                        time.sleep(0.1)
                        flag_new_stab_window = False
                    if curFPS >= frame_limit :
                        if (numero_image % frame_skip) != 1 :
                            flag_display = True
                        else :
                            flag_display = False
                    else :
                        flag_display = True
                    if flag_display == True :
                        if flag_full_res == 0 :
                            if res_cam_x > int(1350*fact_s) :
                                if flag_OpenCvCuda == False :
                                    image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_LINEAR)
                                else :
                                    tmpbase = cv2.cuda_GpuMat()
                                    tmprsz = cv2.cuda_GpuMat()
                                    tmpbase.upload(image_traitee)
                                    tmprsz = cv2.cuda.resize(tmpbase,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_LINEAR)
                                    image_traitee_resize = tmprsz.download()
                                cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                            else :
                                cadre_image.im=PIL.Image.fromarray(image_traitee)
                        else :
                            if res_cam_x < int(1350*fact_s) :
                                image_traitee_resize = cv2.resize(image_traitee,(cam_displ_x,cam_displ_y),interpolation = cv2.INTER_LINEAR)
                                cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                            else :                   
                                old_dzx = delta_zx
                                old_dzy = delta_zy
                                if key_pressed == "ZOOM_UP" :
                                    delta_zy = delta_zy - 20
                                    key_pressed = ""
                                if key_pressed == "ZOOM_DOWN" :
                                    delta_zy = delta_zy + 20
                                    key_pressed = ""
                                if key_pressed == "ZOOM_RIGHT" :
                                    delta_zx = delta_zx + 20
                                    key_pressed = ""
                                if key_pressed == "ZOOM_LEFT" :
                                    delta_zx = delta_zx - 20
                                    key_pressed = ""
                                if key_pressed == "ZOOM_RESET" :
                                    delta_zx = 0
                                    delta_zy = 0
                                    key_pressed = ""
                                rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                                re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                                cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                                ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                                if cs < 0 or ce > res_cam_x :
                                    delta_zx = old_dzx
                                    cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                                    ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                                if rs < 0 or re > res_cam_y :
                                    delta_zy = old_dzy
                                    rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                                    re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                                if rs < 0 :
                                    rs = 0
                                    re = res_cam_y
                                if cs < 0 :
                                    cs = 0
                                    ce = res_cam_x
                                image_crop = image_traitee[rs:re,cs:ce]
                                cadre_image.im=PIL.Image.fromarray(image_crop)
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
                            if flag_HST == 1 and flag_IsColor == True :
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
                            if flag_HST == 1 and flag_IsColor == False :
                                r = cadre_image.im
                                hst_r = r.histogram()
                                histo = PIL.ImageDraw.Draw(cadre_image.im)
                                for x in range(1,256) :
                                    histo.line(((x*3,SY),(x*3,SY-hst_r[x]/200)),fill="white")
                                histo.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                                histo.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                            if flag_IQE == True :
                                transform = PIL.ImageDraw.Draw(cadre_image.im)
                                for x in range(2,256) :
                                    y2 = int((quality[x]/max_quality)*400)
                                    y1 = int((quality[x-1]/max_quality)*400)
                                    transform.line((((x-1)*3,SY-y1),(x*3,SY-y2)),fill="red",width=2)
                                transform.line(((256*3,SY),(256*3,SY-256*2)),fill="blue",width=3)
                                transform.line(((1,SY-256*2),(256*3,SY-256*2)),fill="blue",width=3)
                            if flag_TRSF == 1 and flag_IsColor == True :
                                transform = PIL.ImageDraw.Draw(cadre_image.im)
                                for x in range(2,256) :
                                    transform.line((((x-1)*3,SY-trsf_r[x-1]*2),(x*3,SY-trsf_r[x]*2)),fill="red",width=2)
                                    transform.line((((x-1)*3,SY-trsf_g[x-1]*2),(x*3,SY-trsf_g[x]*2)),fill="green",width=2)
                                    transform.line((((x-1)*3,SY-trsf_b[x-1]*2),(x*3,SY-trsf_b[x]*2)),fill="blue",width=2)
                                transform.line(((256*3,SY),(256*3,SY-256*2)),fill="red",width=3)
                                transform.line(((1,SY-256*2),(256*3,SY-256*2)),fill="red",width=3)
                            if flag_TRSF == 1 and flag_IsColor == False :
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
                        if flag_GO == False :
                            transform = PIL.ImageDraw.Draw(cadre_image.im)
                            transform.line(((0,0),(SX,SY)),fill="red",width=2)
                            transform.line(((0,SY),(SX,0)),fill="red",width=2)
                        if flag_BFR == True and flag_image_mode == False :
                            transform = PIL.ImageDraw.Draw(cadre_image.im)
                            mul_par = (SX-600) / max_qual
                            transform.line(((0,SY - 50),(int(min_qual*mul_par),SY - 50)),fill="red",width=4) # min quality
                            transform.line(((0,SY - 110),(int(max_qual*mul_par),SY - 110)),fill="blue",width=4) # max quality
                            transform.line(((0,SY - 80),(int(img_qual*mul_par),SY - 80)),fill="yellow",width=4) # image quality
                            transform.line(((int(quality_threshold*mul_par),SY - 55),(int(quality_threshold*mul_par),SY - 105)),fill="green",width=6)# threshold quality
                        cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
                        cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
                else :
                    print("No image to display")
            if flag_quitter == False:
                if flag_HDR == False :
                    fenetre_principale.after(2, refresh)
                else :
                    fenetre_principale.after(10, refresh)
        else :
            if flag_premier_demarrage == True :
                flag_premier_demarrage = False
                start_keyboard()
                if flag_image_mode == True :
                    image_brut_read = cv2.imread(Video_Test,cv2.IMREAD_COLOR)
                    image_brute = image_brut_read
                    flag_image_disponible = True
                    flag_image_mode = True
                    res_cam_y,res_cam_x,layer = image_brut_read.shape
                    flag_image_video_loaded = True
                if flag_image_mode == False :
                    video_frame_position = 1
                    if flag_SER_file == False :
                        video = cv2.VideoCapture(Video_Test, cv2.CAP_FFMPEG)
                        property_id = int(cv2.CAP_PROP_FRAME_WIDTH)
                        res_cam_x_base = int(cv2.VideoCapture.get(video, property_id))
                        property_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
                        res_cam_y_base = int(cv2.VideoCapture.get(video, property_id))
                        property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                        video_frame_number = int(cv2.VideoCapture.get(video, property_id))
                    else :
                        video = Serfile.Serfile(Video_Test,NEW=False)
                        res_cam_x_base = video.getWidth()
                        res_cam_y_base = video.getHeight()
                        video_frame_number = video.getLength()
                        SER_depth = video.getpixeldepth()
                    echelle210 = Scale (cadre, from_ = 0, to = video_frame_number, command= choix_position_frame, orient=HORIZONTAL, length = 1350*fact_s, width = 7, resolution = 1, label="",showvalue=1,tickinterval=100,sliderlength=20)
                    echelle210.set(video_frame_position)
                    echelle210.place(anchor="w", x=70,y=h-30)
                    flag_image_mode = False
                    flag_image_video_loaded = True
            if flag_cap_video == True and nb_cap_video == 1 and flag_image_mode == False :
                if flag_SER_file == False :
                    video.release()
                    time.sleep(0.1)
                    video = cv2.VideoCapture(Video_Test, cv2.CAP_FFMPEG)
                    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                    val_nb_capt_video = int(cv2.VideoCapture.get(video, property_id))
                    property_id = int(cv2.CAP_PROP_FRAME_WIDTH)
                    res_cam_x_base = int(cv2.VideoCapture.get(video, property_id))
                    property_id = int(cv2.CAP_PROP_FRAME_HEIGHT)
                    res_cam_y_base = int(cv2.VideoCapture.get(video, property_id))
                    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
                    video_frame_number = int(cv2.VideoCapture.get(video, property_id))
                else :
                    video_frame_position = 0
                    video.setCurrentPosition(video_frame_position)
                    video_frame_number = video.getLength()
                flag_image_video_loaded = True
                time.sleep(0.1)
            if flag_image_mode == False :
                if flag_SER_file == False :
                    video_frame_position = video_frame_position + 1
                    ret,image_brute = video.read()
                    if flag_capture_image_reference == True :
                        Image_Reference = np.asarray(image_brute,dtype=np.uint8)
                        flag_capture_image_reference = False
                        flag_image_reference_OK = True
                    if flag_image_ref_sub == True and flag_image_reference_OK == True :
                        image_2_subtract = np.asarray(image_brute,dtype=np.uint8)
                        image_brute = cv2.subtract(image_2_subtract,Image_Reference)
                else :
                    video_frame_position = video_frame_position + 1
                    image_brute = video.readFrameAtPos(video_frame_position)
                    ret = True
                    if SER_depth == 16 :
                        if flag_HDR == True and flag_filtrage_ON == True and type_debayer > 0 : 
                        
                            if (16 - TH_16B) <= 5 :
                                delta_th = (16 - TH_16B) / 3.0
                            else :
                                delta_th = 5.0 / 3.0
                            
                            thres4 = 2 ** TH_16B - 1
                            thres3 = 2 ** (TH_16B + delta_th) - 1
                            thres2 = 2 ** (TH_16B + delta_th * 2) - 1
                            thres1 = 2 ** (TH_16B + delta_th * 3) - 1

                            image_camera_base = cp.asarray(image_brute,dtype=cp.uint16)
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres1] = thres1
                            image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute1 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres2] = thres2
                            image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute2 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres3] = thres3
                            image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute3 = image_brute_cam_tmp.get()
                            
                            image_brute_cam16 = image_camera_base.copy()
                            image_brute_cam16[image_brute_cam16 > thres4] = thres4
                            image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
                            image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
                            image_brute4 = image_brute_cam_tmp.get()

                            if mode_HDR == "Mertens" :
                                img_list = [image_brute1,image_brute2,image_brute3,image_brute4]                       
                                merge_mertens = cv2.createMergeMertens()
                                res_mertens = merge_mertens.process(img_list)
                                res_mertens_cp = cp.asarray(res_mertens,dtype=cp.float32)
                                image_brute_cp = cp.clip(res_mertens_cp*255, 0, 255).astype('uint8')
                                image_brute = image_brute_cp.get()
                                image_brute = cv2.cvtColor(image_brute, type_debayer)
                            else :
                                img_list = [image_brute1,image_brute2,image_brute3,image_brute4]
                                img_list = cp.asarray(img_list)
                                tempo_hdr = cp.mean(img_list,axis=0)
                                tempo_hdr_cp = cp.asarray(tempo_hdr,dtype=cp.uint8)
                                image_brute = tempo_hdr_cp.get()
                                image_brute = cv2.cvtColor(image_brute, type_debayer)
                        else :
                            image_brute_cam16 = cp.asarray(image_brute,dtype=cp.uint16)
                            image_brute_cam16[image_brute_cam16 > threshold_16bits] = threshold_16bits
                            image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0)
                            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
                            image_brute_cam8_np = image_brute_cam8.get()
                            image_brute = np.asarray(image_brute_cam8_np,dtype=np.uint8)
                            if flag_capture_image_reference == True :
                                Image_Reference = np.asarray(image_brute,dtype=np.uint8)
                                flag_capture_image_reference = False
                                flag_image_reference_OK = True
                            if flag_image_ref_sub == True and flag_image_reference_OK == True :
                                try :
                                    image_2_subtract = np.asarray(image_brute,dtype=np.uint8)
                                    image_brute = cv2.subtract(image_2_subtract,Image_Reference)
                                except :
                                    pass
                    else :
                        image_brute = np.asarray(image_brute,dtype=np.uint8)
                        if flag_capture_image_reference == True :
                            Image_Reference = np.asarray(image_brute,dtype=np.uint8)
                            flag_capture_image_reference = False
                            flag_image_reference_OK = True
                        if flag_image_ref_sub == True and flag_image_reference_OK == True :
                            try :
                                image_2_subtract = np.asarray(image_brute,dtype=np.uint8)
                                image_brute = cv2.subtract(image_2_subtract,Image_Reference)
                            except :
                                pass
                if flag_new_frame_position == False :
                    if flag_SER_file == False :
                        property_id = int(cv2.CAP_PROP_POS_FRAMES)
                        video_frame_position = int(cv2.VideoCapture.get(video, property_id))
                    else :
                        video_frame_number = video.getLength()
                    if (video_frame_position % 5) == 0 :
                        echelle210.set(video_frame_position)
                else :
                    flag_new_frame_position = False
                if video_frame_position >= video_frame_number - 1:
                    if flag_SER_file == False :
                        video_frame_position = 1
                        video.set(cv2.CAP_PROP_POS_FRAMES,frame_position)
                    else :
                        video_frame_position = 1
                        video.setCurrentPosition(video_frame_position)
                    echelle210.set(video_frame_position)       
                Pixel_threshold = 80
                if ret == True :
                    if type_flip == "vertical" or type_flip == "both" :
                        image_brute = cv2.flip(image_brute,0)
                    if type_flip == "horizontal" or type_flip == "both" :
                        image_brute = cv2.flip(image_brute,1)
                    if flag_noir_blanc == 1 and image_brute.ndim == 3 and GPU_BAYER == 0 :
                        image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                    if image_brute.ndim == 3 :
                        flag_IsColor = True
                    else :
                        flag_IsColor = False
                    if flag_IsColor == False :
                        if  flag_hot_pixels == True and flag_noir_blanc == 0 :
                            res_r = cp.zeros_like(image_brute,dtype=cp.uint8)
                            img = cp.asarray(image_brute,dtype=cp.uint8)
                            height,width = image_brute.shape
                            nb_blocksX = (width // nb_ThreadsX) + 1
                            nb_blocksY = (height // nb_ThreadsY) + 1
                            Dead_Pixels_Remove_Mono_GPU((nb_blocksX*2,nb_blocksY*2),(nb_ThreadsX,nb_ThreadsY),(res_r, img,  np.intc(width), np.intc(height), np.intc(Pixel_threshold)))
                            image_brute = res_r.get()
                    if GPU_BAYER != 0 and flag_HDR == False :
                        flag_IsColor = True
                        if image_brute.ndim == 3 :
                            r = image_brute[:,:,0].copy()
                        else :
                            r = image_brute.copy()
                        if  flag_hot_pixels == True :
                            res_r = cp.zeros_like(r,dtype=cp.uint8)
                            img = cp.asarray(r,dtype=cp.uint8)
                            height,width = r.shape
                            nb_blocksX = (width // nb_ThreadsX) + 1
                            nb_blocksY = (height // nb_ThreadsY) + 1
                            Dead_Pixels_Remove_Colour_GPU((nb_blocksX*2,nb_blocksY*2),(nb_ThreadsX,nb_ThreadsY),(res_r, img,  np.intc(width), np.intc(height), np.intc(Pixel_threshold), np.intc(GPU_BAYER)))
                            r = res_r.get()
                        res_r = cp.zeros_like(r,dtype=cp.uint8)
                        res_g = cp.zeros_like(r,dtype=cp.uint8)
                        res_b = cp.zeros_like(r,dtype=cp.uint8)
                        img = cp.asarray(r,dtype=cp.uint8)
                        height,width = r.shape
                        nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                        nb_blocksY = ((height // 2) // nb_ThreadsY) + 1
                        Image_Debayer_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, res_g, res_b, img, np.intc(width), np.intc(height), np.intc(GPU_BAYER)))
                        if key_pressed != "" :
                            if key_pressed == "RED_UP" :
                                delta_RY = delta_RY + 1
                                key_pressed = ""
                            if key_pressed == "RED_DOWN" :
                                delta_RY = delta_RY - 1
                                key_pressed = ""
                            if key_pressed == "RED_RIGHT" :
                                delta_RX = delta_RX - 1
                                key_pressed = ""
                            if key_pressed == "RED_LEFT" :
                                delta_RX = delta_RX + 1
                                key_pressed = ""
                            if key_pressed == "RED_RESET" :
                                delta_RX = 0
                                delta_RY = 0
                                key_pressed = ""
                            if key_pressed == "BLUE_UP" :
                                delta_BY = delta_BY + 1
                                key_pressed = ""
                            if key_pressed == "BLUE_DOWN" :
                                delta_BY = delta_BY - 1
                                key_pressed = ""
                            if key_pressed == "BLUE_RIGHT" :
                                delta_BX = delta_BX - 1
                            if key_pressed == "BLUE_LEFT" :
                                delta_BX = delta_BX + 1
                                key_pressed = ""
                            if key_pressed == "BLUE_RESET" :
                                delta_BX = 0
                                delta_BY = 0
                                key_pressed = ""
                            texte = "Shift Red : " + str(delta_RX) + " : "+str(delta_RY) + "    Shift Blue : " + str(delta_BX) + " : " + str(delta_BY) + "             "
                            labelInfo10.config(text = texte)
                        if delta_RX !=0 or delta_RY !=0 or delta_BX !=0 or delta_BY != 0 :
                            height,width = r.shape
                            nb_blocksX = (width // nb_ThreadsX) + 1
                            nb_blocksY = (height // nb_ThreadsY) + 1
                            img_r = res_r.copy()
                            img_g = res_g.copy()
                            img_b = res_b.copy()
                            RGB_Align_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, res_g, res_b, img_r, img_g, img_b, np.intc(width), np.intc(height), np.intc(delta_RX), np.intc(delta_RY), np.intc(delta_BX), np.intc(delta_BY)))
                        if Dev_system == "Windows" :
                            temp_r = res_r.copy()
                            temp_g = res_g.copy()
                            temp_b = res_b.copy()
                            if flag_noir_blanc == 1 : # and image_brute.ndim == 3 :
                                image_brute = cupy_separateRGB_2_numpy_RGBimage(temp_r,temp_g,temp_b)
                                image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                                flag_IsColor = False
                            else :
                                image_brute = cupy_separateRGB_2_numpy_RGBimage(temp_r,temp_g,temp_b)
                                flag_IsColor = True
                        else :
                            temp_r = res_r.get()
                            temp_g = res_g.get()
                            temp_b = res_b.get()
                            if flag_noir_blanc == 1 : # and image_brute.ndim == 3 :
                                image_brute = cv2.merge((temp_r,temp_g,temp_b))
                                image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                                flag_IsColor = False
                            else :
                                image_brute = cv2.merge((temp_r,temp_g,temp_b))
                                flag_IsColor = True
                    else :
                        if flag_IsColor == True :
                            if key_pressed == "RED_UP" :
                                delta_RY = delta_RY + 1
                                key_pressed = ""
                            if key_pressed == "RED_DOWN" :
                                delta_RY = delta_RY - 1
                                key_pressed = ""
                            if key_pressed == "RED_RIGHT" :
                                delta_RX = delta_RX - 1
                                key_pressed = ""
                            if key_pressed == "RED_LEFT" :
                                delta_RX = delta_RX + 1
                                key_pressed = ""
                            if key_pressed == "RED_RESET" :
                                delta_RX = 0
                                delta_RY = 0
                                key_pressed = ""
                            if key_pressed == "BLUE_UP" :
                                delta_BY = delta_BY + 1
                                key_pressed = ""
                            if key_pressed == "BLUE_DOWN" :
                                delta_BY = delta_BY - 1
                                key_pressed = ""
                            if key_pressed == "BLUE_RIGHT" :
                                delta_BX = delta_BX - 1
                                key_pressed = ""
                            if key_pressed == "BLUE_LEFT" :
                                delta_BX = delta_BX + 1
                                key_pressed = ""
                            if key_pressed == "BLUE_RESET" :
                                delta_BX = 0
                                delta_BY = 0
                                key_pressed = ""
                            texte = "Shift Red : " + str(delta_RX) + " : "+str(delta_RY) + "    Shift Blue : " + str(delta_BX) + " : " + str(delta_BY) + "             "
                            labelInfo10.config(text = texte)
                            if delta_RX !=0 or delta_RY !=0 or delta_BX !=0 or delta_BY != 0 :
                                img_r,img_g,img_b = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                                res_r = cp.zeros_like(img_r,dtype=cp.uint8)
                                res_g = cp.zeros_like(img_g,dtype=cp.uint8)
                                res_b = cp.zeros_like(img_b,dtype=cp.uint8)
                                height,width,layer = image_brute.shape
                                nb_blocksX = (width // nb_ThreadsX) + 1
                                nb_blocksY = (height // nb_ThreadsY) + 1
                                RGB_Align_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, res_g, res_b, img_r, img_g, img_b, np.intc(width), np.intc(height), np.intc(delta_RX), np.intc(delta_RY), np.intc(delta_BX), np.intc(delta_BY)))
                                image_brute = cupy_separateRGB_2_numpy_RGBimage(res_r,res_g,res_b)
                    flag_image_video_loaded = True
                    res_cam_y = res_cam_y_base
                    res_cam_x = res_cam_x_base
                    if mode_BIN == 2 :
                        if flag_HB == False :
                            BIN_mode = 0 # BIN sum
                        else :
                            BIN_mode = 1 # BIN mean
                        if flag_IsColor == True :
                            img_r,img_g,img_b = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                            height,width,layer = image_brute.shape
                            res_r = cp.zeros([height//2,width//2],dtype=cp.uint8)
                            res_g = cp.zeros([height//2,width//2],dtype=cp.uint8)
                            res_b = cp.zeros([height//2,width//2],dtype=cp.uint8)
                            nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                            nb_blocksY = ((height //2) // nb_ThreadsY) + 1
                            BIN_Color_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, res_g, res_b, img_r, img_g, img_b, np.intc(width//2), np.intc(height//2), np.intc(BIN_mode)))
                            image_brute = cupy_separateRGB_2_numpy_RGBimage(res_r,res_g,res_b)
                            res_cam_y = height // 2
                            res_cam_x = width // 2
                        else :
                            img_r = cp.asarray(image_brute,dtype=cp.uint8)
                            height,width = image_brute.shape
                            res_r = cp.zeros([height//2,width//2],dtype=cp.uint8)
                            nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                            nb_blocksY = ((height //2) // nb_ThreadsY) + 1
                            BIN_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, img_r, np.intc(width//2), np.intc(height//2), np.intc(BIN_mode)))
                            image_brute = res_r.get()
                            res_cam_y = height // 2
                            res_cam_x = width // 2  
                    if flag_STAB == True :
                        if flag_IsColor == True :
                            image_brute = Template_tracking(image_brute,3)
                        else :
                            image_brute = Template_tracking(image_brute,1)
                else :
                    flag_image_video_loaded = False
            if flag_image_mode == True :
                image_brute = image_brut_read
                if type_flip == "vertical" or type_flip == "both" :
                    image_brute = cv2.flip(image_brute,0)
                if type_flip == "horizontal" or type_flip == "both" :
                    image_brute = cv2.flip(image_brute,1)
                res_cam_y,res_cam_x,layer = image_brute.shape
                if key_pressed == "RED_UP" :
                    delta_RY = delta_RY + 1
                    key_pressed = ""
                if key_pressed == "RED_DOWN" :
                    delta_RY = delta_RY - 1
                    key_pressed = ""
                if key_pressed == "RED_RIGHT" :
                    delta_RX = delta_RX - 1
                    key_pressed = ""
                if key_pressed == "RED_LEFT" :
                    delta_RX = delta_RX + 1
                    key_pressed = ""
                if key_pressed == "RED_RESET" :
                    delta_RX = 0
                    delta_RY = 0
                    key_pressed = ""
                if key_pressed == "BLUE_UP" :
                    delta_BY = delta_BY + 1
                    key_pressed = ""
                if key_pressed == "BLUE_DOWN" :
                    delta_BY = delta_BY - 1
                    key_pressed = ""
                if key_pressed == "BLUE_RIGHT" :
                    delta_BX = delta_BX - 1
                    key_pressed = ""
                if key_pressed == "BLUE_LEFT" :
                    delta_BX = delta_BX + 1
                    key_pressed = ""
                if key_pressed == "BLUE_RESET" :
                    delta_BX = 0
                    delta_BY = 0
                    key_pressed = ""
                texte = "Shift Red : " + str(delta_RX) + " : "+str(delta_RY) + "    Shift Blue : " + str(delta_BX) + " : " + str(delta_BY) + "             "
                labelInfo10.config(text = texte)
                if delta_RX !=0 or delta_RY !=0 or delta_BX !=0 or delta_BY != 0 :
                    img_r,img_g,img_b = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                    res_r = cp.zeros_like(img_r,dtype=cp.uint8)
                    res_g = cp.zeros_like(img_g,dtype=cp.uint8)
                    res_b = cp.zeros_like(img_b,dtype=cp.uint8)
                    height,width,layer = image_brute.shape
                    nb_blocksX = (width // nb_ThreadsX) + 1
                    nb_blocksY = (height // nb_ThreadsY) + 1
                    RGB_Align_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, res_g, res_b, img_r, img_g, img_b, np.intc(width), np.intc(height), np.intc(delta_RX), np.intc(delta_RY), np.intc(delta_BX), np.intc(delta_BY)))
                    image_brute = cupy_separateRGB_2_numpy_RGBimage(res_r,res_g,res_b)
                flag_image_video_loaded = True
                flag_IsColor = True
            if flag_image_video_loaded == True :
                if flag_IsColor == True :
                    res_rr1,res_gg1,res_bb1 = numpy_RGBImage_2_cupy_separateRGB(image_brute)
                else :
                    res_bb1 = cp.asarray(image_brute,dtype=cp.uint8)
                flag_image_disponible = True
                frame_number = frame_number + 1
                flag_GO = True
                if flag_BFR == True and flag_image_mode == False :
                    if flag_IsColor == True :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_Im = image_brute[rs:re,cs:ce]
                        crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                        img_qual = cv2.Laplacian(crop_im_grey, cv2.CV_64F, ksize=laplacianksize).var()
                        img_qual = Image_Quality(crop_im_grey,IQ_Method)
                        if img_qual > max_qual :
                            max_qual = img_qual
                            if flag_BFREF == True and flag_BFReference == "BestFrame" :
                                BFR_image = image_brute
                                flag_BFREF_image = True
                            else :
                                flag_BFREF_image = False
                        if img_qual < min_qual :
                            min_qual = img_qual
                        quality_threshold = min_qual + (max_qual - min_qual) * (val_BFR / 100)
                        if img_qual < quality_threshold :
                            flag_GO = False
                            SFN = SFN + 1
                        ratio = int((SFN / frame_number) * 1000) / 10
                        texte = "SFN : " + str(SFN) + "   B/T : "+str(ratio) + "  Thres : " + str(int(quality_threshold*10)/10) + "  Qual : " + str(int(img_qual*10)/10) + "             "
                        labelInfo10.config(text = texte)
                    else :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_im_grey = image_brute[rs:re,cs:ce]
                        img_qual = Image_Quality(crop_im_grey,IQ_Method)
                        if img_qual > max_qual :
                            max_qual = img_qual
                            if flag_BFREF == True and flag_BFReference == "BestFrame" :
                                BFREF_image = image_brute
                                flag_BFREF_image = True
                            else :
                                flag_BFREF_image = False
                        if img_qual < min_qual :
                            min_qual = img_qual
                        quality_threshold = min_qual + (max_qual - min_qual) * (val_BFR / 100)
                        if img_qual < quality_threshold :
                            flag_GO = False
                            SFN = SFN + 1
                        ratio = int((SFN / frame_number) * 1000) / 10 
                        texte = "SFN : " + str(SFN) + "   B/T : "+str(ratio) + "  Thres : " + str(int(quality_threshold*10)/10) + "  Qual : " + str(int(img_qual*10)/10) + "             "
                        labelInfo10.config(text = texte)

                if flag_BFR == False and flag_BFREF == True and flag_BFReference == "BestFrame" and flag_image_mode == False :
                    if flag_IsColor == True :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_Im = image_brute[rs:re,cs:ce]
                        crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                        img_qual = Image_Quality(crop_im_grey,IQ_Method)
                        if img_qual > max_qual :
                            max_qual = img_qual
                            BFREF_image = image_brute
                            flag_BFREF_image = True
                    else :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_Im = image_brute[rs:re,cs:ce]
                        img_qual = Image_Quality(crop_Im,IQ_Method)
                        if img_qual > max_qual :
                            max_qual = img_qual
                            BFREF_image = image_brute
                            flag_BFREF_image = True
                if flag_filtrage_ON == True :
                    if flag_GO == True :
                        if flag_IsColor == True :
                            application_filtrage_color(res_bb1,res_gg1,res_rr1)
                        else :
                            application_filtrage_mono(res_bb1)
                    else :
                        if flag_IsColor == True :
                            image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_bb1,res_gg1,res_rr1)
                        else :
                            image_traitee = res_bb1.get()
                    if flag_AI_Craters == True and flag_crater_model_loaded == True:
                        if flag_IsColor == True :
                            image_model = image_traitee
                        else :
                            image_model = cv2.merge((image_traitee,image_traitee,image_traitee))
                        if flag_AI_Trace == True :
                            result_craters = model_craters_track.track(image_model, device = 0, half=True, conf = 0.05, persist = True, verbose=False)
                            result_craters2 = model_craters_track(image_model, conf = 0.05)[0]
                        else :
                            result_craters = model_craters_predict.predict(image_model, device = 0,max_det = 100, half=True, verbose=False)
                            result_craters2 = model_craters_predict(image_model, conf = 0.05)[0]
                        boxes_crater = result_craters2.boxes.xywh.cpu()
                        bboxes_crater = np.array(result_craters2.boxes.xyxy.cpu(), dtype="int")
                        classes_crater = np.array(result_craters2.boxes.cls.cpu(), dtype="int")
                        confidence_crater = result_craters2.boxes.conf.cpu()
                        if flag_AI_Trace == True :
                            track_crater_ids = (result_craters2.boxes.id.int().cpu().tolist() if result_craters2.boxes.id is not None else None)
                        else :
                            track_crater_ids = None
                        if track_crater_ids :
                            for cls, box, track_crater_id in zip(classes_crater, boxes_crater, track_crater_ids):
                                x, y, w1, h1 = box
                                object_name = model_craters_track.names[cls]
                                if object_name == "Small crater":
                                    BOX_COLOUR = (0, 255, 255)
                                if object_name == "Crater":
                                    BOX_COLOUR = (0, 255, 0)
                                if object_name == "Large crater":
                                    BOX_COLOUR = (255, 150, 30)
                                if flag_IsColor == False :
                                    BOX_COLOUR = (255, 255, 255)
                                track_crater = track_crater_history[track_crater_id]
                                track_crater.append((float(x), float(y)))  # x, y center point
                                if len(track_crater) > 30:  # retain 90 tracks for 90 frames
                                    track_crater.pop(0)
                                points = np.hstack(track_crater).astype(np.int32).reshape((-1, 1, 2))
                                if flag_AI_Trace == True :
                                    cv2.polylines(image_traitee, [points], isClosed=False, color=BOX_COLOUR, thickness=1)
                        for cls, bbox in zip(classes_crater, bboxes_crater):
                            (x, y, x2, y2) = bbox
                            if flag_AI_Trace == True :
                                object_name = model_craters_track.names[cls]
                            else :
                                object_name = model_craters_predict.names[cls]
                            if object_name == "Small crater":
                                BOX_COLOUR = (0, 255, 255)
                            if object_name == "Crater":
                                BOX_COLOUR = (0, 255, 0)
                            if object_name == "Large crater":
                                BOX_COLOUR = (255, 150, 30)
                            if flag_IsColor == False :
                                BOX_COLOUR = (255, 255, 255)
                            cv2.rectangle(image_traitee, (x, y), (x2, y2), BOX_COLOUR, 1)
                            cv2.putText(image_traitee, f"{object_name}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, 1)                         
#                            cv2.putText(image_traitee, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, BOX_COLOUR, 2)
                    if flag_AI_Satellites == True and flag_satellites_model_loaded == True :
                        flag_sat_OK,image_model = satellites_tracking_AI()
                        calque_satellites_AI = np.zeros_like(image_traitee)
                        if flag_sat_OK == True :
                            try :
                                if flag_AI_Trace == True :
                                    result_sat = model_satellites_track.track(image_model, tracker = Custom_satellites_model_tracker, device = 0, half=False, conf = 0.01, persist = True, verbose=False)
                                    result_sat2 = model_satellites_track(image_model, conf = 0.01)[0]
                                else :
                                    result_sat = model_satellites_predict.predict(image_model, device = 0,max_det = 100, half=True, verbose=False)
                                    result_sat2 = model_satellites_predict(image_model, conf = 0.1)[0]
                                model_OK = True
                            except :
                                model_OK = False
                            if model_OK == True :
                                boxes_sat = result_sat2.boxes.xywh.cpu()
                                bboxes_sat = np.array(result_sat2.boxes.xyxy.cpu(), dtype="int")
                                classes_sat = np.array(result_sat2.boxes.cls.cpu(), dtype="int")
                                confidence_sat = result_sat2.boxes.conf.cpu()
                                if flag_AI_Trace == True :
                                    track_sat_ids = (result_sat2.boxes.id.int().cpu().tolist() if result_sat2.boxes.id is not None else None)
                                else :
                                    track_sat_ids = None
                                if track_sat_ids :
                                    for cls, box, track_sat_id in zip(classes_sat, boxes_sat, track_sat_ids):
                                        x, y, w1, h1 = box
                                        object_name = model_satellites_track.names[cls]
                                        if object_name == "Shooting star":
                                            BOX_COLOUR = (255, 0, 0)
                                        if object_name == "Plane":
                                            BOX_COLOUR = (255, 255, 0)
                                            ep = 2
                                        if object_name == "Satellite":
                                            BOX_COLOUR = (0, 255, 0)
                                            ep = 1
                                        if flag_IsColor == False :
                                            BOX_COLOUR = (255, 255, 255)
                                        track_sat = track_satellite_history[track_sat_id]
                                        track_sat.append((float(x), float(y)))  # x, y center point
                                        if len(track_sat) > 30:  # retain 90 tracks for 90 frames
                                            track_sat.pop(0)
                                        points = np.hstack(track_sat).astype(np.int32).reshape((-1, 1, 2))
                                        if flag_AI_Trace == True :
                                            cv2.polylines(calque_satellites_AI, [points], isClosed=False, color=BOX_COLOUR, thickness=1)
                                for cls, bbox,conf in zip(classes_sat, bboxes_sat, confidence_sat):
                                    (x, y, x2, y2) = bbox
                                    if flag_AI_Trace == True :
                                        object_name = model_satellites_track.names[cls]
                                    else :
                                        object_name = model_satellites_predict.names[cls]
                                    if object_name == "Shooting star":
                                        box_text = "Sht Star"
                                        ep = 2
                                        BOX_COLOUR = (255, 0, 0)
                                    if object_name == "Plane":
                                        box_text = "Plane"
                                        ep = 2
                                        BOX_COLOUR = (255, 255, 0)
                                    if object_name == "Satellite":
                                        box_text = "Sat"
                                        ep = 1
                                        BOX_COLOUR = (0, 255, 0)
                                    if flag_IsColor == False :
                                        ep = 1
                                        BOX_COLOUR = (255, 255, 255)
                                    cv2.rectangle(calque_satellites_AI, (x, y), (x2, y2), BOX_COLOUR, 1)
                                    cv2.putText(calque_satellites_AI, f"{box_text}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, ep)                         
#                                    cv2.putText(calque_satellites_AI, f"{object_name}: {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, BOX_COLOUR, ep)
                else :
                    if flag_IsColor == True :
                        image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_bb1,res_gg1,res_rr1)
                    else :
                        image_traitee = res_bb1.get()
                if flag_IQE == True :
                    if flag_IsColor == True :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_Im = image_traitee[rs:re,cs:ce]
                        crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                        quality[quality_pos] = int(Image_Quality(crop_im_grey,IQ_Method))
                        if quality[quality_pos] > max_quality :
                            max_quality = quality[quality_pos]
                        quality_pos = quality_pos + 1
                        if quality_pos > 255 :
                            quality_pos = 1
                    else :
                        rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                        re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                        cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                        ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                        crop_Im = image_traitee[rs:re,cs:ce]
                        quality[quality_pos] = int(Image_Quality(crop_Im,IQ_Method))
                        if quality[quality_pos] > max_quality :
                            max_quality = quality[quality_pos]
                        quality_pos = quality_pos + 1
                        if quality_pos > 255 :
                            quality_pos = 1
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

                if flag_false_colours == True :
                    if flag_IsColor == True :
                        tmp_grey = cv2.cvtColor(image_traitee, cv2.COLOR_BGR2GRAY)
                        image_traitee = cv2.applyColorMap(tmp_grey, cv2.COLORMAP_TURBO)
                    else:
                        image_traitee = cv2.applyColorMap(image_traitee, cv2.COLORMAP_TURBO)
                if flag_CONST == 0 :
                    if (flag_TRKSAT == 1 and flag_image_mode == False) and flag_REMSAT == 0 :
                         satellites_tracking()
                    if (flag_REMSAT == 1 and flag_image_mode == False) :
                         remove_satellites()
                    flag_sat_detected = False
                    if flag_DETECT_STARS == 1 and flag_image_mode == False :
                        stars_detection(True)
                    if (flag_TRKSAT == 1  and nb_sat >= 0 and nb_sat < max_sat and flag_image_mode == False) and flag_REMSAT == 0 :
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        size = 0.5
                        for i in range(nb_sat+1):
                            if correspondance[i] >=0:
                                centercircle = (sat_x[i],sat_y[i])
                                center_texte = (sat_x[i]+10,sat_y[i]+10)
                                texte = "Sat"
                                if flag_IsColor == True :
                                    cv2.circle(calque_direction_satellites, centercircle, 7, (0,255,0), 1, cv2.LINE_AA)
                                    cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (0, 255, 0), 1, cv2.LINE_AA)
                                    center_texte = (sat_x[i]+10,sat_y[i]+25)
                                    texte = "Rel Speed " + str(sat_speed[i])
                                    cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 0), 1, cv2.LINE_AA)
                                else :
                                    cv2.circle(calque_direction_satellites, centercircle, 7, (255,255,255), 1, cv2.LINE_AA)
                                    cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 255), 1, cv2.LINE_AA)
                                    center_texte = (sat_x[i]+10,sat_y[i]+25)
                                    texte = "Rel Speed " + str(sat_speed[i])
                                    cv2.putText(calque_direction_satellites, texte,center_texte, font, size, (255, 255, 255), 1, cv2.LINE_AA)
                        image_traitee = cv2.addWeighted(image_traitee, 1, calque_direction_satellites, 1, 0)     
                    if flag_DETECT_STARS == 1 and flag_image_mode == False :
                        image_traitee = cv2.addWeighted(image_traitee, 1, calque_stars, 1, 0)
                    if flag_AI_Satellites and flag_satellites_model_loaded == True :
                        image_traitee = cv2.addWeighted(image_traitee, 1, calque_satellites_AI, 1, 0)
                else :
                    reconstruction_image()
                    image_traitee = image_reconstructed
                    
                if flag_cap_pic == True :
                    pic_capture()
                if flag_cap_video == True and flag_image_mode == False :
                    video_capture(image_traitee)
                if flag_new_stab_window == True :
                    cv2.rectangle(image_traitee,start_point,end_point, (255,0,0), 2, cv2.LINE_AA)
                    time.sleep(0.1)
                    flag_new_stab_window = False
                if (res_cam_x > int(cam_displ_x*fact_s) or res_cam_y > int(cam_displ_y*fact_s)) and flag_full_res == 0 :
                    image_traitee_resize = cv2.resize(image_traitee,(int(cam_displ_x*fact_s),int(cam_displ_y*fact_s)),interpolation = cv2.INTER_LINEAR)
                    cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                if (res_cam_x < int(cam_displ_x*fact_s) or res_cam_y < int(cam_displ_y*fact_s)) and flag_full_res == 1 :
                    image_traitee_resize = cv2.resize(image_traitee,(int(cam_displ_x*fact_s),int(cam_displ_y*fact_s)),interpolation = cv2.INTER_LINEAR)
                    cadre_image.im=PIL.Image.fromarray(image_traitee_resize)
                if (res_cam_x > int(1350*fact_s) or res_cam_y > int(1012*fact_s)) and flag_full_res == 1 :
                    old_dzx = delta_zx
                    old_dzy = delta_zy
                    if key_pressed == "ZOOM_UP" :
                        delta_zy = delta_zy - 20
                        key_pressed = ""
                    if key_pressed == "ZOOM_DOWN" :
                         delta_zy = delta_zy + 20
                         key_pressed = ""
                    if key_pressed == "ZOOM_RIGHT" :
                        delta_zx = delta_zx + 20
                        key_pressed = ""
                    if key_pressed == "ZOOM_LEFT" :
                        delta_zx = delta_zx - 20
                        key_pressed = ""
                    if key_pressed == "ZOOM_RESET" :
                        delta_zx = 0
                        delta_zy = 0
                        key_pressed = ""
                    if (res_cam_x > int(1350*fact_s) and res_cam_y > int(1012*fact_s)) :
                        rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                        re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                        cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                        ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                        if cs < 0 or ce > res_cam_x :
                            delta_zx = old_dzx
                            cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                            ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                        if rs < 0 or re > res_cam_y :
                            delta_zy = old_dzy
                            rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                            re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                    if (res_cam_x > int(1350*fact_s) and res_cam_y <= int(1012*fact_s)) :
                        rs = 0
                        re = res_cam_y
                        cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                        ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                        if cs < 0 or ce > res_cam_x :
                            delta_zx = old_dzx
                            cs = (res_cam_x - int(1350*fact_s)) // 2 - 1 + delta_zx
                            ce = (res_cam_x + int(1350*fact_s)) // 2 + 1 + delta_zx
                    if (res_cam_x <= int(1350*fact_s) and res_cam_y > int(1012*fact_s)) :
                        rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                        re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                        cs = 0
                        ce = res_cam_x
                        if rs < 0 or re > res_cam_y :
                            delta_zy = old_dzy
                            rs = (res_cam_y - int(1012*fact_s)) // 2 - 1 + delta_zy
                            re = (res_cam_y + int(1012*fact_s)) // 2 + 1 + delta_zy
                    image_crop = image_traitee[rs:re,cs:ce]
                    cadre_image.im=PIL.Image.fromarray(image_crop)
                if res_cam_x <= int(cam_displ_x*fact_s) and flag_full_res == 0 :
                    cadre_image.im=PIL.Image.fromarray(image_traitee)                    
                if flag_cross == True :
                    draw = PIL.ImageDraw.Draw(cadre_image.im)
                    SX, SY = cadre_image.im.size
                    draw.line(((SX/2-100,SY/2),(SX/2+100,SY/2)), fill="red", width=1)
                    draw.line(((SX/2,SY/2-100),(SX/2,SY/2+100)), fill="red", width=1)
                if flag_HST == 1 and flag_IsColor == True :
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
                if flag_IQE == True :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(2,256) :
                        y2 = int((quality[x]/max_quality)*400)
                        y1 = int((quality[x-1]/max_quality)*400)
                        transform.line((((x-1)*3,cam_displ_y-y1),(x*3,cam_displ_y-y2)),fill="red",width=2)
                    transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="blue",width=3)
                    transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="blue",width=3)
                if flag_HST == 1 and flag_IsColor == False :
                    r = cadre_image.im
                    hst_r = r.histogram()
                    histo = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(1,256) :
                        histo.line(((x*3,cam_displ_y),(x*3,cam_displ_y-hst_r[x]/100)),fill="white")
                    histo.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    histo.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_TRSF == 1 and flag_IsColor == True :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    for x in range(2,256) :
                        transform.line((((x-1)*3,cam_displ_y-trsf_r[x-1]*3),(x*3,cam_displ_y-trsf_r[x]*3)),fill="red",width=2)
                        transform.line((((x-1)*3,cam_displ_y-trsf_g[x-1]*3),(x*3,cam_displ_y-trsf_g[x]*3)),fill="green",width=2)
                        transform.line((((x-1)*3,cam_displ_y-trsf_b[x-1]*3),(x*3,cam_displ_y-trsf_b[x]*3)),fill="blue",width=2)
                    transform.line(((256*3,cam_displ_y),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                    transform.line(((1,cam_displ_y-256*3),(256*3,cam_displ_y-256*3)),fill="red",width=3)
                if flag_TRSF == 1 and flag_IsColor == False :
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
                if flag_BFR == True and flag_image_mode == False :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    mul_par = (cam_displ_x*fact_s-600) / max_qual
                    transform.line(((0,cam_displ_y*fact_s - 50),(int(min_qual*mul_par),cam_displ_y*fact_s - 50)),fill="red",width=4) # min quality
                    transform.line(((0,cam_displ_y*fact_s - 110),(int(max_qual*mul_par),cam_displ_y*fact_s - 110)),fill="blue",width=4) # max quality
                    transform.line(((0,cam_displ_y*fact_s - 80),(int(img_qual*mul_par),cam_displ_y*fact_s - 80)),fill="yellow",width=4) # image quality
                    transform.line(((int(quality_threshold*mul_par),cam_displ_y*fact_s - 55),(int(quality_threshold*mul_par),cam_displ_y*fact_s - 105)),fill="green",width=6)# threshold quality
                if flag_GO == False :
                    transform = PIL.ImageDraw.Draw(cadre_image.im)
                    transform.line(((0,0),(cam_displ_x*fact_s,cam_displ_y*fact_s)),fill="red",width=2)
                    transform.line(((0,cam_displ_y*fact_s),(cam_displ_x*fact_s,0)),fill="red",width=2)
                cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
                cadre_image.create_image(cam_displ_x*fact_s/2,cam_displ_y*fact_s/2, image=cadre_image.photo)
                flag_image_video_loaded = False
            else :
                if flag_image_mode == False :
                    video.release()
                    flag_premier_demarrage = True
                    flag_cap_video = False
            if flag_image_mode == True :
                fenetre_principale.after(20, refresh)
            else :
                fenetre_principale.after(3, refresh)


def satellites_tracking_AI ():
    global imggrey12,imggrey22,image_traitee,flag_IsColor,cupy_context,sat_frame_count,sat_frame_count_AI,flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,\
           flag_img_sat_buf4_AI,flag_img_sat_buf5_AI,flag_first_sat_pass_AI,img_sat_buf1_AI,img_sat_buf2_AI,img_sat_buf3_AI,img_sat_buf4_AI,img_sat_buf5_AI

    if sat_frame_count_AI < sat_frame_target_AI and flag_first_sat_pass_AI == True :
        sat_frame_count_AI = sat_frame_count_AI + 1
        if sat_frame_count_AI == 1 :
            img_sat_buf1_AI = image_traitee
            flag_img_sat_buf1_AI = True
        if sat_frame_count_AI == 2 :
            img_sat_buf2_AI = image_traitee
            flag_img_sat_buf2_AI = True
        if sat_frame_count_AI == 3 :
            img_sat_buf3_AI = image_traitee
            flag_img_sat_buf3_AI = True
        if sat_frame_count_AI == 4 :
            img_sat_buf4_AI = image_traitee
            flag_img_sat_buf4_AI = True
        if sat_frame_count_AI == 5 :
            flag_img_sat_buf5_AI = True
            flag_first_sat_pass_AI = False
            
    if flag_img_sat_buf5_AI == True :
        img_sat_buf5_AI = image_traitee
        if flag_IsColor == True :
            imggrey22 = cv2.cvtColor(img_sat_buf5_AI, cv2.COLOR_BGR2GRAY)
            imggrey12 = cv2.cvtColor(img_sat_buf1_AI, cv2.COLOR_BGR2GRAY)
        else :
            imggrey22 = img_sat_buf5_AI
            imggrey12 = img_sat_buf1_AI
        height,width = imggrey22.shape
        diff = cv2.subtract(imggrey22,imggrey12)
        seuilb = np.percentile(diff, 99) + 30
        diff[0:90,0:width] = 0
        ret,thresh = cv2.threshold(diff, seuilb , 255, cv2.THRESH_BINARY)
        img_sat_buf1_AI = img_sat_buf2_AI.copy()
        img_sat_buf2_AI = img_sat_buf3_AI.copy()
        img_sat_buf3_AI = img_sat_buf4_AI.copy()
        img_sat_buf4_AI = img_sat_buf5_AI.copy()
        flag_sat2 = True
    else :
        flag_sat2 = False
    if flag_sat2 == True :
        with cupy_context :
            height,width = thresh.shape
            Pixel_threshold = 120
            nb_blocksX = (width // nb_ThreadsX) + 1
            nb_blocksY = (height // nb_ThreadsY) + 1
            res_r = cp.zeros_like(thresh,dtype=cp.uint8)
            img = cp.asarray(thresh,dtype=cp.uint8)
            Dead_Pixels_Remove_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, img, np.intc(width), np.intc(height), np.intc(Pixel_threshold)))
            thresh = res_r.copy()
            thresh_blur = gaussianblur_mono(thresh,1)
            thresh = thresh_blur.get()
            image_sat=cv2.merge((thresh,thresh,thresh))
            result = image_sat
    else :
        result = image_traitee
        
    return flag_sat2,result

        
def satellites_tracking ():
    global imggrey1,imggrey2,image_traitee,calque_satellites,calque_direction_satellites,flag_sat_exist,sat_x,sat_y,sat_s,sat_old_x,sat_old_y,sat_cntdwn,nb_sat,sat_id,compteur_sat, \
           flag_first_sat_pass,nb_trace_sat,sat_speed,max_sat,flag_IsColor,sat_old_dx,sat_old_dy,sat_frame_target,sat_frame_count,img_sat_buf1,img_sat_buf2,img_sat_buf3,img_sat_buf4,\
           img_sat_buf5,flag_img_sat_buf1,flag_img_sat_buf2,flag_img_sat_buf3,flag_img_sat_buf4,flag_img_sat_buf5

    if sat_frame_count < sat_frame_target and flag_first_sat_pass == True :
        sat_frame_count = sat_frame_count + 1
        if sat_frame_count == 1 :
            img_sat_buf1 = image_traitee
            flag_img_sat_buf1 = True
        if sat_frame_count == 2 :
            img_sat_buf2 = image_traitee
            flag_img_sat_buf2 = True
        if sat_frame_count == 3 :
            img_sat_buf3 = image_traitee
            flag_img_sat_buf3 = True
        if sat_frame_count == 4 :
            img_sat_buf4 = image_traitee
            flag_img_sat_buf4 = True
        if sat_frame_count == 5 :
            flag_img_sat_buf5 = True
            
    if flag_img_sat_buf5 == True :
        img_sat_buf5 = image_traitee
        if flag_IsColor == True :
            imggrey2 = cv2.cvtColor(img_sat_buf5, cv2.COLOR_BGR2GRAY)
            imggrey1 = cv2.cvtColor(img_sat_buf1, cv2.COLOR_BGR2GRAY)
        else :
            imggrey2 = img_sat_buf5
            imggrey1 = img_sat_buf1
        correspondance = np.zeros(10000,dtype=int)
        height,width = imggrey2.shape
        calque_satellites = np.zeros_like(image_traitee)
        calque_direction_satellites = np.zeros_like(image_traitee)
        diff = cv2.subtract(imggrey2,imggrey1)
        seuilb = np.percentile(diff, 99) + 30
        diff[0:90,0:width] = 0
        ret,thresh = cv2.threshold(diff, seuilb , 255, cv2.THRESH_BINARY)
        image_sat=cv2.merge((thresh,thresh,thresh))
        seuil_min_blob_sat = 20
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
        img_sat_buf1 = img_sat_buf2.copy()
        img_sat_buf2 = img_sat_buf3.copy()
        img_sat_buf3 = img_sat_buf4.copy()
        img_sat_buf4 = img_sat_buf5.copy()
        flag_first_sat_pass = False
    else :
        flag_sat = False
        
    nb_sat = -1
    if flag_sat == True :
        if flag_first_sat_pass == False :
            for kp_sat in keypoints_sat:
                nb_sat=nb_sat+1
            if nb_sat >= 0 and nb_sat < max_sat :
                nb_sat = -1
                for kp_sat in keypoints_sat:
                    nb_sat = nb_sat+1
                    sat_x[nb_sat] = int(kp_sat.pt[0])
                    sat_y[nb_sat] = int(kp_sat.pt[1])
                    sat_s[nb_sat] = int(kp_sat.size*2)
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
                    if dist_min > 50 :
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
                        delta_x = (sat_x[i] - sat_old_x[correspondance[i]]) * 7
                        delta_x = (delta_x + sat_old_dx[correspondance[i]]) // 2
                        delta_y = (sat_y[i] - sat_old_y[correspondance[i]]) * 7
                        delta_y = (delta_y + sat_old_dy[correspondance[i]]) // 2
                        sat_speed[i] = math.sqrt(delta_x*delta_x+delta_y*delta_y)
                        direction = (sat_x[i] + delta_x,sat_y[i]+delta_y)
                        cv2.line(calque_direction_satellites,end_point,direction,(255,255,0),1)           
                        sat_old_x[correspondance[i]]=sat_x[i]
                        sat_old_y[correspondance[i]]=sat_y[i]
                        sat_old_dx[correspondance[i]]=delta_x
                        sat_old_dy[correspondance[i]]=delta_y
                        sat_old_id[correspondance[i]]=sat_id[i]                        
                    else :
                        sat_old_x[correspondance[i]]= -1
                        sat_old_y[correspondance[i]]= -1
                        sat_old_id[correspondance[i]]= -1                 
            if nb_sat >= max_sat :
                raz_tracking()
                nb_sat = -1       
        if flag_first_sat_pass == True :
            for kp_sat in keypoints_sat:
                nb_sat=nb_sat+1
            if nb_sat >= 0 :
                nb_sat = -1
                for kp_sat in keypoints_sat:
                    nb_sat = nb_sat+1
                    sat_x[nb_sat] = int(kp_sat.pt[0])
                    sat_y[nb_sat] = int(kp_sat.pt[1])
                    sat_s[nb_sat] = int(kp_sat.size*2)
                    sat_id[nb_sat] = nb_sat
                    sat_old_x[nb_sat] = sat_x[nb_sat]
                    sat_old_y[nb_sat] = sat_y[nb_sat]
                    sat_old_id[nb_sat] = nb_sat
                nb_trace_sat = nb_sat
                flag_first_sat_pass = False


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
            if flag_IsColor == True :
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
            else :
                seuilb = abs(np.percentile(mask_sat[:,:], 70))
                axex = range (x1,x2)
                axey = range (y1,y2)
                for i in axex :
                    for j in axey :
                        if image_traitee[j,i] > seuilb :
                            image_traitee[j,i] = abs(seuilb + random.randrange(0,40) - 30)                
        except :
            pass
    
                        
def stars_detection(draw) :
    global image_traitee,calque_stars,stars_x,stars_y,stars_s,nb_stars,flag_IsColor

    if flag_IsColor == True :
        calque_stars = np.zeros_like(image_traitee)
        seuilb = np.percentile(image_traitee[:,:,0], 90)
        seuilg = np.percentile(image_traitee[:,:,1], 90)
        seuilr = np.percentile(image_traitee[:,:,2], 90)
        seuil_min_blob = max(seuilb,seuilg,seuilr) + 15
        height,width,layers = image_traitee.shape
    else :
        calque_stars = np.zeros_like(image_traitee)
        seuilb = np.percentile(image_traitee, 90)
        seuil_min_blob = seuilb + 15
        height,width = image_traitee.shape

    image_stars = image_traitee.copy()
    image_stars[0:50,0:width] = 0
    
    if seuil_min_blob > 160 :
        seuil_min_blob = 160
    params = cv2.SimpleBlobDetector_Params()    
    params.minThreshold = seuil_min_blob;     # Change thresholds
    params.maxThreshold = 255;
    params.thresholdStep = 10 # steps to go through
    params.filterByColor = False    # Filter by color.    
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
            if flag_IsColor == True :
                cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size*1.5), (255,0,0), 1, cv2.LINE_AA)
            else :
                cv2.circle(calque_stars, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size*1.5), (255,255,255), 1, cv2.LINE_AA)

def draw_satellite(x,y) :
    global image_reconstructed

    centercircle = (x,y)
    cv2.circle(image_reconstructed, centercircle, 7, (0,255,0), -1, cv2.LINE_AA)
    start = (x-10,y-10)
    stop =(x+10,y+10)
    if flag_IsColor == True :
        cv2.line(image_reconstructed, start, stop, (0,255,0), 5, cv2.LINE_AA)
        cv2.circle(image_reconstructed, centercircle, 7, (0,255,0), -1, cv2.LINE_AA)
    else :
        cv2.line(image_reconstructed, start, stop, (255,255,255), 5, cv2.LINE_AA)
        cv2.circle(image_reconstructed, centercircle, 7, (255,255,255), -1, cv2.LINE_AA)

def draw_star(x,y,s) :
    global image_reconstructed,calque_reconstruct
    
    if flag_IsColor == True :
        red = image_traitee[y,x,0]
        green = image_traitee[y,x,1]
        blue = image_traitee[y,x,2]
    else :
        red = image_traitee[y,x]
        green = red
        blue = red
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


def Image_Quality(image,IQ_Method):
    if IQ_Method == "Laplacian" :
        image = cv2.GaussianBlur(image,(3,3), 0)
        Image_Qual = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacianksize).var()
    elif IQ_Method == "Sobel" :
        image = cv2.GaussianBlur(image,(3,3), 0)
        Image_Qual = cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=SobelSize).var()
    else :
        image = cv2.GaussianBlur(image,(3,3), 0)
        Image_Qual = cv2.Laplacian(image, cv2.CV_64F, ksize=laplacianksize).var()
        
    return Image_Qual


def Template_tracking(image,dim) :
    global flag_STAB,flag_Template,Template,gsrc,gtmpl,gresult,matcher,delta_tx,delta_ty,start_point,end_point,flag_new_stab_window,key_pressed,DSW

    old_tx = delta_tx
    old_ty = delta_ty
    flag_modif = False
    if key_pressed == "STAB_UP" :
        delta_ty = delta_ty - 30
        key_pressed = ""
        flag_modif = True
    if key_pressed == "STAB_DOWN" :
        delta_ty = delta_ty + 30
        key_pressed = ""
        flag_modif = True
    if key_pressed == "STAB_RIGHT" :
        delta_tx = delta_tx + 30
        key_pressed = ""
        flag_modif = True
    if key_pressed == "STAB_LEFT" :
        delta_tx = delta_tx - 30
        key_pressed = ""
        flag_modif = True
    if key_pressed == "STAB_ZONE_MORE" :
        DSW = DSW - 1
        key_pressed = ""
        flag_modif = True
    if key_pressed == "STAB_ZONE_LESS" :
        DSW = DSW + 1
        key_pressed = ""
        flag_modif = True
    if DSW > 12 :
        DSW = 12
    if DSW < 0 :
        DSW = 0
    if flag_modif == True :
        flag_Template = False
    
    if flag_Template == False :
        if res_cam_x > 1500 :
            rs = res_cam_y // 2 - res_cam_y // (8 + DSW) + delta_ty
            re = res_cam_y // 2 + res_cam_y // (8 + DSW) + delta_ty
            cs = res_cam_x // 2 - res_cam_x // (8 + DSW) + delta_tx
            ce = res_cam_x // 2 + res_cam_x // (8 + DSW) + delta_tx
            if cs < 30 or ce > (res_cam_x - 30) :
                delta_tx = old_tx
                cs = res_cam_x // 2 - res_cam_x // (8 + DSW) + delta_tx
                ce = res_cam_x // 2 + res_cam_x // (8 + DSW) + delta_tx
            if rs < 30 or re > (res_cam_y - 30) :
                delta_ty = old_ty
                rs = res_cam_y // 2 - res_cam_y // (8 + DSW) + delta_ty
                re = res_cam_y // 2 + res_cam_y // (8 + DSW) + delta_ty
        else :
            rs = res_cam_y // 2 - res_cam_y // (3 + DSW) + delta_ty
            re = res_cam_y // 2 + res_cam_y // (3 + DSW) + delta_ty
            cs = res_cam_x // 2 - res_cam_x // (3 + DSW) + delta_tx
            ce = res_cam_x // 2 + res_cam_x // (3 + DSW) + delta_tx
            if cs < 30 or ce > (res_cam_x - 30) :
                delta_tx = old_tx
                cs = res_cam_x // 2 - res_cam_x // (3 + DSW) + delta_tx
                ce = res_cam_x // 2 + res_cam_x // (3 + DSW) + delta_tx
            if rs < 30 or re > (res_cam_y - 30) :
                delta_ty = old_ty
                rs = res_cam_y // 2 - res_cam_y // (3 + DSW) + delta_ty
                re = res_cam_y // 2 + res_cam_y // (3 + DSW) + delta_ty            
        start_point = (cs,rs)
        end_point = (ce,re)
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
        flag_new_stab_window = True
    else :
        flag_new_stab_window = False
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
            DeltaX = image.shape[1] // 2 + delta_tx - midX
            DeltaY = image.shape[0] // 2 + delta_ty - midY
            try :
                rs = int(res_cam_y / 4 + DeltaY) # Y up
                re = int(rs + res_cam_y) # Y down
                cs = int(res_cam_x / 4 + DeltaX) # X left
                ce = int(cs + res_cam_x) # X right
                tmp_image[rs:re,cs:ce] = image
                rs = res_cam_y // 4  # Y up
                re = res_cam_y // 4 + res_cam_y # Y down
                cs = res_cam_x // 4  # X left
                ce = res_cam_x // 4 + res_cam_x  # X right
                new_image = tmp_image[rs:re,cs:ce]
            except :
                new_image = image
        else :
            new_image = image
            
    return new_image
       
    
def application_filtrage_color(res_b1,res_g1,res_r1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,compteur_images,numero_image,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm, g1_sm, g2_sm, g3_sm, g4_sm, g5_sm, r1_sm, r2_sm, r3_sm, r4_sm, r5_sm,\
           Im4OK,Im5OK,flag_hold_picture,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_camera,image_camera_old,image_brute_grey,cupy_context,BFREF_image,flag_BFREF_image, \
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,flag_IsColor,flag_BFReference,BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_filtre_work,flag_AmpSoft,val_ampl,grad_vignet,compteur_AADF,compteur_AADFB,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\
           Im1fsdnOK,Im2fsdnOK,Im1fsdnOKB,Im2fsdnOKB,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,val_reds,val_greens,val_blues,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\
           flag_dyn_AADF,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_g2,res_r2,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\
           imgb1,imgg1,imgr1,imgb2,imgg2,imgr2,imgb3,imgg3,imgr3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,res_b2B,res_g2B,res_r2B,\
           imgb1B,imgg1B,imgr1B,imgb2B,imgg2B,imgr2B,imgb3B,imgg3B,imgr3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,flag_AADFB

    start_time_test = cv2.getTickCount()

    if flag_HDR == False :
        if image_camera > numero_image :
            numero_image = numero_image + 1
            image_camera = numero_image
            image_camera_old = image_camera

    with cupy_context :     
        if flag_filtrage_ON == True :
            flag_filtre_work = True
            if flag_TRSF == 1 :
                for x in range(1,256) :
                    trsf_r[x] = x
                    trsf_g[x] = x
                    trsf_b[x] = x
                                            
            if flag_IsColor == True :

                # Colour image treatment
                
                if flag_DEMO == 1 :
                    if Dev_system == "Windows" :
                        image_base = cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
                    else :
                        image_base = cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
#                        image_base = cv2.merge((res_b1.get(),res_g1.get(),res_r1.get()))
                
                height,width = res_b1.shape
                                
                nb_pixels = height * width
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1

                # Adjust RGB channels soft
                if (val_reds != 1.0 or val_greens != 1.0 or val_blues != 1.0) :              

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                    
                    Set_RGB((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \
                            np.float32(val_reds), np.float32(val_greens), np.float32(val_blues)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                    if flag_TRSF == 1 :                                                      
                        if val_reds != 0 :
                            for x in range(1,256) :
                                trsf_r[x] = (int)(trsf_r[x] * val_reds)
                            trsf_r = np.clip(trsf_r,0,255)
                        if val_greens != 0 :
                            for x in range(1,256) :
                                trsf_g[x] = (int)(trsf_g[x] * val_greens)
                            trsf_g = np.clip(trsf_g,0,255)
                        if val_blues != 0 :
                            for x in range(1,256) :
                                trsf_b[x] = (int)(trsf_b[x] * val_blues)
                            trsf_b = np.clip(trsf_b,0,255)

                # Image negative
                if ImageNeg == 1 :
                    res_b1,res_g1,res_r1 = image_negative_colour(res_b1,res_g1,res_r1)  
                    if flag_TRSF == 1 :                    
                        for x in range(1,256) :
                            trsf_r[x] = (int)(256-trsf_r[x])
                            trsf_g[x] = (int)(256-trsf_g[x])
                            trsf_b[x] = (int)(256-trsf_b[x])
                        trsf_r = np.clip(trsf_r,0,255)
                        trsf_g = np.clip(trsf_g,0,255)
                        trsf_b = np.clip(trsf_b,0,255)

                # Luminance estimation if a mono sensor was used
                if flag_NB_estime == 1 :

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                    
                    color_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1,\
                        np.int_(width), np.int_(height)))

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
                            imgs = [b1_sm,b2_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            imgs = [g1_sm,g2_sm]
                            imgs = cp.asarray(imgs)
                            sum_g = cp.median(imgs,axis=0)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            imgs = [r1_sm,r2_sm]
                            imgs = cp.asarray(imgs)
                            sum_r = cp.median(imgs,axis=0)
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
                            imgs = [b1_sm,b2_sm,b3_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            imgs = [g1_sm,g2_sm,g3_sm]
                            imgs = cp.asarray(imgs)
                            sum_g = cp.median(imgs,axis=0)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            imgs = [r1_sm,r2_sm,r3_sm]
                            imgs = cp.asarray(imgs)
                            sum_r = cp.median(imgs,axis=0)
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
                            imgs = [b1_sm,b2_sm,b3_sm,b4_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            imgs = [g1_sm,g2_sm,g3_sm,g4_sm]
                            imgs = cp.asarray(imgs)
                            sum_g = cp.median(imgs,axis=0)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            imgs = [r1_sm,r2_sm,r3_sm,r4_sm]
                            imgs = cp.asarray(imgs)
                            sum_r = cp.median(imgs,axis=0)
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
                            imgs = [b1_sm,b2_sm,b3_sm,b4_sm,b5_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            imgs = [g1_sm,g2_sm,g3_sm,g4_sm,g5_sm]
                            imgs = cp.asarray(imgs)
                            sum_g = cp.median(imgs,axis=0)
                            res_g1 = cp.asarray(sum_g,dtype=cp.uint8)
                            imgs = [r1_sm,r2_sm,r3_sm,r4_sm,r5_sm]
                            imgs = cp.asarray(imgs)
                            sum_r = cp.median(imgs,axis=0)
                            res_r1 = cp.asarray(sum_r,dtype=cp.uint8)

                # Reduce variation filter (turbulence management) with Previous frame reference
                if flag_reduce_variation == True and flag_BFReference == "PreviousFrame" and flag_hold_picture == 0 and flag_image_mode == False :
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


                # Reduce variation filter (turbulence management) with best frame reference
                if flag_reduce_variation == True and flag_BFReference == "BestFrame" and flag_BFREF_image == True and flag_hold_picture == 0 and flag_image_mode == False :

                    res_r2,res_g2,res_b2 = numpy_RGBImage_2_cupy_separateRGB(BFREF_image)
                    variation = int(255/100*val_reduce_variation)
                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
                        
                    reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                         np.int_(width), np.int_(height),np.int_(variation)))

                    res_r1 = r_gpu
                    res_g1 = g_gpu
                    res_b1 = b_gpu

                # 3 Frames Noise Reduction Filter Front
                if flag_3FNR == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_3FNR < 4 and FNR_First_Start == True:
                        compteur_3FNR = compteur_3FNR + 1
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
                        
                                                     
                # Adaptative Absorber Denoise Filter Front
                if flag_AADF == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_AADF < 3 :
                        compteur_AADF = compteur_AADF + 1
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
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AADF),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))


                        res_b2 = res_b1.copy()
                        res_g2 = res_g1.copy()
                        res_r2 = res_r1.copy()

                        tmp = cp.asarray(r_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_r1 = cp.asarray(tmp,dtype=cp.uint8)
                        tmp = cp.asarray(g_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_g1 = cp.asarray(tmp,dtype=cp.uint8)
                        tmp = cp.asarray(b_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_b1 = cp.asarray(tmp,dtype=cp.uint8)
                                
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
                if flag_NLM2 == 1 :
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

                if flag_BFREFPT == True and flag_image_mode == False :
                    rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                    re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                    cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                    ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                    im_qual_tmp = cupy_separateRGB_2_numpy_RGBimage(res_r1,res_g1,res_b1)
                    crop_Im = im_qual_tmp[rs:re,cs:ce]
                    crop_im_grey = cv2.cvtColor(crop_Im, cv2.COLOR_BGR2GRAY)
                    img_qual_PT = int(Image_Quality(crop_im_grey,IQ_Method))
                    if img_qual_PT > max_qual_PT :
                        max_qual_PT = img_qual_PT
                        BFREF_image_PT = im_qual_tmp
                        flag_BFREF_image_PT = True
                    if flag_BFREF_image_PT == True :
                        res_r2,res_g2,res_b2 = numpy_RGBImage_2_cupy_separateRGB(BFREF_image_PT)
                        variation = int(255/100*val_reduce_variation)
                        b_gpu = res_b1
                        g_gpu = res_g1
                        r_gpu = res_r1
                            
                        reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                             np.int_(width), np.int_(height),np.int_(variation)))

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

                    if flag_TRSF == 1 :                                                      
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

                    if flag_TRSF == 1 :
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
                        img_b = res_b1.copy()
                        img_g = res_g1.copy()
                        img_r = res_r1.copy()
                        img_b[img_b > seuilb] = seuilb
                        img_g[img_g > seuilg] = seuilg
                        img_r[img_r > seuilr] = seuilr
                        niveau_blur = val_NGB*2 + 3
                        img_b,img_g,img_r = gaussianblur_colour(img_b,img_g,img_r,niveau_blur)
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
                        fd_b = res_b1.copy()
                        fd_g = res_g1.copy()
                        fd_r = res_r1.copy()
                        fd_b[fd_b > seuilb] = seuilb
                        fd_g[fd_g > seuilg] = seuilg
                        fd_r[fd_r > seuilr] = seuilr
                        niveau_blur = val_NGB*2 + 3
                        fd_b,fd_g,fd_r = gaussianblur_colour(fd_b,fd_g,fd_r,niveau_blur)
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

                # image saturation enhancement
                if flag_SAT == True :
                    if flag_SAT_Image == True :
                        r_gpu = res_r1.copy()
                        g_gpu = res_g1.copy()
                        b_gpu = res_b1.copy()
                        init_r = res_r1.copy()
                        init_g = res_g1.copy()
                        init_b = res_b1.copy()
                        coul_r,coul_g,coul_b = gaussianblur_colour(r_gpu,g_gpu,b_gpu,3)
                        if ImageNeg == 1 :
                            flag_neg_sat = 1
                        else :
                            flag_neg_sat = 0
                        Saturation_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, coul_r, coul_g, coul_b, np.int_(width), np.int_(height),np.float32(val_SAT), np.int_(flag_neg_sat)))
                        coul_gauss2_r = r_gpu.copy()
                        coul_gauss2_g = g_gpu.copy()
                        coul_gauss2_b = b_gpu.copy()
                        coul_gauss2_r,coul_gauss2_g,coul_gauss2_b = gaussianblur_colour(coul_gauss2_r,coul_gauss2_g,coul_gauss2_b,7)
                        Saturation_Combine_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, init_r, init_g, init_b, coul_gauss2_r,coul_gauss2_g,coul_gauss2_b, np.int_(width), np.int_(height)))
                        res_r1 = r_gpu.copy()
                        res_g1 = g_gpu.copy()
                        res_b1 = b_gpu.copy()
                    else :
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
                        Saturation_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, coul_gauss_r,coul_gauss_g,coul_gauss_b, np.int_(width), np.int_(height),np.float32(val_SAT), np.int_(flag_neg_sat)))
                        coul_gauss_r = r_gpu.copy()
                        coul_gauss_g = g_gpu.copy()
                        coul_gauss_b = b_gpu.copy()
                        coul_gauss_r,coul_gauss_g,coul_gauss_b = gaussianblur_colour(coul_gauss_r,coul_gauss_g,coul_gauss_b,7)
                        Saturation_Combine_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, coul_gauss_r, coul_gauss_g, coul_gauss_b, np.int_(width), np.int_(height)))
                        res_r1 = r_gpu.copy()
                        res_g1 = g_gpu.copy()
                        res_b1 = b_gpu.copy()
                                  
                # 3 Frames Noise Reduction Filter Back
                if flag_3FNRB == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_3FNRB < 4 and FNRB_First_Start == True:
                        compteur_3FNRB = compteur_3FNRB + 1
                        if compteur_3FNRB == 1 :
                            imgb1B = res_b1.copy()
                            imgg1B = res_g1.copy()
                            imgr1B = res_r1.copy()
                            img1_3FNROKB = True
                        if compteur_3FNRB == 2 :
                            imgb2B = res_b1.copy()
                            imgg2B = res_g1.copy()
                            imgr2B = res_r1.copy()
                            img2_3FNROKB = True
                        if compteur_3FNRB == 3 :
                            imgb3B = res_b1.copy()
                            imgg3B = res_g1.copy()
                            imgr3B = res_r1.copy()
                            img3_3FNROKB = True
                            FNRB_First_Start = True
                    if img3_3FNROKB == True :
                        if FNRB_First_Start == False :
                            imgb3B = res_b1.copy()
                            imgg3B = res_g1.copy()
                            imgr3B = res_r1.copy()
                        
                        FNRB_First_Start = False
                        b_gpu = res_b1
                        g_gpu = res_g1
                        r_gpu = res_r1
                
                        FNR_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr1B, imgg1B, imgb1B, imgr2B, imgg2B, imgb2B,\
                                             imgr3B, imgg3B, imgb3B, np.int_(width), np.int_(height)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb1B = imgb2B.copy()
                        imgg1B = imgg2B.copy()
                        imgr1B = imgr2B.copy()
                        imgr2B = r_gpu.copy()
                        imgg2B = g_gpu.copy()
                        imgb2B = b_gpu.copy()


                # Adaptative Absorber Denoise Filter Back
                if flag_AADFB == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_AADFB < 3 :
                        compteur_AADFB = compteur_AADFB + 1
                        if compteur_AADFB == 1 :
                            res_b2B = res_b1.copy()
                            res_g2B = res_g1.copy()
                            res_r2B = res_r1.copy()
                            Im1fsdnOKB = True
                        if compteur_AADFB == 2 :
                            Im2fsdnOKB = True

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
     
                    if Im2fsdnOKB == True :

                        local_dyn = 1
                        local_GR = 0
                        local_VGR = 0
                        
                        adaptative_absorber_denoise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2B, res_g2B, res_b2B,\
                                             np.int_(width), np.int_(height),np.intc(local_dyn),np.intc(local_GR),np.intc(local_VGR)))

                        res_b2B = res_b1.copy()
                        res_g2B = res_g1.copy()
                        res_r2B = res_r1.copy()
                        tmp = cp.asarray(r_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_r1 = cp.asarray(tmp,dtype=cp.uint8)
                        tmp = cp.asarray(g_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_g1 = cp.asarray(tmp,dtype=cp.uint8)
                        tmp = cp.asarray(b_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_b1 = cp.asarray(tmp,dtype=cp.uint8)

                # image sharpen 1
                if flag_sharpen_soft1 == 1 :
                    cupy_context.use()
                    res_b1_blur,res_g1_blur,res_r1_blur = gaussianblur_colour(res_b1,res_g1,res_r1,val_sigma_sharpen)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_g1 = cp.asarray(res_g1).astype(cp.int16)
                    tmp_r1 = cp.asarray(res_r1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen * (tmp_b1 - res_b1_blur)
                    tmp_g1 = tmp_g1 + val_sharpen * (tmp_g1 - res_g1_blur)
                    tmp_r1 = tmp_r1 + val_sharpen * (tmp_r1 - res_r1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    tmp_g1 = cp.clip(tmp_g1,0,255)
                    tmp_r1 = cp.clip(tmp_r1,0,255)
                    if flag_sharpen_soft2 == 1 :
                        res_s1_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                        res_s1_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                        res_s1_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)
                    else :
                        res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                        res_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                        res_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)

                # image sharpen 2
                if flag_sharpen_soft2 == 1 :
                    cupy_context.use()
                    res_b1_blur,res_g1_blur,res_r1_blur = gaussianblur_colour(res_b1,res_g1,res_r1,val_sigma_sharpen2)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_g1 = cp.asarray(res_g1).astype(cp.int16)
                    tmp_r1 = cp.asarray(res_r1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen2 * (tmp_b1 - res_b1_blur)
                    tmp_g1 = tmp_g1 + val_sharpen2 * (tmp_g1 - res_g1_blur)
                    tmp_r1 = tmp_r1 + val_sharpen2 * (tmp_r1 - res_r1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    tmp_g1 = cp.clip(tmp_g1,0,255)
                    tmp_r1 = cp.clip(tmp_r1,0,255)
                    if flag_sharpen_soft1 == 1 :
                        res_s2_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                        res_s2_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                        res_s2_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)
                    else :
                        res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                        res_g1 = cp.asarray(tmp_g1,dtype=cp.uint8)
                        res_r1 = cp.asarray(tmp_r1,dtype=cp.uint8)

                if flag_sharpen_soft1 == 1 and flag_sharpen_soft2 == 1 :
                    res_b1 = res_s1_b1 // 2 + res_s2_b1 // 2
                    res_g1 = res_s1_g1 // 2 + res_s2_g1 // 2
                    res_r1 = res_s1_r1 // 2 + res_s2_r1 // 2

                if flag_reverse_RB == 0 :
                    image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_b1,res_g1,res_r1)
                else :
                    image_traitee = cupy_separateRGB_2_numpy_RGBimage(res_r1,res_g1,res_b1)
                    

                if flag_DEMO == 1 :
                    if flag_demo_side == "Left" :
                        image_traitee[0:height,0:width//2] = image_base[0:height,0:width//2]            
                    if flag_demo_side == "Right" :
                        image_traitee[0:height,width//2:width] = image_base[0:height,width//2:width]            
    
    stop_time_test = cv2.getTickCount()
    time_exec_test= int((stop_time_test-start_time_test)/cv2.getTickFrequency()*1000)

    TTQueue.append(time_exec_test)
    if len(TTQueue) > 10:
        TTQueue.pop(0)
    curTT = (sum(TTQueue)/len(TTQueue))

    flag_filtre_work = False 


def application_filtrage_mono(res_b1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,compteur_images,numero_image,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm,flag_IsColor,flag_BFReference,\
           Im4OK,Im5OK,flag_hold_picture,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_camera,image_camera_old,image_brute_grey,cupy_context, BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,BFREF_image,flag_BFREF_image,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_filtre_work,flag_AmpSoft,val_ampl,grad_vignet,compteur_AADF,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\
           Im1fsdnOK,Im2fsdnOK,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\
           flag_dyn_AADF,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_b2B,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\
           imgb1,imgb2,imgb3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,Im1fsdnOKB,Im2fsdnOKB, \
           imgb1B,imgb2B,imgb3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,compteur_AADFB

    start_time_test = cv2.getTickCount()

    if flag_HDR == False :
        if image_camera > numero_image :
            numero_image = numero_image + 1
            image_camera = numero_image
            image_camera_old = image_camera

    with cupy_context :     
        if flag_filtrage_ON == True :
            flag_filtre_work = True
            for x in range(1,256) :
                trsf_r[x] = x
                trsf_g[x] = x
                trsf_b[x] = x
            
            if flag_IsColor == False :

                # traitement image monochrome

                if flag_DEMO == 1 :
                    image_base = res_b1.get()
                    
                height,width = res_b1.shape
                nb_pixels = height * width
                nb_blocksX = (width // nb_ThreadsX) + 1
                nb_blocksY = (height // nb_ThreadsY) + 1
                
                r_gpu = res_b1.copy()

                # Image Negative
                if ImageNeg == 1 :
                    res_b1 = cp.invert(res_b1,dtype=cp.uint8)
                    for x in range(1,256) :
                        trsf_r[x] = (int)(256-trsf_r[x])
                    trsf_r = np.clip(trsf_r,0,255)

                # Estimate luminance if mono sensor was used - Do not use with a mono sensor
                if flag_NB_estime == 1 :
                    
                    grey_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1,\
                        np.int_(width), np.int_(height)))

                    res_b1 = r_gpu.copy()
     
                if val_FS > 1 and flag_hold_picture == 0 and flag_image_mode == False :
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
                            imgs = [b1_sm,b2_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                    
                    if val_FS == 3 and Im3OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            imgs = [b1_sm,b2_sm,b3_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                            
                    if val_FS == 4 and Im4OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            imgs = [b1_sm,b2_sm,b3_sm,b4_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        
                    if val_FS == 5 and Im5OK == True :
                        if stack_div == 1 :
                            sum_b = b1_sm + b2_sm + b3_sm + b4_sm + b5_sm
                            sum_b = cp.clip(sum_b,0,255)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)
                        else :
                            imgs = [b1_sm,b2_sm,b3_sm,b4_sm,b5_sm]
                            imgs = cp.asarray(imgs)
                            sum_b = cp.median(imgs,axis=0)
                            res_b1 = cp.asarray(sum_b,dtype=cp.uint8)

                # Reduce variation filter (turbulence management) with Previous frame reference
                if flag_reduce_variation == True and flag_BFReference == "PreviousFrame" and flag_hold_picture == 0 and flag_image_mode == False :
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

                # Reduce variation filter (turbulence management) with best frame reference
                if flag_reduce_variation == True and flag_BFReference == "BestFrame" and flag_BFREF_image == True and flag_hold_picture == 0 and flag_image_mode == False :

                    res_b2 = cp.asarray(BFREF_image,dtype=cp.uint8)
                    variation = int(255/100*val_reduce_variation)
                    b_gpu = res_b1
                        
                    reduce_variation_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, res_b1, res_b2, np.int_(width), np.int_(height),np.int_(variation)))

                    res_b1 = b_gpu
     
                # 3 Frames Noise reduction Front
                if flag_3FNR == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_3FNR < 4 and FNR_First_Start == True:
                        compteur_3FNR = compteur_3FNR + 1
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
                    if img3_3FNROK == True :
                        if FNR_First_Start == False :
                            imgb3 = res_b1.copy()
                        
                        FNR_First_Start = False
                        b_gpu = res_b1
                
                        FNR_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb1, imgb2, imgb3, np.int_(width), np.int_(height)))
                                              
                        res_b1 = b_gpu
                        imgb1 = imgb2.copy()
                        imgb2 = b_gpu.copy()

                # Adaptative Absorber Noise Reduction
                if flag_AADF == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_AADF < 3 :
                        compteur_AADF = compteur_AADF + 1
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
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AADF),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))

                        res_b2 = res_b1.copy()
                        tmp = cp.asarray(r_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_b1 = cp.asarray(tmp,dtype=cp.uint8)


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
                if flag_NLM2 == 1 :
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

                if flag_BFREFPT == True and flag_image_mode == False :
                    rs = res_cam_y // 2 - res_cam_y // 8 + delta_ty
                    re = res_cam_y // 2 + res_cam_y // 8 + delta_ty
                    cs = res_cam_x // 2 - res_cam_x // 8 + delta_tx
                    ce = res_cam_x // 2 + res_cam_x // 8 + delta_tx
                    im_qual_tmp = res_b1.get()
                    crop_im_grey = im_qual_tmp[rs:re,cs:ce]
                    img_qual_PT = int(Image_Quality(crop_im_grey,IQ_Method))
                    if img_qual_PT > max_qual_PT :
                        max_qual_PT = img_qual_PT
                        BFREF_image_PT = im_qual_tmp
                        flag_BFREF_image_PT = True
                    if flag_BFREF_image_PT == True :
                        res_b2 = cp.asarray(BFREF_image_PT,dtype=cp.uint8)
                        variation = int(255/100*val_reduce_variation)
                        b_gpu = res_b1
                            
                        reduce_variation_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, res_b1, res_b2,\
                                             np.int_(width), np.int_(height),np.int_(variation)))

                        res_b1 = b_gpu

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
                        img_b = res_b1.copy()
                        img_b[img_b > seuilb] = seuilb
                        niveau_blur = val_NGB*2 + 3
                        img_b = gaussianblur_mono(img_b,niveau_blur)
                        att_b = cp.asarray(img_b) * ((100.0-val_AGR) / 100.0) 
                        resb = cp.subtract(cp.asarray(res_b1),att_b)
                        resb = cp.clip(resb,0,255)
                        res_b1 = cp.asarray(resb,dtype=cp.uint8)
                    else :
                        seuilb = int(cp.percentile(res_b1, val_SGR))
                        fd_b = res_b1.copy()
                        fd_b[fd_b > seuilb] = seuilb
                        niveau_blur = val_NGB*2 + 3
                        fd_b = gaussianblur_mono(fd_b,niveau_blur)
                        pivot_b = int(cp.percentile(cp.asarray(res_b1), val_AGR))
                        corr_b = cp.asarray(res_b1).astype(cp.int16) - cp.asarray(fd_b).astype(cp.int16) + pivot_b
                        corr_b = cp.clip(corr_b,0,255)
                        res_b1 = cp.asarray(corr_b,dtype=cp.uint8)

                # Contrast Low Light
                if flag_CLL == 1 :
                    correction_CLL = cp.asarray(Corr_CLL,dtype=cp.uint8)
                    r_gpu = res_b1
                    Contrast_Low_Light_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), correction_CLL))
                    res_b1 = r_gpu

                # Contrast CLAHE
                if flag_contrast_CLAHE ==1 :
                    if flag_OpenCvCuda == True :
                        clahe = cv2.cuda.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        srcb = cv2.cuda_GpuMat()
                        srcb.upload(res_b1.get())
                        resb = clahe.apply(srcb, cv2.cuda_Stream.Null())
                        resbb = resb.download()
                        res_b1 = cp.asarray(resbb,dtype=cp.uint8)
                    else :        
                        clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(val_grid_CLAHE,val_grid_CLAHE))
                        b = clahe.apply(res_b1.get())
                        res_b1 = cp.asarray(b,dtype=cp.uint8)
                        
                # 3 Frames Noise reduction back
                if flag_3FNRB == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_3FNRB < 4 and FNRB_First_Start == True:
                        compteur_3FNRB = compteur_3FNRB + 1
                        if compteur_3FNRB == 1 :
                            imgb1B = res_b1.copy()
                            img1_3FNROKB = True
                        if compteur_3FNRB == 2 :
                            imgb2B = res_b1.copy()
                            img2_3FNROKB = True
                        if compteur_3FNRB == 3 :
                            imgb3B = res_b1.copy()
                            img3_3FNROKB = True
                            FNRB_First_Start = True
                    if img3_3FNROKB == True :
                        if FNRB_First_Start == False :
                            imgb3B = res_b1.copy()
                        
                        FNRB_First_Start = False
                        b_gpu = res_b1
                
                        FNR_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb1B, imgb2B, imgb3B, np.int_(width), np.int_(height)))
                                              
                        res_b1 = b_gpu.copy()
                        imgb1B = imgb2B.copy()
                        imgb2B = b_gpu.copy()

                # Adaptative Absorber Denoise Filter Back
                if flag_AADFB == True and flag_hold_picture == 0 and flag_image_mode == False :
                    if compteur_AADFB < 3 :
                        compteur_AADFB = compteur_AADFB + 1
                        if compteur_AADFB == 1 :
                            res_b2B = res_b1.copy()
                            Im1fsdnOKB = True
                        if compteur_AADFB == 2 :
                            Im2fsdnOKB = True

                    r_gpu = res_b1
     
                    if Im2fsdnOKB == True :

                        local_dyn = 1
                        local_GR = 0
                        local_VGR = 0
                        
                        adaptative_absorber_denoise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, res_b2B,\
                                             np.int_(width), np.int_(height),np.intc(local_dyn),np.intc(local_GR),np.intc(local_VGR)))

                        res_b2B = res_b1.copy()
                        tmp = cp.asarray(r_gpu).astype(cp.float64) * 1.05
                        tmp = cp.clip(tmp,0,255)
                        res_b1 = cp.asarray(tmp,dtype=cp.uint8)

                # Image Sharpen 1
                if flag_sharpen_soft1 == 1 :
                    cupy_context.use()
                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen * (tmp_b1 - res_b1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    if flag_sharpen_soft2 == 1 :
                        res_s1_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                    else :
                        res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)

                # Image Sharpen 2
                if flag_sharpen_soft2 == 1 :
                    cupy_context.use()
                    res_b1_blur = gaussianblur_mono(res_b1,val_sigma_sharpen2)
                    tmp_b1 = cp.asarray(res_b1).astype(cp.int16)
                    tmp_b1 = tmp_b1 + val_sharpen2 * (tmp_b1 - res_b1_blur)
                    tmp_b1 = cp.clip(tmp_b1,0,255)
                    if flag_sharpen_soft1 == 1 :
                        res_s2_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)
                    else :
                        res_b1 = cp.asarray(tmp_b1,dtype=cp.uint8)

                if flag_sharpen_soft1 == 1 and flag_sharpen_soft2 == 1 :
                    res_b1 = res_s1_b1 // 2 + res_s2_b1 // 2
 
                image_traitee = res_b1.get()
                
                if flag_DEMO == 1 :
                    if flag_demo_side == "Left" :
                        image_traitee[0:height,0:width//2] = image_base[0:height,0:width//2]            
                    if flag_demo_side == "Right" :
                        image_traitee[0:height,width//2:width] = image_base[0:height,width//2:width]            
    
    stop_time_test = cv2.getTickCount()
    time_exec_test= int((stop_time_test-start_time_test)/cv2.getTickFrequency()*1000)

    TTQueue.append(time_exec_test)
    if len(TTQueue) > 10:
        TTQueue.pop(0)
    curTT = (sum(TTQueue)/len(TTQueue))

    flag_filtre_work = False 

 
def mount_info() :
    global azimut,hauteur,flag_mount_connect
    
    if flag_mount_connect == False :
        try :
            smc=synscan2.motors()
            flag_mount_connect = True
        except :
            flag_mount_connect = False
            hauteur = 0.0
            azimut = 0.0
    if flag_mount_connect == True :
        try :
            hauteur = smc.axis_get_pos(1)
            azimut = smc.axis_get_pos(2)
        except :
            flag_mount_connect = False

                        
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
           flag_pause_video,timer1,val_deltat,flag_DETECT_STARS,nb_sat,trig_count_down,flag_TRIGGER,flag_sat_detected,video_frame_number,video_frame_position,flag_GO,\
           image_camera_old,image_camera,flag_camera_ok,previous_frame_number,image_camera
           
    if flag_camera_ok == True :
        if nb_cap_video == 1 :
            previous_frame_number = -1
            if flag_HQ == 0 and flag_filtrage_ON == True :
                if flag_filter_wheel == True:
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.mp4'
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
                        if flag_GO == True :
                            if image_camera > previous_frame_number :
                                if image_sauve.ndim == 3 :
                                    video.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                                else :
                                    video.write(image_sauve)
                                if (nb_cap_video % 5) == 0 :
                                    time_rec = int(nb_cap_video/2.5)/10
                                    labelInfo1.config(text = " frame : " + str (nb_cap_video) + "    " + str (time_rec) + " sec                   ")
                                    previous_frame_number = image_camera
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
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.mp4'
                else :
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
            else :
                if flag_filter_wheel == True:
                    nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '_F' + "%01d" % fw_position + '.mp4'
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
                if flag_GO == True :
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
            flag_cap_video = False
            videoOut.release()
            labelInfo1.config(text = " Acquisition vidéo terminee     ")
            if flag_SER_file == True :
                if nb_cap_video > val_nb_capt_video :
                    video_frame_position = nb_cap_video
                else :
                    video_frame_position = 1
                video.setCurrentPosition(video_frame_position)
                

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
    global flag_premier_demarrage,Video_Test,flag_image_mode,flag_image_video_loaded,flag_SAT_Image,Sat_Vid_Img,flag_SER_file

    flag_image_video_loaded = False
    filetypes = (
            ('Movie files', '*.mp4 *.avi *.ser *.mov'),
            ('Image files', '*.tif *.tiff *.jpg *.jpeg'),
            ('ALL files', '*.*')
        )

    Video_name = fd.askopenfilename(title='Open a file',initialdir=video_path,filetypes=filetypes)
    if Video_name != "" :
        reset_general_FS()
        Video_Test = Video_name
        flag_premier_demarrage = True
        if Video_Test.lower().endswith(('.mp4', '.avi', '.ser', '.mov')) :
            flag_image_mode = False
            flag_SAT_Image = False
            Sat_Vid_Img.set(0)
            if Video_Test.lower().endswith(('.ser')) :
                flag_SER_file = True
            else :
                flag_SER_file = False
            
        if Video_Test.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) :
            flag_image_mode = True
            flag_SAT_Image = True
            Sat_Vid_Img.set(1)
    flag_image_video_loaded = True


def load_image() :
    global flag_premier_demarrage,Video_Test,flag_image_mode,flag_image_video_loaded,flag_SAT_Image,Sat_Vid_Img

    flag_image_video_loaded = False
    filetypes = (
            ('Image files', '*.tif *.tiff *.jpg *.jpeg'),
            ('Movie files', '*.mp4 *.avi *.ser *.mov'),
            ('ALL files', '*.*')
        )

    Video_name = fd.askopenfilename(title='Open a file',initialdir=image_path,filetypes=filetypes)
    if Video_name != "" :
        reset_general_FS()
        Video_Test = Video_name
        flag_premier_demarrage = True
        if Video_Test.lower().endswith(('.mp4', '.avi', '.ser', '.mov')) :
            flag_image_mode = False
            flag_SAT_Image = False
            Sat_Vid_Img.set(0)
        if Video_Test.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) :
            flag_image_mode = True
            flag_SAT_Image = True
            Sat_Vid_Img.set(1)
    flag_image_video_loaded = True


def choose_dir_vid() :
    global video_path

    video_path = fd.askdirectory()


def choose_dir_pic() :
    global image_path

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

    
def HDR_Mertens() :
    global mode_HDR

    mode_HDR = "Mertens"


def HDR_Mean() :
    global mode_HDR

    mode_HDR = "Mean"


def mode_acq_rapide() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,timeoutexp,flag_read_speed,\
           frame_rate,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
        flag_acq_rapide = "Fast"
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
           val_exposition,frame_rate,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
        flag_acq_rapide = "MedF"
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
           val_exposition,frame_rate,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
        flag_acq_rapide = "MedS"
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
           val_exposition,frame_rate,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
        flag_acq_rapide = "Slow"
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
    global timeoutexp,timeout_val,flag_acq_rapide,camera,val_exposition,echelle1,val_resolution,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
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
    global camera,val_gain,echelle2,flag_stop_acquisition

    if flag_camera_ok == True :
        if flag_autoexposure_gain == False :
            val_gain = echelle2.get()
            camera.set_control_value(asi.ASI_GAIN, val_gain)


def choix_BIN1(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_stop_acquisition,mode_BIN,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video,flag_image_disponible,flag_new_image

    if flag_camera_ok == True :
        if flag_cap_video == False :
            flag_image_disponible = False
            flag_new_image = False
            reset_general_FS()
            reset_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_nouvelle_resolution = True
            flag_stop_acquisition=True
            stop_tracking()
            time.sleep(0.5)
            echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 130, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
            val_resolution = 1
            echelle3.set (val_resolution)
            echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)
            time.sleep(0.1)
            mode_BIN = 1
            choix_resolution_camera()
            flag_stop_acquisition=False
    else :
        mode_BIN = 1
            

def choix_BIN2(event=None) :
    global val_FS,camera,val_resolution,echelle3,flag_stop_acquisition,mode_BIN,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video

    if flag_camera_ok == True :
        if flag_cap_video == False :
            reset_general_FS()
            reset_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_nouvelle_resolution = True
            flag_stop_acquisition=True
            stop_tracking()
            time.sleep(0.5)
            echelle3 = Scale (cadre, from_ = 1, to = 7, command= choix_resolution_camera, orient=HORIZONTAL, length = 130, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
            val_resolution = 1
            echelle3.set (val_resolution)
            echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)
            mode_BIN = 2
            choix_resolution_camera()
            flag_stop_acquisition=False
    else :
        mode_BIN = 2
            
    
def choix_resolution_camera(event=None) :
    global val_FS,camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,echelle3,\
           flag_image_disponible,flag_stop_acquisition,\
           flag_nouvelle_resolution,tnr,inSize,backend,choix_TIP,flag_TIP,flag_cap_video
    
    if flag_camera_ok == True :
        if flag_cap_video == False :
            reset_FS()
            reset_general_FS()
            flag_TIP = 0
            choix_TIP.set(0)
            flag_stop_acquisition=True
            time.sleep(0.1)
            stop_tracking()
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
    global FlipV,FlipH,val_FS,GPU_BAYER,type_flip,type_debayer

    if flag_camera_ok == True :
        reset_FS()
        if choix_flipV.get() == 0 :
            FlipV = 0
        else :
            FlipV = 1
        if Camera_Bayer == "RAW" :
            type_debayer = 0
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 1)            
        if Camera_Bayer == "RGGB" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "BGGR" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 3)
                type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GRBG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GBRG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipV.get() == 0 :
            FlipV = 0
        else :
            FlipV = 1
        if Video_Bayer == "RGGB" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "horizontal"
        if Video_Bayer == "BGGR" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "horizontal"
        if Video_Bayer == "GRBG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "horizontal"
        if Video_Bayer == "GBRG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "horizontal"
        if Video_Bayer == "RAW" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 0
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 0
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 0
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 0
                type_flip = "horizontal"
        

def commande_flipH() :
    global FlipH,FlipV,val_FS,GPU_BAYER,type_flip,type_debayer

    if flag_camera_ok == True :
        reset_FS()
        if choix_flipH.get() == 0 :
            FlipH = 0
        else :
            FlipH = 1
        if Camera_Bayer == "RAW" :
            type_debayer = 0
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 0
                camera.set_control_value(asi.ASI_FLIP, 1)            
        if Camera_Bayer == "RGGB" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "BGGR" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 3)
                type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GRBG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GBRG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                type_debayer = cv2.COLOR_BayerGB2RGB
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                type_debayer = cv2.COLOR_BayerGR2RGB
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                type_debayer = cv2.COLOR_BayerRG2RGB
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                type_debayer = cv2.COLOR_BayerBG2RGB
                camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipH.get() == 0 :
            FlipH = 0
        else :
            FlipH = 1
        if Video_Bayer == "RGGB" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "horizontal"
        if Video_Bayer == "BGGR" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "horizontal"
        if Video_Bayer == "GRBG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "horizontal"
        if Video_Bayer == "GBRG" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "horizontal"
        if Video_Bayer == "RAW" :
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 0
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 0
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 0
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 0
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
    global flag_full_res,delta_zx,delta_zy
    
    delta_zx = 0
    delta_zy = 0
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


def commande_NLM2() :
    global flag_NLM2
    
    if choix_NLM2.get() == 0 :
        flag_NLM2 = 0
    else :
        flag_NLM2 = 1


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
    global flag_STAB, flag_Template,delta_tx,delta_ty,DSW
    
    delta_tx = 0
    delta_ty = 0
    DSW = 0
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

      
def commande_AADF() :
    global flag_AADF,compteur_AADF,Im1fsdnOK,Im2fsdnOK
    
    if choix_AADF.get() == 0 :
        flag_AADF = False
    else :
        flag_AADF = True
        compteur_AADF = 0
        Im1fsdnOK = False
        Im2fsdnOK = False


def commande_AADFB() :
    global flag_AADFB,compteur_AADFB,Im1fsdnOKB,Im2fsdnOKB
    
    if choix_AADFB.get() == 0 :
        flag_AADFB = False
    else :
        flag_AADFB = True
        compteur_AADFB = 0
        Im1fsdnOKB = False
        Im2fsdnOKB = False


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


def commande_3FNRB() :
    global flag_3FNRB,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start
    
    if choix_3FNRB.get() == 0 :
        flag_3FNRB = False
    else :
        flag_3FNRB = True
        compteur_3FNRB = 0
        img1_3FNROKB = False
        img2_3FNROKB = False
        img3_3FNROKB = False
        FNRB_First_Start = True


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
    global flag_reduce_variation,val_RV,compteur_RV,Im1rvOK,Im2rvOK,flag_BFREF,flag_BFREF_image,max_qual
    
    if choix_reduce_variation.get() == 0 :
        flag_reduce_variation = False
        flag_BFREF = False
        flag_BFREF_image = False
    else :
        flag_reduce_variation = True
        val_RV = 1
        compteur_RV = 0
        Im1rvOK = False
        Im2rvOK = False
        max_qual = 0
        flag_BFREF = True
        flag_BFREF_image = False


def commande_reduce_variation_post_treatment() :
    global flag_reduce_variation_post_treatment,flag_BFREF_imagePT,max_qual_PT,flag_BFREFPT
    
    if choix_reduce_variation_post_treatment.get() == 0 :
        flag_reduce_variation_post_treatment = False
        flag_BFREFPT = False
        flag_BFREF_image = False
        max_qual_PT = 0
    else :
        flag_reduce_variation_post_treatment = True
        max_qual_PT = 0
        flag_BFREFPT = True
        flag_BFREF_image_PT = False


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


def commande_HOTPIX() :
    global flag_hot_pixels
    
    if choix_HOTPIX.get() == 0 :
        flag_hot_pixels = False
    else :
        flag_hot_pixels = True


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
    global camera,flag_16b,flag_HB

    if flag_camera_ok == True :
        if flag_16b == False :
            if choix_hard_bin.get() == 0 :
                camera.set_control_value(asi.ASI_HARDWARE_BIN,0)
                flag_HB = False
            else :
                camera.set_control_value(asi.ASI_HARDWARE_BIN,1)
                flag_HB = True
            time.sleep(0.1)
        else :
            if choix_hard_bin.get() == 0 :
                flag_HB = False
            else :
                flag_HB = True
    else :
        if choix_hard_bin.get() == 0 :
            flag_HB = False
        else :
            flag_HB = True
            

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


def choix_dyn_high(event=None):
    global flag_dyn_AADF
    
    flag_dyn_AADF = 1 


def choix_dyn_low(event=None):
    global flag_dyn_AADF
    
    flag_dyn_AADF = 0    


def choix_SAT_Vid() :
    global flag_SAT_Image
    
    flag_SAT_Image = False


def choix_SAT_Img() :
    global flag_SAT_Image
    
    flag_SAT_Image = True


def commande_autoexposure() :
    global flag_autoexposure_exposition,flag_autoexposure_gain,camera,controls,echelle1,val_exposition

    if flag_camera_ok == True :
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
    global flag_autoexposure_exposition,flag_autoexposure_gain,camera,controls,val_gain

    if flag_camera_ok == True :
        if choix_autogain.get() == 0 :
            flag_autoexposure_gain = False
            camera.set_control_value(asi.ASI_GAIN,controls['Gain']['DefaultValue'],auto=False)
            camera.set_control_value(asi.ASI_GAIN, val_gain)
        else :
            flag_autoexposure_gain = True
            camera.set_control_value(asi.ASI_GAIN,controls['Gain']['DefaultValue'],auto=True)
            camera.set_control_value(controls['AutoExpMaxGain']['ControlType'], controls['AutoExpMaxGain']['MaxValue'])


def commande_noir_blanc() :
    global val_FS,flag_noir_blanc,flag_NB_estime,format_capture,flag_stop_acquisition,flag_filtrage_ON

    reset_FS()
    reset_general_FS()
    flag_stop_acquisition=True
    flag_restore_filtrage = False
    if choix_noir_blanc.get() == 0 and flag_camera_ok == False :
        flag_noir_blanc = 0
    if choix_noir_blanc.get() == 1 and flag_camera_ok == False :
        flag_noir_blanc = 1
    if flag_camera_ok == True :
        if flag_filtrage_ON == True :
            flag_filtrage_ON == False
            flag_restore_filtrage = True
            time.sleep(0.3)
        if choix_noir_blanc.get() == 0 and flag_colour_camera == True :
            flag_noir_blanc = 0
        else :
            flag_noir_blanc = 1
        if choix_noir_blanc_estime.get() == 0 :
            flag_NB_estime = 0
        else :
            flag_NB_estime = 1
        flag_stop_acquisition = False
        time.sleep(0.2)
        if flag_restore_filtrage == True :
            flag_filtrage_ON = True
    else :
        if choix_noir_blanc_estime.get() == 0 :
            flag_NB_estime = 0
        else :
            flag_NB_estime = 1


def commande_16bLL() :
    global flag_16b,format_capture,flag_stop_acquisition,flag_filtrage_ON,camera

    reset_FS()
    reset_general_FS()
    flag_stop_acquisition=True
    flag_restore_filtrage = False
    if flag_camera_ok == True :
        if flag_filtrage_ON == True :
            flag_filtrage_ON == False
            flag_restore_filtrage = True
            time.sleep(0.3)
        if choix_16bLL.get() == 0 :
            flag_16b = False
            format_capture = asi.ASI_IMG_RAW8
            camera.stop_video_capture()
            time.sleep(1)
            camera.set_image_type(format_capture)
            time.sleep(0.5)
            camera.start_video_capture()
            time.sleep(0.5)            
        else :
            flag_16b = True
            format_capture = asi.ASI_IMG_RAW16
            camera.stop_video_capture()
            time.sleep(1)
            camera.set_image_type(format_capture)
            time.sleep(0.5)
            camera.start_video_capture()
            time.sleep(0.5)            
        flag_stop_acquisition = False
        time.sleep(0.2)
        if flag_restore_filtrage == True :
            flag_filtrage_ON = True
    else :
        if choix_16bLL.get() == 0 :
            flag_16b = False
        else :
            flag_16b = True


def commande_IMQE() :
    global flag_IQE,quality,max_quality,quality_pos
    
    if choix_IMQE.get() == 0 :
        flag_IQE = 0
    else :
        flag_IQE = 1
    for x in range(1,256) :
        quality[x] = 0
    max_quality = 1
    quality_pos = 1
       

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
    global ASIGAMMA, echelle204,camera

    if flag_camera_ok == True :
        ASIGAMMA=echelle204.get()
        camera.set_control_value(asi.ASI_GAMMA, ASIGAMMA)


def choix_TH_16B(event=None) :
    global TH_16B, echelle804,camera,threshold_16bits

    TH_16B=echelle804.get()
    threshold_16bits = 2 ** TH_16B - 1


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


def choix_demo_left(event=None) :
    global flag_demo_side
    
    flag_demo_side = "Left"


def choix_demo_right(event=None) :
    global flag_demo_side
    
    flag_demo_side = "Right"


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


def command_BFReference(event=None) :
    global flag_BFReference
    
    flag_BFReference = "BestFrame"


def command_PFReference(event=None) :
    global flag_BFReference
    
    flag_BFReference = "PreviousFrame"


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
    global Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer
    
    type_debayer = 0
    Camera_Bayer = "RAW"
    Video_Bayer = "RAW"
    GPU_BAYER = 0
    choix_flipV.set(0)
    choix_flipH.set(0)
    commande_flipV()


def choix_bayer_RGGB(event=None) :
    global Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer
    
    type_debayer = cv2.COLOR_BayerBG2RGB
    Camera_Bayer = "RGGB"
    Video_Bayer = "RGGB"
    GPU_BAYER = 1
    choix_flipV.set(0)
    choix_flipH.set(0)
    commande_flipV()


def choix_bayer_BGGR(event=None) :
    global Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer
    
    type_debayer = cv2.COLOR_BayerRG2RGB
    Camera_Bayer = "BGGR"
    Video_Bayer = "BGGR"
    GPU_BAYER = 2
    choix_flipV.set(0)
    choix_flipH.set(0)
    commande_flipV()


def choix_bayer_GBRG(event=None) :
    global Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer
    
    type_debayer = cv2.COLOR_BayerGB2RGB
    Camera_Bayer = "GBRG"
    Video_Bayer = "GBRG"
    GPU_BAYER = 3
    choix_flipV.set(0)
    choix_flipH.set(0)
    commande_flipV()


def choix_bayer_GRBG(event=None) :
    global Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer
    
    type_debayer = cv2.COLOR_BayerGR2RGB
    Camera_Bayer = "GRBG"
    Video_Bayer = "GRBG"
    GPU_BAYER = 4
    choix_flipV.set(0)
    choix_flipH.set(0)
    commande_flipV()


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
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK,stack_div,echelle20,choix_AADF,choix_AADFB,flag_first_sat_pass,nb_sat,\
           flag_AADF,compteur_AADF,Im1fsdnOK,Im2fsdnOK,flag_AADFB,compteur_AADFB,Im1fsdnOKB,Im2fsdnOKB,delta_RX,delta_RY,delta_BX,delta_BY,\
           flag_STAB,flag_Template,choix_STAB,compteur_RV,Im1rvOK,Im2rvOK,flag_reduce_variation,choix_reduce_variation,\
           flag_3FNR,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNRB,compteur_3FNRB,\
           img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNR_First_StartB,flag_BFR,flag_STAB,flag_IQE,choix_IMQE,choix_BFR,labelInfo10,\
           flag_CONST,flag_TRKSAT,flag_REMSAT,flag_DETECT_STARS,choix_CONST,choix_TRKSAT,choix_REMSAT,choix_DETECT_STARS,choix_AI_Craters,flag_AI_Craters,\
           flag_reduce_variation_post_treatment,flag_BFREF_imagePT,max_qual_PT,flag_BFREFPT,choix_HDR,flag_HDR,choix_HOTPIX,flag_hot_pixels,\
           flag_AI_Craters,track_crater_history,flag_AI_Satellites,choix_AI_Satellites,track_satelitte_history,model_craters_track,model_satellites_track,\
           flag_image_disponible,flag_new_image,flag_img_sat_buf1,flag_img_sat_buf2,flag_img_sat_buf3,flag_img_sat_buf4,flag_img_sat_buf5,sat_frame_count,\
           flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,flag_img_sat_buf4_AI,flag_img_sat_buf5_AI,sat_frame_count_AI,flag_first_sat_pass_AI,\
           choix_sub_img_ref,flag_capture_image_reference,flag_image_ref_sub,flag_image_reference_OK

    choix_sub_img_ref.set(0)
    flag_capture_image_reference = False
    flag_image_reference_OK = False
    flag_image_ref_sub = False
    flag_image_disponible = False
    flag_new_image = False
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    echelle20.set(val_FS)
    choix_AADF.set(0)
    choix_AADFB.set(0)
    flag_AADF = False
    compteur_AADF = 0
    Im1fsdnOK = False
    Im2fsdnOK = False
    flag_AADFB = False
    compteur_AADFB = 0
    Im1fsdnOKB = False
    Im2fsdnOKB = False
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
    choix_3FNRB.set(0)
    flag_3FNRB = False
    compteur_3FNRB = 0
    img1_3FNROKB = False
    img2_3FNROKB = False
    img3_3FNROKB = False
    FNRB_First_Start = True
    flag_BFR = False
    choix_BFR.set(0)
    choix_IMQE.set(0)
    flag_IQE = False
    delta_RX = 0
    delta_RY = 0
    delta_BX = 0
    delta_BY =0
    flag_first_sat_pass = True
    flag_first_sat_pass_AI = True
    flag_img_sat_buf1 = False
    flag_img_sat_buf2 = False
    flag_img_sat_buf3 = False
    flag_img_sat_buf4 = False
    flag_img_sat_buf5 = False
    flag_img_sat_buf1_AI = False
    flag_img_sat_buf2_AI = False
    flag_img_sat_buf3_AI = False
    flag_img_sat_buf4_AI = False
    flag_img_sat_buf5_AI = False
    sat_frame_count = 0
    sat_frame_count_AI = 0
    nb_sat = -1
    texte = " "
    labelInfo10.config(text = texte)
    flag_CONST = 0
    choix_CONST.set(0)
    flag_TRKSAT = 0
    choix_TRKSAT.set(0)
    flag_REMSAT = 0
    choix_REMSAT.set(0)
    flag_DETECT_STARS = 0
    choix_DETECT_STARS.set(0)
    time.sleep(0.5)
    flag_AI_Craters = False
    choix_AI_Craters.set(0)
    flag_reduce_variation_post_treatment = False
    flag_BFREFPT = False
    flag_BFREF_image = False
    max_qual_PT = 0
    choix_HOTPIX.set(0)
    flag_hot_pixels = False
    choix_HDR.set(0)
    flag_HDR = False
    choix_AI_Craters.set(0)
    try :
        model_craters_track.predictor.trackers[0].reset()
        model_satellites_track.predictor.trackers[0].reset()
    except :
        pass
    track_crater_history = defaultdict(lambda: [])
    track_satelitte_history = defaultdict(lambda: [])
    flag_AI_Craters = False
    choix_AI_Satellites.set(0)
    flag_AI_Satellites = False
    time.sleep(0.2)


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
    global flag_TRKSAT,flag_nouvelle_resolution,flag_first_sat_pass,flag_img_sat_buf1,flag_img_sat_buf2,flag_img_sat_buf3,flag_img_sat_buf4,flag_img_sat_buf5,sat_frame_count

    if choix_TRKSAT.get() == 0 :
        flag_TRKSAT = 0
    else :
        flag_TRKSAT = 1
        flag_nouvelle_resolution = True
    flag_first_sat_pass = True
    sat_frame_count = 0
    flag_img_sat_buf1 = False
    flag_img_sat_buf2 = False
    flag_img_sat_buf3 = False
    flag_img_sat_buf4 = False
    flag_img_sat_buf5 = False


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


def commande_BFR() :
    global flag_BFR,min_qual,max_qual,val_BFR,echelle300,SFN,frame_number,labelInfo10
    
    if choix_BFR.get() == 0 :
        flag_BFR = False
        max_qual = 0
        min_qual = 10000
        val_BFR = 50
        echelle300.set(val_BFR)
        SFN = 0
        frame_number = 0
        labelInfo10.config(text = "                                             ")
    else :
        flag_BFR = True
        max_qual = 0
        min_qual = 10000
        val_BFR = 50
        echelle300.set(val_BFR)
        SFN = 0
        frame_number = 0
        labelInfo10.config(text = "                                             ")


def choix_val_BFR(event=None) :
    global val_BFR, echelle300
    
    val_BFR = echelle300.get()


def choix_position_frame(event=None) :
    global  echelle210,video,flag_new_frame_position,flag_SER_file,video_frame_position
    
    if flag_cap_video == False and flag_image_mode == False :
        flag_new_frame_position = True
        video_frame_position = echelle210.get()
        if flag_SER_file == False :
            video.set(cv2.CAP_PROP_POS_FRAMES,video_frame_position)
        else :
            video.setCurrentPosition(video_frame_position)


def raz_framecount() :
    global compteur_images,numero_image,image_camera,image_camera_old
    
    compteur_images = 0
    numero_image = 0
    image_camera = 0
    image_camera_old = 0


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


def calc_jour_julien (jours, mois, annee):
    global jour_julien
    
    if mois < 3 :
        mois = mois + 12
        annee = annee - 1
    coef_a = annee // 100
    coef_b = 2 - coef_a + (coef_a // 4)
    coef_c = int(365.25 * annee)
    coef_d = int(30.6001 * (mois + 1))
    jour_julien = coef_b + coef_c + coef_d + jours + 1720994.5


def calc_heure_siderale(jrs_jul,heure_obs, min_obs, zone):
    global HS
    
    TT = (jrs_jul - 2451545) / 36525
    H1 = 24110.54841 + (8640184.812866 * TT) + (0.093104 * (TT * TT)) - (0.0000062 * (TT * TT * TT))
    HSH = H1 / 3600
    HS = ((HSH / 24) - int(HSH / 24)) * 24


# Calcul azimut et hauteur cible - coordonnÃ©es altaz - a partir donnÃ©es cible, lieu observation et date
def calcul_AZ_HT_cible(jours_obs, mois_obs, annee_obs, heure_obs, min_obs, second_obs, lat_obs, long_obs, cible_ASD, cible_DEC):
    global azimut_cible, hauteur_cible, HS
    
    calc_jour_julien(jours_obs, mois_obs, annee_obs)
    calc_heure_siderale(jour_julien,heure_obs, min_obs, zone)

    angleH =  (2 * Pi * HS / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi
    angleT = ((heure_obs - 12 + min_obs / 60 - zone + second_obs / 3600) * 2 * Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi

    H = angleH + angleT - 15*(cible_ASD) + (long_obs)
    
    sinushauteur = math.sin(cible_DEC * conv_rad) * math.sin(lat_obs * conv_rad) - math.cos(cible_DEC * conv_rad) * math.cos(lat_obs * conv_rad) * math.cos(H * conv_rad)
    hauteur_cible = math.asin(sinushauteur)

    cosazimut = (math.sin(cible_DEC * conv_rad) - math.sin(lat_obs * conv_rad) * math.sin(hauteur_cible)) / (math.cos(lat_obs*conv_rad) * math.cos(hauteur_cible))
    sinazimut = (math.cos(cible_DEC * conv_rad) * math.sin(H* conv_rad)) / math.cos(hauteur_cible)

    if sinazimut > 0 :
        azimut_cible = math.acos(cosazimut)
    else :
        azimut_cible = - math.acos(cosazimut)


# Calcul ascension droite et declinaison d'un astre de l'azimut et la hauteur observees d'un astre, du lieu d'observation et de la date
def calcul_ASD_DEC_cible(jours_obs, mois_obs, annee_obs, heure_obs, min_obs, second_obs, lat_obs, long_obs, azimut_cible, hauteur_cible):
    global ASD_calculee, DEC_calculee,HS
    
    calc_jour_julien(jours_obs, mois_obs, annee_obs)
    calc_heure_siderale(jour_julien,heure_obs, min_obs, zone)

    angleH =  (2 * Pi * HS / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi
    angleT = ((heure_obs - 12 + min_obs / 60 - zone + second_obs / 3600) * 2 * Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi

    DEC_calculee = math.asin(math.cos(azimut_cible) * math.cos(lat_obs * conv_rad) * math.cos(hauteur_cible) + math.sin(lat_obs * conv_rad) * math.sin(hauteur_cible))

    if azimut_cible > 0 :
        H = math.acos((math.sin(DEC_calculee) * math.sin(lat_obs * conv_rad) - math.sin(hauteur_cible))/(math.cos(DEC_calculee) * math.cos(lat_obs * conv_rad)))
    else :
        H = - math.asin(math.sin(azimut_cible)*math.cos(hauteur_cible)/math.cos(DEC_calculee)) + Pi

    ASD_calculee = (angleH + angleT + long_obs - H * conv_deg)/15

    DEC_calculee = DEC_calculee


def Mount_calibration() :
    global azimut_moteur,hauteur_moteur,lat_obs,long_obs,Polaris_AD, Polaris_DEC,jour_julien,zone,azimut_monture,hauteur_monture,delta_azimut,delta_hauteur,labelInfo1

    date = datetime.now()
    annee_obs = date.year
    mois_obs = date.month
    jours_obs = date.day
    heure_obs = date.hour
    min_obs = date.minute
    second_obs = date.second

    cible_polaire_ASD = Polaris_AD
    cible_polaire_DEC = Polaris_DEC

    calc_jour_julien(jours_obs, mois_obs, annee_obs)
    calc_heure_siderale(jour_julien,heure_obs, min_obs, zone)

    angleH =  (2 * Pi * HS / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi
    angleT = ((heure_obs - 12 + min_obs / 60 - zone + second_obs / 3600) * 2 * Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / Pi

    H = angleH + angleT - 15*(cible_polaire_ASD) + (long_obs)
    
    sinushauteur = math.sin(cible_polaire_DEC * conv_rad) * math.sin(lat_obs * conv_rad) - math.cos(cible_polaire_DEC * conv_rad) * math.cos(lat_obs * conv_rad) * math.cos(H * conv_rad)
    hauteurcible = math.asin(sinushauteur)

    cosazimut = (math.sin(cible_polaire_DEC * conv_rad) - math.sin(lat_obs * conv_rad) * math.sin(hauteurcible)) / (math.cos(lat_obs*conv_rad) * math.cos(hauteurcible))
    sinazimut = (math.cos(cible_polaire_DEC * conv_rad) * math.sin(H* conv_rad)) / math.cos(hauteurcible)

    if sinazimut > 0 :
        azimut_polaris = math.acos(cosazimut) * conv_deg # compris entre -180 et +180
    else :
        azimut_polaris = - math.acos(cosazimut) * conv_deg # compris entre -180 et +180

    if azimut_polaris < 0 :
        azimut_polaris = 360 + azimut_polaris
        
    hauteur_polaris = hauteurcible * conv_deg # compris entre 0 et 90
           
    texte = "Azimut : %6.2f" %azimut_polaris
    texte = texte + "  "            
    print(texte)
    texte = "Hauteur : %6.2f" %hauteur_polaris
    texte = texte + "  "
    print(texte)
    print(azimut," ",hauteur)
    delta_azimut = azimut_polaris - azimut_monture
    delta_hauteur = hauteur_polaris - hauteur_monture
    print(delta_azimut,delta_hauteur)
    labelInfo1.config(text = " Mount aligned on Polaris     ")


def commande_false_colours() :
    global flag_false_colours
    
    if choix_false_colours.get() == 0 :
        flag_false_colours = False
    else :
        flag_false_colours = True


def commande_AI_Craters() :
    global flag_AI_Craters,model_craters,track_crater_history
    
    if choix_AI_Craters.get() == 0 :
        if flag_crater_model_loaded == True :
            try :
                model_craters_track.predictor.trackers[0].reset()
            except :
                pass
        track_crater_history = defaultdict(lambda: [])
        flag_AI_Craters = False
    else :
        if flag_crater_model_loaded == True :
            try :
                model_craters_track.predictor.trackers[0].reset()
            except :
                pass
        track_crater_history = defaultdict(lambda: [])
        flag_AI_Craters = True


def commande_AI_Satellites() :
    global flag_AI_Satellites,model_Satellites,track_satellite_history,sat_frame_count_AI,flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,\
           flag_img_sat_buf4_AI,flag_img_sat_buf5_AI,flag_first_sat_pass_AI
           
    if choix_AI_Satellites.get() == 0 :
        if flag_satellites_model_loaded == True :
            try :
                model_satellites_track.predictor.trackers[0].reset()
            except :
                pass
        track_satellite_history = defaultdict(lambda: [])
        flag_AI_Satellites = False
    else :
        if flag_satellites_model_loaded == True :
            try :
                model_satellites_track.predictor.trackers[0].reset()
            except :
                pass
        track_satellite_history = defaultdict(lambda: [])
        flag_AI_Satellites = True
    sat_frame_count_AI = 0
    flag_img_sat_buf1_AI = False
    flag_img_sat_buf2_AI = False
    flag_img_sat_buf3_AI = False
    flag_img_sat_buf4_AI = False
    flag_img_sat_buf5_AI = False
    flag_first_sat_pass_AI = True
    

def commande_AI_Trace() :
    global flag_AI_Trace,track_crater_history,track_satellite_history,model_craters_track,model_satellites_track
    
    if choix_AI_Trace.get() == 0 :
        flag_AI_Trace = False
        try :
            model_satellites_track.predictor.trackers[0].reset()
            model_craters_track.predictor.trackers[0].reset()
        except :
            pass
        track_satellite_history = defaultdict(lambda: [])
        track_crater_history = defaultdict(lambda: [])        
    else :
        flag_AI_Trace = True
        try :
            model_satellites_track.predictor.trackers[0].reset()
            model_craters_track.predictor.trackers[0].reset()
        except :
            pass
        track_satellite_history = defaultdict(lambda: [])
        track_crater_history = defaultdict(lambda: [])


def Capture_Ref_Img() :
    global flag_capture_image_reference
    
    flag_capture_image_reference = True


def commande_sub_img_ref() :
    global flag_image_ref_sub
    
    if choix_sub_img_ref.get() == 0 :
        flag_image_ref_sub = False
    else :
        flag_image_ref_sub = True


# initialisation des boites scrolbar, buttonradio et checkbutton

xS3=1690
yS3=80

xS1=1490
yS1=270


###########################
#     VARIOUS WIDGETS     #
###########################

# Bandwidth USB
labelParam50 = Label (cadre, text = "USB")
labelParam50.place(anchor="w", x=15,y=255)
echelle50 = Scale (cadre, from_ = 40, to = 100, command= choix_USB, orient=VERTICAL, length = 80, width = 7, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle50.set(val_USB)
echelle50.place(anchor="c", x=20,y=310)

# Flip Vertical
CBFV = Checkbutton(cadre,text="FlipV", variable=choix_flipV,command=commande_flipV,onvalue = 1, offvalue = 0)
CBFV.place(anchor="w",x=5, y=20)

# Flip Horizontal
CBFH = Checkbutton(cadre,text="FlipH", variable=choix_flipH,command=commande_flipH,onvalue = 1, offvalue = 0)
CBFH.place(anchor="w",x=5, y=45)

# Hot Pixel removal
CBHOTPIX = Checkbutton(cadre,text="Hot Pix", variable=choix_HOTPIX,command=commande_HOTPIX,onvalue = 1, offvalue = 0)
CBHOTPIX.place(anchor="w",x=5, y=180)

# Filter Wheel position
labelFW = Label (cadre, text = "FW :")
labelFW.place (anchor="w",x=5, y=400)
RBEFW1 = Radiobutton(cadre,text="#1", variable=fw_position_,command=choix_position_EFW0,value=0)
RBEFW1.place(anchor="w",x=5, y=420)
RBEFW2 = Radiobutton(cadre,text="#2", variable=fw_position_,command=choix_position_EFW1,value=1)
RBEFW2.place(anchor="w",x=5, y=440)
RBEFW3 = Radiobutton(cadre,text="#3", variable=fw_position_,command=choix_position_EFW2,value=2)
RBEFW3.place(anchor="w",x=5, y=460)
RBEFW4 = Radiobutton(cadre,text="#4", variable=fw_position_,command=choix_position_EFW3,value=3)
RBEFW4.place(anchor="w",x=5, y=480)
RBEFW5 = Radiobutton(cadre,text="#5", variable=fw_position_,command=choix_position_EFW4,value=4)
RBEFW5.place(anchor="w",x=5, y=500)

# Bayer Matrix
labelBM = Label (cadre, text = "Debayer :")
labelBM.place (anchor="w",x=5, y=525)
RBEBM1 = Radiobutton(cadre,text="RAW", variable=bayer_sensor,command=choix_bayer_RAW,value=1)
RBEBM1.place(anchor="w",x=5, y=545)
RBEBM2 = Radiobutton(cadre,text="RGGB", variable=bayer_sensor,command=choix_bayer_RGGB,value=2)
RBEBM2.place(anchor="w",x=5, y=565)
RBEBM3 = Radiobutton(cadre,text="BGGR", variable=bayer_sensor,command=choix_bayer_BGGR,value=3)
RBEBM3.place(anchor="w",x=5, y=585)
RBEBM4 = Radiobutton(cadre,text="GRBG", variable=bayer_sensor,command=choix_bayer_GRBG,value=4)
RBEBM4.place(anchor="w",x=5, y=605)
RBEBM5 = Radiobutton(cadre,text="GBRG", variable=bayer_sensor,command=choix_bayer_GBRG,value=5)
RBEBM5.place(anchor="w",x=5, y=625)

labelTRK = Label (cadre, text = "Tracking :")
labelTRK.place (anchor="w",x=0, y=650)

# Find Stars
CBDTCSTARS = Checkbutton(cadre,text="Stars", variable=choix_DETECT_STARS,command=commande_DETECT_STARS,onvalue = 1, offvalue = 0)
CBDTCSTARS.place(anchor="w",x=0, y=670)

# Track Satellites
CBTRKSAT = Checkbutton(cadre,text="Sat Detect", variable=choix_TRKSAT,command=commande_TRKSAT,onvalue = 1, offvalue = 0)
CBTRKSAT.place(anchor="w",x=0, y=690)

# Remove Satellites
CBREMSAT = Checkbutton(cadre,text="Sat Remov", variable=choix_REMSAT,command=commande_REMSAT,onvalue = 1, offvalue = 0)
CBREMSAT.place(anchor="w",x=0, y=710)

# Track Meteor
CBTRIGGER = Checkbutton(cadre,text="Trigger", variable=choix_TRIGGER,command=commande_TRIGGER,onvalue = 1, offvalue = 0)
CBTRIGGER.place(anchor="w",x=0, y=730)

# Image reconstruction
CBCONST = Checkbutton(cadre,text="Reconst", variable=choix_CONST,command=commande_CONST,onvalue = 1, offvalue = 0)
CBCONST.place(anchor="w",x=0, y=750)

# Artifical intelligence object detection
labelAI = Label (cadre, text = " A I detect:")
labelAI.place (anchor="w",x=0, y=780)

CBAICTR = Checkbutton(cadre,text="Craters", variable=choix_AI_Craters,command=commande_AI_Craters,onvalue = 1, offvalue = 0)
CBAICTR.place(anchor="w",x=0, y=800)
CBAISAT = Checkbutton(cadre,text="Satellites", variable=choix_AI_Satellites,command=commande_AI_Satellites,onvalue = 1, offvalue = 0)
CBAISAT.place(anchor="w",x=0, y=820)
CBAITRC = Checkbutton(cadre,text="Tracking", variable=choix_AI_Trace,command=commande_AI_Trace,onvalue = 1, offvalue = 0)
CBAITRC.place(anchor="w",x=0, y=840)

# Capture reference image
Button7 = Button (cadre,text = "Ref Img Cap", command = Capture_Ref_Img,padx=10,pady=0)
Button7.place(anchor="w", x=0,y=870)
CBSUBIMREF = Checkbutton(cadre,text="Sub Img Ref", variable=choix_sub_img_ref,command=commande_sub_img_ref,onvalue = 1, offvalue = 0)
CBSUBIMREF.place(anchor="w",x=0, y=900)

# Set 16 bits threshold
labelParam804 = Label (cadre, text = "16bit Th")
labelParam804.place(anchor="w", x=1840+delta_s,y=20)
echelle804 = Scale (cadre, from_ = 8, to = 16, command= choix_TH_16B, orient=VERTICAL, length = 80, width = 7, resolution = 0.5, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle804.set(TH_16B)
echelle804.place(anchor="c", x=1860+delta_s,y=70)

# Set Gamma value ZWO camera
labelParam204 = Label (cadre, text = "Gamma Cam")
labelParam204.place(anchor="w", x=1840+delta_s,y=125)
echelle204 = Scale (cadre, from_ = 0, to = 100, command= choix_ASI_GAMMA, orient=VERTICAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle204.set(ASIGAMMA)
echelle204.place(anchor="c", x=1860+delta_s,y=190)

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
CBSAT.place(anchor="w",x=1840+delta_s,y=640)
RBSATVI1 = Radiobutton(cadre,text="Vid", variable=Sat_Vid_Img,command=choix_SAT_Vid,value=0)
RBSATVI1.place(anchor="w",x=1840+delta_s, y=660)
RBSATVI2 = Radiobutton(cadre,text="Img", variable=Sat_Vid_Img,command=choix_SAT_Img,value=1)
RBSATVI2.place(anchor="w",x=1840+delta_s, y=680)
echelle70 = Scale (cadre, from_ = 0, to = 40, command= choix_val_SAT, orient=VERTICAL, length = 200, width = 7, resolution = 0.01, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle70.set(val_SAT)
echelle70.place(anchor="c", x=1855+delta_s,y=800)

# DEMO
CBDEMO = Checkbutton(cadre,text="Demo", variable=choix_DEMO,command=commande_DEMO,onvalue = 1, offvalue = 0)
CBDEMO.place(anchor="w",x=1840+delta_s, y=960)
RBDEML = Radiobutton(cadre,text="Left", variable=demo_side,command=choix_demo_left,value=0)
RBDEML.place(anchor="w",x=1840+delta_s, y=985)
RBDEMR = Radiobutton(cadre,text="Right", variable=demo_side,command=choix_demo_right,value=1)
RBDEMR.place(anchor="w",x=1840+delta_s, y=1010)

# Choix filtrage ON
CBF = Checkbutton(cadre,text="Filters ON", variable=choix_filtrage_ON,command=commande_filtrage_ON,onvalue = 1, offvalue = 0)
CBF.place(anchor="w",x=1440+delta_s, y=50)

# Fulres displaying
CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=1510+delta_s, y=50)

# Choix forcage N&B
CBFNB = Checkbutton(cadre,text="Set B&W", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=1570+delta_s, y=50)

# Choix forcage N&B Estimate
CBFNBE = Checkbutton(cadre,text="B&W Est", variable=choix_noir_blanc_estime,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNBE.place(anchor="w",x=1630+delta_s, y=50)

# Choix forcage N&B Estimate
CBRRB = Checkbutton(cadre,text="R-B Rev", variable=choix_reverse_RB,command=commande_reverse_RB,onvalue = 1, offvalue = 0)
CBRRB.place(anchor="w",x=1695+delta_s, y=50)

# Choix False colours
CBFC = Checkbutton(cadre,text="False Col", variable=choix_false_colours,command=commande_false_colours,onvalue = 1, offvalue = 0)
CBFC.place(anchor="w",x=1750+delta_s, y=50)

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
labelMode_Acq.place (anchor="w",x=1440+delta_s, y=240)
RBMA1 = Radiobutton(cadre,text="Fast", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=1495+delta_s, y=240)
RBMA2 = Radiobutton(cadre,text="MedF", variable=mode_acq,command=mode_acq_mediumF,value=2)
RBMA2.place(anchor="w",x=1540+delta_s, y=240)
RBMA3 = Radiobutton(cadre,text="MedS", variable=mode_acq,command=mode_acq_mediumS,value=3)
RBMA3.place(anchor="w",x=1585+delta_s, y=240)
RBMA4 = Radiobutton(cadre,text="Slow", variable=mode_acq,command=mode_acq_lente,value=4)
RBMA4.place(anchor="w",x=1630+delta_s, y=240)

# Choix HDR
CBHDR = Checkbutton(cadre,text="HDR", variable=choix_HDR,command=commande_HDR,onvalue = 1, offvalue = 0)
CBHDR.place(anchor="w",x=1690+delta_s, y=240)
RBHDR1 = Radiobutton(cadre,text="Mertens", variable=mode_HDR_select,command=HDR_Mertens,value=1)
RBHDR1.place(anchor="w",x=1730+delta_s, y=240)
RBHDR2 = Radiobutton(cadre,text="Mean", variable=mode_HDR_select,command=HDR_Mean,value=2)
RBHDR2.place(anchor="w",x=1790+delta_s, y=240)


# Automatic exposition time
CBOE = Checkbutton(cadre,text="AE", variable=choix_autoexposure,command=commande_autoexposure,onvalue = 1, offvalue = 0)
CBOE.place(anchor="w",x=1440+delta_s, y=yS1)

# Exposition setting
echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 330, width = 7, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
echelle1.set(val_exposition)
echelle1.place(anchor="w", x=xS1+delta_s,y=yS1)

# Choix du mode BINNING - 1, 2 ou 3
labelBIN = Label (cadre, text = "BIN : ")
labelBIN.place (anchor="w",x=1440+delta_s, y=80)
RBB1 = Radiobutton(cadre,text="BIN1", variable=choix_bin,command=choix_BIN1,value=1)
RBB1.place(anchor="w",x=1470+delta_s, y=80)
RBB2 = Radiobutton(cadre,text="BIN2", variable=choix_bin,command=choix_BIN2,value=2)
RBB2.place(anchor="w",x=1510+delta_s, y=80)

# Choix Hardware Bin
CBHB = Checkbutton(cadre,text="HB", variable=choix_hard_bin,command=commande_hard_bin,onvalue = 1, offvalue = 0)
CBHB.place(anchor="w",x=1550+delta_s, y=80)


# Choix 16 bits Low Light
CBHDR = Checkbutton(cadre,text="16bLL", variable=choix_16bLL,command=commande_16bLL,onvalue = 1, offvalue = 0)
CBHDR.place(anchor="w",x=1600+delta_s, y=80)

# Resolution setting
labelParam3 = Label (cadre, text = "RES : ")
labelParam3.place(anchor="w", x=1660+delta_s,y=80)
echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 130, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle3.set (val_resolution)
echelle3.place(anchor="w", x=xS3+delta_s,y=yS3)

# choix hold picture
#CBOP = Checkbutton(cadre,text="Hold Picture", variable=choix_hold_picture,command=commande_hold_picture,onvalue = 1, offvalue = 0)
#CBOP.place(anchor="w",x=1730+delta_s, y=240)
    
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
labelParam100.place(anchor="w", x=1440+delta_s,y=355)
echelle100 = Scale (cadre, from_ = 0, to = 2, command= choix_w_reds, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0.1,sliderlength=20)
echelle100.set(val_reds)
echelle100.place(anchor="w", x=1440+delta_s+15,y=355)

# Software Green balance
labelParam101 = Label (cadre, text = "G") # choix balance rouge
labelParam101.place(anchor="w", x=1440+delta_s,y=385)
echelle101 = Scale (cadre, from_ = 0, to = 2, command= choix_w_greens, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0.1,sliderlength=20)
echelle101.set(val_greens)
echelle101.place(anchor="w", x=1440+delta_s+15,y=385)

# Software Blue balance
labelParam102 = Label (cadre, text = "B") # choix balance bleue
labelParam102.place(anchor="w", x=1440+delta_s,y=415)
echelle102 = Scale (cadre, from_ = 0, to = 2, command= choix_w_blues, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0.1,sliderlength=20)
echelle102.set(val_blues)
echelle102.place(anchor="w", x=1440+delta_s+15,y=415)


###########################
# SHARPEN DENOISE WIDGETS #
###########################

# Choix Sharpen 1 & 2
CBSS1 = Checkbutton(cadre,text="Sharpen 1  Val/Sigma", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=1450+delta_s, y=455)
echelle152 = Scale (cadre, from_ = 0, to = 10, command= choix_val_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.2, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle152.set(val_sharpen)
echelle152.place(anchor="w", x=1560+delta_s,y=455)
echelle153 = Scale (cadre, from_ = 1, to = 9, command= choix_val_sigma_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle153.set(val_sigma_sharpen)
echelle153.place(anchor="w", x=1690+delta_s,y=455)

CBSS2 = Checkbutton(cadre,text="Sharpen 2  Val/Sigma", variable=choix_sharpen_soft2,command=commande_sharpen_soft2,onvalue = 1, offvalue = 0)
CBSS2.place(anchor="w",x=1450+delta_s, y=485)
echelle154 = Scale (cadre, from_ = 0, to = 10, command= choix_val_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.2, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle154.set(val_sharpen2)
echelle154.place(anchor="w", x=1560+delta_s,y=485)
echelle155 = Scale (cadre, from_ = 1, to = 9, command= choix_val_sigma_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle155.set(val_sigma_sharpen2)
echelle155.place(anchor="w", x=1690+delta_s,y=485)

# Choix filtre Denoise Paillou image
CBEPF = Checkbutton(cadre,text="NR P1", variable=choix_denoise_Paillou,command=commande_denoise_Paillou,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1450+delta_s, y=525)

# Choix filtre Denoise Paillou 2 image
CBEPF2 = Checkbutton(cadre,text="NR P2", variable=choix_denoise_Paillou2,command=commande_denoise_Paillou2,onvalue = 1, offvalue = 0)
CBEPF2.place(anchor="w",x=1500+delta_s, y=525)

# Choix filtre Denoise KNN
CBEPF = Checkbutton(cadre,text="KNN", variable=choix_denoise_KNN,command=choix_KNN,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1550+delta_s, y=525)
echelle30 = Scale (cadre, from_ = 0.05, to = 1.2, command= choix_val_KNN, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle30.set(val_denoise_KNN)
echelle30.place(anchor="w", x=1600+delta_s,y=525)

# Choix filtre Denoise NLM2
CBDS = Checkbutton(cadre,text="NLM2", variable=choix_NLM2,command=commande_NLM2,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=1690+delta_s, y=525)
echelle4 = Scale (cadre, from_ = 0.1, to = 1.2, command= choix_valeur_denoise, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle4.set(val_denoise)
echelle4.place(anchor="w", x=1740+delta_s,y=525)


# Choix filtre Denoise Paillou 3FNR Front
CB3FNR = Checkbutton(cadre,text="3FNRF", variable=choix_3FNR,command=commande_3FNR,onvalue = 1, offvalue = 0)
CB3FNR.place(anchor="w",x=1450+delta_s, y=565)

# Choix filtre Denoise Paillou 3FNR Back
CB3FNR = Checkbutton(cadre,text="3FNRB", variable=choix_3FNRB,command=commande_3FNRB,onvalue = 1, offvalue = 0)
CB3FNR.place(anchor="w",x=1500+delta_s, y=565)

# Choix filtre Denoise Paillou AADF Front
CBEPFS = Checkbutton(cadre,text="AANRF", variable=choix_AADF,command=commande_AADF,onvalue = 1, offvalue = 0)
CBEPFS.place(anchor="w",x=1550+delta_s, y=565)

# AADF Mode
RBAADP1 = Radiobutton(cadre,text="H", variable=choix_dyn_AADP,command=choix_dyn_high,value=1)
RBAADP1.place(anchor="w",x=1600+delta_s, y=565)
RBAADP2 = Radiobutton(cadre,text="L", variable=choix_dyn_AADP,command=choix_dyn_low,value=2)
RBAADP2.place(anchor="w",x=1630+delta_s, y=565)

# AADF ghost reducer
CBGR = Checkbutton(cadre,text="GR", variable=choix_ghost_reducer,command=commande_ghost_reducer,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1631+30+delta_s, y=565)
echelle130 = Scale (cadre, from_ = 20, to = 70, command= choix_val_ghost_reducer, orient=HORIZONTAL, length = 70, width = 7, resolution = 2, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle130.set(val_ghost_reducer)
echelle130.place(anchor="w", x=1695+delta_s,y=558)

# Choix filtre Denoise Paillou AADF Back
CBEPFSB = Checkbutton(cadre,text="AANRB", variable=choix_AADFB,command=commande_AADFB,onvalue = 1, offvalue = 0)
CBEPFSB.place(anchor="w",x=1775+delta_s, y=565)

# Stabilization
CBSTAB = Checkbutton(cadre,text="STAB", variable=choix_STAB,command=commande_STAB,onvalue = 1, offvalue = 0)
CBSTAB.place(anchor="w",x=1840+delta_s, y=525)

# Image Quality Estimate
CBIMQE = Checkbutton(cadre,text="IQE", variable=choix_IMQE,command=commande_IMQE,onvalue = 1, offvalue = 0)
CBIMQE .place(anchor="w",x=1840+delta_s, y=565)

# BFR Bad Frame Remove
CBBFR = Checkbutton(cadre,text="RmBF", variable=choix_BFR,command=commande_BFR,onvalue = 1, offvalue = 0)
CBBFR.place(anchor="w",x=1450+delta_s, y=600)
echelle300 = Scale (cadre, from_ = 0, to = 100, command= choix_val_BFR, orient=HORIZONTAL, length = 100, width = 7, resolution = 5, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle300.set(val_BFR)
echelle300.place(anchor="w", x=1500+delta_s,y=600)

# Choix 2 frames variation Reduction Filter for turbulence pre treatment
CBRV = Checkbutton(cadre,text="VAR", variable=choix_reduce_variation,command=commande_reduce_variation,onvalue = 1, offvalue = 0)
CBRV.place(anchor="w",x=1615+delta_s, y=600)
RBRVBPF1 = Radiobutton(cadre,text="BF", variable=choix_BFReference,command=command_BFReference,value=1) # Best frame reference
RBRVBPF1.place(anchor="w",x=1655+delta_s, y=600)
RBRVBPF2 = Radiobutton(cadre,text="PF", variable=choix_BFReference,command=command_PFReference,value=2)  # Previous frame reference
RBRVBPF2.place(anchor="w",x=1688+delta_s, y=600)

echelle270 = Scale (cadre, from_ = 0.5, to = 3, command= choix_val_reduce_variation, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle270.set(val_reduce_variation)
echelle270.place(anchor="w", x=1720+delta_s,y=600)

# Choix 2 frames variation Reduction Filter for turbulence post treatment
CBRVPT = Checkbutton(cadre,text="VARPT", variable=choix_reduce_variation_post_treatment,command=commande_reduce_variation_post_treatment,onvalue = 1, offvalue = 0)
CBRVPT.place(anchor="w",x=1840+delta_s, y=600)


#####################
# HISTOGRAM WIDGETS #
#####################

# Choix filtre Gradient Removal
CBGR = Checkbutton(cadre,text="Grad/Vignet", variable=choix_GR,command=commande_GR,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1450+delta_s, y=640)

# Choix du mode gradient ou vignetting
RBGV1 = Radiobutton(cadre,text="Gr", variable=gradient_vignetting,command=mode_gradient,value=1)
RBGV1.place(anchor="w",x=1530+delta_s, y=640)
RBGV2 = Radiobutton(cadre,text="Vig", variable=gradient_vignetting,command=mode_vignetting,value=2)
RBGV2.place(anchor="w",x=1560+delta_s, y=640)

# Choix Parametre Seuil Gradient Removal
echelle60 = Scale (cadre, from_ = 0, to = 100, command= choix_SGR, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=20,sliderlength=20)
echelle60.set(val_SGR)
echelle60.place(anchor="w", x=1600+delta_s,y=640)

# Choix Parametre Atenuation Gradient Removal
labelParam61 = Label (cadre, text = "At")
labelParam61.place(anchor="e", x=1735+delta_s,y=640)
echelle61 = Scale (cadre, from_ = 0, to = 100, command= choix_AGR, orient=HORIZONTAL, length = 80, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle61.set(val_AGR)
echelle61.place(anchor="w", x=1740+delta_s,y=640)

# Choix du mode image en négatif
CBIN = Checkbutton(cadre,text="Img Neg", variable=choix_img_Neg,command=commande_img_Neg,onvalue = 1, offvalue = 0)
CBIN.place(anchor="w",x=1450+delta_s, y=680)

# Histogram equalize
CBHE2 = Checkbutton(cadre,text="Gamma", variable=choix_histogram_equalize2,command=commande_histogram_equalize2,onvalue = 1, offvalue = 0)
CBHE2.place(anchor="w",x=1520+delta_s, y=680)
echelle16 = Scale (cadre, from_ = 0.1, to = 4, command= choix_heq2, orient=HORIZONTAL, length = 240, width = 7, resolution = 0.05, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle16.set(val_heq2)
echelle16.place(anchor="w", x=1580+delta_s,y=680)

# Choix histogramme stretch
CBHS = Checkbutton(cadre,text="Histo Stretch", variable=choix_histogram_stretch,command=commande_histogram_stretch,onvalue = 1, offvalue = 0)
CBHS.place(anchor="w",x=1450+delta_s, y=720)

labelParam5 = Label (cadre, text = "Min") # choix valeur histogramme strech minimum
labelParam5.place(anchor="w", x=1555+delta_s,y=720)
echelle5 = Scale (cadre, from_ = 0, to = 150, command= choix_histo_min, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle5.set(val_histo_min)
echelle5.place(anchor="w", x=1580+delta_s,y=720)

labelParam6 = Label (cadre, text = "Max") # choix valeur histogramme strech maximum
labelParam6.place(anchor="w", x=1700+delta_s,y=720)
echelle6 = Scale (cadre, from_ = 155, to = 255, command= choix_histo_max, orient=HORIZONTAL, length = 100, width = 7, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle6.set(val_histo_max)
echelle6.place(anchor="w", x=1720+delta_s,y=720)

#Choix histogramme Sigmoide
CBHPT = Checkbutton(cadre,text="Histo Sigmoide", variable=choix_histogram_phitheta,command=commande_histogram_phitheta,onvalue = 1, offvalue = 0)
CBHPT.place(anchor="w",x=1450+delta_s, y=760)

labelParam12 = Label (cadre, text = "Pnt") # choix valeur histogramme Signoide param 1
labelParam12.place(anchor="w", x=1555+delta_s,y=760)
echelle12 = Scale (cadre, from_ = 0.5, to = 3, command= choix_phi, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0.5,sliderlength=20)
echelle12.set(val_phi)
echelle12.place(anchor="w", x=1580+delta_s,y=760)

labelParam13 = Label (cadre, text = "Dec") # choix valeur histogramme Signoide param 2
labelParam13.place(anchor="w", x=1700+delta_s,y=760)
echelle13 = Scale (cadre, from_ = 50, to = 200, command= choix_theta, orient=HORIZONTAL, length = 100, width = 7, resolution = 2, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle13.set(val_theta)
echelle13.place(anchor="w", x=1720+delta_s,y=760)

# Choix contrast CLAHE
CBCC = Checkbutton(cadre,text="Contrast", variable=choix_contrast_CLAHE,command=commande_contrast_CLAHE,onvalue = 1, offvalue = 0)
CBCC.place(anchor="w",x=1450+delta_s, y=800)

labelParam9 = Label (cadre, text = "Clip") # choix valeur contrate CLAHE
labelParam9.place(anchor="w", x=1555+delta_s,y=800)
echelle9 = Scale (cadre, from_ = 0.1, to = 4, command= choix_valeur_CLAHE, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle9.set(val_contrast_CLAHE)
echelle9.place(anchor="w", x=1580+delta_s,y=800)
labelParam109 = Label (cadre, text = "Grid") # choix valeur contrate CLAHE
labelParam109.place(anchor="w", x=1700+delta_s,y=800)
echelle109 = Scale (cadre, from_ = 4, to = 24, command= choix_grid_CLAHE, orient=HORIZONTAL, length = 100, width = 7, resolution = 2, label="",showvalue=1,tickinterval=8,sliderlength=20)
echelle109.set(val_grid_CLAHE)
echelle109.place(anchor="w", x=1720+delta_s,y=800)

# Choix Contrast Low Light
CBCLL = Checkbutton(cadre,text="CLL", variable=choix_CLL,command=commande_CLL,onvalue = 1, offvalue = 0)
CBCLL.place(anchor="w",x=1450+delta_s, y=840)

labelParam200 = Label (cadre, text = "µ") # choix Mu CLL
labelParam200.place(anchor="w", x=1500+delta_s,y=840)
echelle200 = Scale (cadre, from_ = 0, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle200.set(val_MuCLL)
echelle200.place(anchor="w", x=1510+delta_s,y=840)

labelParam201 = Label (cadre, text = "Ro") # choix Ro CLL
labelParam201.place(anchor="w", x=1605+delta_s,y=840)
echelle201 = Scale (cadre, from_ = 0.5, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle201.set(val_RoCLL)
echelle201.place(anchor="w", x=1625+delta_s,y=840)

labelParam202 = Label (cadre, text = "amp") # choix Amplification CLL
labelParam202.place(anchor="w", x=1715+delta_s,y=840)
echelle202 = Scale (cadre, from_ = 0.5, to = 5.0, command= choix_Var_CLL, orient=HORIZONTAL, length = 80, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle202.set(val_AmpCLL)
echelle202.place(anchor="w", x=1740+delta_s,y=840)


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

labelInfo10 = Label (cadre, text = text_info10) # label info n°10
labelInfo10.place(anchor="w", x=1450+delta_s,y=1030)

labelParam100 = Label (cadre, text = "Treatment time : ")
labelParam100.place(anchor="w", x=1450+delta_s,y=20)

labelInfo2 = Label (cadre, text = "") # label info n°2
labelInfo2.place(anchor="w", x=1520+delta_s,y=20)


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

# RAZ frame counter
Button10 = Button (cadre,text = "RZ Fr Cnt", command = raz_framecount,padx=10,pady=0)
Button10.place(anchor="w", x=5,y=370)

if flag_camera_ok == False :
    Button12 = Button (cadre,text = "Load Vid", command = load_video,padx=10,pady=0)
    Button12.place(anchor="w", x=5,y=945)
    Button17 = Button (cadre,text = "Load Pic", command = load_image,padx=10,pady=0)
    Button17.place(anchor="w", x=5,y=970)

Button15 = Button (cadre,text = "Dir Vid", command = choose_dir_vid,padx=10,pady=0)
Button15.place(anchor="w", x=5,y=995)

Button16 = Button (cadre,text = "Dir Pic", command = choose_dir_pic,padx=10,pady=0)
Button16.place(anchor="w", x=5,y=1020)


Button (fenetre_principale, text = "Quit", command = quitter,padx=10,pady=5).place(x=1700+delta_s,y=1030, anchor="w")


def init_camera() :
    global camera,flag_colour_camera,format_capture,controls,res_cam_x,res_cam_y, \
           cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,sensor_ratio,sensor_factor, \
           val_blue,val_red,val_USB,val_maxgain,val_resolution, \
           flag_supported_camera,flag_camera_ok,RES_X_BIN1_4_3,RES_Y_BIN1_4_3,RES_X_BIN1_16_9,RES_Y_BIN1_16_9,\
           RES_X_BIN1_1_1,RES_Y_BIN1_1_1,RES_X_BIN2_4_3,RES_Y_BIN2_4_3,RES_X_BIN2_16_9,RES_Y_BIN2_16_9,\
           RES_X_BIN2_1_1,RES_Y_BIN2_1_1,sensor_bits_depth,Camera_Bayer,GPU_BAYER,Video_Bayer,choix_flipV,choix_flipH,type_flip,type_debayer

    try :
        if env_filename_camera:
            asi.init(env_filename_camera)
            time.sleep(0.5)
            flag_ASI_OK = True
        else:
            print('The SDK library for ASI camera is required')
            flag_ASI_OK = False
    except :
        print('The SDK library for ASI camera is required')
        flag_ASI_OK = False
        
    if flag_ASI_OK == True :
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
                val_USB = USBCam
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

            if cameras_found[0] == "ZWO ASI294MC" or cameras_found[0] == "ZWO ASI294MC Pro" :
                res_cam_x = 4144
                res_cam_y = 2822
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(1012*fact_s)
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "4_3"
                sensor_ratio.set(0)

                RES_X_BIN1_4_3 = [4144,3240,2880,2400,1800,1536,1200,960,600]
                RES_Y_BIN1_4_3 = [2822,2430,2160,1800,1350,1152,900,720,450]

                RES_X_BIN1_16_9 = [4144,3240,2880,2400,1800,1536,1216,960,608]
                RES_Y_BIN1_16_9 = [2340,1824,1620,1350,1018,864,684,540,342]

                RES_X_BIN1_1_1 = [2816,2432,2160,1800,1352,1152,904,720,448]
                RES_Y_BIN1_1_1 = [2816,2432,2160,1800,1352,1152,904,720,448]

                RES_X_BIN2_4_3 = [2072,1624,1440,1200,900,768,600]
                RES_Y_BIN2_4_3 = [1410,1216,1080,900,674,576,450]
                
                RES_X_BIN2_16_9 = [2072,1620,1440,1200,900,768,608]
                RES_Y_BIN2_16_9 = [1170,912,810,674,510,432,342]
                
                RES_X_BIN2_1_1 = [1408,1216,1080,900,676,576,452]
                RES_Y_BIN2_1_1 = [1408,1216,1080,900,676,576,452]

                RES_X_BIN1 = RES_X_BIN1_4_3
                RES_Y_BIN1 = RES_Y_BIN1_4_3
                RES_X_BIN2 = RES_X_BIN2_4_3
                RES_Y_BIN2 = RES_Y_BIN2_4_3          

            if cameras_found[0] == "ZWO ASI294MM" or cameras_found[0] == "ZWO ASI294MM Pro" :
                res_cam_x = 8288
                res_cam_y = 5644
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(1012*fact_s)
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "4_3"
                sensor_ratio.set(0)

                RES_X_BIN1_4_3 = [8288,6480,5760,4800,3600,3072,2400,1920,1200]
                RES_Y_BIN1_4_3 = [5644,4860,4320,3600,2700,2304,1800,1440,900]

                RES_X_BIN1_16_9 = [8288,6480,5760,4800,3600,3072,2432,1920,1216]
                RES_Y_BIN1_16_9 = [4680,3648,3240,2700,2036,1728,1368,1080,684]

                RES_X_BIN1_1_1 = [5632,4864,4320,3600,2704,2304,1808,1440,896]
                RES_Y_BIN1_1_1 = [5632,4864,4320,3600,2704,2304,1808,1440,896]

                RES_X_BIN2_4_3 = [4144,3248,2880,2400,1800,1536,1200]
                RES_Y_BIN2_4_3 = [2820,2432,2160,1800,1348,1152,900]
                
                RES_X_BIN2_16_9 = [4144,3240,2880,2400,1800,1536,1216]
                RES_Y_BIN2_16_9 = [2340,1824,1620,1358,1020,864,684]
                
                RES_X_BIN2_1_1 = [2816,2432,2160,1800,1352,1152,904]
                RES_Y_BIN2_1_1 = [2816,2432,2160,1800,1352,1152,904]

                RES_X_BIN1 = RES_X_BIN1_4_3
                RES_Y_BIN1 = RES_Y_BIN1_4_3
                RES_X_BIN2 = RES_X_BIN2_4_3
                RES_Y_BIN2 = RES_Y_BIN2_4_3          
     
            if cameras_found[0] == "ZWO ASI178MC" or cameras_found[0] == "ZWO ASI178MM" or cameras_found[0] == "ZWO ASI178MM Pro" :
                res_cam_x = 3096
                res_cam_y = 2080
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(1012*fact_s)
                val_USB = USBCam
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

            if cameras_found[0] == "ZWO ASI485MC" or cameras_found[0] == "ZWO ASI585MC" or cameras_found[0] == "ZWO ASI585MM" or cameras_found[0] == "ZWO ASI678MC" or cameras_found[0] == "ZWO ASI678MM" or cameras_found[0] == "ZWO ASI715MC":
                res_cam_x = 3840
                res_cam_y = 2160
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(760*fact_s)
                val_USB = USBCam
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
                val_USB = USBCam
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
                val_USB = USBCam
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
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "4_3"
                sensor_ratio.set(0)

                RES_X_BIN1_4_3 = [1304,1280,1024,800,640,320,320,320,320]
                RES_Y_BIN1_4_3 = [976,960,768,600,480,240,240,240,240]

                RES_X_BIN1_16_9 = [1304,1280,1024,800,640,320,320,320,320]
                RES_Y_BIN1_16_9 = [732,720,576,450,360,180,180,180,180]

                RES_X_BIN1_1_1 = [976,960,768,600,480,240,240,240,240]
                RES_Y_BIN1_1_1 = [976,960,768,600,480,240,240,240,240]

                RES_X_BIN2_4_3 = [652,640,512,400,320,160,160]
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
                val_USB = USBCam
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
                RES_Y_BIN2_16_9 = [846,720,540,450,360,288,224]
                
                RES_X_BIN2_1_1 = [1504,1280,960,800,640,512,400]
                RES_Y_BIN2_1_1 = [1504,1280,960,800,640,512,400]

                RES_X_BIN1 = RES_X_BIN1_1_1
                RES_Y_BIN1 = RES_Y_BIN1_1_1
                RES_X_BIN2 = RES_X_BIN2_1_1
                RES_Y_BIN2 = RES_Y_BIN2_1_1

            if cameras_found[0] == "ZWO ASI676MC" :
                res_cam_x = 3552
                res_cam_y = 3552
                cam_displ_x = int(1012*fact_s)
                cam_displ_y = int(1012*fact_s)
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "1_1"
                sensor_ratio.set(2)

                RES_X_BIN1_4_3 = [3552,3008,2560,1920,1600,1280,1024,800,640]
                RES_Y_BIN1_4_3 = [2664,2256,1920,1440,1200,960,768,600,480]

                RES_X_BIN1_16_9 = [3552,3008,2560,1920,1600,1280,1024,800,640]
                RES_Y_BIN1_16_9 = [1998,1692,1440,1080,900,720,576,450,360]

                RES_X_BIN1_1_1 = [3552,3008,2560,1920,1600,1280,1024,800,640]
                RES_Y_BIN1_1_1 = [3552,3008,2560,1920,1600,1280,1024,800,640]

                RES_X_BIN2_4_3 = [1776,1504,1624,960,800,640,512]
                RES_Y_BIN2_4_3 = [1328,1128,960,720,600,480,384]
                
                RES_X_BIN2_16_9 = [1776,1504,1280,960,800,640,512]
                RES_Y_BIN2_16_9 = [992,846,720,540,450,360,288]
                
                RES_X_BIN2_1_1 = [1776,1504,1280,960,800,640,512]
                RES_Y_BIN2_1_1 = [1776,1504,1280,960,800,640,512]

                RES_X_BIN1 = RES_X_BIN1_1_1
                RES_Y_BIN1 = RES_Y_BIN1_1_1
                RES_X_BIN2 = RES_X_BIN2_1_1
                RES_Y_BIN2 = RES_Y_BIN2_1_1

            if cameras_found[0] == "ZWO ASI120MC" or cameras_found[0] == "ZWO ASI120MM" :
                res_cam_x = 1280
                res_cam_y = 960
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(760*fact_s)
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "4_3"
                sensor_ratio.set(0)

                RES_X_BIN1_4_3 = [1280,1024,800,640,320,320,320,320,320]
                RES_Y_BIN1_4_3 = [960,768,600,480,240,240,240,240,240]

                RES_X_BIN1_16_9 = [1280,1024,800,640,320,320,320,320,320]
                RES_Y_BIN1_16_9 = [720,576,450,360,180,180,180,180,180]

                RES_X_BIN1_1_1 = [960,768,600,480,240,240,240,240,240]
                RES_Y_BIN1_1_1 = [960,768,600,480,240,240,240,240,240]

                RES_X_BIN2_4_3 = [640,512,400,320,160,160,160]
                RES_Y_BIN2_4_3 = [480,384,300,240,120,120,120]
                
                RES_X_BIN2_16_9 = [640,512,400,320,160,160,160]
                RES_Y_BIN2_16_9 = [360,288,674,510,90,90,90]
                
                RES_X_BIN2_1_1 = [480,384,300,240,120,120,120]
                RES_Y_BIN2_1_1 = [480,384,300,240,120,120,120]

                RES_X_BIN1 = RES_X_BIN1_4_3
                RES_Y_BIN1 = RES_Y_BIN1_4_3
                RES_X_BIN2 = RES_X_BIN2_4_3
                RES_Y_BIN2 = RES_Y_BIN2_4_3            

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

                if flag_colour_camera == True :
                    if camera_info['BayerPattern'] == 0 :
                        choix_bayer_RGGB()
                        bayer_sensor.set(2)
                    if camera_info['BayerPattern'] == 1 :
                        choix_bayer_BGGR()
                        bayer_sensor.set(3)
                    if camera_info['BayerPattern'] == 2 :
                        choix_bayer_GRBG()
                        bayer_sensor.set(4)
                    if camera_info['BayerPattern'] == 3 :
                        choix_bayer_GBRG()
                        bayer_sensor.set(5)
                else :
                    choix_bayer_RAW()
                    bayer_sensor.set(1)

                print("Camera Bayer Pattern : ",Camera_Bayer)

                sensor_bits_depth = camera_info['BitDepth']
                print("Sensor bits depth = ",sensor_bits_depth)
                
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

                camera.start_video_capture()
            else :
                print("Camera not supported")
                flag_camera_ok = Flase
    else :
        print('No cameras found - Video treatment mode activated')
        flag_camera_ok = False


init_camera()


if flag_camera_ok == True :
    # Mount coordinates
    CBMNT = Checkbutton(cadre,text="AZ / H", variable=choix_mount,command=commande_mount,onvalue = 1, offvalue = 0)
    CBMNT.place(anchor="w",x=5, y=200)
    # Mount coordinates calibration
    Button20 = Button (cadre,text = "Mount Cal", command = Mount_calibration,padx=10,pady=0)
    Button20.place(anchor="w", x=5,y=220)

    # Image format
    labelSR = Label (cadre, text = "Sensor Ratio :")
    labelSR.place (anchor="w",x=5, y=80)
    RBSR1 = Radiobutton(cadre,text="4/3", variable=sensor_ratio,command=choix_sensor_ratio_4_3,value=0)
    RBSR1.place(anchor="w",x=5, y=100)
    RBSR2 = Radiobutton(cadre,text="16/9", variable=sensor_ratio,command=choix_sensor_ratio_16_9,value=1)
    RBSR2.place(anchor="w",x=5, y=120)
    RBSR3 = Radiobutton(cadre,text="1/1", variable=sensor_ratio,command=choix_sensor_ratio_1_1,value=2)
    RBSR3.place(anchor="w",x=5, y=140)
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
        flag_SAT_Image = False
        Sat_Vid_Img.set(0)
        if Video_Test.lower().endswith(('.ser')) :
            flag_SER_file = True
        else :
            flag_SER_file = False

    if Video_Test.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg')) :
        flag_image_mode = True
        flag_SAT_Image = True
        Sat_Vid_Img.set(1)
    flag_image_video_loaded = True
    
    
if Dev_system == "Linux" :
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    with listener:
        fenetre_principale.after(500, refresh)
        fenetre_principale.mainloop()
else :
    fenetre_principale.after(500, refresh)
    fenetre_principale.mainloop()


fenetre_principale.destroy()

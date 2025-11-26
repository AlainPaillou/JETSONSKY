
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
#                             Code optimization and cleaning : Giray Yillikci                              #
#                                                                                                          #
#                  Free of use for personal and non commercial or professional use                         #
#                                                                                                          #
#       This software or part of this software is NOT free of use for commercial or professional use       #
#                                                                                                          #
############################################################################################################



JetsonSky_version = "V53_07RC"



# Supported ZWO cameras :  
#
# ASI120MC, ASI120MM
# ASI178MC, ASI178MM, ASI178MM Pro
# ASI183MC, ASI183MM, ASI183MC Pro, ASI183MM Pro
# ASI224MC
# ASI290MC, ASI290MM, ASI290MM Mini
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
# New 3 frames noise removal front apply                                                                                                        
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
# Color saturation enhancement with 2 pass option                                                                                                       
# New 3 frames noise removal back apply                                                                                                        
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

# Libraries import

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
import platform
import gui_widgets
import image_utils

my_os = sys.platform

if my_os == "win32" :
    import keyboard
    flag_pynput = False
    Dev_system = "Windows"
    nb_ThreadsX = 32
    nb_ThreadsY = 32
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
    nb_ThreadsX = 32
    nb_ThreadsY = 32
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
    if platform.machine() == "aarch64" :
        env_filename_camera = os.path.join(os.getcwd(), 'Lib','libASICamera2.so')
        env_filename_efw = os.path.join(os.getcwd(), 'Lib','libEFWFilter.so')
    elif platform.machine() == "x86_64" :
        env_filename_camera = os.path.join(os.getcwd(), 'x64_Lib','libASICamera2.so.1.27')
        env_filename_efw = os.path.join(os.getcwd(), 'Lib','libEFWFilter.so.1.7')

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
val_USB = USBCam
flag_HQ = 0
frame_limit = 20
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
FlipV = 0
FlipH = 0
ImageNeg = 0
val_red = 63
val_blue = 74
val_FS = 1
compteur_FS = 0
compteur_AANR = 0
compteur_AANRB = 0
compteur_AANR2 = 0
compteur_AANR2B = 0
compteur_RV = 0
val_denoise_KNN = 0.2
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
val_3FNR_Thres = 0.5

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
        thread_1.stop()
        thread_2.stop()
        flag_autorise_acquisition = False
        flag_image_disponible = False
        flag_quitter = True
        flag_acquisition_mount = False
        time.sleep(0.5)
        if Dev_system == "Windows" :
            thread_3.stop()
        time.sleep(0.5)
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
        time.sleep(0.5)
        fenetre_principale.quit()


# Splash screen - Show before main window
try:
    from PIL import Image, ImageTk
    splash_root = Tk()
    splash_root.title("JetsonSky")
    splash_root.overrideredirect(True)
    
    # Get screen dimensions
    screen_width = splash_root.winfo_screenwidth()
    screen_height = splash_root.winfo_screenheight()
    
    # Load and display splash image
    splash_image = Image.open('JetsonSky_Logo.jpg')
    splash_photo = ImageTk.PhotoImage(splash_image)
    splash_label = Label(splash_root, image=splash_photo)
    splash_label.image = splash_photo
    splash_label.pack()
    
    # Center splash screen
    splash_width = splash_image.width
    splash_height = splash_image.height
    x = (screen_width - splash_width) // 2
    y = (screen_height - splash_height) // 2
    splash_root.geometry(f"{splash_width}x{splash_height}+{x}+{y}")
    
    # Add instruction text
    instruction_label = Label(splash_root, text="Press any key to continue...", 
                             font=("Arial", 12), bg="white", fg="black")
    instruction_label.pack(pady=10)
    
    # Wait for key press
    splash_root.bind("<Key>", lambda e: splash_root.destroy())
    splash_root.focus_force()
    splash_root.mainloop()
except:
    pass  # Skip splash if image not found or PIL not available

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

choix_SAT2PASS = IntVar () # réglage saturation couleurs 2 pass
choix_SAT2PASS.set(0)

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

choix_AANR = IntVar()
choix_AANR.set(0) # Initialisation Filtrage AANR Front

choix_AANRB = IntVar()
choix_AANRB.set(0) # Initialisation Filtrage AANR Back

choix_3FNR = IntVar()
choix_3FNR.set(0) # Initialisation Filtrage 3 frames noise removal 1 Front

choix_3FNR2 = IntVar()
choix_3FNR2.set(0) # Initialisation Filtrage 3 frames noise removal 2 Front

choix_3FNRB = IntVar()
choix_3FNRB.set(0) # Initialisation Filtrage 3 frames noise removal 1 Back

choix_3FNR2B = IntVar()
choix_3FNR2B.set(0) # Initialisation Filtrage 3 frames noise removal 2 Back

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

choix_Blur_img_ref = IntVar()
choix_Blur_img_ref.set(0) # Initialisation Imge reference subtract

choix_GBL = IntVar()
choix_GBL.set(0) # Initialisation Gaussian Blur


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
flag_denoise_KNN = False
flag_denoise_Paillou = False
flag_denoise_Paillou2 = False
flag_AANR = False
flag_AANRB = False
flag_GR = False
flag_TIP = False
flag_cross = False
flag_SAT = False
flag_SAT2PASS = False
flag_pause_video = False
flag_quitter = False
flag_autoexposure_exposition = False
flag_autoexposure_gain = False
flag_stacking = "Mean"
flag_HST = 0
flag_TRSF = 0
flag_DEMO = 0
flag_DETECT_STARS = 0
flag_dyn_AANR = 1
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
FNR2_First_Start = False
img1_3FNR2OK  = False
img2_3FNR2OK  = False
img3_3FNR2OK = False
flag_3FNR2 = False
FNR2B_First_Start = False
img1_3FNR2OKB  = False
img2_3FNR2OKB  = False
img3_3FNR2OKB = False
flag_3FNR2B = False
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
flag_blur_image_ref_sub = False
image_captured = False
New_camera_image = False
flag_acquisition_thread_OK = False
flag_GaussBlur = False



# Import CUDA kernels from separate module
from cuda_kernels import *



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
        zoom_up = 'up' # UP ARROW
        zoom_down = 'down' # DOWN ARROW
        zoom_right = 'right' # RIGHT ARROW
        zoom_left = 'left' # LEFT ARROW
        zoom_reset = 'space' # SPACE KEY

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


def opencv_color_debayer(image,debayer_pattern,cuda_flag) :

    if cuda_flag == False :
        debayer_image = cv2.cvtColor(image, debayer_pattern)
    else :
        tmpbase = cv2.cuda_GpuMat()
        tmprsz = cv2.cuda_GpuMat()
        tmpbase.upload(image)
        tmprsz = cv2.cuda.cvtColor(tmpbase, debayer_pattern)
        debayer_image = tmprsz.download()

    return debayer_image


def HDR_compute(mono_colour,image_16b,method,threshold_16b,BIN_mode,Hard_BIN,type_bayer) :

    if (16 - threshold_16b) <= 5 :
        delta_th = (16 - threshold_16b) / 3.0
    else :
        delta_th = 5.0 / 3.0
                            
    thres4 = 2 ** threshold_16b - 1
    thres3 = 2 ** (threshold_16b + delta_th) - 1
    thres2 = 2 ** (threshold_16b + delta_th * 2) - 1
    thres1 = 2 ** (threshold_16b + delta_th * 3) - 1
                                                        
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres1] = thres1
    if BIN_mode == 2 :
        if Hard_BIN == False :
            image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
        else :
            image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
    else :
        image_brute_cam8 = (image_brute_cam16 / thres1 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
    image_brute1 = image_brute_cam_tmp.get()

    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres2] = thres2
    if BIN_mode == 2 :
        if Hard_BIN == False :
            image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
        else :
            image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
    else :
        image_brute_cam8 = (image_brute_cam16 / thres2 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
    image_brute2 = image_brute_cam_tmp.get()

    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres3] = thres3
    if BIN_mode == 2 :
        if Hard_BIN == False :
            image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
        else :
            image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
    else :
        image_brute_cam8 = (image_brute_cam16 / thres3 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
    image_brute3 = image_brute_cam_tmp.get()
    
    image_brute_cam16 = image_16b.copy()
    image_brute_cam16[image_brute_cam16 > thres4] = thres4
    if BIN_mode == 2 :
        if Hard_BIN == False :
            image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0) * 4.0
            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
        else :
            image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
    else :
        image_brute_cam8 = (image_brute_cam16 / thres4 * 255.0)
    image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
    image_brute_cam_tmp = cp.asarray(image_brute_cam8,dtype=cp.uint8)
    image_brute4 = image_brute_cam_tmp.get()

    img_list = [image_brute1,image_brute2,image_brute3,image_brute4]                       
                            
    if method == "Mertens" :
        merge_mertens = cv2.createMergeMertens()
        res_mertens = merge_mertens.process(img_list)
        res_mertens_cp = cp.asarray(res_mertens,dtype=cp.float32)
        image_brute_cp = cp.clip(res_mertens_cp*255, 0, 255).astype('uint8')
    if method == "Median" :
        img_list = cp.asarray(img_list)
        tempo_hdr = cp.median(img_list,axis=0)
        image_brute_cp = cp.asarray(tempo_hdr,dtype=cp.uint8)
    if method == "Mean" :
        img_list = cp.asarray(img_list)
        tempo_hdr = cp.mean(img_list,axis=0)
        image_brute_cp = cp.asarray(tempo_hdr,dtype=cp.uint8)

    HDR_image = image_brute_cp.get()
    HDR_image = opencv_color_debayer(HDR_image,type_bayer,flag_OpenCvCuda)
    if mono_colour == "Mono" :
        HDR_image = cv2.cvtColor(HDR_image, cv2.COLOR_BGR2GRAY)
            
    return HDR_image


def start_acquisition() :
    global flag_camera_ok,flag_autorise_acquisition,thread_1,flag_stop_acquisition,flag_acquisition_thread_OK
    
    if flag_camera_ok == True :
        flag_acquisition_thread_OK = True
        flag_autorise_acquisition = True
        flag_stop_acquisition = False
        thread_1 = acquisition("1")
        thread_1.start()

def stop_acquisition() :
    global flag_camera_ok,flag_autorise_acquisition,thread_1,flag_stop_acquisition,flag_acquisition_thread_OK
    
    if flag_camera_ok == True :
        flag_acquisition_thread_OK = False
        flag_autorise_acquisition = False
        flag_stop_acquisition = False
        time.sleep(1)
             
class acquisition(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global flag_16b,cupy_context,flag_autorise_acquisition,image_brute_camera,timeoutexp,image_captured,New_camera_image,flag_acquisition_thread_OK,nb_erreur,my_os

        with cupy_context :
            while flag_acquisition_thread_OK == True :
                if flag_autorise_acquisition == True :
                    try :
                        if my_os == "win32" :
                            if flag_16b == True :
                                image_camera_tmp=camera.capture_video_frame_RAW16_CUPY(filename=None,timeout=timeoutexp)
                            else :    
                                image_camera_tmp = camera.capture_video_frame_RAW8_CUPY(filename=None,timeout=timeoutexp)
                        else :
                            if flag_16b == True :
                                image_camera_tmp=camera.capture_video_frame_RAW16_NUMPY(filename=None,timeout=timeoutexp)
                            else :    
                                image_camera_tmp = camera.capture_video_frame_RAW8_NUMPY(filename=None,timeout=timeoutexp)                            
                        image_captured = False
                        New_camera_image = True
                        image_brute_camera = image_camera_tmp.copy()                      
                        image_captured = True
                    except Exception as error :
                        image_captured = False
                        New_camera_image = False
                        camera.stop_video_capture()
                        camera.stop_exposure()
                        time.sleep(0.5)
                        nb_erreur += 1
                        print("An error occurred : ", error)
                        print("Capture error : ",nb_erreur)
                        camera.start_video_capture()
                        time.sleep(0.5)
                else :
                    time.sleep(0.5)

    def stop(self) :
        global flag_start_acquisition,flag_autorise_acquisition,flag_acquisition_thread_OK
        
        flag_acquisition_thread_OK = False
        time.sleep(0.5)
        flag_start_acquisition = False
        flag_autorise_acquisition = False
        

        
def camera_acquisition() :
    global camera,nb_erreur,flag_autorise_acquisition,key_pressed,flag_stop_acquisition,GPU_BAYER,\
           flag_noir_blanc,cupy_context,delta_RX,delta_RY,delta_BX,delta_BY,mode_BIN,flag_HB,\
           timeoutexp,flag_filtrage_ON,flag_IsColor,labelInfo10,image_captured,image_brute_camera,\
           threshold_16bits,choix_hard_bin,flag_capture_image_reference,Image_Reference,flag_image_reference_OK,\
           New_camera_image,my_os
        
    with cupy_context :
        if image_captured == True :
            if New_camera_image == True :
                New_camera_image = False
                ret = True
                if my_os == "linux" :
                    if flag_16b == True :
                        image_brute_cam = cp.asarray(image_brute_camera,dtype=cp.uint16)
                    else :
                        image_brute_cam = cp.asarray(image_brute_camera,dtype=cp.uint8)
                else :
                    image_brute_cam = image_brute_camera
                if flag_16b == True :
                    image_brute_cam16 = image_brute_cam.copy()
                    image_brute_cam16[image_brute_cam16 > threshold_16bits] = threshold_16bits
                    if mode_BIN == 2 :
                        if flag_HB == False :
                            image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0) * 4.0
                            image_brute_cam8 = cp.clip(image_brute_cam8,0,255)
                        else :
                            image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0)
                    else :
                        image_brute_cam8 = (image_brute_cam16 / threshold_16bits * 255.0)                                   
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
                    res_gg = 0
                    res_rr = 0
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
                            img = res_rr.copy()
                        nb_blocksX = ((width // 2) // nb_ThreadsX) + 1
                        nb_blocksY = ((height // 2) // nb_ThreadsY) + 1
                        Image_Debayer_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_rr, res_gg, res_bb, img, np.intc(width), np.intc(height), np.intc(GPU_BAYER)))
                        flag_newdelta = False
                        if key_pressed != "" :
                            if key_pressed == "RED_UP" :
                                delta_RY = delta_RY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_DOWN" :
                                delta_RY = delta_RY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RIGHT" :
                                delta_RX = delta_RX - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_LEFT" :
                                delta_RX = delta_RX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RESET" :
                                delta_RX = 0
                                delta_RY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_UP" :
                                delta_BY = delta_BY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_DOWN" :
                                delta_BY = delta_BY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RIGHT" :
                                delta_BX = delta_BX - 1
                                key_pressed = ""
                                flag_newdelta = True
                                flag_newdelta = True
                            if key_pressed == "BLUE_LEFT" :
                                delta_BX = delta_BX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RESET" :
                                delta_BX = 0
                                delta_BY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if flag_newdelta == True :
                                texte = "Shift Red : " + str(delta_RX) + " : "+str(delta_RY) + "    Shift Blue : " + str(delta_BX) + " : " + str(delta_BY) + "             "
                                labelInfo10.config(text = texte)
                        if delta_RX !=0 or delta_RY !=0 or delta_BX !=0 or delta_BY != 0 :
                            img_r = res_rr.copy()
                            img_g = res_gg.copy()
                            img_b = res_bb.copy()
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
                    res_rr = 0
                    res_gg = 0 
                ret = True
            else :
                ret = False
                res_rr = 0
                res_gg = 0
                res_bb = 0
        else :
            ret = False
            res_rr = 0
            res_gg = 0
            res_bb = 0
    return ret,res_rr,res_gg,res_bb
     

def refresh() :
    global video,flag_cap_video,camera,traitement,cadre_image,image_brute,flag_image_disponible,flag_quitter,timeoutexp,curFPS,fpsQueue,image_brut_read,delta_zx,delta_zy,flag_new_stab_window,\
           flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2,image_traitee,val_SAT,exposition,total_start,total_stop,flag_image_video_loaded,flag_BFReference,\
           val_gain, echelle2,val_exposition,echelle1,imggrey1,imggrey2,flag_DETECT_STARS,flag_TRKSAT,flag_REMSAT,flag_CONST,labelInfo2,Date_hour_image,val_nb_capt_video,flag_image_mode,\
           calque_stars,calque_satellites,calque_TIP,flag_nouvelle_resolution,nb_sat,sat_x,sat_y,sat_s,res_bb1,res_gg1,res_rr1,cupy_context,flag_TRIGGER,flag_filtrage_ON,\
           nb_erreur,image_brute_grey,flag_sat_detected,start_time_video,stop_time_video,time_exec_test,total_time,font,flag_colour_camera,flag_iscolor,\
           res_cam_x,res_cam_y,flag_premier_demarrage,Video_Test,TTQueue,curTT,max_sat,video_frame_number,video_frame_position,image_reconstructed,stars_x,strs_y,stars_s,nb_stars,\
           quality,max_quality,quality_pos,tmp_qual,flag_IsColor,mean_quality,SFN,min_qual,max_qual,val_BFR,SFN,frame_number,delta_tx,delta_ty,labelInfo10,flag_GO,BFREF_image,flag_BFREF_image,\
           delta_RX,delta_RY,delta_BX,delta_BY,track,track_crater_history,track_sat_history,track_sat,track_crater,key_pressed,DSW,echelle210,frame_position,h,flag_new_frame_position,\
           curFPS,SER_depth,flag_SER_file,res_cam_x_base,res_cam_y_base,flag_capture_image_reference,Image_Reference,flag_image_reference_OK,previous_frame_number,flag_blur_image_ref_sub

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
                            img_list = [image_brute1,image_brute2,image_brute3,image_brute4]                       
                            Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                            merge_mertens = cv2.createMergeMertens()
                            res_mertens = merge_mertens.process(img_list)
                            image_brute = np.clip(res_mertens*255, 0, 255).astype('uint8')
                            image_brute = opencv_color_debayer(image_brute,type_debayer,flag_OpenCvCuda)
                        else :
                            if type_debayer != 0 and flag_IsColor == True:
                                mono_colour = "Colour"
                            else :
                                mono_colour = "Mono"
                            image_camera_base = camera.capture_video_frame_RAW16_CUPY(filename=None,timeout=timeoutexp)
                            if mode_HDR == "Mean" :
                                res_r = cp.asarray(image_camera_base,dtype=cp.uint8)
                                height,width = image_camera_base.shape
                                nb_blocksX = (width // nb_ThreadsX) + 1
                                nb_blocksY = (height // nb_ThreadsY) + 1
                                type_method = 0
                                if flag_HB == True :
                                    Hard_BIN = 1
                                else :
                                    Hard_BIN = 0
                                HDR_compute_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(res_r, image_camera_base,  np.intc(width), np.intc(height), np.float32(TH_16B), \
                                            np.intc(type_method), np.intc(mode_BIN), np.intc(Hard_BIN)))
                                image_brute = res_r.get()
                                image_brute = cv2.cvtColor(image_brute,type_debayer)
                            else :
                                image_brute = HDR_compute(mono_colour,image_camera_base,mode_HDR,TH_16B,mode_BIN,flag_HB,type_debayer)
                            if flag_noir_blanc == 1 and flag_colour_camera == True :
                                if image_brute.ndim == 3 :
                                    image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                                    Dim = 1
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
                        ret_img = True
                    except Exception as error :
                        nb_erreur += 1
                        print("An error occurred : ", error)
                        print("Capture error : ",nb_erreur)
                        time.sleep(0.01)
                        ret_img = False
                else :
                    ret_img,res_rr1,res_gg1,res_bb1 = camera_acquisition()
                Date_hour_image = datetime.now().strftime('Date %Y/%m/%d  Time %H:%M:%S.%f')[:-3]
                if ret_img == True :
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
                                    image_traitee = cv2.applyColorMap(tmp_grey, cv2.COLORMAP_JET)
                                else:
                                    image_traitee = cv2.applyColorMap(image_traitee, cv2.COLORMAP_JET)
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
                    if len(fpsQueue) > 10:
                        fpsQueue.pop(0)
                    curFPS = (sum(fpsQueue)/len(fpsQueue))
                    texte_TIP1 = Date_hour_image + "  Frame nbr : " + str(frame_number) + "  FPS : " + str(int(curFPS*10)/10)
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
                        if (frame_number % frame_skip) != 1 :
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
                    pass
            if flag_quitter == False:
                if flag_HDR == False :
                    fenetre_principale.after(4, refresh)
                else :
                    fenetre_principale.after(6, refresh)
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
                else :
                    video_frame_position = video_frame_position + 1
                    image_brute = video.readFrameAtPos(video_frame_position)
                    ret = True
                    if SER_depth == 16 :
                        if flag_HDR == True and flag_filtrage_ON == True and type_debayer > 0 : 
                            mono_colour = "Colour"
                            image_brute = HDR_compute(mono_colour,image_brute,mode_HDR,TH_16B,mode_BIN,flag_HB,type_debayer)

                            if flag_noir_blanc == 1 :
                                if image_brute.ndim == 3 :
                                    image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                                    flag_IsColor = False
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
                    else :
                        image_brute = np.asarray(image_brute,dtype=np.uint8)
                        if flag_capture_image_reference == True :
                            Image_Reference = np.asarray(image_brute,dtype=np.uint8)
                            flag_capture_image_reference = False
                            flag_image_reference_OK = True
                if (video_frame_position % 5) == 0 :
                    echelle210.set(video_frame_position)
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
                    if flag_noir_blanc == 1 and image_brute.ndim == 3 and GPU_BAYER == 0 :
                        image_brute = cv2.cvtColor(image_brute, cv2.COLOR_BGR2GRAY)
                    if type_flip == "vertical" or type_flip == "both" :
                        image_brute = cv2.flip(image_brute,0)
                    if type_flip == "horizontal" or type_flip == "both" :
                        image_brute = cv2.flip(image_brute,1)
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
                        flag_newdelta = False
                        if key_pressed != "" :
                            if key_pressed == "RED_UP" :
                                delta_RY = delta_RY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_DOWN" :
                                delta_RY = delta_RY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RIGHT" :
                                delta_RX = delta_RX - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_LEFT" :
                                delta_RX = delta_RX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RESET" :
                                delta_RX = 0
                                delta_RY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_UP" :
                                delta_BY = delta_BY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_DOWN" :
                                delta_BY = delta_BY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RIGHT" :
                                delta_BX = delta_BX - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_LEFT" :
                                delta_BX = delta_BX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RESET" :
                                delta_BX = 0
                                delta_BY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if flag_newdelta == True :
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
                            flag_newdelta = False
                            if key_pressed == "RED_UP" :
                                delta_RY = delta_RY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_DOWN" :
                                delta_RY = delta_RY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RIGHT" :
                                delta_RX = delta_RX - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_LEFT" :
                                delta_RX = delta_RX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "RED_RESET" :
                                delta_RX = 0
                                delta_RY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_UP" :
                                delta_BY = delta_BY + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_DOWN" :
                                delta_BY = delta_BY - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RIGHT" :
                                delta_BX = delta_BX - 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_LEFT" :
                                delta_BX = delta_BX + 1
                                key_pressed = ""
                                flag_newdelta = True
                            if key_pressed == "BLUE_RESET" :
                                delta_BX = 0
                                delta_BY = 0
                                key_pressed = ""
                                flag_newdelta = True
                            if flag_newdelta == True :
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
                    if flag_capture_image_reference == True :
                        Image_Reference = np.asarray(image_brute,dtype=np.uint8)
                        flag_capture_image_reference = False
                        flag_image_reference_OK = True
                    if flag_image_ref_sub == True and flag_image_reference_OK == True :
                        image_2_subtract = np.asarray(image_brute,dtype=np.uint8)
                        image_brute = cv2.subtract(image_2_subtract,Image_Reference)
                        if flag_blur_image_ref_sub == True :
                            image_brute = cv2.GaussianBlur(image_brute,(7,7),0)
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
                        image_traitee = cv2.applyColorMap(tmp_grey, cv2.COLORMAP_JET)
                    else:
                        image_traitee = cv2.applyColorMap(image_traitee, cv2.COLORMAP_JET)
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
                fenetre_principale.after(10, refresh)
            else :
                fenetre_principale.after(4, refresh)


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
    
    
# Image conversion utilities moved to image_utils.py
cupy_RGBImage_2_cupy_separateRGB = image_utils.cupy_RGBImage_2_cupy_separateRGB
numpy_RGBImage_2_numpy_separateRGB = image_utils.numpy_RGBImage_2_numpy_separateRGB
numpy_RGBImage_2_cupy_separateRGB = image_utils.numpy_RGBImage_2_cupy_separateRGB
cupy_RGBImage_2_numpy_separateRGB = image_utils.cupy_RGBImage_2_numpy_separateRGB
cupy_separateRGB_2_numpy_RGBimage = image_utils.cupy_separateRGB_2_numpy_RGBimage
cupy_separateRGB_2_cupy_RGBimage = image_utils.cupy_separateRGB_2_cupy_RGBimage
numpy_separateRGB_2_numpy_RGBimage = image_utils.numpy_separateRGB_2_numpy_RGBimage
gaussianblur_mono = image_utils.gaussianblur_mono
gaussianblur_colour = image_utils.gaussianblur_colour
image_negative_colour = image_utils.image_negative_colour
Image_Quality = image_utils.Image_Quality


def Template_tracking(image,dim) :
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
            try :
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
    global compteur_FS,Im1OK,Im2OK,Im3OK,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm, g1_sm, g2_sm, g3_sm, g4_sm, g5_sm, r1_sm, r2_sm, r3_sm, r4_sm, r5_sm,\
           Im4OK,Im5OK,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_brute_grey,cupy_context,BFREF_image,flag_BFREF_image,flag_SAT2PASS,\
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,flag_IsColor,flag_BFReference,BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_AmpSoft,val_ampl,grad_vignet,compteur_AANR,compteur_AANRB,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\
           Im1fsdnOK,Im2fsdnOK,Im1fsdnOKB,Im2fsdnOKB,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,val_reds,val_greens,val_blues,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\
           flag_dyn_AANR,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_g2,res_r2,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\
           imgb1,imgg1,imgr1,imgb2,imgg2,imgr2,imgb3,imgg3,imgr3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,res_b2B,res_g2B,res_r2B,\
           imgb1B,imgg1B,imgr1B,imgb2B,imgg2B,imgr2B,imgb3B,imgg3B,imgr3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,flag_AANRB,val_3FNR_Thres,\
           compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2,compteur_3FNR2B,img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2B_First_Start,flag_3FNR2B,\
           imgb21,imgg21,imgr21,imgb22,imgg22,imgr22,imgb23,imgg23,imgr23,imgb21B,imgg21B,imgr21B,imgb22B,imgg22B,imgr22B,imgb23B,imgg23B,imgr23B

    start_time_test = cv2.getTickCount()

    with cupy_context :     
        if flag_filtrage_ON == True :
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

                # Gaussian Blur
                if flag_GaussBlur == True :
                    res_b1,res_g1,res_r1 = gaussianblur_colour(res_b1,res_g1,res_r1,3)

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

                if val_FS > 1 and flag_image_mode == False :
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
                if flag_reduce_variation == True and flag_BFReference == "PreviousFrame" and flag_image_mode == False :
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
                if flag_reduce_variation == True and flag_BFReference == "BestFrame" and flag_BFREF_image == True and flag_image_mode == False :

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

                # 3 Frames Noise Reduction 1 Filter Front
                if flag_3FNR == True and flag_image_mode == False :
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
                                             imgr3, imgg3, imgb3, np.int_(width), np.int_(height),np.float32(val_3FNR_Thres)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb1 = imgb2.copy()
                        imgg1 = imgg2.copy()
                        imgr1 = imgr2.copy()
                        imgr2 = r_gpu.copy()
                        imgg2 = g_gpu.copy()
                        imgb2 = b_gpu.copy()
                        

                # 3 Frames Noise Reduction 2 Filter Front
                if flag_3FNR2 == True and flag_image_mode == False :
                    if compteur_3FNR2 < 4 and FNR2_First_Start == True:
                        compteur_3FNR2 = compteur_3FNR2 + 1
                        if compteur_3FNR2 == 1 :
                            imgb21 = res_b1.copy()
                            imgg21 = res_g1.copy()
                            imgr21 = res_r1.copy()
                            img1_3FNR2OK = True
                        if compteur_3FNR2 == 2 :
                            imgb22 = res_b1.copy()
                            imgg22 = res_g1.copy()
                            imgr22 = res_r1.copy()
                            img2_3FNR2OK = True
                        if compteur_3FNR2 == 3 :
                            imgb23 = res_b1.copy()
                            imgg23 = res_g1.copy()
                            imgr23 = res_r1.copy()
                            img3_3FNR2OK = True
                            FNR2_First_Start = True
                    if img3_3FNR2OK == True :
                        if FNR2_First_Start == False :
                            imgb23 = res_b1.copy()
                            imgg23 = res_g1.copy()
                            imgr23 = res_r1.copy()
                        
                        FNR2_First_Start = False
                        b_gpu = res_b1
                        g_gpu = res_g1
                        r_gpu = res_r1
                
                        FNR2_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr21, imgg21, imgb21, imgr22, imgg22, imgb22,\
                                             imgr23, imgg23, imgb23, np.int_(width), np.int_(height)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb21 = imgb22.copy()
                        imgg21 = imgg22.copy()
                        imgr21 = imgr22.copy()
                        imgr22 = r_gpu.copy()
                        imgg22 = g_gpu.copy()
                        imgb22 = b_gpu.copy()
                                                                           
                # Adaptative Absorber Denoise Filter Front
                if flag_AANR == True and flag_image_mode == False :
                    if compteur_AANR < 3 :
                        compteur_AANR = compteur_AANR + 1
                        if compteur_AANR == 1 :
                            res_b2 = res_b1.copy()
                            res_g2 = res_g1.copy()
                            res_r2 = res_r1.copy()
                            Im1fsdnOK = True
                        if compteur_AANR == 2 :
                            Im2fsdnOK = True

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1
     
                    if Im2fsdnOK == True :
                        
                        adaptative_absorber_denoise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AANR),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))


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
                        coul_r,coul_g,coul_b = gaussianblur_colour(r_gpu,g_gpu,b_gpu,5)
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
                        if flag_SAT2PASS == True :
                            r_gpu = res_r1.copy()
                            g_gpu = res_g1.copy()
                            b_gpu = res_b1.copy()
                            init_r = res_r1.copy()
                            init_g = res_g1.copy()
                            init_b = res_b1.copy()
                            coul_r,coul_g,coul_b = gaussianblur_colour(r_gpu,g_gpu,b_gpu,11)
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
                        if flag_SAT2PASS == True :
                            r_gpu = res_r1.copy()
                            g_gpu = res_g1.copy()
                            b_gpu = res_b1.copy()
                            coul_gauss_r = res_r1.copy()
                            coul_gauss_g = res_g1.copy()
                            coul_gauss_b = res_b1.copy()
                            coul_gauss_r,coul_gauss_g,coul_gauss_b = gaussianblur_colour(coul_gauss_r,coul_gauss_g,coul_gauss_b,11)
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
                if flag_3FNRB == True and flag_image_mode == False :
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
                                             imgr3B, imgg3B, imgb3B, np.int_(width), np.int_(height),np.float32(val_3FNR_Thres)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb1B = imgb2B.copy()
                        imgg1B = imgg2B.copy()
                        imgr1B = imgr2B.copy()
                        imgr2B = r_gpu.copy()
                        imgg2B = g_gpu.copy()
                        imgb2B = b_gpu.copy()


                # 3 Frames Noise Reduction 2 Filter Back
                if flag_3FNR2B == True and flag_image_mode == False :
                    if compteur_3FNR2B < 4 and FNR2B_First_Start == True:
                        compteur_3FNR2B = compteur_3FNR2B + 1
                        if compteur_3FNR2B == 1 :
                            imgb21B = res_b1.copy()
                            imgg21B = res_g1.copy()
                            imgr21B = res_r1.copy()
                            img1_3FNR2OKB = True
                        if compteur_3FNR2B == 2 :
                            imgb22B = res_b1.copy()
                            imgg22B = res_g1.copy()
                            imgr22B = res_r1.copy()
                            img2_3FNR2OKB = True
                        if compteur_3FNR2B == 3 :
                            imgb23B = res_b1.copy()
                            imgg23B = res_g1.copy()
                            imgr23B = res_r1.copy()
                            img3_3FNR2OKB = True
                            FNR2B_First_Start = True
                    if img3_3FNR2OKB == True :
                        if FNR2B_First_Start == False :
                            imgb23B = res_b1.copy()
                            imgg23B = res_g1.copy()
                            imgr23B = res_r1.copy()
                        
                        FNR2B_First_Start = False
                        b_gpu = res_b1
                        g_gpu = res_g1
                        r_gpu = res_r1
                
                        FNR2_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr21B, imgg21B, imgb21B, imgr22B, imgg22B, imgb22B,\
                                             imgr23B, imgg23B, imgb23B, np.int_(width), np.int_(height)))
                
                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                        imgb21B = imgb22B.copy()
                        imgg21B = imgg22B.copy()
                        imgr21B = imgr22B.copy()
                        imgr22B = r_gpu.copy()
                        imgg22B = g_gpu.copy()
                        imgb22B = b_gpu.copy()


                # Adaptative Absorber Denoise Filter Back
                if flag_AANRB == True and flag_image_mode == False :
                    if compteur_AANRB < 3 :
                        compteur_AANRB = compteur_AANRB + 1
                        if compteur_AANRB == 1 :
                            res_b2B = res_b1.copy()
                            res_g2B = res_g1.copy()
                            res_r2B = res_r1.copy()
                            Im1fsdnOKB = True
                        if compteur_AANRB == 2 :
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
                    res_b1 = cp.minimum(res_s1_b1,res_s2_b1)
                    res_g1 = cp.minimum(res_s1_g1,res_s2_g1)
                    res_r1 = cp.minimum(res_s1_r1,res_s2_r1)

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


def application_filtrage_mono(res_b1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm,flag_IsColor,flag_BFReference,\
           Im4OK,Im5OK,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_brute_grey,cupy_context, BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,BFREF_image,flag_BFREF_image,\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_AmpSoft,val_ampl,grad_vignet,compteur_AANR,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\
           Im1fsdnOK,Im2fsdnOK,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\
           flag_dyn_AANR,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_b2B,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\
           imgb1,imgb2,imgb3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,Im1fsdnOKB,Im2fsdnOKB, \
           imgb1B,imgb2B,imgb3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,compteur_AANRB,val_3FNR_Thres,\
           compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2,compteur_3FNR2B,img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2B_First_Start,flag_3FNR2B,\
           imgb21,imgb22,imgb23,imgb21B,imgb22B,imgb23B


    start_time_test = cv2.getTickCount()

    with cupy_context :     
        if flag_filtrage_ON == True :
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
                
                # Gaussian Blur
                if flag_GaussBlur == True :
                    res_b1 = gaussianblur_mono(res_b1,3)

                # Image Negative
                if ImageNeg == 1 :
                    res_b1 = cp.invert(res_b1,dtype=cp.uint8)
                    for x in range(1,256) :
                        trsf_r[x] = (int)(256-trsf_r[x])
                    trsf_r = np.clip(trsf_r,0,255)

                r_gpu = res_b1.copy()

                # Estimate luminance if mono sensor was used - Do not use with a mono sensor
                if flag_NB_estime == 1 :
                    
                    grey_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1,\
                        np.int_(width), np.int_(height)))

                    res_b1 = r_gpu.copy()
     
                if val_FS > 1 and flag_image_mode == False :
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
                if flag_reduce_variation == True and flag_BFReference == "PreviousFrame" and flag_image_mode == False :
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
                if flag_reduce_variation == True and flag_BFReference == "BestFrame" and flag_BFREF_image == True and flag_image_mode == False :

                    res_b2 = cp.asarray(BFREF_image,dtype=cp.uint8)
                    variation = int(255/100*val_reduce_variation)
                    b_gpu = res_b1
                        
                    reduce_variation_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, res_b1, res_b2, np.int_(width), np.int_(height),np.int_(variation)))

                    res_b1 = b_gpu
     
                # 3 Frames Noise reduction Front
                if flag_3FNR == True and flag_image_mode == False :
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
                
                        FNR_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb1, imgb2, imgb3, np.int_(width), np.int_(height),np.float32(val_3FNR_Thres)))
                                              
                        res_b1 = b_gpu
                        imgb1 = imgb2.copy()
                        imgb2 = b_gpu.copy()


                # 3 Frames Noise Reduction 2 Filter Front
                if flag_3FNR2 == True and flag_image_mode == False :
                    if compteur_3FNR2 < 4 and FNR2_First_Start == True:
                        compteur_3FNR2 = compteur_3FNR2 + 1
                        if compteur_3FNR2 == 1 :
                            imgb21 = res_b1.copy()
                            img1_3FNR2OK = True
                        if compteur_3FNR2 == 2 :
                            imgb22 = res_b1.copy()
                            img2_3FNR2OK = True
                        if compteur_3FNR2 == 3 :
                            imgb23 = res_b1.copy()
                            img3_3FNR2OK = True
                            FNR2_First_Start = True
                    if img3_3FNR2OK == True :
                        if FNR2_First_Start == False :
                            imgb23 = res_b1.copy()
                        
                        FNR2_First_Start = False
                        b_gpu = res_b1
                
                        FNR2_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb21, imgb22, imgb23, np.int_(width), np.int_(height)))
                
                        res_b1 = b_gpu

                        imgb21 = imgb22.copy()
                        imgb22 = b_gpu.copy()


                # Adaptative Absorber Noise Reduction
                if flag_AANR == True and flag_image_mode == False :
                    if compteur_AANR < 3 :
                        compteur_AANR = compteur_AANR + 1
                        if compteur_AANR == 1 :
                            res_b2 = res_b1.copy()
                            Im1fsdnOK = True
                        if compteur_AANR == 2 :
                            Im2fsdnOK = True

                    b_gpu = res_b1
                
                    if Im2fsdnOK == True :
                        nb_images = 2
                        divise = 2.0

                        adaptative_absorber_denoise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, res_b2,\
                                             np.int_(width), np.int_(height),np.intc(flag_dyn_AANR),np.intc(flag_ghost_reducer),np.intc(val_ghost_reducer)))

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
                if flag_3FNRB == True and flag_image_mode == False :
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
                
                        FNR_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, imgb1B, imgb2B, imgb3B, np.int_(width), np.int_(height),np.float32(val_3FNR_Thres)))
                                              
                        res_b1 = b_gpu.copy()
                        imgb1B = imgb2B.copy()
                        imgb2B = b_gpu.copy()

                # Adaptative Absorber Denoise Filter Back
                if flag_AANRB == True and flag_image_mode == False :
                    if compteur_AANRB < 3 :
                        compteur_AANRB = compteur_AANRB + 1
                        if compteur_AANRB == 1 :
                            res_b2B = res_b1.copy()
                            Im1fsdnOKB = True
                        if compteur_AANRB == 2 :
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
           flag_camera_ok,previous_frame_number,frame_number
           
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
                            if frame_number > previous_frame_number :
                                if image_sauve.ndim == 3 :
                                    video.write(cv2.cvtColor(image_sauve, cv2.COLOR_BGR2RGB))
                                else :
                                    video.write(image_sauve)
                                if (nb_cap_video % 5) == 0 :
                                    time_rec = int(nb_cap_video/2.5)/10
                                    labelInfo1.config(text = " frame : " + str (nb_cap_video) + "    " + str (time_rec) + " sec                   ")
                                    previous_frame_number = frame_number
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
    global nb_cap_video,flag_cap_video,start_video,val_nb_capt_video,timer1,frame_number,flag_camera_ok,flag_SER_file,video,video_frame_position
    
    flag_cap_video = True
    nb_cap_video =1
    if val_nb_capt_video == 0 :
        val_nb_capt_video = 10000
    start_video = datetime.now()
    if flag_camera_ok == True :
        timer1 = time.time()
        frame_number = 0
    if flag_camera_ok == False :
        if flag_SER_file == False :
            video_frame_position = 1
            video.set(cv2.CAP_PROP_POS_FRAMES,frame_position)
        else :
            video_frame_position = 1
            video.setCurrentPosition(video_frame_position)
    
 
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


def HDR_Median() :
    global mode_HDR

    mode_HDR = "Median"


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
    global val_FS,camera,val_resolution,echelle3,flag_stop_acquisition,mode_BIN,flag_nouvelle_resolution,flag_TIP,choix_TIP,flag_cap_video,flag_image_disponible

    if flag_camera_ok == True :
        if flag_cap_video == False :
            flag_image_disponible = False
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
           flag_image_disponible,flag_stop_acquisition,flag_autorise_acquisition,flag_HDR,\
           flag_nouvelle_resolution,tnr,inSize,backend,choix_TIP,flag_TIP,flag_cap_video
    
    if flag_camera_ok == True :
        if flag_cap_video == False :
            flag_autorise_acquisition = False
            reset_FS()
            reset_general_FS()
            time.sleep(0.5)
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
            if flag_HDR == False :
                flag_autorise_acquisition = True
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


def commande_SAT2PASS() :
    global flag_SAT2PASS
    
    if choix_SAT2PASS.get() == 0 :
        flag_SAT2PASS = False
    else :
        flag_SAT2PASS = True


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

      
def commande_AANR() :
    global flag_AANR,compteur_AANR,Im1fsdnOK,Im2fsdnOK
    
    if choix_AANR.get() == 0 :
        flag_AANR = False
    else :
        flag_AANR = True
        compteur_AANR = 0
        Im1fsdnOK = False
        Im2fsdnOK = False


def commande_AANRB() :
    global flag_AANRB,compteur_AANRB,Im1fsdnOKB,Im2fsdnOKB
    
    if choix_AANRB.get() == 0 :
        flag_AANRB = False
    else :
        flag_AANRB = True
        compteur_AANRB = 0
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

def commande_3FNR2() :
    global flag_3FNR2,compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start
    
    if choix_3FNR2.get() == 0 :
        flag_3FNR2 = False
    else :
        flag_3FNR2 = True
        compteur_3FNR2 = 0
        img1_3FNR2OK = False
        img2_3FNR2OK = False
        img3_3FNR2OK = False
        FNR2_First_Start = True


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

def commande_3FNR2B() :
    global flag_3FNR2B,compteur_3FNR2B,img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2B_First_Start
    
    if choix_3FNR2B.get() == 0 :
        flag_3FNR2B = False
    else :
        flag_3FNR2B = True
        compteur_3FNR2B = 0
        img1_3FNR2OKB = False
        img2_3FNR2OKB = False
        img3_3FNR2OKB = False
        FNR2B_First_Start = True


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
    global flag_HDR,flag_autorise_acquisition
    
    if choix_HDR.get() == 0 :
        flag_HDR = False
        flag_autorise_acquisition = True
    else :
        flag_HDR = True
        flag_autorise_acquisition = False
    time.sleep(1)


def commande_HOTPIX() :
    global flag_hot_pixels
    
    if choix_HOTPIX.get() == 0 :
        flag_hot_pixels = False
    else :
        flag_hot_pixels = True


def choix_val_ghost_reducer(event=None) :
    global val_ghost_reducer,echelle130
    
    val_ghost_reducer = echelle130.get()


def choix_val_3FNR_Thres(event=None) :
    global val_3FNR_Thres,echelle330
    
    val_3FNR_Thres = echelle330.get()


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
            

def choix_mean_stacking(event=None):
    global flag_stacking
    
    flag_stacking = "Mean"
    reset_FS()


def choix_sum_stacking(event=None):
    global flag_stacking
    
    flag_stacking = "Sum"
    reset_FS()


def choix_dyn_high(event=None):
    global flag_dyn_AANR
    
    flag_dyn_AANR = 1 


def choix_dyn_low(event=None):
    global flag_dyn_AANR
    
    flag_dyn_AANR = 0    


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
    global flag_16b,format_capture,flag_stop_acquisition,flag_filtrage_ON,camera,flag_autorise_acquisition,flag_HDR

    flag_autorise_acquisition = False
    reset_FS()
    reset_general_FS()
    time.sleep(0.5)
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
            time.sleep(0.5)
            camera.set_image_type(format_capture)
            time.sleep(0.2)
            camera.start_video_capture()
        else :
            flag_16b = True
            format_capture = asi.ASI_IMG_RAW16
            camera.stop_video_capture()
            time.sleep(0.5)
            camera.set_image_type(format_capture)
            time.sleep(0.2)
            camera.start_video_capture()
        flag_stop_acquisition = False
        time.sleep(0.2)
        if flag_restore_filtrage == True :
            flag_filtrage_ON = True
        if flag_HDR == False :
            flag_autorise_acquisition = True
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
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK,stack_div,echelle20,choix_AANR,choix_AANRB,flag_first_sat_pass,nb_sat,\
           flag_AANR,compteur_AANR,Im1fsdnOK,Im2fsdnOK,flag_AANRB,compteur_AANRB,Im1fsdnOKB,Im2fsdnOKB,delta_RX,delta_RY,delta_BX,delta_BY,\
           flag_STAB,flag_Template,choix_STAB,compteur_RV,Im1rvOK,Im2rvOK,flag_reduce_variation,choix_reduce_variation,\
           flag_3FNR,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNRB,compteur_3FNRB,\
           img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNR_First_StartB,flag_BFR,flag_STAB,flag_IQE,choix_IMQE,choix_BFR,labelInfo10,\
           flag_3FNR2,compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2B,compteur_3FNR2B,\
           img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2_First_StartB,choix_3FNR2B,\
           flag_CONST,flag_TRKSAT,flag_REMSAT,flag_DETECT_STARS,choix_CONST,choix_TRKSAT,choix_REMSAT,choix_DETECT_STARS,choix_AI_Craters,flag_AI_Craters,\
           flag_reduce_variation_post_treatment,flag_BFREF_imagePT,max_qual_PT,flag_BFREFPT,choix_HDR,flag_HDR,choix_HOTPIX,flag_hot_pixels,\
           flag_AI_Craters,track_crater_history,flag_AI_Satellites,choix_AI_Satellites,track_satelitte_history,model_craters_track,model_satellites_track,\
           flag_image_disponible,flag_img_sat_buf1,flag_img_sat_buf2,flag_img_sat_buf3,flag_img_sat_buf4,flag_img_sat_buf5,sat_frame_count,\
           flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,flag_img_sat_buf4_AI,flag_img_sat_buf5_AI,sat_frame_count_AI,flag_first_sat_pass_AI,\
           choix_sub_img_ref,flag_capture_image_reference,flag_image_ref_sub,flag_image_reference_OK

    choix_sub_img_ref.set(0)
    flag_capture_image_reference = False
    flag_image_reference_OK = False
    flag_image_ref_sub = False
    flag_image_disponible = False
    val_FS = 1
    compteur_FS = 0
    Im1OK = False
    Im2OK = False
    Im3OK = False
    Im4OK = False
    Im5OK = False
    echelle20.set(val_FS)
    choix_AANR.set(0)
    choix_AANRB.set(0)
    flag_AANR = False
    compteur_AANR = 0
    Im1fsdnOK = False
    Im2fsdnOK = False
    flag_AANRB = False
    compteur_AANRB = 0
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
    choix_3FNR2.set(0)
    flag_3FNR2 = False
    compteur_3FNR2 = 0
    img1_3FNR2OK = False
    img2_3FNR2OK = False
    img3_3FNR2OK = False
    FNR2_First_Start = True
    choix_3FNR2B.set(0)
    flag_3FNR2B = False
    compteur_3FNR2B = 0
    img1_3FNR2OKB = False
    img2_3FNR2OKB = False
    img3_3FNR2OKB = False
    FNR2B_First_Start = True
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
    global frame_number
    
    frame_number = 0


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


def commande_Blur_img_ref() :
    global flag_blur_image_ref_sub
    
    if choix_Blur_img_ref.get() == 0 :
        flag_blur_image_ref_sub = False
    else :
        flag_blur_image_ref_sub = True


def commande_GBL() :
    global flag_GaussBlur
    
    if choix_GBL.get() == 0 :
        flag_GaussBlur = False
    else :
        flag_GaussBlur = True


# initialisation des boites scrolbar, buttonradio et checkbutton

xS3=1690
yS3=80

xS1=1490
yS1=270

# Various widgets - loaded from gui_widgets module
exec(gui_widgets.create_various_widgets(), globals())

# Top row widgets - loaded from gui_widgets module
exec(gui_widgets.create_top_row_widgets(), globals())

# Exposition settings widgets - loaded from gui_widgets module
exec(gui_widgets.create_exposition_widgets(), globals())

# Sharpen/Denoise widgets - loaded from gui_widgets module
exec(gui_widgets.create_sharpen_denoise_widgets(), globals())

# Histogram widgets - loaded from gui_widgets module
exec(gui_widgets.create_histogram_widgets(), globals())

# Capture widgets and buttons - loaded from gui_widgets module
exec(gui_widgets.create_capture_widgets(), globals())


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


            if cameras_found[0] == "ZWO ASI183MC" or cameras_found[0] == "ZWO ASI183MM" or cameras_found[0] == "ZWO ASI183MC Pro" or cameras_found[0] == "ZWO ASI183MM Pro" :
                res_cam_x = 5496
                res_cam_y = 3672
                cam_displ_x = int(1350*fact_s)
                cam_displ_y = int(1012*fact_s)
                val_USB = USBCam
                flag_supported_camera = True

                sensor_factor = "4_3"
                sensor_ratio.set(0)

                RES_X_BIN1_4_3 = [4896,4400,4160,3680,3120,2560,1920,1600,1280]
                RES_Y_BIN1_4_3 = [3672,3300,3120,2760,2340,1920,1440,1200,960]

                RES_X_BIN1_16_9 = [5496,4800,4160,3680,3120,2560,1920,1600,1280]
                RES_Y_BIN1_16_9 = [3092,2700,2340,2070,1756,1440,1080,900,720]

                RES_X_BIN1_1_1 = [3672,3200,2640,2240,1920,1440,1200,960,768]
                RES_Y_BIN1_1_1 = [3672,3200,2640,2240,1920,1440,1200,960,768]

                RES_X_BIN2_4_3 = [2448,2200,2080,1840,1560,1280,960]
                RES_Y_BIN2_4_3 = [1836,1650,1560,1380,1170,960,720]
                
                RES_X_BIN2_16_9 = [2748,2400,2080,1840,1560,1280,960]
                RES_Y_BIN2_16_9 = [1546,1350,1170,1035,878,720,540]
                
                RES_X_BIN2_1_1 = [1836,1600,1320,1120,960,720,600]
                RES_Y_BIN2_1_1 = [1836,1600,1320,1120,960,720,600]

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

            if cameras_found[0] == "ZWO ASI290MM Mini" or cameras_found[0] == "ZWO ASI290MC" or cameras_found[0] == "ZWO ASI290MM" or cameras_found[0] == "ZWO ASI462MC" or cameras_found[0] == "ZWO ASI385MC" :
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
start_acquisition()

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


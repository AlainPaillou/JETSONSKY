
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
import gui_callbacks
import image_utils
import astronomy_utils
import filter_pipeline
import refresh_loop

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

# Initialize astronomy calculator with observer location
astro_calc = astronomy_utils.AstronomyCalculator(
    lat_obs=lat_obs,
    long_obs=long_obs,
    alt_obs=alt_obs,
    zone=zone,
    polaris_ad=Polaris_AD,
    polaris_dec=Polaris_DEC
)

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


# Note: GUI callback functions have been moved to gui_callbacks.py module
# They are loaded via exec() below in the GUI initialization section

# initialisation des boites scrolbar, buttonradio et checkbutton

xS3=1690
yS3=80

xS1=1490
yS1=270

# Filter Pipeline - loaded from filter_pipeline module
exec(filter_pipeline.create_filter_pipeline_color(), globals())
exec(filter_pipeline.create_filter_pipeline_mono(), globals())

# Refresh Loop - loaded from refresh_loop module
exec(refresh_loop.create_refresh_loop(), globals())

# GUI Callbacks - loaded from gui_callbacks module (BEFORE widgets since widgets reference callbacks in command= params)
exec(gui_callbacks.create_mode_callbacks(), globals())
exec(gui_callbacks.create_acquisition_mode_callbacks(), globals())
exec(gui_callbacks.create_camera_control_callbacks(), globals())
exec(gui_callbacks.create_flip_callbacks(), globals())
exec(gui_callbacks.create_filter_toggle_callbacks(), globals())
exec(gui_callbacks.create_slider_callbacks(), globals())
exec(gui_callbacks.create_stacking_callbacks(), globals())
exec(gui_callbacks.create_sensor_ratio_callbacks(), globals())
exec(gui_callbacks.create_bayer_callbacks(), globals())
exec(gui_callbacks.create_filter_wheel_callbacks(), globals())
exec(gui_callbacks.create_tracking_callbacks(), globals())
exec(gui_callbacks.create_misc_callbacks(), globals())
exec(gui_callbacks.create_reset_general_fs(), globals())

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

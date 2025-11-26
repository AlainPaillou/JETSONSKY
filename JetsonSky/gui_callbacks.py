"""
GUI Callback Functions Module for JetsonSky

This module contains all GUI callback functions for buttons, checkboxes, sliders,
and other interactive widgets. These callbacks are loaded into the main script's
namespace using exec() to maintain access to global variables.

The callbacks are organized into logical groups:
- Mode selection callbacks (linear, gaussian, HDR modes)
- Acquisition mode callbacks (fast, medium, slow)
- Camera control callbacks (exposure, gain, resolution, BIN)
- Filter control callbacks (flip, sharpen, denoise, etc.)
- Display callbacks (histogram, demo, cross)
- Tracking callbacks (satellites, stars, AI detection)
- File/capture callbacks
- Miscellaneous settings callbacks

Copyright Alain Paillou 2018-2025
"""


def create_mode_callbacks():
    """
    Returns callback functions for mode selection:
    - Linear/Gaussian/Stars amplification modes
    - HDR modes (Mertens, Median, Mean)
    - Gradient/Vignetting modes
    """
    return '''
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


def mode_gradient() :
    global grad_vignet

    grad_vignet = 1


def mode_vignetting() :
    global grad_vignet

    grad_vignet = 2
'''


def create_acquisition_mode_callbacks():
    """
    Returns callback functions for acquisition speed modes:
    - Fast mode (100-10000 microseconds)
    - Medium Fast mode (1-400 ms)
    - Medium Slow mode (1-1000 ms)
    - Slow mode (500-20000 ms)
    """
    return '''
def mode_acq_rapide() :
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,timeoutexp,flag_read_speed,\\
           frame_rate,flag_stop_acquisition,exposition

    if flag_camera_ok == True :
        flag_acq_rapide = "Fast"
        flag_stop_acquisition=True
        time.sleep(1)
        exp_min=100 #us
        exp_max=10000 #us
        exp_delta=100 #us
        exp_interval=2000 #us
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
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\\
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
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\\
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
    global labelParam1,flag_acq_rapide,camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,timeoutexp,flag_read_speed,\\
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
'''


def create_camera_control_callbacks():
    """
    Returns callback functions for camera controls:
    - Exposure and gain settings
    - BIN mode selection (1 or 2)
    - Resolution selection
    - USB bandwidth
    - Gamma settings
    - Read speed (fast/slow)
    - White balance (red/blue)
    """
    return '''
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
    global val_FS,camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,echelle3,\\
           flag_image_disponible,flag_stop_acquisition,flag_autorise_acquisition,flag_HDR,\\
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
'''


def create_flip_callbacks():
    """
    Returns callback functions for flip controls (FlipV, FlipH).
    These handle vertical and horizontal image flipping with Bayer pattern adjustments.
    """
    return '''
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
            type_debayer = cv2.COLOR_BayerBG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "BGGR" :
            type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GRBG" :
            type_debayer = cv2.COLOR_BayerGR2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GBRG" :
            type_debayer = cv2.COLOR_BayerGB2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipV.get() == 0 :
            FlipV = 0
        else :
            FlipV = 1
        if Video_Bayer == "RAW" :
            type_debayer = 0
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
        if Video_Bayer == "RGGB" :
            type_debayer = cv2.COLOR_BayerBG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "horizontal"
        if Video_Bayer == "BGGR" :
            type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "horizontal"
        if Video_Bayer == "GRBG" :
            type_debayer = cv2.COLOR_BayerGR2RGB
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
            type_debayer = cv2.COLOR_BayerGB2RGB
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
            type_debayer = cv2.COLOR_BayerBG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "BGGR" :
            type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GRBG" :
            type_debayer = cv2.COLOR_BayerGR2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 1)
        if Camera_Bayer == "GBRG" :
            type_debayer = cv2.COLOR_BayerGB2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 4
                camera.set_control_value(asi.ASI_FLIP, 3)
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 3
                camera.set_control_value(asi.ASI_FLIP, 0)
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 1
                camera.set_control_value(asi.ASI_FLIP, 2)
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 2
                camera.set_control_value(asi.ASI_FLIP, 1)
    else :
        if choix_flipH.get() == 0 :
            FlipH = 0
        else :
            FlipH = 1
        if Video_Bayer == "RAW" :
            type_debayer = 0
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
        if Video_Bayer == "RGGB" :
            type_debayer = cv2.COLOR_BayerBG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 2
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 1
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 4
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 3
                type_flip = "horizontal"
        if Video_Bayer == "BGGR" :
            type_debayer = cv2.COLOR_BayerRG2RGB
            if FlipV == 1 and FlipH == 1 :
                GPU_BAYER = 1
                type_flip = "both"
            if FlipV == 0 and FlipH == 0 :
                GPU_BAYER = 2
                type_flip = "none"
            if FlipV == 1 and FlipH == 0 :
                GPU_BAYER = 3
                type_flip = "vertical"
            if FlipV == 0 and FlipH == 1 :
                GPU_BAYER = 4
                type_flip = "horizontal"
        if Video_Bayer == "GRBG" :
            type_debayer = cv2.COLOR_BayerGR2RGB
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
            type_debayer = cv2.COLOR_BayerGB2RGB
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
'''


def create_filter_toggle_callbacks():
    """
    Returns callback functions for filter toggles (checkboxes):
    - Image negative, TIP, saturation
    - Sharpen, NLM2, denoise Paillou
    - CLAHE, CLL, histogram equalize
    - AANR, 3FNR filters
    - Ghost reducer, KNN
    - Gradient removal, HDR, hot pixels
    """
    return '''
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


def commande_histogram_equalize2() :
    global flag_histogram_equalize2

    if choix_histogram_equalize2.get() == 0 :
        flag_histogram_equalize2 = 0
    else :
        flag_histogram_equalize2 = 1


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
        flag_contrast_CLAHE = 0
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

    if choix_filtrage_ON.get() == 0 :
        flag_filtrage_ON = False
    else :
        flag_filtrage_ON = True


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
    global choix_hard_bin,flag_HB

    if choix_hard_bin.get() == 0 :
        flag_HB = False
    else :
        flag_HB = True
'''


def create_slider_callbacks():
    """
    Returns callback functions for slider (Scale) controls:
    - Denoise, CLAHE grid, ghost reducer, 3FNR threshold
    - Histogram min/max, phi, theta, heq2
    - KNN, amplification, SGR, AGR
    - Sharpen values, saturation
    - RGB software adjustments
    - CLL parameters (Mu, Ro, Amp)
    """
    return '''
def choix_valeur_denoise(event=None) :
    global val_denoise

    val_denoise=echelle4.get()
    if val_denoise == 0 :
        val_denoise += 1


def choix_grid_CLAHE(event=None) :
    global val_grid_CLAHE

    val_grid_CLAHE=echelle109.get()


def choix_valeur_CLAHE(event=None) :
    global val_contrast_CLAHE,echelle9

    val_contrast_CLAHE=echelle9.get()


def choix_val_ghost_reducer(event=None) :
    global val_ghost_reducer,echelle130

    val_ghost_reducer = echelle130.get()


def choix_val_3FNR_Thres(event=None) :
    global val_3FNR_Thres,echelle330

    val_3FNR_Thres = echelle330.get()


def choix_histo_min(event=None) :
    global val_histo_min, echelle5

    val_histo_min=echelle5.get()


def choix_phi(event=None) :
    global val_phi, echelle12

    val_phi=echelle12.get()


def choix_theta(event=None) :
    global val_theta, echelle13

    val_theta=echelle13.get()


def choix_histo_max(event=None) :
    global val_histo_max, echelle6

    val_histo_max=echelle6.get()


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


def choix_TH_16B(event=None) :
    global TH_16B, echelle804,camera,threshold_16bits

    TH_16B=echelle804.get()
    threshold_16bits = 2 ** TH_16B - 1


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


def choix_val_BFR(event=None) :
    global val_BFR, echelle300

    val_BFR = echelle300.get()
'''


def create_stacking_callbacks():
    """
    Returns callback functions for stacking and frame selection:
    - Mean/Sum stacking modes
    - Dynamic high/low AANR modes
    - Frame stacking controls
    - Best/Previous frame reference
    """
    return '''
def choix_mean_stacking(event=None):
    global flag_stacking,stack_div,val_FS

    flag_stacking = "Mean"
    stack_div = val_FS


def choix_sum_stacking(event=None):
    global flag_stacking,stack_div

    flag_stacking = "Sum"
    stack_div = 1


def choix_dyn_high(event=None):
    global flag_dyn_AANR

    flag_dyn_AANR = 1


def choix_dyn_low(event=None):
    global flag_dyn_AANR

    flag_dyn_AANR = 0


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


def command_BFReference(event=None) :
    global flag_BFReference

    flag_BFReference = "BestFrame"


def command_PFReference(event=None) :
    global flag_BFReference

    flag_BFReference = "PreviousFrame"
'''


def create_sensor_ratio_callbacks():
    """
    Returns callback functions for sensor aspect ratio selection:
    - 4:3, 16:9, 1:1 ratios
    """
    return '''
def choix_sensor_ratio_4_3(event=None) :
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\\
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
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\\
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
    global sensor_factor,cam_displ_x,cam_displ_y,RES_X_BIN1,RES_Y_BIN1,RES_X_BIN2,RES_Y_BIN2,\\
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
'''


def create_bayer_callbacks():
    """
    Returns callback functions for Bayer pattern selection:
    - RAW, RGGB, BGGR, GBRG, GRBG patterns
    """
    return '''
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
'''


def create_filter_wheel_callbacks():
    """
    Returns callback functions for filter wheel positions (0-4).
    """
    return '''
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


def commande_FW() :
    time.sleep(0.01)
'''


def create_tracking_callbacks():
    """
    Returns callback functions for satellite/star tracking and AI detection:
    - Track satellites, remove satellites, reconstruct image
    - AI craters detection, AI satellites detection
    - AI trace toggle
    """
    return '''
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
    global flag_AI_Satellites,model_Satellites,track_satellite_history,sat_frame_count_AI,flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,\\
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
'''


def create_misc_callbacks():
    """
    Returns miscellaneous callback functions:
    - Demo side (left/right)
    - SAT video/image mode
    - Autoexposure/autogain
    - Black and white mode
    - 16-bit low light mode
    - Image quality estimation
    - Reverse R/B
    - Capture counts
    - Position frame slider
    - Reset functions
    - Image reference subtraction
    - Gaussian blur
    """
    return '''
def choix_demo_left(event=None) :
    global flag_demo_side

    flag_demo_side = "Left"


def choix_demo_right(event=None) :
    global flag_demo_side

    flag_demo_side = "Right"


def choix_SAT_Vid() :
    global flag_SAT_Image

    flag_SAT_Image = False


def choix_SAT_Img() :
    global flag_SAT_Image

    flag_SAT_Image = True


def commande_autoexposure() :
    global flag_autoexposure_exposition

    if choix_autoexposure.get() == 0 :
        flag_autoexposure_exposition = False
        if flag_camera_ok == True :
            camera.set_control_value(asi.ASI_EXPOSURE, exposition,auto=False)
    else :
        flag_autoexposure_exposition = True
        if flag_camera_ok == True :
            camera.set_control_value(asi.ASI_EXPOSURE, exposition,auto=True)
            camera.set_control_value(asi.ASI_AUTO_MAX_EXP,250) # 250ms max exposure


def commande_autogain() :
    global flag_autoexposure_gain

    if choix_autogain.get() == 0 :
        flag_autoexposure_gain = False
        if flag_camera_ok == True :
            camera.set_control_value(asi.ASI_GAIN, val_gain,auto=False)
    else :
        flag_autoexposure_gain = True
        if flag_camera_ok == True :
            camera.set_control_value(asi.ASI_GAIN, val_gain,auto=True)
            camera.set_control_value(asi.ASI_AUTO_MAX_GAIN,val_maxgain)


def commande_noir_blanc() :
    global flag_noir_blanc,flag_colour_camera

    if choix_noir_blanc.get() == 0 :
        flag_noir_blanc = 0
        flag_colour_camera = True
        if flag_camera_ok == True :
            if flag_IsColor == True :
                camera.set_control_value(asi.ASI_MONO_BIN,0)
    else :
        flag_noir_blanc = 1
        flag_colour_camera = False
        if flag_camera_ok == True :
            if flag_IsColor == True :
                camera.set_control_value(asi.ASI_MONO_BIN,1)


def commande_16bLL() :
    global flag_16b,flag_cap_video

    if flag_camera_ok == True :
        if flag_cap_video == False :
            if choix_16bLL.get() == 0 :
                flag_16b = False
            else :
                flag_16b = True


def commande_IMQE() :
    global flag_IQE,quality,max_quality,quality_pos

    if choix_IMQE.get() == 0 :
        flag_IQE = False
    else :
        flag_IQE = True
        for x in range(1,258) :
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
'''


def create_reset_general_fs():
    """
    Returns the reset_general_FS function which resets all filter states.
    This is a large function that resets many global variables.
    """
    return '''
def reset_general_FS():
    global val_FS,compteur_FS,Im1OK,Im2OK,Im3OK,Im4OK,Im5OK,stack_div,echelle20,choix_AANR,choix_AANRB,flag_first_sat_pass,nb_sat,\\
           flag_AANR,compteur_AANR,Im1fsdnOK,Im2fsdnOK,flag_AANRB,compteur_AANRB,Im1fsdnOKB,Im2fsdnOKB,delta_RX,delta_RY,delta_BX,delta_BY,\\
           flag_STAB,flag_Template,choix_STAB,compteur_RV,Im1rvOK,Im2rvOK,flag_reduce_variation,choix_reduce_variation,\\
           flag_3FNR,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNRB,compteur_3FNRB,\\
           img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNR_First_StartB,flag_BFR,flag_STAB,flag_IQE,choix_IMQE,choix_BFR,labelInfo10,\\
           flag_3FNR2,compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2B,compteur_3FNR2B,\\
           img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2_First_StartB,choix_3FNR2B,\\
           flag_CONST,flag_TRKSAT,flag_REMSAT,flag_DETECT_STARS,choix_CONST,choix_TRKSAT,choix_REMSAT,choix_DETECT_STARS,choix_AI_Craters,flag_AI_Craters,\\
           flag_reduce_variation_post_treatment,flag_BFREF_imagePT,max_qual_PT,flag_BFREFPT,choix_HDR,flag_HDR,choix_HOTPIX,flag_hot_pixels,\\
           flag_AI_Craters,track_crater_history,flag_AI_Satellites,choix_AI_Satellites,track_satelitte_history,model_craters_track,model_satellites_track,\\
           flag_image_disponible,flag_img_sat_buf1,flag_img_sat_buf2,flag_img_sat_buf3,flag_img_sat_buf4,flag_img_sat_buf5,sat_frame_count,\\
           flag_img_sat_buf1_AI,flag_img_sat_buf2_AI,flag_img_sat_buf3_AI,flag_img_sat_buf4_AI,flag_img_sat_buf5_AI,sat_frame_count_AI,flag_first_sat_pass_AI,\\
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
'''

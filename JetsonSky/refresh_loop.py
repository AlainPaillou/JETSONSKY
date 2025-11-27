"""
Refresh Loop Module for JetsonSky

This module contains the main display loop function that:
- Handles camera acquisition and video processing
- Applies the filter pipeline to incoming frames
- Performs AI-based object detection (craters, satellites)
- Manages display rendering and user interaction
- Controls video/image capture operations

The refresh() function is the heart of the application, called repeatedly
via Tkinter's after() method to process and display frames.

Main flow:
1. Camera mode (flag_camera_ok == True):
   - Capture frame from camera
   - HDR processing (if enabled)
   - Image stabilization (if enabled)
   - Apply filter pipeline
   - AI detection (craters, satellites)
   - Display and capture

2. Video/Image mode (flag_camera_ok == False):
   - Load frame from video/image file
   - Apply debayering and preprocessing
   - Apply filter pipeline
   - Display and capture

Copyright Alain Paillou 2018-2025
"""


def create_refresh_loop():
    """
    Returns the main refresh loop function.
    This function (~1440 lines) handles frame acquisition, processing, and display.
    """
    return ''' 
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
                        texte_mount = astro_calc.angle2degminsec(azimut)
                        cv2.putText(image_traitee, texte_mount, (posAX,posAY), font, size, (255, 255, 255), 1, cv2.LINE_AA)
                        texte_mount = astro_calc.angle2degminsec(hauteur)
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


''' 

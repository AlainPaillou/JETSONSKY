"""
Filter Pipeline Module for JetsonSky

This module contains the GPU-accelerated image filtering pipeline functions:
- application_filtrage_color: Color image filter processing pipeline
- application_filtrage_mono: Monochrome image filter processing pipeline

These functions are loaded into the main script's namespace using exec() to maintain
access to global variables and CUDA kernels.

The filter pipeline processes frames in this fixed order:
1. RGB software adjustment
2. Image negative
3. Luminance estimate (mono from color)
4. 2-5 images SUM or MEAN stacking
5. Reduce consecutive variation
6. 3FNR1 front (3-frame noise removal)
7. AANR front (Adaptive Absorber Noise Removal)
8. NR P1 (Noise Removal Paillou 1)
9. NR P2 (Noise Removal Paillou 2)
10. NLM2 (Non-Local Means 2)
11. KNN (K-Nearest Neighbors)
12. Luminance adjust
13. Image amplification (Linear/Gaussian)
14. Star amplification
15. Gradient/vignetting management
16. CLL (Contrast Low Light)
17. CLAHE contrast
18. Color saturation enhancement (2-pass option)
19. 3FNR2 back
20. AANR back (High dynamic only)
21. Sharpen 1
22. Sharpen 2

Copyright Alain Paillou 2018-2025
"""


def create_filter_pipeline_color():
    """
    Returns the color image filter pipeline function.
    This is a large function (~970 lines) that applies GPU-accelerated filters
    to color images in a specific order.
    """
    return '''
def application_filtrage_color(res_b1,res_g1,res_r1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm, g1_sm, g2_sm, g3_sm, g4_sm, g5_sm, r1_sm, r2_sm, r3_sm, r4_sm, r5_sm,\\
           Im4OK,Im5OK,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_brute_grey,cupy_context,BFREF_image,flag_BFREF_image,flag_SAT2PASS,\\
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,flag_IsColor,flag_BFReference,BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_AmpSoft,val_ampl,grad_vignet,compteur_AANR,compteur_AANRB,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\\
           Im1fsdnOK,Im2fsdnOK,Im1fsdnOKB,Im2fsdnOKB,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,val_reds,val_greens,val_blues,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\\
           flag_dyn_AANR,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_g2,res_r2,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\\
           imgb1,imgg1,imgr1,imgb2,imgg2,imgr2,imgb3,imgg3,imgr3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,res_b2B,res_g2B,res_r2B,\\
           imgb1B,imgg1B,imgr1B,imgb2B,imgg2B,imgr2B,imgb3B,imgg3B,imgr3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,flag_AANRB,val_3FNR_Thres,\\
           compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2,compteur_3FNR2B,img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2B_First_Start,flag_3FNR2B,\\
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

                    Set_RGB((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \\
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

                    color_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1,\\
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

                        reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\\
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

                    reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\\
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

                        FNR_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr1, imgg1, imgb1, imgr2, imgg2, imgb2,\\
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

                        FNR2_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr21, imgg21, imgb21, imgr22, imgg22, imgb22,\\
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

                        adaptative_absorber_denoise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\\
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

                    Denoise_Paillou_Colour((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.intc(cell_size), \\
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

                    NLM2_Colour_GPU((nb_blocksXs,nb_blocksYs),(nb_ThreadsXs,nb_ThreadsYs),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.float32(Noise), \\
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

                    KNN_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.intc(width), np.intc(height), np.float32(Noise), \\
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

                        reduce_variation_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2, res_g2, res_b2,\\
                                             np.int_(width), np.int_(height),np.int_(variation)))

                        res_r1 = r_gpu
                        res_g1 = g_gpu
                        res_b1 = b_gpu

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) or\\
                   (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) :

                    b_gpu = res_b1
                    g_gpu = res_g1
                    r_gpu = res_r1

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                    # Histo equalize 2
                    # Histo stretch
                    # Histo Phi Theta

                    Histo_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \\
                       np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \\
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

                        Colour_ampsoft_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \\
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

                    Colour_staramp_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, imagegrey, imagegreyblur,np.int_(width), np.int_(height), \\
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

                    Contrast_Low_Light_Colour_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, np.int_(width), np.int_(height), \\
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

                        FNR_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr1B, imgg1B, imgb1B, imgr2B, imgg2B, imgb2B,\\
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

                        FNR2_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, imgr21B, imgg21B, imgb21B, imgr22B, imgg22B, imgb22B,\\
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

                        adaptative_absorber_denoise_Color((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, g_gpu, b_gpu, res_r1, res_g1, res_b1, res_r2B, res_g2B, res_b2B,\\
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
'''


def create_filter_pipeline_mono():
    """
    Returns the monochrome image filter pipeline function.
    This is a large function (~500 lines) that applies GPU-accelerated filters
    to monochrome images in a specific order.
    """
    return '''
def application_filtrage_mono(res_b1) :
    global compteur_FS,Im1OK,Im2OK,Im3OK,b1_sm, b2_sm, b3_sm, b4_sm, b5_sm,flag_IsColor,flag_BFReference,\\
           Im4OK,Im5OK,val_denoise,val_denoise_KNN,val_histo_min,val_histo_max,image_brute_grey,cupy_context, BFREF_image_PT,max_qual_PT,flag_BFREF_image_PT,\\
           flag_cap_pic,flag_traitement,flag_CLL,val_contrast_CLAHE,flag_histogram_phitheta,image_traitee,Date_hour_image,image_brute,BFREF_image,flag_BFREF_image,\\
           val_heq2,val_SGR,val_NGB,val_AGR,flag_AmpSoft,val_ampl,grad_vignet,compteur_AANR,compteur_RV,flag_SAT,val_SAT,flag_NB_estime,TTQueue,curTT,\\
           Im1fsdnOK,Im2fsdnOK,Im1rvOK,Im2rvOK,image_traiteefsdn1,image_traiteefsdn2,old_image,trsf_r,trsf_g,trsf_b,val_sigma_sharpen,val_sigma_sharpen2,\\
           flag_dyn_AANR,Corr_GS,azimut,hauteur,val_ghost_reducer,res_b2,res_b2B,time_exec_test,flag_HDR,val_sharpen,val_sharpen2,flag_reduce_variation,val_reduce_variation,\\
           imgb1,imgb2,imgb3,compteur_3FNR,img1_3FNROK,img2_3FNROK,img3_3FNROK,FNR_First_Start,flag_3FNR,Corr_CLL,Im1fsdnOKB,Im2fsdnOKB, \\
           imgb1B,imgb2B,imgb3B,compteur_3FNRB,img1_3FNROKB,img2_3FNROKB,img3_3FNROKB,FNRB_First_Start,flag_3FNRB,compteur_AANRB,val_3FNR_Thres,\\
           compteur_3FNR2,img1_3FNR2OK,img2_3FNR2OK,img3_3FNR2OK,FNR2_First_Start,flag_3FNR2,compteur_3FNR2B,img1_3FNR2OKB,img2_3FNR2OKB,img3_3FNR2OKB,FNR2B_First_Start,flag_3FNR2B,\\
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

                    grey_estimate_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1,\\
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

                        adaptative_absorber_denoise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, res_b2,\\
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
                    Denoise_Paillou_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.intc(width), np.intc(height), np.intc(cell_size), \\
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
                    NLM2_Mono_GPU((nb_blocksXs,nb_blocksYs),(nb_ThreadsXs,nb_ThreadsYs),(r_gpu, res_b1, np.intc(width),np.intc(height), np.float32(Noise), \\
                         np.float32(lerpC)))

                    res_b1 = r_gpu

                # Denoise KNN
                if flag_denoise_KNN == 1 :
                    param=float(val_denoise_KNN)
                    Noise = 1.0/(param*param)
                    lerpC = 0.4
                    r_gpu = res_b1
                    KNN_Mono_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.intc(width),np.intc(height), np.float32(Noise), \\
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

                        reduce_variation_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(b_gpu, res_b1, res_b2,\\
                                             np.int_(width), np.int_(height),np.int_(variation)))

                        res_b1 = b_gpu

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) or (flag_AmpSoft == 1  and (flag_lin_gauss == 1 or flag_lin_gauss == 2)) :

                    r_gpu = res_b1

                if (flag_histogram_stretch == 1 or flag_histogram_equalize2 == 1 or flag_histogram_phitheta == 1) :
                    # Histo equalize 2 CUDA
                    # Histo stretch CUDA
                    # Histo Phi Theta CUDA

                    Histo_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), \\
                       np.intc(flag_histogram_stretch),np.float32(val_histo_min), np.float32(val_histo_max), \\
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

                        Mono_ampsoft_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, np.int_(width), np.int_(height), \\
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

                    Mono_staramp_GPU((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, imagegreyblur,np.int_(width), np.int_(height), \\
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

                        adaptative_absorber_denoise_Mono((nb_blocksX,nb_blocksY),(nb_ThreadsX,nb_ThreadsY),(r_gpu, res_b1, res_b2B,\\
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
'''

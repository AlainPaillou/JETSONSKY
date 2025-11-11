def create_top_row_widgets():
    return '''
# Top row widgets
CBF = Checkbutton(cadre,text="Filters ON", variable=choix_filtrage_ON,command=commande_filtrage_ON,onvalue = 1, offvalue = 0)
CBF.place(anchor="w",x=1440+delta_s, y=50)
CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=1510+delta_s, y=50)
CBFNB = Checkbutton(cadre,text="Set B&W", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=1570+delta_s, y=50)
CBFNBE = Checkbutton(cadre,text="B&W Est", variable=choix_noir_blanc_estime,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNBE.place(anchor="w",x=1630+delta_s, y=50)
CBRRB = Checkbutton(cadre,text="R-B Rev", variable=choix_reverse_RB,command=commande_reverse_RB,onvalue = 1, offvalue = 0)
CBRRB.place(anchor="w",x=1695+delta_s, y=50)
CBFC = Checkbutton(cadre,text="False Col", variable=choix_false_colours,command=commande_false_colours,onvalue = 1, offvalue = 0)
CBFC.place(anchor="w",x=1750+delta_s, y=50)
RBSM1 = Radiobutton(cadre,text="MEAN", variable=choix_stacking,command=choix_mean_stacking,value=1)
RBSM1.place(anchor="w",x=1610+delta_s, y=20)
RBSM2 = Radiobutton(cadre,text="SUM", variable=choix_stacking,command=choix_sum_stacking,value=2)
RBSM2.place(anchor="w",x=1660+delta_s, y=20)
echelle20 = Scale (cadre, from_ = 1, to = 5, command= choix_FS, orient=HORIZONTAL, length = 80, width = 7, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle20.set(val_FS)
echelle20.place(anchor="w", x=1720+delta_s,y=20)
'''


def create_various_widgets():
    return '''
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
CBSUBIMREF.place(anchor="w",x=0, y=895)
CBBLURIMREF = Checkbutton(cadre,text="Blur Res", variable=choix_Blur_img_ref,command=commande_Blur_img_ref,onvalue = 1, offvalue = 0)
CBBLURIMREF.place(anchor="w",x=0, y=920)

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
CBGBL = Checkbutton(cadre,text="GBL", variable=choix_GBL,command=commande_GBL,onvalue = 1, offvalue = 0)
CBGBL.place(anchor="w",x=1840+delta_s, y=495)

# Saturation
CBSAT = Checkbutton(cadre,text="SAT", variable=choix_SAT,command=commande_SAT,onvalue = 1, offvalue = 0)
CBSAT.place(anchor="w",x=1840+delta_s,y=640)
RBSATVI1 = Radiobutton(cadre,text="Vid", variable=Sat_Vid_Img,command=choix_SAT_Vid,value=0)
RBSATVI1.place(anchor="w",x=1840+delta_s, y=660)
RBSATVI2 = Radiobutton(cadre,text="Img", variable=Sat_Vid_Img,command=choix_SAT_Img,value=1)
RBSATVI2.place(anchor="w",x=1840+delta_s, y=680)
CBSAT2PASS = Checkbutton(cadre,text="2pass", variable=choix_SAT2PASS,command=commande_SAT2PASS,onvalue = 1, offvalue = 0)
CBSAT2PASS.place(anchor="w",x=1840+delta_s,y=700)
echelle70 = Scale (cadre, from_ = 0, to = 20, command= choix_val_SAT, orient=VERTICAL, length = 150, width = 7, resolution = 0.01, label="",showvalue=1,tickinterval=None,sliderlength=10)
echelle70.set(val_SAT)
echelle70.place(anchor="c", x=1855+delta_s,y=800)

# DEMO
CBDEMO = Checkbutton(cadre,text="Demo", variable=choix_DEMO,command=commande_DEMO,onvalue = 1, offvalue = 0)
CBDEMO.place(anchor="w",x=1840+delta_s, y=960)
RBDEML = Radiobutton(cadre,text="Left", variable=demo_side,command=choix_demo_left,value=0)
RBDEML.place(anchor="w",x=1840+delta_s, y=985)
RBDEMR = Radiobutton(cadre,text="Right", variable=demo_side,command=choix_demo_right,value=1)
RBDEMR.place(anchor="w",x=1840+delta_s, y=1010)
'''


def create_exposition_widgets():
    return '''
###########################
#   EXPOSITION SETTINGS   #
###########################

# Speed mode acquisition
labelMode_Acq = Label (cadre, text = "Speed")
labelMode_Acq.place (anchor="w",x=1430+delta_s, y=240)
RBMA1 = Radiobutton(cadre,text="Fast", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=1460+delta_s, y=240)
RBMA2 = Radiobutton(cadre,text="MedF", variable=mode_acq,command=mode_acq_mediumF,value=2)
RBMA2.place(anchor="w",x=1505+delta_s, y=240)
RBMA3 = Radiobutton(cadre,text="MedS", variable=mode_acq,command=mode_acq_mediumS,value=3)
RBMA3.place(anchor="w",x=1550+delta_s, y=240)
RBMA4 = Radiobutton(cadre,text="Slow", variable=mode_acq,command=mode_acq_lente,value=4)
RBMA4.place(anchor="w",x=1595+delta_s, y=240)

# Choix HDR
CBHDR = Checkbutton(cadre,text="HDR", variable=choix_HDR,command=commande_HDR,onvalue = 1, offvalue = 0)
CBHDR.place(anchor="w",x=1650+delta_s, y=240)
RBHDR1 = Radiobutton(cadre,text="Mertens", variable=mode_HDR_select,command=HDR_Mertens,value=1)
RBHDR1.place(anchor="w",x=1690+delta_s, y=240)
RBHDR2 = Radiobutton(cadre,text="Median", variable=mode_HDR_select,command=HDR_Median,value=2)
RBHDR2.place(anchor="w",x=1745+delta_s, y=240)
RBHDR3 = Radiobutton(cadre,text="Mean", variable=mode_HDR_select,command=HDR_Mean,value=3)
RBHDR3.place(anchor="w",x=1795+delta_s, y=240)

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

# Automatic gain
CBOG = Checkbutton(cadre,text="Auto Gain", variable=choix_autogain,command=commande_autogain,onvalue = 1, offvalue = 0)
CBOG.place(anchor="w",x=1440+delta_s, y=120)

echelle2 = Scale (cadre, from_ = 0, to = val_maxgain , command= valeur_gain, orient=HORIZONTAL, length = 320, width = 7, resolution = 1, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle2.set(val_gain)
echelle2.place(anchor="w", x=1500+delta_s,y=120)

# Signal amplification soft
CBAS = Checkbutton(cadre,text="Amplif Soft", variable=choix_AmpSoft,command=commande_AmpSoft,onvalue = 1, offvalue = 0)
CBAS.place(anchor="w",x=1450+delta_s, y=160)
echelle80 = Scale (cadre, from_ = 0, to = 20.0, command= choix_amplif, orient=HORIZONTAL, length = 280, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle80.set(val_ampl)
echelle80.place(anchor="w", x=1540+delta_s,y=160)

RBMuRo1 = Radiobutton(cadre,text="Lin", variable=mode_Lin_Gauss,command=mode_Lineaire,value=1)
RBMuRo1.place(anchor="w",x=1440+delta_s, y=200)
RBMuRo2 = Radiobutton(cadre,text="Gauss", variable=mode_Lin_Gauss,command=mode_Gauss,value=2)
RBMuRo2.place(anchor="w",x=1480+delta_s, y=200)
RBMuRo3 = Radiobutton(cadre,text="Stars", variable=mode_Lin_Gauss,command=mode_Stars,value=3)
RBMuRo3.place(anchor="w",x=1525+delta_s, y=200)

labelParam82 = Label (cadre, text = "µX")
labelParam82.place(anchor="w", x=1573+delta_s,y=200)
echelle82 = Scale (cadre, from_ = -5.0, to = 5.0, command= choix_Mu, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle82.set(val_Mu)
echelle82.place(anchor="w", x=1593+delta_s,y=200)

labelParam84 = Label (cadre, text = "Ro")
labelParam84.place(anchor="w", x=1705+delta_s,y=200)
echelle84 = Scale (cadre, from_ = 0.2, to = 5.0, command= choix_Ro, orient=HORIZONTAL, length = 100, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle84.set(val_Ro)
echelle84.place(anchor="w", x=1720+delta_s,y=200)

# Camera Red balance
labelParam14 = Label (cadre, text = "CRed")
labelParam14.place(anchor="w", x=1450+delta_s,y=305)
echelle14 = Scale (cadre, from_ = 1, to = 99, command= choix_w_red, orient=HORIZONTAL, length = 140, width = 7, resolution = 1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle14.set(val_red)
echelle14.place(anchor="w", x=1485+delta_s,y=305)

# Camera Blue balance
labelParam15 = Label (cadre, text = "CBlue")
labelParam15.place(anchor="w", x=1645+delta_s,y=305)
echelle15 = Scale (cadre, from_ = 1, to = 99, command= choix_w_blue, orient=HORIZONTAL, length = 140, width = 7, resolution = 1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle15.set(val_blue)
echelle15.place(anchor="w", x=1680+delta_s,y=305)

# Software Red balance
labelParam100 = Label (cadre, text = "R")
labelParam100.place(anchor="w", x=1440+delta_s,y=340)
echelle100 = Scale (cadre, from_ = 0, to = 2, command= choix_w_reds, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle100.set(val_reds)
echelle100.place(anchor="w", x=1440+delta_s+15,y=340)

# Software Green balance
labelParam101 = Label (cadre, text = "G")
labelParam101.place(anchor="w", x=1440+delta_s,y=370)
echelle101 = Scale (cadre, from_ = 0, to = 2, command= choix_w_greens, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle101.set(val_greens)
echelle101.place(anchor="w", x=1440+delta_s+15,y=370)

# Software Blue balance
labelParam102 = Label (cadre, text = "B")
labelParam102.place(anchor="w", x=1440+delta_s,y=400)
echelle102 = Scale (cadre, from_ = 0, to = 2, command= choix_w_blues, orient=HORIZONTAL, length = 360, width = 7, resolution = 0.005, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle102.set(val_blues)
echelle102.place(anchor="w", x=1440+delta_s+15,y=400)
'''

def create_sharpen_denoise_widgets():
    return '''
###########################
# SHARPEN DENOISE WIDGETS #
###########################

# Choix Sharpen 1 & 2
CBSS1 = Checkbutton(cadre,text="Sharpen 1  Val/Sigma", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=1450+delta_s, y=435)
echelle152 = Scale (cadre, from_ = 0, to = 10, command= choix_val_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.2, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle152.set(val_sharpen)
echelle152.place(anchor="w", x=1560+delta_s,y=435)
echelle153 = Scale (cadre, from_ = 1, to = 9, command= choix_val_sigma_sharpen, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle153.set(val_sigma_sharpen)
echelle153.place(anchor="w", x=1690+delta_s,y=435)

CBSS2 = Checkbutton(cadre,text="Sharpen 2  Val/Sigma", variable=choix_sharpen_soft2,command=commande_sharpen_soft2,onvalue = 1, offvalue = 0)
CBSS2.place(anchor="w",x=1450+delta_s, y=460)
echelle154 = Scale (cadre, from_ = 0, to = 10, command= choix_val_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 0.2, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle154.set(val_sharpen2)
echelle154.place(anchor="w", x=1560+delta_s,y=460)
echelle155 = Scale (cadre, from_ = 1, to = 9, command= choix_val_sigma_sharpen2, orient=HORIZONTAL, length = 120, width = 7, resolution = 1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle155.set(val_sigma_sharpen2)
echelle155.place(anchor="w", x=1690+delta_s,y=460)

# Choix filtre Denoise Paillou image
CBEPF = Checkbutton(cadre,text="NR P1", variable=choix_denoise_Paillou,command=commande_denoise_Paillou,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1450+delta_s, y=495)

# Choix filtre Denoise Paillou 2 image
CBEPF2 = Checkbutton(cadre,text="NR P2", variable=choix_denoise_Paillou2,command=commande_denoise_Paillou2,onvalue = 1, offvalue = 0)
CBEPF2.place(anchor="w",x=1500+delta_s, y=495)

# Choix filtre Denoise KNN
CBEPF = Checkbutton(cadre,text="KNN", variable=choix_denoise_KNN,command=choix_KNN,onvalue = 1, offvalue = 0)
CBEPF.place(anchor="w",x=1550+delta_s, y=495)
echelle30 = Scale (cadre, from_ = 0.05, to = 1.2, command= choix_val_KNN, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle30.set(val_denoise_KNN)
echelle30.place(anchor="w", x=1600+delta_s,y=490)

# Choix filtre Denoise NLM2
CBDS = Checkbutton(cadre,text="NLM2", variable=choix_NLM2,command=commande_NLM2,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=1690+delta_s, y=495)
echelle4 = Scale (cadre, from_ = 0.1, to = 1.2, command= choix_valeur_denoise, orient=HORIZONTAL, length = 70, width = 7, resolution = 0.1, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle4.set(val_denoise)
echelle4.place(anchor="w", x=1740+delta_s,y=490)


# Choix filtre Denoise Paillou 3FNR 1 Front
CB3FNRF = Checkbutton(cadre,text="3FNR 1F", variable=choix_3FNR,command=commande_3FNR,onvalue = 1, offvalue = 0)
CB3FNRF.place(anchor="w",x=1450+delta_s, y=525)

# Choix filtre Denoise Paillou 3FNR 1 Back
CB3FNRB = Checkbutton(cadre,text="3FNR 1B", variable=choix_3FNRB,command=commande_3FNRB,onvalue = 1, offvalue = 0)
CB3FNRB.place(anchor="w",x=1505+delta_s, y=525)

echelle330 = Scale (cadre, from_ = 0.2, to = 0.8, command= choix_val_3FNR_Thres, orient=HORIZONTAL, length = 110, width = 7, resolution = 0.05, label="",showvalue=1,tickinterval=0,sliderlength=20)
echelle330.set(val_3FNR_Thres)
echelle330.place(anchor="w", x=1565+delta_s,y=520)

# Choix filtre Denoise Paillou 3FNR 2 Front
CB3FNR2F = Checkbutton(cadre,text="3FNR 2F", variable=choix_3FNR2,command=commande_3FNR2,onvalue = 1, offvalue = 0)
CB3FNR2F.place(anchor="w",x=1705+delta_s, y=525)

# Choix filtre Denoise Paillou 3FNR Back
CB3FNR2B = Checkbutton(cadre,text="3FNR 2B", variable=choix_3FNR2B,command=commande_3FNR2B,onvalue = 1, offvalue = 0)
CB3FNR2B.place(anchor="w",x=1760+delta_s, y=525)

# Choix filtre Denoise Paillou AANRF 2 Front
CBEPFS = Checkbutton(cadre,text="AANRF", variable=choix_AANR,command=commande_AANR,onvalue = 1, offvalue = 0)
CBEPFS.place(anchor="w",x=1450+delta_s, y=560)

# AANR Mode
RBAADP1 = Radiobutton(cadre,text="H", variable=choix_dyn_AADP,command=choix_dyn_high,value=1)
RBAADP1.place(anchor="w",x=1500+delta_s, y=560)
RBAADP2 = Radiobutton(cadre,text="L", variable=choix_dyn_AADP,command=choix_dyn_low,value=2)
RBAADP2.place(anchor="w",x=1530+delta_s, y=560)

# AANR ghost reducer
CBGR = Checkbutton(cadre,text="GR", variable=choix_ghost_reducer,command=commande_ghost_reducer,onvalue = 1, offvalue = 0)
CBGR.place(anchor="w",x=1531+30+delta_s, y=560)
echelle130 = Scale (cadre, from_ = 20, to = 70, command= choix_val_ghost_reducer, orient=HORIZONTAL, length = 130, width = 7, resolution = 2, label="",showvalue=1,tickinterval=10,sliderlength=20)
echelle130.set(val_ghost_reducer)
echelle130.place(anchor="w", x=1600+delta_s,y=560)

# Choix filtre Denoise Paillou AANRF Back
CBEPFSB = Checkbutton(cadre,text="AANRB", variable=choix_AANRB,command=commande_AANRB,onvalue = 1, offvalue = 0)
CBEPFSB.place(anchor="w",x=1750+delta_s, y=560)

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
'''

def create_histogram_widgets():
    return '''
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
'''

def create_capture_widgets():
    return '''
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
'''

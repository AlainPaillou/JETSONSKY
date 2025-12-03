"""
GUI Widgets - Direct widget creation functions.

This module provides functions to create Tkinter widgets directly,
receiving a globals dictionary for accessing variables and callbacks.

Refactored from exec() string pattern to direct function calls.
Now supports screen scaling for different display sizes.
"""

from tkinter import Checkbutton, Radiobutton, Scale, Label, Button, HORIZONTAL, VERTICAL


def _sx(g, x):
    """Scale x coordinate using globals scale function."""
    scale_x_pos = g.get('scale_x_pos')
    if scale_x_pos:
        return scale_x_pos(x)
    return x

def _sy(g, y):
    """Scale y coordinate using globals scale function."""
    scale_y_pos = g.get('scale_y_pos')
    if scale_y_pos:
        return scale_y_pos(y)
    return y

def _ss(g, size):
    """Scale size using globals scale function."""
    scale_size = g.get('scale_size')
    if scale_size:
        return scale_size(size)
    return size


def create_top_row_widgets(g: dict):
    """
    Create top row widgets (filters, full res, B&W, stacking).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']

    # Filters ON
    CBF = Checkbutton(cadre, text="Filters ON", variable=g['choix_filtrage_ON'],
                      command=g['commande_filtrage_ON'], onvalue=1, offvalue=0)
    CBF.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 50))
    g['CBF'] = CBF

    # Full Res
    CBMFR = Checkbutton(cadre, text="Full Res", variable=g['choix_mode_full_res'],
                        command=g['commande_mode_full_res'], onvalue=1, offvalue=0)
    CBMFR.place(anchor="w", x=_sx(g, 1510)+delta_s, y=_sy(g, 50))
    g['CBMFR'] = CBMFR

    # Set B&W
    CBFNB = Checkbutton(cadre, text="Set B&W", variable=g['choix_noir_blanc'],
                        command=g['commande_noir_blanc'], onvalue=1, offvalue=0)
    CBFNB.place(anchor="w", x=_sx(g, 1570)+delta_s, y=_sy(g, 50))
    g['CBFNB'] = CBFNB

    # B&W Est
    CBFNBE = Checkbutton(cadre, text="B&W Est", variable=g['choix_noir_blanc_estime'],
                         command=g['commande_noir_blanc'], onvalue=1, offvalue=0)
    CBFNBE.place(anchor="w", x=_sx(g, 1630)+delta_s, y=_sy(g, 50))
    g['CBFNBE'] = CBFNBE

    # R-B Rev
    CBRRB = Checkbutton(cadre, text="R-B Rev", variable=g['choix_reverse_RB'],
                        command=g['commande_reverse_RB'], onvalue=1, offvalue=0)
    CBRRB.place(anchor="w", x=_sx(g, 1695)+delta_s, y=_sy(g, 50))
    g['CBRRB'] = CBRRB

    # False Col
    CBFC = Checkbutton(cadre, text="False Col", variable=g['choix_false_colours'],
                       command=g['commande_false_colours'], onvalue=1, offvalue=0)
    CBFC.place(anchor="w", x=_sx(g, 1750)+delta_s, y=_sy(g, 50))
    g['CBFC'] = CBFC

    # Stacking mode (MEAN/SUM)
    RBSM1 = Radiobutton(cadre, text="MEAN", variable=g['choix_stacking'],
                        command=g['choix_mean_stacking'], value=1)
    RBSM1.place(anchor="w", x=_sx(g, 1610)+delta_s, y=_sy(g, 20))
    g['RBSM1'] = RBSM1

    RBSM2 = Radiobutton(cadre, text="SUM", variable=g['choix_stacking'],
                        command=g['choix_sum_stacking'], value=2)
    RBSM2.place(anchor="w", x=_sx(g, 1660)+delta_s, y=_sy(g, 20))
    g['RBSM2'] = RBSM2

    # Frame stacking slider
    echelle20 = Scale(cadre, from_=1, to=5, command=g['choix_FS'], orient=HORIZONTAL,
                      length=_ss(g, 80), width=_ss(g, 7), resolution=1, label="", showvalue=1,
                      tickinterval=1, sliderlength=_ss(g, 20))
    echelle20.set(g['val_FS'])
    echelle20.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 20))
    g['echelle20'] = echelle20


def create_various_widgets(g: dict):
    """
    Create various widgets (USB, flip, filter wheel, debayer, tracking, AI).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']

    ###########################
    #     VARIOUS WIDGETS     #
    ###########################

    # Bandwidth USB
    labelParam50 = Label(cadre, text="USB")
    labelParam50.place(anchor="w", x=_sx(g, 15), y=_sy(g, 255))
    g['labelParam50'] = labelParam50

    echelle50 = Scale(cadre, from_=40, to=100, command=g['choix_USB'], orient=VERTICAL,
                      length=_ss(g, 80), width=_ss(g, 7), resolution=1, label="", showvalue=1,
                      tickinterval=None, sliderlength=_ss(g, 10))
    echelle50.set(g['val_USB'])
    echelle50.place(anchor="c", x=_sx(g, 20), y=_sy(g, 310))
    g['echelle50'] = echelle50

    # Flip Vertical
    CBFV = Checkbutton(cadre, text="FlipV", variable=g['choix_flipV'],
                       command=g['commande_flipV'], onvalue=1, offvalue=0)
    CBFV.place(anchor="w", x=_sx(g, 5), y=_sy(g, 20))
    g['CBFV'] = CBFV

    # Flip Horizontal
    CBFH = Checkbutton(cadre, text="FlipH", variable=g['choix_flipH'],
                       command=g['commande_flipH'], onvalue=1, offvalue=0)
    CBFH.place(anchor="w", x=_sx(g, 5), y=_sy(g, 45))
    g['CBFH'] = CBFH

    # Hot Pixel removal
    CBHOTPIX = Checkbutton(cadre, text="Hot Pix", variable=g['choix_HOTPIX'],
                           command=g['commande_HOTPIX'], onvalue=1, offvalue=0)
    CBHOTPIX.place(anchor="w", x=_sx(g, 5), y=_sy(g, 180))
    g['CBHOTPIX'] = CBHOTPIX

    # Filter Wheel position
    labelFW = Label(cadre, text="FW :")
    labelFW.place(anchor="w", x=_sx(g, 5), y=_sy(g, 400))
    g['labelFW'] = labelFW

    RBEFW1 = Radiobutton(cadre, text="#1", variable=g['fw_position_'],
                         command=g['choix_position_EFW0'], value=0)
    RBEFW1.place(anchor="w", x=_sx(g, 5), y=_sy(g, 420))
    g['RBEFW1'] = RBEFW1

    RBEFW2 = Radiobutton(cadre, text="#2", variable=g['fw_position_'],
                         command=g['choix_position_EFW1'], value=1)
    RBEFW2.place(anchor="w", x=_sx(g, 5), y=_sy(g, 440))
    g['RBEFW2'] = RBEFW2

    RBEFW3 = Radiobutton(cadre, text="#3", variable=g['fw_position_'],
                         command=g['choix_position_EFW2'], value=2)
    RBEFW3.place(anchor="w", x=_sx(g, 5), y=_sy(g, 460))
    g['RBEFW3'] = RBEFW3

    RBEFW4 = Radiobutton(cadre, text="#4", variable=g['fw_position_'],
                         command=g['choix_position_EFW3'], value=3)
    RBEFW4.place(anchor="w", x=_sx(g, 5), y=_sy(g, 480))
    g['RBEFW4'] = RBEFW4

    RBEFW5 = Radiobutton(cadre, text="#5", variable=g['fw_position_'],
                         command=g['choix_position_EFW4'], value=4)
    RBEFW5.place(anchor="w", x=_sx(g, 5), y=_sy(g, 500))
    g['RBEFW5'] = RBEFW5

    # Bayer Matrix
    labelBM = Label(cadre, text="Debayer :")
    labelBM.place(anchor="w", x=_sx(g, 5), y=_sy(g, 525))
    g['labelBM'] = labelBM

    RBEBM1 = Radiobutton(cadre, text="RAW", variable=g['bayer_sensor'],
                         command=g['choix_bayer_RAW'], value=1)
    RBEBM1.place(anchor="w", x=_sx(g, 5), y=_sy(g, 545))
    g['RBEBM1'] = RBEBM1

    RBEBM2 = Radiobutton(cadre, text="RGGB", variable=g['bayer_sensor'],
                         command=g['choix_bayer_RGGB'], value=2)
    RBEBM2.place(anchor="w", x=_sx(g, 5), y=_sy(g, 565))
    g['RBEBM2'] = RBEBM2

    RBEBM3 = Radiobutton(cadre, text="BGGR", variable=g['bayer_sensor'],
                         command=g['choix_bayer_BGGR'], value=3)
    RBEBM3.place(anchor="w", x=_sx(g, 5), y=_sy(g, 585))
    g['RBEBM3'] = RBEBM3

    RBEBM4 = Radiobutton(cadre, text="GRBG", variable=g['bayer_sensor'],
                         command=g['choix_bayer_GRBG'], value=4)
    RBEBM4.place(anchor="w", x=_sx(g, 5), y=_sy(g, 605))
    g['RBEBM4'] = RBEBM4

    RBEBM5 = Radiobutton(cadre, text="GBRG", variable=g['bayer_sensor'],
                         command=g['choix_bayer_GBRG'], value=5)
    RBEBM5.place(anchor="w", x=_sx(g, 5), y=_sy(g, 625))
    g['RBEBM5'] = RBEBM5

    labelTRK = Label(cadre, text="Tracking :")
    labelTRK.place(anchor="w", x=_sx(g, 0), y=_sy(g, 650))
    g['labelTRK'] = labelTRK

    # Find Stars
    CBDTCSTARS = Checkbutton(cadre, text="Stars", variable=g['choix_DETECT_STARS'],
                             command=g['commande_DETECT_STARS'], onvalue=1, offvalue=0)
    CBDTCSTARS.place(anchor="w", x=_sx(g, 0), y=_sy(g, 670))
    g['CBDTCSTARS'] = CBDTCSTARS

    # Track Satellites
    CBTRKSAT = Checkbutton(cadre, text="Sat Detect", variable=g['choix_TRKSAT'],
                           command=g['commande_TRKSAT'], onvalue=1, offvalue=0)
    CBTRKSAT.place(anchor="w", x=_sx(g, 0), y=_sy(g, 690))
    g['CBTRKSAT'] = CBTRKSAT

    # Remove Satellites
    CBREMSAT = Checkbutton(cadre, text="Sat Remov", variable=g['choix_REMSAT'],
                           command=g['commande_REMSAT'], onvalue=1, offvalue=0)
    CBREMSAT.place(anchor="w", x=_sx(g, 0), y=_sy(g, 710))
    g['CBREMSAT'] = CBREMSAT

    # Track Meteor
    CBTRIGGER = Checkbutton(cadre, text="Trigger", variable=g['choix_TRIGGER'],
                            command=g['commande_TRIGGER'], onvalue=1, offvalue=0)
    CBTRIGGER.place(anchor="w", x=_sx(g, 0), y=_sy(g, 730))
    g['CBTRIGGER'] = CBTRIGGER

    # Image reconstruction
    CBCONST = Checkbutton(cadre, text="Reconst", variable=g['choix_CONST'],
                          command=g['commande_CONST'], onvalue=1, offvalue=0)
    CBCONST.place(anchor="w", x=_sx(g, 0), y=_sy(g, 750))
    g['CBCONST'] = CBCONST

    # Artificial intelligence object detection
    labelAI = Label(cadre, text=" A I detect:")
    labelAI.place(anchor="w", x=_sx(g, 0), y=_sy(g, 780))
    g['labelAI'] = labelAI

    CBAICTR = Checkbutton(cadre, text="Craters", variable=g['choix_AI_Craters'],
                          command=g['commande_AI_Craters'], onvalue=1, offvalue=0)
    CBAICTR.place(anchor="w", x=_sx(g, 0), y=_sy(g, 800))
    g['CBAICTR'] = CBAICTR

    CBAISAT = Checkbutton(cadre, text="Satellites", variable=g['choix_AI_Satellites'],
                          command=g['commande_AI_Satellites'], onvalue=1, offvalue=0)
    CBAISAT.place(anchor="w", x=_sx(g, 0), y=_sy(g, 820))
    g['CBAISAT'] = CBAISAT

    CBAITRC = Checkbutton(cadre, text="Tracking", variable=g['choix_AI_Trace'],
                          command=g['commande_AI_Trace'], onvalue=1, offvalue=0)
    CBAITRC.place(anchor="w", x=_sx(g, 0), y=_sy(g, 840))
    g['CBAITRC'] = CBAITRC

    # Capture reference image
    Button7 = Button(cadre, text="Ref Img Cap", command=g['Capture_Ref_Img'],
                     padx=_ss(g, 10), pady=0)
    Button7.place(anchor="w", x=_sx(g, 0), y=_sy(g, 870))
    g['Button7'] = Button7

    CBSUBIMREF = Checkbutton(cadre, text="Sub Img Ref", variable=g['choix_sub_img_ref'],
                             command=g['commande_sub_img_ref'], onvalue=1, offvalue=0)
    CBSUBIMREF.place(anchor="w", x=_sx(g, 0), y=_sy(g, 895))
    g['CBSUBIMREF'] = CBSUBIMREF

    CBBLURIMREF = Checkbutton(cadre, text="Blur Res", variable=g['choix_Blur_img_ref'],
                              command=g['commande_Blur_img_ref'], onvalue=1, offvalue=0)
    CBBLURIMREF.place(anchor="w", x=_sx(g, 0), y=_sy(g, 920))
    g['CBBLURIMREF'] = CBBLURIMREF

    # Set 16 bits threshold
    labelParam804 = Label(cadre, text="16bit Th")
    labelParam804.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 20))
    g['labelParam804'] = labelParam804

    echelle804 = Scale(cadre, from_=8, to=16, command=g['choix_TH_16B'], orient=VERTICAL,
                       length=_ss(g, 80), width=7, resolution=0.5, label="", showvalue=1,
                       tickinterval=None, sliderlength=_ss(g, 10))
    echelle804.set(g['TH_16B'])
    echelle804.place(anchor="c", x=_sx(g, 1860)+delta_s, y=_sy(g, 70))
    g['echelle804'] = echelle804

    # Set Gamma value ZWO camera
    labelParam204 = Label(cadre, text="Gamma Cam")
    labelParam204.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 125))
    g['labelParam204'] = labelParam204

    echelle204 = Scale(cadre, from_=0, to=100, command=g['choix_ASI_GAMMA'], orient=VERTICAL,
                       length=_ss(g, 100), width=7, resolution=1, label="", showvalue=1,
                       tickinterval=None, sliderlength=_ss(g, 10))
    echelle204.set(g['ASIGAMMA'])
    echelle204.place(anchor="c", x=_sx(g, 1860)+delta_s, y=_sy(g, 190))
    g['echelle204'] = echelle204

    # Set camera read speed
    labelCamSpeed = Label(cadre, text="Cam Speed")
    labelCamSpeed.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 260))
    g['labelCamSpeed'] = labelCamSpeed

    RBCS1 = Radiobutton(cadre, text="Fast", variable=g['cam_read_speed'],
                        command=g['choix_read_speed_fast'], value=0)
    RBCS1.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 280))
    g['RBCS1'] = RBCS1

    RBCS2 = Radiobutton(cadre, text="Slow", variable=g['cam_read_speed'],
                        command=g['choix_read_speed_slow'], value=1)
    RBCS2.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 300))
    g['RBCS2'] = RBCS2

    # Text in picture
    CBTIP = Checkbutton(cadre, text="TIP", variable=g['choix_TIP'],
                        command=g['commande_TIP'], onvalue=1, offvalue=0)
    CBTIP.place(anchor="w", x=_sx(g, 1840)+delta_s, y=95+240)
    g['CBTIP'] = CBTIP

    # Cross
    CBCR = Checkbutton(cadre, text="Cr", variable=g['choix_cross'],
                       command=g['commande_cross'], onvalue=1, offvalue=0)
    CBCR.place(anchor="w", x=_sx(g, 1840)+delta_s, y=135+240)
    g['CBCR'] = CBCR

    # Histogram
    CBHST = Checkbutton(cadre, text="Hst", variable=g['choix_HST'],
                        command=g['commande_HST'], onvalue=1, offvalue=0)
    CBHST.place(anchor="w", x=_sx(g, 1840)+delta_s, y=165+240)
    g['CBHST'] = CBHST

    # Transfer function display
    CBTRSF = Checkbutton(cadre, text="Trsf", variable=g['choix_TRSF'],
                         command=g['commande_TRSF'], onvalue=1, offvalue=0)
    CBTRSF.place(anchor="w", x=_sx(g, 1840)+delta_s, y=190+240)
    g['CBTRSF'] = CBTRSF

    # Affichage fonction de transfert amplification soft
    CBTRGS = Checkbutton(cadre, text="TrGS", variable=g['choix_TRGS'],
                         command=g['commande_TRGS'], onvalue=1, offvalue=0)
    CBTRGS.place(anchor="w", x=_sx(g, 1840)+delta_s, y=210+240)
    g['CBTRGS'] = CBTRGS

    # Affichage fonction de transfert Contrast Low Light
    CBGBL = Checkbutton(cadre, text="GBL", variable=g['choix_GBL'],
                        command=g['commande_GBL'], onvalue=1, offvalue=0)
    CBGBL.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 495))
    g['CBGBL'] = CBGBL

    # Saturation
    CBSAT = Checkbutton(cadre, text="SAT", variable=g['choix_SAT'],
                        command=g['commande_SAT'], onvalue=1, offvalue=0)
    CBSAT.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 640))
    g['CBSAT'] = CBSAT

    RBSATVI1 = Radiobutton(cadre, text="Vid", variable=g['Sat_Vid_Img'],
                           command=g['choix_SAT_Vid'], value=0)
    RBSATVI1.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 660))
    g['RBSATVI1'] = RBSATVI1

    RBSATVI2 = Radiobutton(cadre, text="Img", variable=g['Sat_Vid_Img'],
                           command=g['choix_SAT_Img'], value=1)
    RBSATVI2.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 680))
    g['RBSATVI2'] = RBSATVI2

    CBSAT2PASS = Checkbutton(cadre, text="2pass", variable=g['choix_SAT2PASS'],
                             command=g['commande_SAT2PASS'], onvalue=1, offvalue=0)
    CBSAT2PASS.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 700))
    g['CBSAT2PASS'] = CBSAT2PASS

    echelle70 = Scale(cadre, from_=0, to=20, command=g['choix_val_SAT'], orient=VERTICAL,
                      length=_ss(g, 150), width=7, resolution=0.01, label="", showvalue=1,
                      tickinterval=None, sliderlength=_ss(g, 10))
    echelle70.set(g['val_SAT'])
    echelle70.place(anchor="c", x=_sx(g, 1855)+delta_s, y=_sy(g, 800))
    g['echelle70'] = echelle70

    # DEMO
    CBDEMO = Checkbutton(cadre, text="Demo", variable=g['choix_DEMO'],
                         command=g['commande_DEMO'], onvalue=1, offvalue=0)
    CBDEMO.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 960))
    g['CBDEMO'] = CBDEMO

    RBDEML = Radiobutton(cadre, text="Left", variable=g['demo_side'],
                         command=g['choix_demo_left'], value=0)
    RBDEML.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 985))
    g['RBDEML'] = RBDEML

    RBDEMR = Radiobutton(cadre, text="Right", variable=g['demo_side'],
                         command=g['choix_demo_right'], value=1)
    RBDEMR.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 1010))
    g['RBDEMR'] = RBDEMR


def create_exposition_widgets(g: dict):
    """
    Create exposition settings widgets (speed mode, HDR, exposure, gain, binning).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']

    ###########################
    #   EXPOSITION SETTINGS   #
    ###########################

    # Speed mode acquisition
    labelMode_Acq = Label(cadre, text="Speed")
    labelMode_Acq.place(anchor="w", x=_sx(g, 1430)+delta_s, y=_sy(g, 240))
    g['labelMode_Acq'] = labelMode_Acq

    RBMA1 = Radiobutton(cadre, text="Fast", variable=g['mode_acq'],
                        command=g['mode_acq_rapide'], value=1)
    RBMA1.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 240))
    g['RBMA1'] = RBMA1

    RBMA2 = Radiobutton(cadre, text="MedF", variable=g['mode_acq'],
                        command=g['mode_acq_mediumF'], value=2)
    RBMA2.place(anchor="w", x=_sx(g, 1505)+delta_s, y=_sy(g, 240))
    g['RBMA2'] = RBMA2

    RBMA3 = Radiobutton(cadre, text="MedS", variable=g['mode_acq'],
                        command=g['mode_acq_mediumS'], value=3)
    RBMA3.place(anchor="w", x=_sx(g, 1550)+delta_s, y=_sy(g, 240))
    g['RBMA3'] = RBMA3

    RBMA4 = Radiobutton(cadre, text="Slow", variable=g['mode_acq'],
                        command=g['mode_acq_lente'], value=4)
    RBMA4.place(anchor="w", x=_sx(g, 1595)+delta_s, y=_sy(g, 240))
    g['RBMA4'] = RBMA4

    # Choix HDR
    CBHDR = Checkbutton(cadre, text="HDR", variable=g['choix_HDR'],
                        command=g['commande_HDR'], onvalue=1, offvalue=0)
    CBHDR.place(anchor="w", x=_sx(g, 1650)+delta_s, y=_sy(g, 240))
    g['CBHDR'] = CBHDR

    RBHDR1 = Radiobutton(cadre, text="Mertens", variable=g['mode_HDR_select'],
                         command=g['HDR_Mertens'], value=1)
    RBHDR1.place(anchor="w", x=_sx(g, 1690)+delta_s, y=_sy(g, 240))
    g['RBHDR1'] = RBHDR1

    RBHDR2 = Radiobutton(cadre, text="Median", variable=g['mode_HDR_select'],
                         command=g['HDR_Median'], value=2)
    RBHDR2.place(anchor="w", x=_sx(g, 1745)+delta_s, y=_sy(g, 240))
    g['RBHDR2'] = RBHDR2

    RBHDR3 = Radiobutton(cadre, text="Mean", variable=g['mode_HDR_select'],
                         command=g['HDR_Mean'], value=3)
    RBHDR3.place(anchor="w", x=_sx(g, 1795)+delta_s, y=_sy(g, 240))
    g['RBHDR3'] = RBHDR3

    # Automatic exposition time
    CBOE = Checkbutton(cadre, text="AE", variable=g['choix_autoexposure'],
                       command=g['commande_autoexposure'], onvalue=1, offvalue=0)
    CBOE.place(anchor="w", x=_sx(g, 1440)+delta_s, y=g['yS1'])
    g['CBOE'] = CBOE

    # Exposition setting
    echelle1 = Scale(cadre, from_=g['exp_min'], to=g['exp_max'], command=g['valeur_exposition'],
                     orient=HORIZONTAL, length=_ss(g, 330), width=7, resolution=g['exp_delta'],
                     label="", showvalue=1, tickinterval=g['exp_interval'], sliderlength=_ss(g, 20))
    echelle1.set(g['val_exposition'])
    echelle1.place(anchor="w", x=g['xS1']+delta_s, y=g['yS1'])
    g['echelle1'] = echelle1

    # Choix du mode BINNING - 1, 2 ou 3
    labelBIN = Label(cadre, text="BIN : ")
    labelBIN.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 80))
    g['labelBIN'] = labelBIN

    RBB1 = Radiobutton(cadre, text="BIN1", variable=g['choix_bin'],
                       command=g['choix_BIN1'], value=1)
    RBB1.place(anchor="w", x=_sx(g, 1470)+delta_s, y=_sy(g, 80))
    g['RBB1'] = RBB1

    RBB2 = Radiobutton(cadre, text="BIN2", variable=g['choix_bin'],
                       command=g['choix_BIN2'], value=2)
    RBB2.place(anchor="w", x=_sx(g, 1510)+delta_s, y=_sy(g, 80))
    g['RBB2'] = RBB2

    # Choix Hardware Bin
    CBHB = Checkbutton(cadre, text="HB", variable=g['choix_hard_bin'],
                       command=g['commande_hard_bin'], onvalue=1, offvalue=0)
    CBHB.place(anchor="w", x=_sx(g, 1550)+delta_s, y=_sy(g, 80))
    g['CBHB'] = CBHB

    # Choix 16 bits Low Light
    CBHDR_LL = Checkbutton(cadre, text="16bLL", variable=g['choix_16bLL'],
                           command=g['commande_16bLL'], onvalue=1, offvalue=0)
    CBHDR_LL.place(anchor="w", x=_sx(g, 1600)+delta_s, y=_sy(g, 80))
    g['CBHDR_LL'] = CBHDR_LL

    # Resolution setting
    labelParam3 = Label(cadre, text="RES : ")
    labelParam3.place(anchor="w", x=_sx(g, 1660)+delta_s, y=_sy(g, 80))
    g['labelParam3'] = labelParam3

    echelle3 = Scale(cadre, from_=1, to=9, command=g['choix_resolution_camera'],
                     orient=HORIZONTAL, length=_ss(g, 130), width=7, resolution=1, label="",
                     showvalue=1, tickinterval=1, sliderlength=_ss(g, 20))
    echelle3.set(g['val_resolution'])
    echelle3.place(anchor="w", x=g['xS3']+delta_s, y=g['yS3'])
    g['echelle3'] = echelle3

    # Automatic gain
    CBOG = Checkbutton(cadre, text="Auto Gain", variable=g['choix_autogain'],
                       command=g['commande_autogain'], onvalue=1, offvalue=0)
    CBOG.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 120))
    g['CBOG'] = CBOG

    echelle2 = Scale(cadre, from_=0, to=g['val_maxgain'], command=g['valeur_gain'],
                     orient=HORIZONTAL, length=_ss(g, 320), width=7, resolution=1, label="",
                     showvalue=1, tickinterval=50, sliderlength=_ss(g, 20))
    echelle2.set(g['val_gain'])
    echelle2.place(anchor="w", x=_sx(g, 1500)+delta_s, y=_sy(g, 120))
    g['echelle2'] = echelle2

    # Signal amplification soft
    CBAS = Checkbutton(cadre, text="Amplif Soft", variable=g['choix_AmpSoft'],
                       command=g['commande_AmpSoft'], onvalue=1, offvalue=0)
    CBAS.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 160))
    g['CBAS'] = CBAS

    echelle80 = Scale(cadre, from_=0, to=20.0, command=g['choix_amplif'],
                      orient=HORIZONTAL, length=_ss(g, 280), width=7, resolution=0.1, label="",
                      showvalue=1, tickinterval=1, sliderlength=_ss(g, 20))
    echelle80.set(g['val_ampl'])
    echelle80.place(anchor="w", x=_sx(g, 1540)+delta_s, y=_sy(g, 160))
    g['echelle80'] = echelle80

    RBMuRo1 = Radiobutton(cadre, text="Lin", variable=g['mode_Lin_Gauss'],
                          command=g['mode_Lineaire'], value=1)
    RBMuRo1.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 200))
    g['RBMuRo1'] = RBMuRo1

    RBMuRo2 = Radiobutton(cadre, text="Gauss", variable=g['mode_Lin_Gauss'],
                          command=g['mode_Gauss'], value=2)
    RBMuRo2.place(anchor="w", x=_sx(g, 1480)+delta_s, y=_sy(g, 200))
    g['RBMuRo2'] = RBMuRo2

    RBMuRo3 = Radiobutton(cadre, text="Stars", variable=g['mode_Lin_Gauss'],
                          command=g['mode_Stars'], value=3)
    RBMuRo3.place(anchor="w", x=_sx(g, 1525)+delta_s, y=_sy(g, 200))
    g['RBMuRo3'] = RBMuRo3

    labelParam82 = Label(cadre, text="µX")
    labelParam82.place(anchor="w", x=_sx(g, 1573)+delta_s, y=_sy(g, 200))
    g['labelParam82'] = labelParam82

    echelle82 = Scale(cadre, from_=-5.0, to=5.0, command=g['choix_Mu'],
                      orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=0.1, label="",
                      showvalue=1, tickinterval=2, sliderlength=_ss(g, 20))
    echelle82.set(g['val_Mu'])
    echelle82.place(anchor="w", x=_sx(g, 1593)+delta_s, y=_sy(g, 200))
    g['echelle82'] = echelle82

    labelParam84 = Label(cadre, text="Ro")
    labelParam84.place(anchor="w", x=_sx(g, 1705)+delta_s, y=_sy(g, 200))
    g['labelParam84'] = labelParam84

    echelle84 = Scale(cadre, from_=0.2, to=5.0, command=g['choix_Ro'],
                      orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=0.1, label="",
                      showvalue=1, tickinterval=1, sliderlength=_ss(g, 20))
    echelle84.set(g['val_Ro'])
    echelle84.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 200))
    g['echelle84'] = echelle84

    # Camera Red balance
    labelParam14 = Label(cadre, text="CRed")
    labelParam14.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 305))
    g['labelParam14'] = labelParam14

    echelle14 = Scale(cadre, from_=1, to=99, command=g['choix_w_red'],
                      orient=HORIZONTAL, length=_ss(g, 140), width=7, resolution=1, label="",
                      showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle14.set(g['val_red'])
    echelle14.place(anchor="w", x=_sx(g, 1485)+delta_s, y=_sy(g, 305))
    g['echelle14'] = echelle14

    # Camera Blue balance
    labelParam15 = Label(cadre, text="CBlue")
    labelParam15.place(anchor="w", x=_sx(g, 1645)+delta_s, y=_sy(g, 305))
    g['labelParam15'] = labelParam15

    echelle15 = Scale(cadre, from_=1, to=99, command=g['choix_w_blue'],
                      orient=HORIZONTAL, length=_ss(g, 140), width=7, resolution=1, label="",
                      showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle15.set(g['val_blue'])
    echelle15.place(anchor="w", x=_sx(g, 1680)+delta_s, y=_sy(g, 305))
    g['echelle15'] = echelle15

    # Software Red balance
    labelParam100 = Label(cadre, text="R")
    labelParam100.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 340))
    g['labelParam100_rgb'] = labelParam100

    echelle100 = Scale(cadre, from_=0, to=2, command=g['choix_w_reds'],
                       orient=HORIZONTAL, length=_ss(g, 360), width=7, resolution=0.005, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle100.set(g['val_reds'])
    echelle100.place(anchor="w", x=_sx(g, 1440)+delta_s+15, y=_sy(g, 340))
    g['echelle100'] = echelle100

    # Software Green balance
    labelParam101 = Label(cadre, text="G")
    labelParam101.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 370))
    g['labelParam101'] = labelParam101

    echelle101 = Scale(cadre, from_=0, to=2, command=g['choix_w_greens'],
                       orient=HORIZONTAL, length=_ss(g, 360), width=7, resolution=0.005, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle101.set(g['val_greens'])
    echelle101.place(anchor="w", x=_sx(g, 1440)+delta_s+15, y=_sy(g, 370))
    g['echelle101'] = echelle101

    # Software Blue balance
    labelParam102 = Label(cadre, text="B")
    labelParam102.place(anchor="w", x=_sx(g, 1440)+delta_s, y=_sy(g, 400))
    g['labelParam102'] = labelParam102

    echelle102 = Scale(cadre, from_=0, to=2, command=g['choix_w_blues'],
                       orient=HORIZONTAL, length=_ss(g, 360), width=7, resolution=0.005, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle102.set(g['val_blues'])
    echelle102.place(anchor="w", x=_sx(g, 1440)+delta_s+15, y=_sy(g, 400))
    g['echelle102'] = echelle102


def create_sharpen_denoise_widgets(g: dict):
    """
    Create sharpen and denoise widgets (sharpen, NR P1/P2, KNN, NLM2, 3FNR, AANR).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']

    ###########################
    # SHARPEN DENOISE WIDGETS #
    ###########################

    # Choix Sharpen 1 & 2
    CBSS1 = Checkbutton(cadre, text="Sharpen 1  Val/Sigma", variable=g['choix_sharpen_soft1'],
                        command=g['commande_sharpen_soft1'], onvalue=1, offvalue=0)
    CBSS1.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 435))
    g['CBSS1'] = CBSS1

    echelle152 = Scale(cadre, from_=0, to=10, command=g['choix_val_sharpen'],
                       orient=HORIZONTAL, length=_ss(g, 120), width=7, resolution=0.2, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle152.set(g['val_sharpen'])
    echelle152.place(anchor="w", x=_sx(g, 1560)+delta_s, y=_sy(g, 435))
    g['echelle152'] = echelle152

    echelle153 = Scale(cadre, from_=1, to=9, command=g['choix_val_sigma_sharpen'],
                       orient=HORIZONTAL, length=_ss(g, 120), width=7, resolution=1, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle153.set(g['val_sigma_sharpen'])
    echelle153.place(anchor="w", x=_sx(g, 1690)+delta_s, y=_sy(g, 435))
    g['echelle153'] = echelle153

    CBSS2 = Checkbutton(cadre, text="Sharpen 2  Val/Sigma", variable=g['choix_sharpen_soft2'],
                        command=g['commande_sharpen_soft2'], onvalue=1, offvalue=0)
    CBSS2.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 460))
    g['CBSS2'] = CBSS2

    echelle154 = Scale(cadre, from_=0, to=10, command=g['choix_val_sharpen2'],
                       orient=HORIZONTAL, length=_ss(g, 120), width=7, resolution=0.2, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle154.set(g['val_sharpen2'])
    echelle154.place(anchor="w", x=_sx(g, 1560)+delta_s, y=_sy(g, 460))
    g['echelle154'] = echelle154

    echelle155 = Scale(cadre, from_=1, to=9, command=g['choix_val_sigma_sharpen2'],
                       orient=HORIZONTAL, length=_ss(g, 120), width=7, resolution=1, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle155.set(g['val_sigma_sharpen2'])
    echelle155.place(anchor="w", x=_sx(g, 1690)+delta_s, y=_sy(g, 460))
    g['echelle155'] = echelle155

    # Choix filtre Denoise Paillou image
    CBEPF = Checkbutton(cadre, text="NR P1", variable=g['choix_denoise_Paillou'],
                        command=g['commande_denoise_Paillou'], onvalue=1, offvalue=0)
    CBEPF.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 495))
    g['CBEPF'] = CBEPF

    # Choix filtre Denoise Paillou 2 image
    CBEPF2 = Checkbutton(cadre, text="NR P2", variable=g['choix_denoise_Paillou2'],
                         command=g['commande_denoise_Paillou2'], onvalue=1, offvalue=0)
    CBEPF2.place(anchor="w", x=_sx(g, 1500)+delta_s, y=_sy(g, 495))
    g['CBEPF2'] = CBEPF2

    # Choix filtre Denoise KNN
    CBEPF_KNN = Checkbutton(cadre, text="KNN", variable=g['choix_denoise_KNN'],
                            command=g['choix_KNN'], onvalue=1, offvalue=0)
    CBEPF_KNN.place(anchor="w", x=_sx(g, 1550)+delta_s, y=_sy(g, 495))
    g['CBEPF_KNN'] = CBEPF_KNN

    echelle30 = Scale(cadre, from_=0.05, to=1.2, command=g['choix_val_KNN'],
                      orient=HORIZONTAL, length=_ss(g, 70), width=7, resolution=0.1, label="",
                      showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle30.set(g['val_denoise_KNN'])
    echelle30.place(anchor="w", x=_sx(g, 1600)+delta_s, y=_sy(g, 490))
    g['echelle30'] = echelle30

    # Choix filtre Denoise NLM2
    CBDS = Checkbutton(cadre, text="NLM2", variable=g['choix_NLM2'],
                       command=g['commande_NLM2'], onvalue=1, offvalue=0)
    CBDS.place(anchor="w", x=_sx(g, 1690)+delta_s, y=_sy(g, 495))
    g['CBDS'] = CBDS

    echelle4 = Scale(cadre, from_=0.1, to=1.2, command=g['choix_valeur_denoise'],
                     orient=HORIZONTAL, length=_ss(g, 70), width=7, resolution=0.1, label="",
                     showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle4.set(g['val_denoise'])
    echelle4.place(anchor="w", x=_sx(g, 1740)+delta_s, y=_sy(g, 490))
    g['echelle4'] = echelle4

    # Choix filtre Denoise Paillou 3FNR 1 Front
    CB3FNRF = Checkbutton(cadre, text="3FNR 1F", variable=g['choix_3FNR'],
                          command=g['commande_3FNR'], onvalue=1, offvalue=0)
    CB3FNRF.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 525))
    g['CB3FNRF'] = CB3FNRF

    # Choix filtre Denoise Paillou 3FNR 1 Back
    CB3FNRB = Checkbutton(cadre, text="3FNR 1B", variable=g['choix_3FNRB'],
                          command=g['commande_3FNRB'], onvalue=1, offvalue=0)
    CB3FNRB.place(anchor="w", x=_sx(g, 1505)+delta_s, y=_sy(g, 525))
    g['CB3FNRB'] = CB3FNRB

    echelle330 = Scale(cadre, from_=0.2, to=0.8, command=g['choix_val_3FNR_Thres'],
                       orient=HORIZONTAL, length=_ss(g, 110), width=7, resolution=0.05, label="",
                       showvalue=1, tickinterval=0, sliderlength=_ss(g, 20))
    echelle330.set(g['val_3FNR_Thres'])
    echelle330.place(anchor="w", x=_sx(g, 1565)+delta_s, y=_sy(g, 520))
    g['echelle330'] = echelle330

    # Choix filtre Denoise Paillou 3FNR 2 Front
    CB3FNR2F = Checkbutton(cadre, text="3FNR 2F", variable=g['choix_3FNR2'],
                           command=g['commande_3FNR2'], onvalue=1, offvalue=0)
    CB3FNR2F.place(anchor="w", x=_sx(g, 1705)+delta_s, y=_sy(g, 525))
    g['CB3FNR2F'] = CB3FNR2F

    # Choix filtre Denoise Paillou 3FNR Back
    CB3FNR2B = Checkbutton(cadre, text="3FNR 2B", variable=g['choix_3FNR2B'],
                           command=g['commande_3FNR2B'], onvalue=1, offvalue=0)
    CB3FNR2B.place(anchor="w", x=_sx(g, 1760)+delta_s, y=_sy(g, 525))
    g['CB3FNR2B'] = CB3FNR2B

    # Choix filtre Denoise Paillou AANRF 2 Front
    CBEPFS = Checkbutton(cadre, text="AANRF", variable=g['choix_AANR'],
                         command=g['commande_AANR'], onvalue=1, offvalue=0)
    CBEPFS.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 560))
    g['CBEPFS'] = CBEPFS

    # AANR Mode
    RBAADP1 = Radiobutton(cadre, text="H", variable=g['choix_dyn_AADP'],
                          command=g['choix_dyn_high'], value=1)
    RBAADP1.place(anchor="w", x=_sx(g, 1500)+delta_s, y=_sy(g, 560))
    g['RBAADP1'] = RBAADP1

    RBAADP2 = Radiobutton(cadre, text="L", variable=g['choix_dyn_AADP'],
                          command=g['choix_dyn_low'], value=2)
    RBAADP2.place(anchor="w", x=_sx(g, 1530)+delta_s, y=_sy(g, 560))
    g['RBAADP2'] = RBAADP2

    # AANR ghost reducer
    CBGR_AANR = Checkbutton(cadre, text="GR", variable=g['choix_ghost_reducer'],
                            command=g['commande_ghost_reducer'], onvalue=1, offvalue=0)
    CBGR_AANR.place(anchor="w", x=_sx(g, 1531)+30+delta_s, y=_sy(g, 560))
    g['CBGR_AANR'] = CBGR_AANR

    echelle130 = Scale(cadre, from_=20, to=70, command=g['choix_val_ghost_reducer'],
                       orient=HORIZONTAL, length=_ss(g, 130), width=7, resolution=2, label="",
                       showvalue=1, tickinterval=10, sliderlength=_ss(g, 20))
    echelle130.set(g['val_ghost_reducer'])
    echelle130.place(anchor="w", x=_sx(g, 1600)+delta_s, y=_sy(g, 560))
    g['echelle130'] = echelle130

    # Choix filtre Denoise Paillou AANRF Back
    CBEPFSB = Checkbutton(cadre, text="AANRB", variable=g['choix_AANRB'],
                          command=g['commande_AANRB'], onvalue=1, offvalue=0)
    CBEPFSB.place(anchor="w", x=_sx(g, 1750)+delta_s, y=_sy(g, 560))
    g['CBEPFSB'] = CBEPFSB

    # Stabilization
    CBSTAB = Checkbutton(cadre, text="STAB", variable=g['choix_STAB'],
                         command=g['commande_STAB'], onvalue=1, offvalue=0)
    CBSTAB.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 525))
    g['CBSTAB'] = CBSTAB

    # Image Quality Estimate
    CBIMQE = Checkbutton(cadre, text="IQE", variable=g['choix_IMQE'],
                         command=g['commande_IMQE'], onvalue=1, offvalue=0)
    CBIMQE.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 565))
    g['CBIMQE'] = CBIMQE

    # BFR Bad Frame Remove
    CBBFR = Checkbutton(cadre, text="RmBF", variable=g['choix_BFR'],
                        command=g['commande_BFR'], onvalue=1, offvalue=0)
    CBBFR.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 600))
    g['CBBFR'] = CBBFR

    echelle300 = Scale(cadre, from_=0, to=100, command=g['choix_val_BFR'],
                       orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=5, label="",
                       showvalue=1, tickinterval=20, sliderlength=_ss(g, 20))
    echelle300.set(g['val_BFR'])
    echelle300.place(anchor="w", x=_sx(g, 1500)+delta_s, y=_sy(g, 600))
    g['echelle300'] = echelle300

    # Choix 2 frames variation Reduction Filter for turbulence pre treatment
    CBRV = Checkbutton(cadre, text="VAR", variable=g['choix_reduce_variation'],
                       command=g['commande_reduce_variation'], onvalue=1, offvalue=0)
    CBRV.place(anchor="w", x=_sx(g, 1615)+delta_s, y=_sy(g, 600))
    g['CBRV'] = CBRV

    RBRVBPF1 = Radiobutton(cadre, text="BF", variable=g['choix_BFReference'],
                           command=g['command_BFReference'], value=1)  # Best frame reference
    RBRVBPF1.place(anchor="w", x=_sx(g, 1655)+delta_s, y=_sy(g, 600))
    g['RBRVBPF1'] = RBRVBPF1

    RBRVBPF2 = Radiobutton(cadre, text="PF", variable=g['choix_BFReference'],
                           command=g['command_PFReference'], value=2)  # Previous frame reference
    RBRVBPF2.place(anchor="w", x=_sx(g, 1688)+delta_s, y=_sy(g, 600))
    g['RBRVBPF2'] = RBRVBPF2

    echelle270 = Scale(cadre, from_=0.5, to=3, command=g['choix_val_reduce_variation'],
                       orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=0.1, label="",
                       showvalue=1, tickinterval=2, sliderlength=_ss(g, 20))
    echelle270.set(g['val_reduce_variation'])
    echelle270.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 600))
    g['echelle270'] = echelle270

    # Choix 2 frames variation Reduction Filter for turbulence post treatment
    CBRVPT = Checkbutton(cadre, text="VARPT", variable=g['choix_reduce_variation_post_treatment'],
                         command=g['commande_reduce_variation_post_treatment'], onvalue=1, offvalue=0)
    CBRVPT.place(anchor="w", x=_sx(g, 1840)+delta_s, y=_sy(g, 600))
    g['CBRVPT'] = CBRVPT


def create_histogram_widgets(g: dict):
    """
    Create histogram widgets (gradient, gamma, histogram stretch, sigmoide, CLAHE, CLL).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']

    #####################
    # HISTOGRAM WIDGETS #
    #####################

    # Choix filtre Gradient Removal
    CBGR = Checkbutton(cadre, text="Grad/Vignet", variable=g['choix_GR'],
                       command=g['commande_GR'], onvalue=1, offvalue=0)
    CBGR.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 640))
    g['CBGR'] = CBGR

    # Choix du mode gradient ou vignetting
    RBGV1 = Radiobutton(cadre, text="Gr", variable=g['gradient_vignetting'],
                        command=g['mode_gradient'], value=1)
    RBGV1.place(anchor="w", x=_sx(g, 1530)+delta_s, y=_sy(g, 640))
    g['RBGV1'] = RBGV1

    RBGV2 = Radiobutton(cadre, text="Vig", variable=g['gradient_vignetting'],
                        command=g['mode_vignetting'], value=2)
    RBGV2.place(anchor="w", x=_sx(g, 1560)+delta_s, y=_sy(g, 640))
    g['RBGV2'] = RBGV2

    # Choix Parametre Seuil Gradient Removal
    echelle60 = Scale(cadre, from_=0, to=100, command=g['choix_SGR'],
                      orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=1, label="",
                      showvalue=1, tickinterval=20, sliderlength=_ss(g, 20))
    echelle60.set(g['val_SGR'])
    echelle60.place(anchor="w", x=_sx(g, 1600)+delta_s, y=_sy(g, 640))
    g['echelle60'] = echelle60

    # Choix Parametre Atenuation Gradient Removal
    labelParam61 = Label(cadre, text="At")
    labelParam61.place(anchor="e", x=_sx(g, 1735)+delta_s, y=_sy(g, 640))
    g['labelParam61'] = labelParam61

    echelle61 = Scale(cadre, from_=0, to=100, command=g['choix_AGR'],
                      orient=HORIZONTAL, length=_ss(g, 80), width=7, resolution=1, label="",
                      showvalue=1, tickinterval=25, sliderlength=_ss(g, 20))
    echelle61.set(g['val_AGR'])
    echelle61.place(anchor="w", x=_sx(g, 1740)+delta_s, y=_sy(g, 640))
    g['echelle61'] = echelle61

    # Choix du mode image en négatif
    CBIN = Checkbutton(cadre, text="Img Neg", variable=g['choix_img_Neg'],
                       command=g['commande_img_Neg'], onvalue=1, offvalue=0)
    CBIN.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 680))
    g['CBIN'] = CBIN

    # Histogram equalize (Gamma)
    CBHE2 = Checkbutton(cadre, text="Gamma", variable=g['choix_histogram_equalize2'],
                        command=g['commande_histogram_equalize2'], onvalue=1, offvalue=0)
    CBHE2.place(anchor="w", x=_sx(g, 1520)+delta_s, y=_sy(g, 680))
    g['CBHE2'] = CBHE2

    echelle16 = Scale(cadre, from_=0.1, to=4, command=g['choix_heq2'],
                      orient=HORIZONTAL, length=_ss(g, 240), width=7, resolution=0.05, label="",
                      showvalue=1, tickinterval=0.5, sliderlength=_ss(g, 20))
    echelle16.set(g['val_heq2'])
    echelle16.place(anchor="w", x=_sx(g, 1580)+delta_s, y=_sy(g, 680))
    g['echelle16'] = echelle16

    # Choix histogramme stretch
    CBHS = Checkbutton(cadre, text="Histo Stretch", variable=g['choix_histogram_stretch'],
                       command=g['commande_histogram_stretch'], onvalue=1, offvalue=0)
    CBHS.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 720))
    g['CBHS'] = CBHS

    labelParam5 = Label(cadre, text="Min")  # choix valeur histogramme strech minimum
    labelParam5.place(anchor="w", x=_sx(g, 1555)+delta_s, y=_sy(g, 720))
    g['labelParam5'] = labelParam5

    echelle5 = Scale(cadre, from_=0, to=150, command=g['choix_histo_min'],
                     orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=1, label="",
                     showvalue=1, tickinterval=50, sliderlength=_ss(g, 20))
    echelle5.set(g['val_histo_min'])
    echelle5.place(anchor="w", x=_sx(g, 1580)+delta_s, y=_sy(g, 720))
    g['echelle5'] = echelle5

    labelParam6 = Label(cadre, text="Max")  # choix valeur histogramme strech maximum
    labelParam6.place(anchor="w", x=_sx(g, 1700)+delta_s, y=_sy(g, 720))
    g['labelParam6'] = labelParam6

    echelle6 = Scale(cadre, from_=155, to=255, command=g['choix_histo_max'],
                     orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=1, label="",
                     showvalue=1, tickinterval=25, sliderlength=_ss(g, 20))
    echelle6.set(g['val_histo_max'])
    echelle6.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 720))
    g['echelle6'] = echelle6

    # Choix histogramme Sigmoide
    CBHPT = Checkbutton(cadre, text="Histo Sigmoide", variable=g['choix_histogram_phitheta'],
                        command=g['commande_histogram_phitheta'], onvalue=1, offvalue=0)
    CBHPT.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 760))
    g['CBHPT'] = CBHPT

    labelParam12 = Label(cadre, text="Pnt")  # choix valeur histogramme Signoide param 1
    labelParam12.place(anchor="w", x=_sx(g, 1555)+delta_s, y=_sy(g, 760))
    g['labelParam12'] = labelParam12

    echelle12 = Scale(cadre, from_=0.5, to=3, command=g['choix_phi'],
                      orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=0.1, label="",
                      showvalue=1, tickinterval=0.5, sliderlength=_ss(g, 20))
    echelle12.set(g['val_phi'])
    echelle12.place(anchor="w", x=_sx(g, 1580)+delta_s, y=_sy(g, 760))
    g['echelle12'] = echelle12

    labelParam13 = Label(cadre, text="Dec")  # choix valeur histogramme Signoide param 2
    labelParam13.place(anchor="w", x=_sx(g, 1700)+delta_s, y=_sy(g, 760))
    g['labelParam13'] = labelParam13

    echelle13 = Scale(cadre, from_=50, to=200, command=g['choix_theta'],
                      orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=2, label="",
                      showvalue=1, tickinterval=50, sliderlength=_ss(g, 20))
    echelle13.set(g['val_theta'])
    echelle13.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 760))
    g['echelle13'] = echelle13

    # Choix contrast CLAHE
    CBCC = Checkbutton(cadre, text="Contrast", variable=g['choix_contrast_CLAHE'],
                       command=g['commande_contrast_CLAHE'], onvalue=1, offvalue=0)
    CBCC.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 800))
    g['CBCC'] = CBCC

    labelParam9 = Label(cadre, text="Clip")  # choix valeur contrate CLAHE
    labelParam9.place(anchor="w", x=_sx(g, 1555)+delta_s, y=_sy(g, 800))
    g['labelParam9'] = labelParam9

    echelle9 = Scale(cadre, from_=0.1, to=4, command=g['choix_valeur_CLAHE'],
                     orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=0.1, label="",
                     showvalue=1, tickinterval=1, sliderlength=_ss(g, 20))
    echelle9.set(g['val_contrast_CLAHE'])
    echelle9.place(anchor="w", x=_sx(g, 1580)+delta_s, y=_sy(g, 800))
    g['echelle9'] = echelle9

    labelParam109 = Label(cadre, text="Grid")  # choix valeur contrate CLAHE
    labelParam109.place(anchor="w", x=_sx(g, 1700)+delta_s, y=_sy(g, 800))
    g['labelParam109'] = labelParam109

    echelle109 = Scale(cadre, from_=4, to=24, command=g['choix_grid_CLAHE'],
                       orient=HORIZONTAL, length=_ss(g, 100), width=7, resolution=2, label="",
                       showvalue=1, tickinterval=8, sliderlength=_ss(g, 20))
    echelle109.set(g['val_grid_CLAHE'])
    echelle109.place(anchor="w", x=_sx(g, 1720)+delta_s, y=_sy(g, 800))
    g['echelle109'] = echelle109

    # Choix Contrast Low Light
    CBCLL = Checkbutton(cadre, text="CLL", variable=g['choix_CLL'],
                        command=g['commande_CLL'], onvalue=1, offvalue=0)
    CBCLL.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 840))
    g['CBCLL'] = CBCLL

    labelParam200 = Label(cadre, text="µ")  # choix Mu CLL
    labelParam200.place(anchor="w", x=_sx(g, 1500)+delta_s, y=_sy(g, 840))
    g['labelParam200'] = labelParam200

    echelle200 = Scale(cadre, from_=0, to=5.0, command=g['choix_Var_CLL'],
                       orient=HORIZONTAL, length=_ss(g, 80), width=7, resolution=0.1, label="",
                       showvalue=1, tickinterval=2, sliderlength=_ss(g, 20))
    echelle200.set(g['val_MuCLL'])
    echelle200.place(anchor="w", x=_sx(g, 1510)+delta_s, y=_sy(g, 840))
    g['echelle200'] = echelle200

    labelParam201 = Label(cadre, text="Ro")  # choix Ro CLL
    labelParam201.place(anchor="w", x=_sx(g, 1605)+delta_s, y=_sy(g, 840))
    g['labelParam201'] = labelParam201

    echelle201 = Scale(cadre, from_=0.5, to=5.0, command=g['choix_Var_CLL'],
                       orient=HORIZONTAL, length=_ss(g, 80), width=7, resolution=0.1, label="",
                       showvalue=1, tickinterval=2, sliderlength=_ss(g, 20))
    echelle201.set(g['val_RoCLL'])
    echelle201.place(anchor="w", x=_sx(g, 1625)+delta_s, y=_sy(g, 840))
    g['echelle201'] = echelle201

    labelParam202 = Label(cadre, text="amp")  # choix Amplification CLL
    labelParam202.place(anchor="w", x=_sx(g, 1715)+delta_s, y=_sy(g, 840))
    g['labelParam202'] = labelParam202

    echelle202 = Scale(cadre, from_=0.5, to=5.0, command=g['choix_Var_CLL'],
                       orient=HORIZONTAL, length=_ss(g, 80), width=7, resolution=0.1, label="",
                       showvalue=1, tickinterval=2, sliderlength=_ss(g, 20))
    echelle202.set(g['val_AmpCLL'])
    echelle202.place(anchor="w", x=_sx(g, 1740)+delta_s, y=_sy(g, 840))
    g['echelle202'] = echelle202


def create_capture_widgets(g: dict):
    """
    Create capture widgets (RAW capture, video/image capture, buttons).

    Args:
        g: Globals dictionary containing cadre, delta_s, IntVars, and callbacks
    """
    cadre = g['cadre']
    delta_s = g['delta_s']
    fenetre_principale = g['fenetre_principale']

    ####################
    # CAPTURES WIDGETS #
    ####################

    # Choix HQ Capture
    CBHQC = Checkbutton(cadre, text="RAW Capture", variable=g['choix_HQ_capt'],
                        command=g['commande_HQ_capt'], onvalue=1, offvalue=0)
    CBHQC.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 1010))
    g['CBHQC'] = CBHQC

    # Number of pictures to capture
    echelle8 = Scale(cadre, from_=1, to=501, command=g['choix_nb_captures'],
                     orient=HORIZONTAL, length=_ss(g, 250), width=7, resolution=1, label="",
                     showvalue=1, tickinterval=50, sliderlength=_ss(g, 20))
    echelle8.set(g['val_nb_captures'])
    echelle8.place(anchor="w", x=_sx(g, 1570)+delta_s, y=_sy(g, 883))
    g['echelle8'] = echelle8

    # Frames number Video
    echelle11 = Scale(cadre, from_=0, to=4000, command=g['choix_nb_video'],
                      orient=HORIZONTAL, length=_ss(g, 250), width=7, resolution=20, label="",
                      showvalue=1, tickinterval=500, sliderlength=_ss(g, 20))
    echelle11.set(g['val_nb_capt_video'])
    echelle11.place(anchor="w", x=_sx(g, 1570)+delta_s, y=_sy(g, 930))
    g['echelle11'] = echelle11

    labelParam65 = Label(cadre, text="Delta T")  # choix valeur delta T
    labelParam65.place(anchor="w", x=_sx(g, 1535)+delta_s, y=_sy(g, 975))
    g['labelParam65'] = labelParam65

    echelle65 = Scale(cadre, from_=0, to=60, command=g['choix_deltat'],
                      orient=HORIZONTAL, length=_ss(g, 250), width=7, resolution=1, label="",
                      showvalue=1, tickinterval=10, sliderlength=_ss(g, 20))
    echelle65.set(g['val_deltat'])
    echelle65.place(anchor="w", x=_sx(g, 1570)+delta_s, y=_sy(g, 975))
    g['echelle65'] = echelle65

    labelInfo1 = Label(cadre, text=g['text_info1'])  # label info n°1
    labelInfo1.place(anchor="w", x=_sx(g, 1550)+delta_s, y=_sy(g, 1010))
    g['labelInfo1'] = labelInfo1

    labelInfo10 = Label(cadre, text=g['text_info10'])  # label info n°10
    labelInfo10.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 1030))
    g['labelInfo10'] = labelInfo10

    labelParam100_tt = Label(cadre, text="Treatment time : ")
    labelParam100_tt.place(anchor="w", x=_sx(g, 1450)+delta_s, y=_sy(g, 20))
    g['labelParam100_tt'] = labelParam100_tt

    labelInfo2 = Label(cadre, text="")  # label info n°2
    labelInfo2.place(anchor="w", x=_sx(g, 1520)+delta_s, y=_sy(g, 20))
    g['labelInfo2'] = labelInfo2

    ####################
    #      BUTTONS     #
    ####################

    Button1 = Button(cadre, text="Start CAP", command=g['start_pic_capture'],
                     padx=10, pady=0)
    Button1.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 875))
    g['Button1'] = Button1

    Button2 = Button(cadre, text="Stop CAP", command=g['stop_pic_capture'],
                     padx=10, pady=0)
    Button2.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 900))
    g['Button2'] = Button2

    Button3 = Button(cadre, text="Start REC", command=g['start_video_capture'],
                     padx=10, pady=0)
    Button3.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 935))
    g['Button3'] = Button3

    Button4 = Button(cadre, text="Stop REC", command=g['stop_video_capture'],
                     padx=10, pady=0)
    Button4.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 960))
    g['Button4'] = Button4

    Button5 = Button(cadre, text="Pause REC", command=g['pause_video_capture'],
                     padx=10, pady=0)
    Button5.place(anchor="w", x=_sx(g, 1460)+delta_s, y=_sy(g, 985))
    g['Button5'] = Button5

    # RAZ frame counter
    Button10 = Button(cadre, text="RZ Fr Cnt", command=g['raz_framecount'],
                      padx=10, pady=0)
    Button10.place(anchor="w", x=_sx(g, 5), y=_sy(g, 370))
    g['Button10'] = Button10

    if g['flag_camera_ok'] == False:
        Button12 = Button(cadre, text="Load Vid", command=g['load_video'],
                          padx=10, pady=0)
        Button12.place(anchor="w", x=_sx(g, 5), y=_sy(g, 945))
        g['Button12'] = Button12

        Button17 = Button(cadre, text="Load Pic", command=g['load_image'],
                          padx=10, pady=0)
        Button17.place(anchor="w", x=_sx(g, 5), y=_sy(g, 970))
        g['Button17'] = Button17

    Button15 = Button(cadre, text="Dir Vid", command=g['choose_dir_vid'],
                      padx=10, pady=0)
    Button15.place(anchor="w", x=_sx(g, 5), y=_sy(g, 995))
    g['Button15'] = Button15

    Button16 = Button(cadre, text="Dir Pic", command=g['choose_dir_pic'],
                      padx=10, pady=0)
    Button16.place(anchor="w", x=_sx(g, 5), y=_sy(g, 1020))
    g['Button16'] = Button16

    # Quit button
    quit_button = Button(fenetre_principale, text="Quit", command=g['quitter'],
                         padx=10, pady=5)
    quit_button.place(x=_sx(g, 1700)+delta_s, y=1030, anchor="w")
    g['quit_button'] = quit_button

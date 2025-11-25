"""
Astronomy Utilities Module

This module provides astronomical calculations for celestial mechanics,
coordinate conversions, and mount calibration.

Functions:
- Angle conversions (degrees/minutes/seconds)
- Julian day calculations
- Sidereal time calculations
- Azimuth/altitude (AltAz) to Right Ascension/Declination conversions
- Right Ascension/Declination to Azimuth/altitude conversions
- Telescope mount calibration

Copyright Alain Paillou 2018-2025
"""

import math
from datetime import datetime


class AstronomyCalculator:
    """
    Astronomy calculator for celestial coordinate conversions and mount control.

    Attributes:
        lat_obs: Observer latitude in degrees
        long_obs: Observer longitude in degrees
        alt_obs: Observer altitude in meters
        zone: Time zone offset from UTC
        polaris_ad: Polaris Right Ascension in hours
        polaris_dec: Polaris Declination in degrees
    """

    def __init__(self, lat_obs=48.0175, long_obs=-4.0340, alt_obs=0, zone=2,
                 polaris_ad=2.507, polaris_dec=89.25):
        """
        Initialize astronomy calculator with observer location.

        Args:
            lat_obs: Observer latitude in degrees (default: 48.0175)
            long_obs: Observer longitude in degrees (default: -4.0340)
            alt_obs: Observer altitude in meters (default: 0)
            zone: Time zone offset from UTC (default: 2)
            polaris_ad: Polaris Right Ascension in hours (default: 2.507)
            polaris_dec: Polaris Declination in degrees (default: 89.25)
        """
        self.lat_obs = lat_obs
        self.long_obs = long_obs
        self.alt_obs = alt_obs
        self.zone = zone
        self.polaris_ad = polaris_ad
        self.polaris_dec = polaris_dec

        # Constants
        self.Pi = math.pi
        self.conv_rad = self.Pi / 180
        self.conv_deg = 180 / self.Pi

        # Working variables
        self.jour_julien = 0.0
        self.HS = 0.0  # Sidereal hour

    def angle2degminsec(self, angle):
        """
        Convert decimal angle to degrees, minutes, seconds format.

        Args:
            angle: Angle in decimal degrees

        Returns:
            String formatted as "Xd Y' Z''"
        """
        deg = int(angle)
        minute = int((angle - int(angle)) * 60)
        sec = int((angle - (deg + minute / 60)) * 3600)
        result = str(deg) + "d " + str(abs(minute)) + "' " + str(abs(sec)) + "''"
        return result

    def calc_jour_julien(self, jours, mois, annee):
        """
        Calculate Julian day from date.

        Args:
            jours: Day of month
            mois: Month (1-12)
            annee: Year

        Returns:
            Julian day number
        """
        if mois < 3:
            mois = mois + 12
            annee = annee - 1
        coef_a = annee // 100
        coef_b = 2 - coef_a + (coef_a // 4)
        coef_c = int(365.25 * annee)
        coef_d = int(30.6001 * (mois + 1))
        self.jour_julien = coef_b + coef_c + coef_d + jours + 1720994.5
        return self.jour_julien

    def calc_heure_siderale(self, jrs_jul, heure_obs, min_obs):
        """
        Calculate sidereal hour from Julian day and observation time.

        Args:
            jrs_jul: Julian day
            heure_obs: Observation hour (0-23)
            min_obs: Observation minutes (0-59)

        Returns:
            Sidereal hour
        """
        TT = (jrs_jul - 2451545) / 36525
        H1 = 24110.54841 + (8640184.812866 * TT) + (0.093104 * (TT * TT)) - (0.0000062 * (TT * TT * TT))
        HSH = H1 / 3600
        self.HS = ((HSH / 24) - int(HSH / 24)) * 24
        return self.HS

    def calcul_AZ_HT_cible(self, jours_obs, mois_obs, annee_obs, heure_obs,
                          min_obs, second_obs, cible_ASD, cible_DEC):
        """
        Calculate azimuth and altitude (AltAz coordinates) from target's
        Right Ascension and Declination.

        Args:
            jours_obs: Day of observation
            mois_obs: Month of observation
            annee_obs: Year of observation
            heure_obs: Hour of observation (0-23)
            min_obs: Minutes of observation (0-59)
            second_obs: Seconds of observation (0-59)
            cible_ASD: Target Right Ascension in hours
            cible_DEC: Target Declination in degrees

        Returns:
            Tuple (azimut_cible, hauteur_cible) in radians
        """
        self.calc_jour_julien(jours_obs, mois_obs, annee_obs)
        self.calc_heure_siderale(self.jour_julien, heure_obs, min_obs)

        angleH = (2 * self.Pi * self.HS / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi
        angleT = ((heure_obs - 12 + min_obs / 60 - self.zone + second_obs / 3600) *
                  2 * self.Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi

        H = angleH + angleT - 15 * cible_ASD + self.long_obs

        sinushauteur = (math.sin(cible_DEC * self.conv_rad) * math.sin(self.lat_obs * self.conv_rad) -
                       math.cos(cible_DEC * self.conv_rad) * math.cos(self.lat_obs * self.conv_rad) *
                       math.cos(H * self.conv_rad))
        hauteur_cible = math.asin(sinushauteur)

        cosazimut = ((math.sin(cible_DEC * self.conv_rad) -
                     math.sin(self.lat_obs * self.conv_rad) * math.sin(hauteur_cible)) /
                    (math.cos(self.lat_obs * self.conv_rad) * math.cos(hauteur_cible)))
        sinazimut = ((math.cos(cible_DEC * self.conv_rad) * math.sin(H * self.conv_rad)) /
                    math.cos(hauteur_cible))

        if sinazimut > 0:
            azimut_cible = math.acos(cosazimut)
        else:
            azimut_cible = -math.acos(cosazimut)

        return azimut_cible, hauteur_cible

    def calcul_ASD_DEC_cible(self, jours_obs, mois_obs, annee_obs, heure_obs,
                            min_obs, second_obs, azimut_cible, hauteur_cible):
        """
        Calculate Right Ascension and Declination from observed azimuth and altitude.

        Args:
            jours_obs: Day of observation
            mois_obs: Month of observation
            annee_obs: Year of observation
            heure_obs: Hour of observation (0-23)
            min_obs: Minutes of observation (0-59)
            second_obs: Seconds of observation (0-59)
            azimut_cible: Target azimuth in radians
            hauteur_cible: Target altitude in radians

        Returns:
            Tuple (ASD_calculee, DEC_calculee) - Right Ascension (hours), Declination (radians)
        """
        self.calc_jour_julien(jours_obs, mois_obs, annee_obs)
        self.calc_heure_siderale(self.jour_julien, heure_obs, min_obs)

        angleH = (2 * self.Pi * self.HS / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi
        angleT = ((heure_obs - 12 + min_obs / 60 - self.zone + second_obs / 3600) *
                  2 * self.Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi

        DEC_calculee = math.asin(math.cos(azimut_cible) * math.cos(self.lat_obs * self.conv_rad) *
                                math.cos(hauteur_cible) + math.sin(self.lat_obs * self.conv_rad) *
                                math.sin(hauteur_cible))

        if azimut_cible > 0:
            H = math.acos((math.sin(DEC_calculee) * math.sin(self.lat_obs * self.conv_rad) -
                          math.sin(hauteur_cible)) /
                         (math.cos(DEC_calculee) * math.cos(self.lat_obs * self.conv_rad)))
        else:
            H = -math.asin(math.sin(azimut_cible) * math.cos(hauteur_cible) /
                          math.cos(DEC_calculee)) + self.Pi

        ASD_calculee = (angleH + angleT + self.long_obs - H * self.conv_deg) / 15

        return ASD_calculee, DEC_calculee

    def mount_calibration(self, azimut_monture, hauteur_monture):
        """
        Calculate mount alignment error by comparing Polaris position
        with mount reported position.

        Args:
            azimut_monture: Mount-reported azimuth in degrees
            hauteur_monture: Mount-reported altitude in degrees

        Returns:
            Tuple (azimut_polaris, hauteur_polaris, delta_azimut, delta_hauteur)
            - azimut_polaris: Calculated Polaris azimuth (degrees, 0-360)
            - hauteur_polaris: Calculated Polaris altitude (degrees, 0-90)
            - delta_azimut: Azimuth error (degrees)
            - delta_hauteur: Altitude error (degrees)
        """
        date = datetime.now()
        annee_obs = date.year
        mois_obs = date.month
        jours_obs = date.day
        heure_obs = date.hour
        min_obs = date.minute
        second_obs = date.second

        cible_polaire_ASD = self.polaris_ad
        cible_polaire_DEC = self.polaris_dec

        self.calc_jour_julien(jours_obs, mois_obs, annee_obs)
        self.calc_heure_siderale(self.jour_julien, heure_obs, min_obs)

        angleH = (2 * self.Pi * self.HS / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi
        angleT = ((heure_obs - 12 + min_obs / 60 - self.zone + second_obs / 3600) *
                  2 * self.Pi / (23 + 56 / 60 + 4 / 3600)) * 180 / self.Pi

        H = angleH + angleT - 15 * cible_polaire_ASD + self.long_obs

        sinushauteur = (math.sin(cible_polaire_DEC * self.conv_rad) *
                       math.sin(self.lat_obs * self.conv_rad) -
                       math.cos(cible_polaire_DEC * self.conv_rad) *
                       math.cos(self.lat_obs * self.conv_rad) *
                       math.cos(H * self.conv_rad))
        hauteurcible = math.asin(sinushauteur)

        cosazimut = ((math.sin(cible_polaire_DEC * self.conv_rad) -
                     math.sin(self.lat_obs * self.conv_rad) * math.sin(hauteurcible)) /
                    (math.cos(self.lat_obs * self.conv_rad) * math.cos(hauteurcible)))
        sinazimut = ((math.cos(cible_polaire_DEC * self.conv_rad) * math.sin(H * self.conv_rad)) /
                    math.cos(hauteurcible))

        if sinazimut > 0:
            azimut_polaris = math.acos(cosazimut) * self.conv_deg  # -180 to +180
        else:
            azimut_polaris = -math.acos(cosazimut) * self.conv_deg  # -180 to +180

        if azimut_polaris < 0:
            azimut_polaris = 360 + azimut_polaris

        hauteur_polaris = hauteurcible * self.conv_deg  # 0 to 90

        delta_azimut = azimut_polaris - azimut_monture
        delta_hauteur = hauteur_polaris - hauteur_monture

        return azimut_polaris, hauteur_polaris, delta_azimut, delta_hauteur

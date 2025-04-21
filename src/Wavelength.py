import pandas as pd
import numpy as np
from functools import reduce


path = "src/CSV_wavelength/"

correction_matrices = [
    pd.read_csv(path + f"matrice_corr_diode_{i}.csv", sep=',', decimal='.').values
    for i in range(6)
]
photodiode_ratios_450 = [
    pd.read_csv(path + "ratios_photodiodes_450.csv", sep=';', decimal=',')[col].values
    for col in pd.read_csv(path + "ratios_photodiodes_450.csv", sep=';', decimal=',').columns
]
photodiode_ratios_976 = [
    pd.read_csv(path + "ratios_photodiodes_976.csv", sep=';', decimal=',')[col].values
    for col in pd.read_csv(path + "ratios_photodiodes_976.csv", sep=';', decimal=',').columns
]
photodiode_ratios_1976 = pd.read_csv(path + "ratios_photodiodes_1976.csv", sep=';', decimal=',').values

def id_pos(pos):
    """Trouve l'indice le plus proche dans la matrice de correction pour une position donnée."""
    extremas = 6
    inter = np.linspace(-extremas, extremas, len(correction_matrices[0]))
    delta = np.abs(inter - np.array(pos)[:, None])
    return delta[0].argmin(), delta[1].argmin()


def indexes(array, target, threshold=0.1):
    """Retourne les indices où la valeur est proche du target selon un threshold."""
    return np.where(np.abs(array - target) <= np.maximum(np.abs(target) * threshold, threshold))[0]


def get_visible_wavelength(V_corr, threshold=0.1):
    """Détermine la longueur d'onde visible à partir des tensions corrigées."""
    V_corr[-2] = 0

    ratios_corr = np.divide(
        V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0
    )

    ratio_ids_corr = [
        indexes(photodiode_ratios_450[i], ratio, threshold)
        for i, ratio in enumerate(ratios_corr)
    ]

    if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
        return np.array([])

    return reduce(np.intersect1d, ratio_ids_corr)


def get_NIR_wavelength(V_corr, threshold=0.1):
    """Détermine la longueur d'onde NIR à partir des tensions corrigées."""
    ratios_corr = np.divide(
        V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0
    )

    ratio_ids_corr = [
        indexes(photodiode_ratios_976[i], ratio, threshold)
        for i, ratio in enumerate(ratios_corr)
    ]

    if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
        return np.array([])

    return reduce(np.intersect1d, ratio_ids_corr)


def get_IR_wavelength(V_corr, puissance, threshold):
    """Détermine la longueur d'onde IR à partir de la tension corrigée et la puissance."""
    ratio = V_corr / puissance
    return indexes(photodiode_ratios_1976, ratio, threshold)


def precise_wavelength(func, *args, threshold, threshold_mult, max_iter=20):
    """Affinage de la recherche de longueur d'onde avec ajustement du threshold."""
    for _ in range(max_iter):
        wavelength = func(*args, threshold)

        if len(wavelength) == 1:
            return wavelength

        if len(wavelength) == 0:
            return precise_wavelength(
                func,
                *args,
                threshold=threshold / threshold_mult,
                threshold_mult=np.sqrt(threshold_mult)
            )

        threshold *= threshold_mult

    return wavelength


def get_wavelength(position, V_photodiodes, puissance, threshold, threshold_mult):
    """Détermine la longueur d'onde selon la position et les tensions mesurées."""
    pos = id_pos(position)

    V_corr = np.array([
        V * correction_matrices[i][pos] for i, V in enumerate(V_photodiodes)
    ])

    index_max = np.argmax(V_corr)

    if all(V < 0.1 for V in V_corr):
        print("Longueur d'onde non trouvée")
        light_type = "Unknown"
        wavelength = 0

    elif index_max == 0:
        print("UV")
        light_type = "UV"
        wavelength = -200

    elif index_max == 1:
        print("VIS")
        light_type = "VIS"
        wavelength = precise_wavelength(
            get_visible_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult
        )

    elif index_max == 5:
        print("IR")
        light_type = "IR"
        wavelength =  precise_wavelength(
            get_IR_wavelength, V_corr[-1], puissance, threshold=threshold, threshold_mult=threshold_mult
        )

    else:
        print("NIR")
        light_type = "NIR"
        wavelength =  precise_wavelength(
            get_NIR_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult
        )

    return np.mean(wavelength) + 200

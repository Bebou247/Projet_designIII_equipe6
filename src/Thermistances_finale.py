import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import re
import time

# Constantes du circuit
VREF = 3.3  # Tension de référence en Volts
R_FIXED = 4700  # Résistance de référence (10 kΩ), à ajuster si besoin

# Coefficients Steinhart-Hart chargés depuis le fichier "coefficients.npy"
coefficients = np.load("coefficients.npy", allow_pickle=True)


# Fonction pour calculer la température à partir de la résistance
def steinhart_hart_temperature(R, A, B, C):
    return 1 / (A + B * np.log(R) + C * (np.log(R))**3)

# Fonction pour lire la tension et convertir en résistance
def compute_resistance(voltage):
    if voltage >= VREF:  # Vérification pour éviter une division par zéro
        return float('inf')
    return R_FIXED * (voltage / (VREF - voltage))

# Fonction pour convertir la résistance en température
def compute_temperature_from_resistance(R, coeffs):
    A, B, C = coeffs
    temperature_kelvin = steinhart_hart_temperature(R, A, B, C)
    return temperature_kelvin - 273.15

# Spécification manuelle du port série
arduino_port = "/dev/cu.usbmodem14201"  # Remplacez par le nom exact du port de votre Arduino

# Ouverture de la connexion série avec l'Arduino
try:
    ser = serial.Serial(arduino_port, 9600, timeout=1)
except Exception as e:
    # print(f"Erreur lors de l'ouverture du port série : {e}")
    exit()

thermistor_positions_named = [
    ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
    ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
    ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R12", (-4.5, 8)),
    ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
    ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
    ("R21", (4.5, -8)), ("R22", (-4.5, -8)), ("R23", (3.5, -11.25)), ("R24", (-3.5, -11.25))
]

# Boucle principale modifiée avec affichage terminal
while True:
    try:
        voltages_dict = {}
        
        # Lecture des données
        while True:
            line = ser.readline().decode().strip()
            
            if not line:
                continue
                
            if "Fin du balayage" in line:
                break
                
            match = re.search(r"Canal (\d+): ([\d.]+) V", line)
            if match:
                canal_num = int(match.group(1))
                if 0 <= canal_num <= 23:
                    voltages_dict[canal_num] = float(match.group(2))

        if len(voltages_dict) == 24:
            voltages = [voltages_dict[i] for i in range(24)]
            
            # Calculs
            resistances = {
                name: compute_resistance(voltages[i])
                for i, (name, _) in enumerate(thermistor_positions_named)
            }
            
            temperatures = {
                name: compute_temperature_from_resistance(resistances[name], coefficients[i])
                for i, (name, _) in enumerate(thermistor_positions_named)
            }

            # Affichage terminal
            # print("\n" + "="*60)
            # print(f"{'Thermistor':<10} | {'Temp (°C)':<10} | {'Resistance (Ω)':<15}")
            # print("-"*60)
            # for (name, _), temp in zip(thermistor_positions_named, temperatures.values()):
            #     res = resistances[name]
            #     print(f"{name:<10} | {temp:.2f}{'°C':<8} | {res:.2f} Ω")
            # print("="*60 + "\n")
            
            # Mise à jour graphique
            # create_or_update_heatmap(list(temperatures.values()))
            # plt.pause(0.1)
            
        else:
            # print(f"Données incomplètes : {len(voltages_dict)}/24 canaux reçus")
            pass
            
        time.sleep(1)

    except KeyboardInterrupt:
        # print("Arrêt de la lecture.")
        ser.close()
        break

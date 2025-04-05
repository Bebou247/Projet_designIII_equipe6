import serial
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
import re
import time

# Constantes du circuit
VREF = 3.3  # Tension de référence en Volts
R_FIXED = 4700  # Résistance fixe en ohms

# Chargement des coefficients Steinhart-Hart
coefficients = np.load("coefficients.npy", allow_pickle=True)

# Fonction Steinhart-Hart pour la conversion résistance -> température
def steinhart_hart_temperature(R, A, B, C):
    return 1 / (A + B * np.log(R) + C * (np.log(R))**3) - 273.15

# Conversion tension -> résistance
def compute_resistance(voltage):
    if voltage >= VREF or voltage <= 0:
        return float('inf')
    return R_FIXED * (voltage / (VREF - voltage))

# Positions des thermistances
themistor_positions = [
    (11, 0), (3, 0), (-3, 0), (-11, 0), (8, 2.5), (0, 2.5), (-8, 2.5), (8, 5.5),
    (0, 5.5), (-8, 5.5), (4.5, 8), (-4.5, 8), (4, 11.25), (-4, 11.25), (8, -2.5), (0, -2.5),
    (-8, -2.5), (8, -5.5), (0, -5.5), (-8, -5.5), (4.5, -8), (-4.5, -8), (3.5, -11.25), (-3.5, -11.25)
]

# Définition du maillage pour l'interpolation
xi = np.linspace(-12.5, 12.5, 500)
yi = np.linspace(-12.5, 12.5, 500)
Xi, Yi = np.meshgrid(xi, yi)

# Création du masque circulaire
radius = 12.5
mask = np.sqrt(Xi**2 + Yi**2) <= radius

# Initialisation du port série
arduino_port = "/dev/cu.usbmodem3"  # Adapter selon votre Arduino
ser = serial.Serial(arduino_port, baudrate=9600, timeout=1)

# Configuration de l'affichage
plt.ion()
fig, ax = plt.subplots()
heatmap = ax.imshow(np.zeros_like(Xi), extent=[-12.5, 12.5, -12.5, 12.5], origin='lower', cmap='inferno', vmin=0, vmax=50)
plt.colorbar(heatmap, ax=ax, label="Température (°C)")
ax.set_xlim(-12.5, 12.5)
ax.set_ylim(-12.5, 12.5)
ax.set_title("Interpolation des Températures")
ax.set_xlabel("Position X (mm)")
ax.set_ylabel("Position Y (mm)")

# Boucle principale
try:
    while True:
        voltages_dict = {}
        while True:
            line = ser.readline().decode().strip()
            if not line:
                continue
            if "Fin du balayage" in line:
                break
            match = re.search(r"Canal (\d+): ([\d.]+) V", line)
            if match:
                canal_num = int(match.group(1))  # Canal 0 à 23
                if 0 <= canal_num <= 23:  # Lire uniquement les canaux de 0 à 23
                    voltage = float(match.group(2))
                    voltages_dict[canal_num] = voltage
                    
                    # Affichage du canal et de la valeur de la tension
                    print(f"Canal {canal_num} : {voltage:.2f} V")
        
        if len(voltages_dict) == 24:
            resistances = [compute_resistance(voltages_dict[i]) for i in range(24)]
            temperatures = [steinhart_hart_temperature(resistances[i], *coefficients[i]) for i in range(24)]
            
            # Affichage des températures relevées
            for i, temp in enumerate(temperatures):
                print(f"Thermistance {i+1} (Canal {i}): {temp:.2f}°C")
            
            x, y = zip(*themistor_positions)
            rbf = Rbf(x, y, temperatures, function='multiquadric', smooth=0.5, epsilon=2.0)
            Zi = rbf(Xi, Yi)
            
            # Appliquer le masque circulaire
            Zi[~mask] = np.nan
            
            heatmap.set_data(Zi)
            plt.draw()
            plt.pause(0.005)

except KeyboardInterrupt:
    print("Arrêt du programme.")
    ser.close()

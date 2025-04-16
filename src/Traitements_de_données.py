import serial
import numpy as np
from scipy.interpolate import Rbf, interp1d
import re
import time
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from pathlib import Path
import pandas as pd

class TraitementDonnees:
    VREF = 3.002
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        self.positions = [
            ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
            ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
            ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R24", (-3.5, -11.25)),
            ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
            ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
            ("R21", (4.5, -8))
        ]

        self.indices_à_garder = list(range(21)) + [24]
        self.canaux_photodiodes = list(range(25, 31))

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Aucune connexion série établie.")
        else:
            try:
                self.ser = serial.Serial(self.port, 9600, timeout=1)
                print(f"[INFO] Port série connecté sur {self.port}")
            except Exception as e:
                print(f"[ERREUR] Impossible d'ouvrir le port série : {e}")
                self.ser = None

    def est_connecte(self):
        return self.ser is not None

    def steinhart_hart_temperature(self, R, A, B, C):
        return 1 / (A + B * np.log(R) + C * (np.log(R))**3)

    def compute_resistance(self, voltage):
        if voltage >= self.VREF:
            return float('inf')
        return self.R_FIXED * (voltage / (self.VREF - voltage))

    def compute_temperature(self, resistance, coeffs):
        A, B, C = coeffs
        kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
        return kelvin - 273.15

    def estimate_laser_power(self, temp_initial, temp_measured, time, position, laser_center=(0, 0)):
        delta_t = temp_measured - temp_initial
        K = 0.8411
        tau = 0.9987
        coeff = 0.9999

        dx = position[0] - laser_center[0]
        dy = position[1] - laser_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        attenuation = np.exp(-distance / 5.0)
        adjusted_coeff = coeff * attenuation

        denominator = K * (1 - np.exp(-time / tau))
        denominator = np.where(denominator < 1e-10, 1e-10, denominator)

        estimated_power = (delta_t / denominator) * adjusted_coeff
        return estimated_power

    def lire_donnees(self):
        canaux_requis = self.indices_à_garder + self.canaux_photodiodes

        if self.simulation:
            return {i: np.random.uniform(0.4, 2.6) for i in canaux_requis}

        if self.ser is None:
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        start_time = time.time()
        timeout_sec = 1

        while True:
            if time.time() - start_time > timeout_sec:
                print("⚠️ Temps de lecture dépassé, données incomplètes.")
                return None

            try:
                line = self.ser.readline().decode(errors='ignore').strip()
            except Exception:
                continue

            if not line:
                continue

            if "Fin du balayage" in line:
                break

            match = re.search(r"Canal (\d+): ([\d.]+) V", line)
            if match:
                canal = int(match.group(1))
                if canal in canaux_requis:
                    voltages_dict[canal] = float(match.group(2))

        if len(voltages_dict) != len(canaux_requis):
            print(f"Seulement {len(voltages_dict)}/{len(canaux_requis)} canaux reçus.")
            return None

        return voltages_dict

    def get_temperatures(self, data):
        if data is None:
            return None

        temperatures = []
        noms = []

        for i in self.indices_à_garder:
            if i not in data:
                continue
            if i == 24:
                coeffs = self.coefficients[24]
                nom = "R25"
            elif i == 11:
                coeffs = self.coefficients[23]
                nom = "R24"
            else:
                coeffs = self.coefficients[i]
                nom = self.positions[i][0]

            resistance = self.compute_resistance(data[i])
            temp = self.compute_temperature(resistance, coeffs)
            temperatures.append(temp)
            noms.append(nom)

        return dict(zip(noms, temperatures))

    def afficher_heatmap_dans_figure(self, temperature_dict, fig):
        fig.clear()
        ax = fig.add_subplot(111)
        x, y, t = [], [], []
        for nom, (xi, yi) in dict(self.positions).items():
            if nom in temperature_dict:
                x.append(xi)
                y.append(yi)
                t.append(temperature_dict[nom])

        if len(x) < 3:
            print("Pas assez de données pour générer la heatmap.")
            return

        rbf = Rbf(x, y, t, function='multiquadric', smooth=0.1)
        grid_size = 200
        r_max = 12.25
        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )
        ti = rbf(xi, yi)
        mask = xi**2 + yi**2 > (r_max**2)
        ti_masked = np.ma.array(ti, mask=mask)
        contour = ax.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour, ax=ax, label="Température (°C")
        ax.scatter(x, y, color='black', marker='o', s=25)
        for i, nom in enumerate(x):
            ax.annotate(list(temperature_dict.keys())[i], (x[i], y[i]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)
        ax.set_aspect('equal')
        ax.set_title("Heatmap des thermistances")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        fig.tight_layout()
        plt.pause(0.001)

    def demarrer_acquisition_live(self, interval=0.1):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("Acquisition live, Ctrl+C pour arrêter ")
        fig = plt.figure(figsize=(6, 6))
        plt.ion()
        fig.show()

        all_data = []
        noms = [self.positions[i][0] if i != 24 else "R25" for i in self.indices_à_garder]
        photodiode_headers = [f"PD{i}" for i in self.canaux_photodiodes]
        headers = noms + photodiode_headers + ["Puissance estimée (W)", "T_ref", "timestamp"]

        try:
            while True:
                data_raw = self.lire_donnees()
                if data_raw is None:
                    print("⚠️ Données incomplètes ou non reçues.")
                    time.sleep(interval)
                    continue

                temp_data = self.get_temperatures(data_raw)
                if temp_data:
                    os.system("clear")
                    print("=" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    for name, temp in temp_data.items():
                        print(f"{name:<6} : {temp:6.2f} °C")
                    print("-" * 60)
                    print("Tensions photodiodes :")
                    for i in self.canaux_photodiodes:
                        tension = data_raw.get(i)
                        if tension is not None:
                            print(f"PD{i:<2} : {tension:.3f} V")
                    print("=" * 60)

                    self.afficher_heatmap_dans_figure(temp_data, fig)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    t_max = max(temp_data.values())
                    max_pos = [pos for nom, pos in self.positions if nom in temp_data and temp_data[nom] == t_max]
                    max_position = max_pos[0] if max_pos else (0, 0)
                    puissance = self.estimate_laser_power(25.0, t_max, 3.0, max_position)

                    ligne = [temp_data.get(name, "--") for name in noms]
                    ligne += [data_raw.get(i, "--") for i in self.canaux_photodiodes]
                    ligne += [puissance, 25.0, datetime.now().isoformat(timespec='seconds')]
                    all_data.append(ligne)

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée. Sauvegarde du fichier CSV...")

            desktop_path = Path.home() / "Desktop"
            filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = desktop_path / filename

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(all_data)

            print(f"Données sauvegardées dans : {csv_path}")
        
        
if __name__ == "__main__":
    td = TraitementDonnees(simulation=False)
    td.demarrer_acquisition_live()

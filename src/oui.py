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
    VREF = 3.02
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        # Positions des thermistances (sauf R25, hors matrice)
        self.positions = [
            ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
            ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
            ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R24", (-3.5, -11.25)),
            ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
            ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
            ("R21", (4.5, -8))
        ]

        self.indices_à_garder = list(range(21)) + [24]  # Canal 24 = R25 (dans l’ailette)

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

    def lire_donnees(self):
        if self.simulation:
            return {i: np.random.uniform(0.4, 2.6) for i in self.indices_à_garder}

        if self.ser is None:
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        start_time = time.time()
        timeout_sec = 2

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
                if canal in self.indices_à_garder:
                    voltages_dict[canal] = float(match.group(2))

        if len(voltages_dict) != len(self.indices_à_garder):
            print(f"Seulement {len(voltages_dict)}/{len(self.indices_à_garder)} canaux reçus.")
            return None

        return voltages_dict

    def get_temperatures(self):
        data = self.lire_donnees()
        if data is None:
            return None

        temperatures = []
        noms = []

        for i in self.indices_à_garder:
            if i == 24:
                coeffs = self.coefficients[24]  # Canal 24 = R25
                nom = "R25"
            elif i == 11:
                coeffs = self.coefficients[23]  # Canal 11 = R24
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
        for i in self.indices_à_garder:
            if i == 24:
                continue  # R25 n’a pas de position → exclue de la heatmap
            name, pos = self.positions[i]
            x.append(pos[0])
            y.append(pos[1])
            t.append(temperature_dict[name])

        rbf = Rbf(x, y, t, function='multiquadric', smooth=0.09)
        grid_size = 200
        r_max = 12.5

        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )

        ti = rbf(xi, yi)
        mask = xi**2 + yi**2 > r_max**2
        ti_masked = np.ma.array(ti, mask=mask)

        contour = ax.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour, ax=ax, label="Température (°C)")
        ax.scatter(x, y, color='black', marker='o', s=25)

        for i in self.indices_à_garder:
            if i == 24:
                continue
            name, pos = self.positions[i]
            ax.annotate(name, (pos[0], pos[1]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)

        ax.set_aspect('equal')
        ax.set_title("Heatmap des thermistances")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        fig.tight_layout()

    def estimer_puissance(self):
        file_path = "data/raw/Vrai-Aural2.csv"
        df = pd.read_csv(file_path, skiprows=2)
        df.columns = ["Longueur_donde_refl", "Reflectance", "Longueur_donde_abs", "Absorbance", "Unused"]
        df = df.drop(columns=["Unused"])

        df["Longueur_donde_refl"] = pd.to_numeric(df["Longueur_donde_refl"], errors='coerce')
        df["Reflectance"] = pd.to_numeric(df["Reflectance"], errors='coerce')
        df["Longueur_donde_abs"] = pd.to_numeric(df["Longueur_donde_abs"], errors='coerce')
        df["Absorbance"] = pd.to_numeric(df["Absorbance"], errors='coerce')

        df = df.dropna()
        df["Absorbance"] = (1 - 10 ** (-df["Absorbance"])) * 100
        self.interpolate_abs = interp1d(df["Longueur_donde_abs"], df["Absorbance"], kind='linear', fill_value="extrapolate")

    def get_absorbance_at_wavelength(self, longueur_donde):
        absorbance_donnee = self.interpolate_abs(longueur_donde).item()
        print(f"L'absorbance à {longueur_donde} nm est {absorbance_donnee:.2f}%")
        return absorbance_donnee

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
        headers = noms + ["T_ref", "timestamp"]

        try:
            while True:
                data = self.get_temperatures()
                if data:
                    os.system("clear")
                    print("=" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    for name, temp in data.items():
                        print(f"{name:<6} : {temp:6.2f} °C")
                    print("=" * 60)

                    self.afficher_heatmap_dans_figure(data, fig)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    ligne = [data[name] for name in noms]
                    ligne.append(25.0)
                    ligne.append(datetime.now().isoformat(timespec='seconds'))
                    all_data.append(ligne)

                else:
                    print("⚠️ Données incomplètes ou non reçues.")
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

            print(f"✅ Données sauvegardées dans : {csv_path}")


if __name__ == "__main__":
    td = TraitementDonnees(simulation=False)
    td.demarrer_acquisition_live(interval=0.05)
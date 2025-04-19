import serial
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from scipy.interpolate import Rbf
from scipy.ndimage import gaussian_filter
import re
import time
import os
import csv

class TraitementDonnees:
    VREF = 3.003
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        # Décalage à appliquer
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas

        self.positions = [
            ("R1", (11 + decalage_x, 0 + decalage_y)), ("R2", (3 + decalage_x, 0 + decalage_y)), ("R3", (-3 + decalage_x, 0 + decalage_y)), ("R4", (-11 + decalage_x, 0 + decalage_y)),
            ("R5", (8 + decalage_x, 2.5 + decalage_y)), ("R6", (0 + decalage_x, 2.5 + decalage_y)), ("R7", (-8 + decalage_x, 2.5 + decalage_y)), ("R8", (8 + decalage_x, 5.5 + decalage_y)),
            ("R9", (0 + decalage_x, 5.5 + decalage_y)), ("R10", (-8 + decalage_x, 5.5 + decalage_y)), ("R11", (4.5 + decalage_x, 8 + decalage_y)), ("R24", (-3.5 + decalage_x, -11.25 + decalage_y)),
            ("R13", (4 + decalage_x, 11.25 + decalage_y)), ("R14", (-4 + decalage_x, 11.25 + decalage_y)), ("R15", (8 + decalage_x, -2.5 + decalage_y)), ("R16", (0 + decalage_x, -2.5 + decalage_y)),
            ("R17", (-8 + decalage_x, -2.5 + decalage_y)), ("R18", (8 + decalage_x, -5.5 + decalage_y)), ("R19", (0 + decalage_x, -5.5 + decalage_y)), ("R20", (-8 + decalage_x, -5.5 + decalage_y)),
            ("R21", (4.5 + decalage_x, -8 + decalage_y)), ("R25", (0 + decalage_x, -11.5 + decalage_y))
        ]

        self.indices_a_garder = [i for i, (nom, _) in enumerate(self.positions) if nom != "R25"]
        self.simulation_columns = [nom for i, (nom, _) in enumerate(self.positions) if i in self.indices_a_garder]
        self.simulation_data = None
        self.simulation_index = 0

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                simulation_file_path = Path(__file__).parent.parent / "data" / "10 W centre (hauteur 2 à 6).csv"
                df = pd.read_csv(simulation_file_path, sep=";", decimal=",", engine="python")
                df.dropna(axis=1, how='all', inplace=True)

                idx_tref = df.columns.get_loc("T_ref")
                self.simulation_columns = df.columns[:idx_tref].tolist()
                self.simulation_data = df

                print(f"[SIMULATION] Chargement : {simulation_file_path.resolve()}")
                print(f"[SIMULATION] {len(self.simulation_data)} lignes chargées.")

            except Exception as e:
                print(f"[ERREUR] Chargement simulation : {e}")
                self.simulation_data = None
        else:
            try:
                self.ser = serial.Serial(self.port, 9600, timeout=1)
                print(f"[INFO] Port série connecté : {self.port}")
            except Exception as e:
                print(f"[ERREUR] Connexion série : {e}")
                self.ser = None

    def est_connecte(self):
        return self.ser is not None

    def steinhart_hart_temperature(self, R, A, B, C):
        return 1 / (A + B * np.log(R) + C * (np.log(R))**3)

    def compute_resistance(self, voltage):
        if voltage >= self.VREF:
            return float('inf')
        return self.R_FIXED * (voltage / (self.VREF - voltage))

    def compute_temperature(self, R, coeffs):
        A, B, C = coeffs
        return self.steinhart_hart_temperature(R, A, B, C) - 273.15

    def get_temperatures(self):
        if self.simulation:
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0
                row = self.simulation_data.iloc[self.simulation_index]
                self.simulation_index += 1
                temp_dict = {name: row.get(name, np.nan) for name in self.simulation_columns}
                timestamp = row.get("timestamp", "")
                return temp_dict, timestamp
            return None, ""
        else:
            return None, ""

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, index=0, timestamp="", utiliser_bords=True):
        fig.clear()
        ax = fig.add_subplot(111)

        x, y, t = [], [], []
        for i in self.indices_a_garder:
            name, (xi, yi) = self.positions[i]
            temp = temperature_dict.get(name, np.nan)
            if pd.notna(temp):
                x.append(xi)
                y.append(yi)
                t.append(temp)

        if len(x) < 3:
            print("[SKIP] Trop peu de données valides.")
            return

        r_max = 12.5

        if utiliser_bords:
            edge_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
            edge_x = r_max * np.cos(edge_angles)
            edge_y = r_max * np.sin(edge_angles)
            edge_t = [np.mean(t) - 2.0] * len(edge_x)

            x_all = x + list(edge_x)
            y_all = y + list(edge_y)
            t_all = t + edge_t
        else:
            x_all = x[:]
            y_all = y[:]
            t_all = t[:]

        points_centre = [(-2, 2), (2, -2), (0, 0)]
        temp_moy = np.mean(t)
        for cx, cy in points_centre:
            x_all.append(cx)
            y_all.append(cy)
            t_all.append(temp_moy - 0.5)

        rbf = Rbf(x_all, y_all, t_all, function='multiquadric', smooth=0.5)
        grid_size = 500
        xi, yi = np.meshgrid(np.linspace(-r_max, r_max, grid_size),
                             np.linspace(-r_max, r_max, grid_size))
        ti = rbf(xi, yi)
        ti_filtered = gaussian_filter(ti, sigma=1.2)
        mask = xi**2 + yi**2 > r_max**2
        ti_masked = np.ma.array(ti_filtered, mask=mask)

        max_idx = np.unravel_index(np.nanargmax(ti_masked), ti_masked.shape)
        x_laser, y_laser = xi[max_idx], yi[max_idx]
        temp_peak = ti_masked[max_idx]

        contour = ax.contourf(xi, yi, ti_masked, levels=400, cmap="plasma")
        fig.colorbar(contour, ax=ax, label="Température (°C)")

        ax.scatter(x, y, color='black', s=25)
        ax.scatter(x_laser, y_laser, color='red', marker='x', s=80)
        ax.annotate(f"x={x_laser:.1f}, y={y_laser:.1f}",
                    (x_laser, y_laser),
                    textcoords="offset points",
                    xytext=(10, 10),
                    fontsize=9,
                    color='white',
                    bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))

        for xi_, yi_, nom in zip(x, y, [self.positions[i][0] for i in self.indices_a_garder]):
            ax.annotate(nom, (xi_, yi_), textcoords="offset points", xytext=(4, 4),
                        ha='left', fontsize=8, color='white')

        ax.set_title(f"Frame {index} | t = {timestamp} | Laser = {temp_peak:.1f}°C")
        ax.set_aspect('equal')
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        fig.tight_layout()
        plt.pause(0.001)

    def demarrer_acquisition_live(self, interval=1.0, real_interval_csv=0.05, utiliser_bords=False):
        if not self.simulation and not self.est_connecte():
            print("[ERREUR] Aucun Arduino connecté.")
            return

        if self.simulation and self.simulation_data is None:
            print("[ERREUR] Aucun fichier de simulation chargé. Vérifie le nom du fichier CSV.")
            return

        fig = plt.figure(figsize=(6, 6))
        plt.ion()
        fig.show()

        try:
            frame_index = 0
            skip_n = int(interval / real_interval_csv)

            while True:
                self.simulation_index += skip_n
                if self.simulation_index >= len(self.simulation_data):
                    print("✅ Fin du fichier CSV atteinte.")
                    break

                data, timestamp = self.get_temperatures()
                if data:
                    self.afficher_heatmap_dans_figure(data, fig, index=frame_index, timestamp=timestamp, utiliser_bords=utiliser_bords)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    frame_index += 1
                else:
                    print("[WARN] Données invalides")

        except KeyboardInterrupt:
            print("\n🛑 Simulation interrompue.")
            plt.close()

if __name__ == "__main__":
    traitement = TraitementDonnees(simulation=True)
    traitement.demarrer_acquisition_live(interval=1.0, utiliser_bords=True)
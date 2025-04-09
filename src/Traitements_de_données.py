import serial
import numpy as np
from scipy.interpolate import Rbf
import re


class TraitementDonnees:
    VREF = 3.3
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        self.positions = [
            ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
            ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
            ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R12", (-4.5, 8)),
            ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
            ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
            ("R21", (4.5, -8)), ("R22", (-4.5, -8)), ("R23", (3.5, -11.25)), ("R24", (-3.5, -11.25))
        ]

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
            return {i: np.random.uniform(0.4, 2.6) for i in range(24)}

        if self.ser is None:
            return None

        voltages_dict = {}
        while True:
            line = self.ser.readline().decode().strip()
            if not line:
                continue
            if "Fin du balayage" in line:
                break

            match = re.search(r"Canal (\d+): ([\d.]+) V", line)
            if match:
                canal = int(match.group(1))
                if 0 <= canal <= 23:
                    voltages_dict[canal] = float(match.group(2))

        return voltages_dict if len(voltages_dict) == 24 else None

    def get_temperatures(self):
        data = self.lire_donnees()
        if data is None:
            return None

        voltages = [data[i] for i in range(24)]
        resistances = [self.compute_resistance(v) for v in voltages]
        temperatures = [
            self.compute_temperature(resistances[i], self.coefficients[i])
            for i in range(24)
        ]
        return dict((name, temp) for (name, _), temp in zip(self.positions, temperatures))

    def afficher_heatmap_dans_figure(self, temperature_dict, fig):
        import matplotlib.pyplot as plt

        fig.clear()
        ax = fig.add_subplot(111)

        x, y, t = [], [], []
        for (name, pos) in self.positions:
            x.append(pos[0])
            y.append(pos[1])
            t.append(temperature_dict[name])

        rbf = Rbf(x, y, t, function='multiquadric', smooth=0.1)
        grid_size = 200
        r_max = max(np.hypot(np.array(x), np.array(y))) + 1

        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )

        ti = rbf(xi, yi)
        mask = xi**2 + yi**2 > r_max**2
        ti_masked = np.ma.array(ti, mask=mask)

        contour = ax.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour, ax=ax, label="Température (°C)")
        ax.set_aspect('equal')
        ax.set_title("Map de chaleur des températures des thermistances")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        fig.tight_layout()

    def demarrer_acquisition_live(self, interval=0.2):
        import matplotlib.pyplot as plt
        import time
        import os
        import csv
        from datetime import datetime
        from pathlib import Path

        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté. Wake up le moron!")
            return

        print("🚀 Acquisition live shit")
        fig = plt.figure(figsize=(6, 6))
        plt.ion()
        fig.show()

        # Stocker les données en mémoire
        all_data = []
        headers = [name for name, _ in self.positions] + ["T_ref", "timestamp"]

        try:
            while True:
                data = self.get_temperatures()

                if data:
                    os.system("clear")  # ou "cls" pour Windows
                    print("=" * 60)
                    print("Températures des 24 thermistances")
                    print("-" * 60)
                    for name, temp in data.items():
                        print(f"{name:<6} : {temp:6.2f} °C")
                    print("=" * 60)

                    self.afficher_heatmap_dans_figure(data, fig)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    ligne = [data[name] for name, _ in self.positions]
                    ligne.append(25.0)  # T_ref
                    ligne.append(datetime.now().isoformat(timespec='seconds'))
                    all_data.append(ligne)

                else:
                    print("Données incomplètes ou non reçues.")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n Acquisition stoppée. Sauvegarde en cours...")

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
    td.demarrer_acquisition_live(interval=0.2)

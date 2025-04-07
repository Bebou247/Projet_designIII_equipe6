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
        fig.clear()
        ax = fig.add_subplot(111)

        x, y, t = [], [], []
        for (name, pos) in self.positions:
            x.append(pos[0])
            y.append(pos[1])
            t.append(temperature_dict[name])

        rbf = Rbf(x, y, t, function='linear')
        xi, yi = np.meshgrid(np.linspace(min(x), max(x), 100),
                             np.linspace(min(y), max(y), 100))
        ti = rbf(xi, yi)

        contour = ax.contourf(xi, yi, ti, levels=100)
        fig.colorbar(contour, ax=ax, label="Température (°C)")
        ax.set_title("Heatmap des températures")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        fig.tight_layout()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    td = TraitementDonnees(simulation=False) 
    if not td.est_connecte():
        print("Arduino non connecté. Aucune acquisition, aucune heatmap, connecte le el cave")
    else:
        data = td.get_temperatures()

        if data:
            fig = plt.figure(figsize=(6, 6))
            td.afficher_heatmap_dans_figure(data, fig)
            plt.show()
        else:
            print("⚠️ Aucune donnée reçue.")



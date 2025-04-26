import serial
import numpy as np
from scipy.interpolate import Rbf
import re
import time
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import pandas as pd
from pathlib import Path
from scipy.ndimage import gaussian_filter
import math
from functools import reduce
import pandas as pd
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter



class TraitementDonnees:
    # Constantes
    VREF = 3.003 #[V]
    R_FIXED = 4700 #[Omhs]

    def __init__(self, port="/dev/cu.usbmodem101",path = "data/", coeffs_path="data/raw/coefficients.npy", simulation=False, fichier_simulation=None):
        # Initialisation des variables
        self.path = path
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)
        self.puissance = 0
        self.data_photodiodes = [0,0,0,0,0,0]
        self.puissance_hist = [0,0,0,0]
        self.puissance_hist_2 = [0,0,0,0]
        self.puissance_P = [0,0,0,0]
        self.puissance_I = [0,0,0,0]
        self.puissance_D = [0,0,0,0]
        self.puissance_DD = [0,0,0,0]
        self.time_test = [0, 0, 0]

        # fichier des ratios et des corrections des photodiodes
        self.correction_matrices = [pd.read_csv(self.path + f"matrice_corr_diode_{i}.csv", sep=',', decimal='.').values for i in range(6)]
        self.photodiode_ratios_VIS = [pd.read_csv(self.path + "ratios_photodiodes_VIS.csv", sep=';', decimal=',')[col].values
                                for col in pd.read_csv(self.path + "ratios_photodiodes_VIS.csv", sep=';', decimal=',').columns]
        self.photodiode_ratios_NIR = [pd.read_csv(self.path + "ratios_photodiodes_NIR.csv", sep=';', decimal=',')[col].values
                                for col in pd.read_csv(self.path + "ratios_photodiodes_NIR.csv", sep=';', decimal=',').columns]
        self.photodiode_ratios_IR = pd.read_csv(self.path + "ratios_photodiodes_IR.csv", sep=';', decimal=',').values
        self.photodiode_tensions_VIS = [pd.read_csv(self.path + "tensions_photodiodes_VIS.csv", sep=';', decimal=',')[col].values
                                    for col in pd.read_csv(self.path + "tensions_photodiodes_VIS.csv", sep=';', decimal=',').columns]
        self.photodiode_tensions_NIR = [pd.read_csv(self.path + "tensions_photodiodes_NIR.csv", sep=';', decimal=',')[col].values
                                    for col in pd.read_csv(self.path + "tensions_photodiodes_NIR.csv", sep=';', decimal=',').columns]

        # Décalage à appliquer aux positions des thermistances
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas en y


        self.tension_photodidodes = [0,0,0,0,0,0]

        self.positions = [
            ("R1", (11 + decalage_x, 0 + decalage_y)), ("R2", (3 + 1, 0 + decalage_y)), ("R3", (-3 + decalage_x, 0 + decalage_y)), ("R4", (-11 + decalage_x, 0 + decalage_y)),
            ("R5", (8 + 1, 2.5 - decalage_y)), ("R6", (0 + decalage_x, 2.5 + decalage_y)), ("R7", (-8 + decalage_x, 2.5 + decalage_y)), ("R8", (8 + 1, 5.5 - decalage_y)),
            ("R9", (0 + decalage_x, 5.5 + decalage_y)), ("R10", (-8 + decalage_x, 5.5 + decalage_y)), ("R11", (4.5 + decalage_x, 8 + decalage_y)), ("R24", (-3.5 + decalage_x, -11.25 + decalage_y)), # Note: R24 est sur le canal 11 physiquement
            ("R13", (4 + decalage_x, 11.25 + decalage_y)), ("R14", (-4 + decalage_x, 11.25 + decalage_y)), ("R15", (8 + 1, -2.5 - decalage_y)), ("R16", (0 + decalage_x, -2.5 + decalage_y)),
            ("R17", (-8 + decalage_x, -2.5 + decalage_y)), ("R18", (8 + 1, -5.5 - decalage_y)), ("R19", (0 + decalage_x, -5.5 + decalage_y)), ("R20", (-8 + decalage_x, -5.5 + decalage_y)),
            ("R21", (4.5 + decalage_x, -8 + decalage_y)), ("R25", (0 + decalage_x, -11.5 + decalage_y)),
           # thermistors virtuels pour arranger la carte thermique
            ("R_Virtuel", (-4.9, 7.8))
        ]

        self.photodiodes = ["PD25","PD26","PD27","PD28","PD29","PD30"]

        self.indices_à_garder = list(range(31)) 


        self.simulation_data = None
        self.simulation_index = 150
        self.simulation_columns = [p[0] for i, p in enumerate(self.positions) if p[0] != "R_Virtuel" and i in self.indices_à_garder]
        if "R25" in [p[0] for p in self.positions] and 24 not in self.indices_à_garder:
             if any(p[0] == "R25" for p in self.positions):
                 self.simulation_columns.append("R25")

        self.simulation_columns += self.photodiodes

        self.previous_ti_filtered = None
        self.previous_ti_filtered = None


        self.position_history = [] # Historique pour la médiane mobile
        self.history_length = 5    # Nombre de positions à garder 
        self.last_valid_raw_pos = None # Dernière position brute jugée valide 
        self.last_filtered_pos = (None, None) # Dernière position filtrée
        self.max_speed_mm_per_interval = 3.0 # Max déplacement en mm entre frames 
        self.min_heating_threshold = 0.05
        self.fichier_simulation = fichier_simulation

        if self.simulation:
            self.ser = None
            #print("SIMULATION! Mode simulation activé.")
            try:
                if self.fichier_simulation:
                    simulation_file_path = Path(self.fichier_simulation)
                else:
                    simulation_file_path = Path(__file__).parent.parent / "data" / "Échelons 976 nm.csv"

                self.simulation_data = pd.read_csv(simulation_file_path, sep=',', decimal='.')
                self.sauvegarde_resultats = []
                #print(f"SIMULATION! Chargement du fichier CSV : {simulation_file_path.resolve()}")

                # Nettoyage et validation
                for col in self.simulation_data.columns:
                    self.simulation_data[col] = pd.to_numeric(self.simulation_data[col], errors='coerce')
                if self.simulation_data.isnull().values.any():
                    #print("SIMULATION! Attention : des valeurs manquantes ont été trouvées dans le fichier CSV.")
                    self.simulation_data.dropna(inplace=True)

            except FileNotFoundError:
                print(f"ERREUR SIMULATIO] Fichier non trouvé : {simulation_file_path}")
                self.simulation_data = None
            except Exception as e:
                print(f"ERREUR SIMULATION Problème lors du chargement du fichier CSV : {e}")
                self.simulation_data = None
        else:
            try:
                self.ser = serial.Serial(self.port, 9600, timeout=0.2)
                print(f"INFO Port série connecté sur {self.port}")
            except Exception as e:
                print(f"ERREUR Impossible d'ouvrir le port série : {e}")
                self.ser = None

    # Vérifier si la connexion série est établie
    def est_connecte(self):
        return self.ser is not None


    # Calculer la température à partir de la résistance
    def steinhart_hart_temperature(self, R, A, B, C):
        if R <= 0:
            return np.nan
        with np.errstate(invalid='ignore'):
            log_R = np.log(R)
            denominator = A + B * log_R + C * (log_R**3)
            if denominator == 0:
                return np.nan
            temp_K = 1 / denominator
        return temp_K

    # Calculer la résistance à partir de la tension
    def compute_resistance(self, voltage):
        if voltage <= 0:
             return -1
        if voltage >= self.VREF:
            return float('inf')
        denominator = self.VREF - voltage
        if abs(denominator) < 1e-9:
            return float('inf')
        resistance = self.R_FIXED * (voltage / denominator)
        return resistance

    # Calculer la température à partir de la résistance
    def compute_temperature(self, resistance, coeffs):
        if resistance == float('inf') or resistance <= 0 or pd.isna(resistance):
            return np.nan
        A, B, C = coeffs
        kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
        if pd.isna(kelvin):
            return np.nan
        return kelvin - 273.15

    # Lire les données de la connexion série
    def lire_donnees(self):
        if self.simulation:
            return self.simulation_data is not None and not self.simulation_data.empty

        if self.ser is None:
            print("[ERREUR] Connexion série non établie.")
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        start_time = time.time()
        timeout_sec = 2 

        while True:
            current_time = time.time()

            if current_time - start_time > timeout_sec:
                print(f"Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                return voltages_dict if voltages_dict else None
            

            try:
                if self.ser.in_waiting >= 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line:
                        continue

                    if "Fin du balayage" in line:
                        break 

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        canal = int(match.group(1))
                        if canal in self.indices_à_garder:
                            try:
                                 voltages_dict[canal] = float(match.group(2))
                            except ValueError:
                                 print(f"[AVERTISSEMENT] Impossible de convertir la tension '{match.group(2)}' pour le canal {canal}")
                                 voltages_dict[canal] = np.nan

                    time.sleep(0.01)

            except serial.SerialException as e:
                print(f"Erreur série pendant la lecture : {e}")
                self.ser = None 
                return None
            except Exception as e:
                print(f"Erreur inattendue pendant la lecture série : {e}")
                continue
        #Vérification des canaux reçus
        canaux_attendus = set(self.indices_à_garder)
        canaux_recus = set(voltages_dict.keys())

        if canaux_recus != canaux_attendus:
             canaux_manquants = canaux_attendus - canaux_recus
             print(f"⚠️ Seulement {len(canaux_recus)}/{len(canaux_attendus)} canaux requis reçus. Manquants: {sorted(list(canaux_manquants))}")
             return None 


        data_phot = [0,0,0,0,0,0]

        # Prendre les tensions des photodiodes
        for i in range(25, 31):
            data_phot[i-25] = voltages_dict[i]


        self.data_photodiodes = data_phot


        # Retourner un dictionnaire avec les tensions des canaux
        return voltages_dict


    def get_temperatures(self):
        real_temps_dict = {}  # Dictionnaire pour les températures réelles
        real_tension_dict = {}

        if self.simulation:
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0

                current_data_row = self.simulation_data.iloc[self.simulation_index]

                self.data_photodiodes = [current_data_row.get(f"PD{i}", 0.0) for i in range(25, 31)]

                self.simulation_index += 1
                valid_data_found = False

            
                for i, (name, _) in enumerate(self.positions):
                    if name in ["R_Virtuel", "R24"]: continue

                    if name in self.simulation_columns:
                        if name in current_data_row and pd.notna(current_data_row[name]):
                            real_temps_dict[name] = current_data_row[name]
                            valid_data_found = True
                        else:
                            real_temps_dict[name] = np.nan

                    elif name == "R25" and "R25" in self.simulation_data.columns:
                        if "R25" in current_data_row and pd.notna(current_data_row["R25"]):
                            real_temps_dict["R25"] = current_data_row["R25"]
                            valid_data_found = True
                        else:
                            real_temps_dict["R25"] = np.nan

    
                for i, name in enumerate(self.photodiodes):
                    if name in self.simulation_columns:
                        if name in current_data_row and pd.notna(current_data_row[name]):
                            real_temps_dict[name] = current_data_row[name]
                            valid_data_found = True
                        else:
                            real_temps_dict[name] = 0.0  

                if not valid_data_found:
                    real_temps_dict["R24"] = np.nan
                    real_temps_dict["R_Virtuel"] = np.nan
                    return real_temps_dict

                weighted_sum_r24 = 0.0
                total_weight_r24 = 0.0
                thermistors_r24_weights = {"R19": 0.1, "R20": 0.15, "R21": 0.15}
                other_thermistors_for_r24 = []

                for name, temp in real_temps_dict.items():
                    if name != "R25" and name not in thermistors_r24_weights and pd.notna(temp):
                        other_thermistors_for_r24.append(name)

                weight_per_other_r24 = 0.0
                if other_thermistors_for_r24:
                    weight_per_other_r24 = 0.6 / len(other_thermistors_for_r24)

                for name, temp in real_temps_dict.items():
                    if name == "R25":
                        continue
                    if pd.notna(temp):
                        if name in thermistors_r24_weights:
                            weight = thermistors_r24_weights[name]
                        elif name in other_thermistors_for_r24:
                            weight = weight_per_other_r24
                        else:
                            continue
                        weighted_sum_r24 += temp * weight
                        total_weight_r24 += weight

                if total_weight_r24 > 1e-6:
                    real_temps_dict["R24"] = weighted_sum_r24 / total_weight_r24
                else:
                    real_temps_dict["R24"] = np.nan

            else:
                print("[SIMULATION] Données CSV non disponibles ou vides, génération de températures aléatoires.")


        else:
            data_voltages = self.lire_donnees()
    

        return real_temps_dict


    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        ax = fig.add_subplot(111) 

        x_all_points, y_all_points, t_all_points = [], [], []
        valid_temps_list = []
        thermistor_data_for_plot = [] 

        for name, pos in self.positions:
            if name == "R25": continue
            temp_val = temperature_dict.get(name, np.nan)
            if pd.notna(temp_val):
                x_all_points.append(pos[0])
                y_all_points.append(pos[1])
                t_all_points.append(temp_val)
                valid_temps_list.append(temp_val)
                thermistor_data_for_plot.append({"name": name, "pos": pos, "temp": temp_val})

        if not valid_temps_list:
            baseline_temp = 20.0
        else:
            baseline_temp = min(valid_temps_list) - 0.5

        r_max = 12.5 #mm
        num_edge_points = 12
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [baseline_temp] * num_edge_points

        x_combined = x_all_points + list(edge_x)
        y_combined = y_all_points + list(edge_y)
        t_combined = t_all_points + edge_t

        ti_filtered = None
        xi, yi = None, None
        mask = None
        grad_magnitude = None
        grad_magnitude_masked = None
        raw_laser_x, raw_laser_y = None, None
        raw_pos_found_this_frame = False
        final_laser_pos_found = False

        if len(x_combined) < 3:
            ax.set_title("Pas assez de données (hors R25) pour faire l'interpolation")
            return

        try:
            rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
            grid_size = 100
            xi, yi = np.meshgrid(
                np.linspace(-r_max, r_max, grid_size),
                np.linspace(-r_max, r_max, grid_size)
            )
            ti = rbf(xi, yi)
            sigma_filter_temp = 1.2
            ti_filtered = gaussian_filter(ti, sigma=sigma_filter_temp)
            mask = xi**2 + yi**2 > r_max**2

            grad_y, grad_x = np.gradient(ti_filtered)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_magnitude_masked = np.ma.array(grad_magnitude, mask=mask)

            if self.previous_ti_filtered is not None and self.previous_ti_filtered.shape == ti_filtered.shape:
                difference_map = ti_filtered - self.previous_ti_filtered
                sigma_diff_filter = 1.5
                filtered_difference_map = gaussian_filter(difference_map, sigma=sigma_diff_filter)
                filtered_difference_map_masked = np.ma.array(filtered_difference_map, mask=mask)
                try:
                    max_diff_idx_flat = np.nanargmax(filtered_difference_map_masked.filled(np.nan))
                    max_diff_idx = np.unravel_index(max_diff_idx_flat, filtered_difference_map_masked.shape)
                    max_diff_val = filtered_difference_map_masked[max_diff_idx]

                    if max_diff_val < self.min_heating_threshold:
                        raw_pos_found_this_frame = False
                    else:
                        search_radius_pixels = 20
                        rows, cols = np.indices(grad_magnitude.shape)
                        dist_sq_from_max_diff = (rows - max_diff_idx[0])**2 + (cols - max_diff_idx[1])**2
                        in_search_area = dist_sq_from_max_diff <= search_radius_pixels**2
                        grad_search_map = grad_magnitude.copy()
                        grad_search_map[~in_search_area | mask] = np.nan
                        min_grad_idx_flat = np.nanargmin(grad_search_map)

                        if not np.isnan(grad_search_map.flat[min_grad_idx_flat]):
                            min_grad_idx = np.unravel_index(min_grad_idx_flat, grad_search_map.shape)
                            potential_laser_x = xi[min_grad_idx]
                            potential_laser_y = yi[min_grad_idx]
                            distance_from_center = math.sqrt(potential_laser_x**2 + potential_laser_y**2)
                            max_allowed_distance = 9.5
                            is_within_radius = distance_from_center <= max_allowed_distance
                            is_plausible_move = True
                            if self.last_valid_raw_pos is not None and is_within_radius:
                                prev_x, prev_y = self.last_valid_raw_pos
                                dist_moved_sq = (potential_laser_x - prev_x)**2 + (potential_laser_y - prev_y)**2
                                if dist_moved_sq > self.max_speed_mm_per_interval**2:
                                    is_plausible_move = False

                            if is_within_radius and is_plausible_move:
                                raw_laser_x = potential_laser_x
                                raw_laser_y = potential_laser_y
                                raw_pos_found_this_frame = True
                                self.last_valid_raw_pos = (raw_laser_x, raw_laser_y)
                            else:
                                raw_pos_found_this_frame = False
                        else:
                            raw_pos_found_this_frame = False
                except (ValueError, IndexError):
                    raw_pos_found_this_frame = False
            else:
                raw_pos_found_this_frame = False
                self.last_valid_raw_pos = None
                if self.previous_ti_filtered is None:
                    print("Aucune carte de température précédente pour la comparaison.")
                elif self.previous_ti_filtered.shape != ti_filtered.shape:
                     self.previous_ti_filtered = None

            if raw_pos_found_this_frame:
                self.position_history.append((raw_laser_x, raw_laser_y))

            self.position_history = self.position_history[-self.history_length:]
            filtered_laser_x, filtered_laser_y = None, None
            if len(self.position_history) > 0:
                valid_x = [p[0] for p in self.position_history]
                valid_y = [p[1] for p in self.position_history]
                if len(valid_x) >= 3:
                    filtered_laser_x = np.median(valid_x)
                    filtered_laser_y = np.median(valid_y)
                elif len(valid_x) > 0:
                     filtered_laser_x = np.mean(valid_x)
                     filtered_laser_y = np.mean(valid_y)

            if filtered_laser_x is not None:
                self.last_filtered_pos = (filtered_laser_x, filtered_laser_y)
                final_laser_pos_found = True
            else:
                self.last_filtered_pos = (None, None)
                final_laser_pos_found = False

        except Exception as e:
             ax.set_title("Erreur Calcul Gradient/Laser")
             self.previous_ti_filtered = None
             self.last_valid_raw_pos = None
             self.position_history = []
             self.last_filtered_pos = (None, None)
             final_laser_pos_found = False
             return 

        if ti_filtered is not None:
            self.previous_ti_filtered = ti_filtered.copy()

        if grad_magnitude_masked is not None:
            contour = ax.contourf(xi, yi, grad_magnitude_masked, levels=50, cmap="viridis")
            fig.colorbar(contour, ax=ax, label="Magnitude Gradient Temp. (°C/mm)", shrink=0.8) # Ajusté shrink
            ax.scatter(x_all_points, y_all_points, color='white', marker='.', s=10, alpha=0.5, label='Thermistances')

            plot_x, plot_y = self.last_filtered_pos
            if final_laser_pos_found:

                label_laser = f'Laser (Médiane {len(self.position_history)}/{self.history_length}) @ ({plot_x:.1f}, {plot_y:.1f})'
                ax.plot(plot_x, plot_y, 'rx', markersize=10, label=label_laser) # Croix rouge dans la carte

            ax.set_aspect('equal')
            ax.set_title(f"Gradient Température (Tps: {elapsed_time:.2f} s)", fontsize=10) # Ajusté fontsize
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)
            ax.legend(fontsize=8, loc='upper right') 

        else:
            ax.set_title("Gradient non calculé")
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)


    def demarrer_acquisition_simulation(self, interval=0.3):
        if not hasattr(self, "simulation_data") or self.simulation_data is None:
            return


        self.sauvegarde_resultats = []

        fig = plt.figure(figsize=(12, 6))  
        plt.ion()
        fig.show()

        start_time = time.time()

        for idx, row in self.simulation_data.iterrows():
            elapsed_time = time.time() - start_time

            V_photodiodes = [row[f"PD{i}"] for i in range(25, 31)]
            self.data_photodiodes = V_photodiodes

            light_type, wavelength, puissance = self.get_wavelength()



            # On récupère les températures
            temperature_dict = {name: row[name] for name, _ in self.positions if name in row}

            # On affiche la carte de température
            try:
                self.afficher_heatmap_dans_figure(temperature_dict, fig, row['temps_ecoule_s'])
                fig.canvas.draw()
                fig.canvas.flush_events()
            except Exception as e:
                print(f"[ERREUR HEATMAP] {e}")

            self.sauvegarde_resultats.append({
                "temps_ecoule_s": row["temps_ecoule_s"],
                "spectre": light_type,
                "longueur_onde_nm": wavelength,
                "puissance_W": puissance
            })

            time.sleep(interval)

    # Sauvegarder les résultats dans un fichier CSV
        df_resultats = pd.DataFrame(self.sauvegarde_resultats)
        fichier_resultat = Path("data") / "resultats_photodiodes_simulation.csv"
        df_resultats.to_csv(fichier_resultat, index=False)

        print(f"Résultats de la simulation sauvegardés dans : {fichier_resultat.resolve()}")


    def id_pos(self, pos):
        extremas = 6
        inter = np.linspace(-extremas, extremas, len(self.correction_matrices[0]))
        delta = np.abs(inter - np.array(pos)[:, None])
        return delta[0].argmin(), delta[1].argmin()

    def indexes(self, array, target, threshold=0.1):
        return np.where(np.abs(array - target) <= np.maximum(np.abs(target) * threshold, threshold))[0]

    def precise_wavelength(self, func, *args, threshold, threshold_mult, max_iter=20):
        for _ in range(max_iter):
            wavelength = func(*args, threshold)
            if len(wavelength) == 1:
                return wavelength
            if len(wavelength) == 0:
                return self.precise_wavelength(func, *args, threshold=threshold / threshold_mult, threshold_mult=np.sqrt(threshold_mult))
            threshold *= threshold_mult
        return wavelength

    def get_VIS_wavelength(self, V_corr, threshold=0.1):
        V_corr[-2] = 0
        ratios_corr = np.divide(V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0)
        ratio_ids_corr = [self.indexes(self.photodiode_ratios_VIS[i], ratio, threshold) for i, ratio in enumerate(ratios_corr)]
        if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
            return np.array([])
        return reduce(np.intersect1d, ratio_ids_corr)

    def get_NIR_wavelength(self, V_corr, threshold=0.1):
        ratios_corr = np.divide(V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0)
        ratio_ids_corr = [self.indexes(self.photodiode_ratios_NIR[i], ratio, threshold) for i, ratio in enumerate(ratios_corr)]
        if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
            return np.array([])
        return reduce(np.intersect1d, ratio_ids_corr)

    def get_IR_wavelength(self, V_corr, puissance, threshold):
        if puissance != 0:
            ratio = V_corr / puissance
        else:
            return [0]
        return self.indexes(self.photodiode_ratios_IR, ratio, threshold)

    def get_VIS_power(self, wavelength, V_corr):
        V_corr[-2] = 0
        V_corr[-1] = 0
        V_ratio = [10 * V_corr[i] / self.photodiode_tensions_VIS[i][int(wavelength) - 200] for i in range(6)
                if self.photodiode_tensions_VIS[i][int(wavelength) - 200] != 0 and V_corr[i] != 0]
        return np.mean(V_ratio)

    def get_NIR_power(self, wavelength, V_corr):
        V_ratio = [10 * V_corr[i] / self.photodiode_tensions_NIR[i][int(wavelength) - 200] for i in range(6)
                if self.photodiode_tensions_NIR[i][int(wavelength) - 200] != 0 and V_corr[i] != 0]
        return np.mean(V_ratio)
    
    def get_IR_power(self):
        return self.puissance 

    def get_wavelength(self, threshold=0.5, threshold_mult=0.5):
        if self.last_valid_raw_pos is None:
            y, x = (0, 0)
        else:
            y, x = self.last_valid_raw_pos

        pos = self.id_pos((x, y))

        V_photodiodes = self.data_photodiodes


        for i, V in enumerate(V_photodiodes):
            if V < 0.05:
                V_photodiodes[i] = 0
   
        V_corr = np.array([V * self.correction_matrices[i][pos] for i, V in enumerate(V_photodiodes)])
        index_max = np.argmax(V_corr)

        if all(V < 0.1 for V in V_corr):
            return "inconnu", 0, 0
        elif index_max == 0:
            return "UV", 358, V_corr[0]/0.04
        elif index_max == 1:
            self.wavelength = np.mean(self.precise_wavelength(self.get_VIS_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "VIS", self.wavelength, self.get_VIS_power(self.wavelength, V_corr)
        elif index_max == 5:
            self.wavelength = np.mean(self.precise_wavelength(self.get_IR_wavelength, V_corr[-1], self.puissance, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "IR", self.wavelength, self.get_IR_power()
        else:
            self.wavelength = np.mean(self.precise_wavelength(self.get_NIR_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "NIR", self.wavelength, self.get_NIR_power(self.wavelength, V_corr)




## Estimation de la puissance à partir des données de la ligne si on est dans l'IR
    def estimate_power_from_row(self, row, dt):
        try:
            temperatures = list(row.values())[:25]
            T_max = np.nanmax(temperatures)

            T_ref = float(row["R25"]) if pd.notna(row["R25"]) else 25.0
            delta_T = T_max - T_ref

            kp = 0.56
            ki = -0.012
            kd = 12
            kdd = -4
            bias = -0.5
    
            if not hasattr(self, "delta_T_hist"):
                self.delta_T_hist = []
            self.delta_T_hist.append(delta_T)
   

            if len(self.delta_T_hist) > 100:
                self.delta_T_hist.pop(0)

            delta_T_array = np.array(self.delta_T_hist)
            delta_T_filt = 1/3*(delta_T_array[:-2] + delta_T_array[1:-1] + delta_T_array[2:])
            d_delta_T_dt = np.gradient(delta_T_filt, dt)
            integral = np.cumsum(delta_T_filt) * dt
            dd_delta_T_dt = np.gradient(d_delta_T_dt, dt)
            P = kp * delta_T_filt[-1]
            D = kd * d_delta_T_dt[-1]
            DD = kdd * dd_delta_T_dt[-1]
            I = ki * integral[-1]
            self.puissance_hist.append(max(0, P + D + DD + I + bias))
            self.puissance_P.append(delta_T_filt[-1])
            self.puissance_I.append(integral[-1])
            self.puissance_D.append(d_delta_T_dt[-1])
            self.puissance_DD.append(dd_delta_T_dt[-1])
            puissance = np.mean(self.puissance_hist[-10:])

            if puissance > 0.5:
                self.puissance = puissance
            else:
                self.puissance = 0
            self.puissance_hist_2.append(self.puissance)
        except Exception as e:
            self.puissance_est_temp_live = 0

if __name__ == "__main__":
    td = TraitementDonnees(simulation=True) 
    #td.demarrer_acquisition_simulation(interval=0.1)
    #td.afficher_heatmap_dans_figure()


    
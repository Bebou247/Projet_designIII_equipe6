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
            ("R1", (11 + decalage_x, 0 + decalage_y)), ("R2", (3 + 1, 0 + decalage_y)), ("R3", (-3 + decalage_x, 0 + decalage_y)), ("R4", (-11 + decalage_x, 0 + decalage_y)),
            ("R5", (8 + decalage_x, 2.5 + decalage_y)), ("R6", (0 + decalage_x, 2.5 + decalage_y)), ("R7", (-8 + decalage_x, 2.5 + decalage_y)), ("R8", (8 + decalage_x, 5.5 + decalage_y)),
            ("R9", (0 + decalage_x, 5.5 + decalage_y)), ("R10", (-8 + decalage_x, 5.5 + decalage_y)), ("R11", (4.5 + decalage_x, 8 + decalage_y)), ("R24", (-3.5 + decalage_x, -11.25 + decalage_y)), # Note: R24 est sur le canal 11 physiquement
            ("R13", (4 + decalage_x, 11.25 + decalage_y)), ("R14", (-4 + decalage_x, 11.25 + decalage_y)), ("R15", (8 + decalage_x, -2.5 + decalage_y)), ("R16", (0 + decalage_x, -2.5 + decalage_y)),
            ("R17", (-8 + decalage_x, -2.5 + decalage_y)), ("R18", (8 + decalage_x, -5.5 + decalage_y)), ("R19", (0 + decalage_x, -5.5 + decalage_y)), ("R20", (-8 + decalage_x, -5.5 + decalage_y)),
            ("R21", (4.5 + decalage_x, -8 + decalage_y)), ("R25", (0 + decalage_x, -11.5 + decalage_y)), # R25 est la référence, souvent sur canal 24
            # --- NOUVELLE THERMISTANCE VIRTUELLE ---
            ("R_Virtuel", (-4.9, 7.8))
        ]

        # Indices des *vraies* thermistances à lire (0 à 20 + R25 si elle est lue)
        # R24 est gérée spécialement car elle utilise le canal 11
        # R_Virtuel n'est PAS lue, elle est calculée.
        self.indices_à_garder = list(range(21)) # R1-R11, R13-R21 (R24 est sur canal 11)
        # Si R25 est lue physiquement (ex: canal 24), ajoutez son index ici:
        # self.indices_à_garder.append(24) # Exemple si R25 est sur canal 24

        self.simulation_data = None
        self.simulation_index = 100
        # Noms des colonnes pour la simulation (basés sur les *vraies* thermistances)
        self.simulation_columns = [self.positions[i][0] for i, (nom, _) in enumerate(self.positions) if nom != "R_Virtuel" and i in self.indices_à_garder]
        # Ajouter R25 si elle est simulée depuis le CSV
        if "R25" in [p[0] for p in self.positions] and 24 not in self.indices_à_garder: # Si R25 existe mais n'est pas lue directement
             if any(p[0] == "R25" for p in self.positions): # Vérifie si R25 est dans la liste
                 self.simulation_columns.append("R25")
        self.previous_ti_filtered = None
        self.grid_shape = None

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                script_dir = Path(__file__).parent
                simulation_file_path = script_dir.parent / "data" / "Hauteur 1.csv"
                self.simulation_data = pd.read_csv(simulation_file_path)
                print(f"[SIMULATION] Chargement du fichier CSV : {simulation_file_path.resolve()}")

                # Vérification des colonnes nécessaires pour la simulation
                missing_cols = [col for col in self.simulation_columns if col not in self.simulation_data.columns]
                if missing_cols:
                    print(f"[ERREUR SIMULATION] Colonnes manquantes dans {simulation_file_path.name}: {missing_cols}")
                    self.simulation_data = None
                else:
                    for col in self.simulation_columns:
                        self.simulation_data[col] = pd.to_numeric(self.simulation_data[col], errors='coerce')
                    print(f"[SIMULATION] Fichier CSV chargé. {len(self.simulation_data)} lignes trouvées.")
                    if self.simulation_data.isnull().values.any():
                        print("[AVERTISSEMENT SIMULATION] Le fichier CSV contient des valeurs non numériques après conversion.")

            except FileNotFoundError:
                print(f"[ERREUR SIMULATION] Fichier non trouvé : {simulation_file_path.resolve()}")
                self.simulation_data = None
            except Exception as e:
                print(f"[ERREUR SIMULATION] Impossible de charger ou lire le fichier CSV : {e}")
                self.simulation_data = None
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
        # Ajout d'une vérification pour R <= 0
        if R <= 0:
            # print(f"[DEBUG] Résistance invalide pour log: {R}")
            return np.nan # Ou une autre valeur indiquant une erreur
        with np.errstate(invalid='ignore'): # Ignore les avertissements pour log(R) si R est très petit mais > 0
            log_R = np.log(R)
            denominator = A + B * log_R + C * (log_R**3)
            if denominator == 0:
                return np.nan # Éviter la division par zéro
            temp_K = 1 / denominator
        return temp_K

    def compute_resistance(self, voltage):
        # Ajout de vérifications pour tension invalide
        if voltage <= 0:
             # print(f"[DEBUG] Tension négative ou nulle: {voltage}")
             return -1 # Ou une autre valeur signalant une erreur
        if voltage >= self.VREF:
            # print(f"[DEBUG] Tension >= VREF: {voltage}")
            return float('inf')
        # Vérifier si VREF - voltage est trop proche de zéro
        denominator = self.VREF - voltage
        if abs(denominator) < 1e-9: # Seuil très petit pour éviter l'instabilité
            # print(f"[DEBUG] Dénominateur proche de zéro: {denominator}")
            return float('inf') # Considérer comme résistance infinie
        resistance = self.R_FIXED * (voltage / denominator)
        # print(f"[DEBUG] V={voltage:.3f} -> R={resistance:.2f}")
        return resistance


    def compute_temperature(self, resistance, coeffs):
        if resistance == float('inf') or resistance <= 0 or pd.isna(resistance):
            return np.nan
        A, B, C = coeffs
        kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
        if pd.isna(kelvin):
            return np.nan
        return kelvin - 273.15

    def lire_donnees(self):
        if self.simulation:
            if self.simulation_data is not None and not self.simulation_data.empty:
                return True
            else:
                return None

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
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                return voltages_dict if voltages_dict else None

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line:
                        continue

                    if "Fin du balayage" in line:
                        break

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        canal = int(match.group(1))
                        # Lire seulement les canaux correspondant aux VRAIES thermistances
                        if canal in self.indices_à_garder:
                             try:
                                 voltages_dict[canal] = float(match.group(2))
                             except ValueError:
                                 print(f"[AVERTISSEMENT] Impossible de convertir la tension '{match.group(2)}' pour le canal {canal}")
                                 voltages_dict[canal] = np.nan # Marquer comme invalide

                else:
                    time.sleep(0.01)

            except serial.SerialException as e:
                print(f"Erreur série pendant la lecture : {e}")
                self.ser = None
                return None
            except Exception as e:
                print(f"Erreur inattendue pendant la lecture série : {e}")
                continue

        # Vérifier si tous les canaux *attendus* ont été lus
        canaux_attendus = set(self.indices_à_garder)
        canaux_recus = set(voltages_dict.keys())

        if canaux_recus != canaux_attendus:
             canaux_manquants = canaux_attendus - canaux_recus
             print(f"⚠️ Seulement {len(canaux_recus)}/{len(canaux_attendus)} canaux requis reçus. Manquants: {canaux_manquants}")
             # Optionnel: Remplir les manquants avec NaN si on veut quand même continuer
             # for canal_manquant in canaux_manquants:
             #     voltages_dict[canal_manquant] = np.nan
             # return voltages_dict # Retourner données partielles + NaN
             return None # Préférable de retourner None si incomplet

        return voltages_dict

    def get_temperatures(self):
        real_temps_dict = {} # Dictionnaire pour les températures réelles

        if self.simulation:
            # --- Logique Simulation CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0
                    print("[SIMULATION] Fin du fichier CSV atteinte, retour au début.")

                current_data_row = self.simulation_data.iloc[self.simulation_index]
                self.simulation_index += 1
                valid_data_found = False

                # Lire les températures simulées pour les thermistances réelles (SAUF R24 pour l'instant)
                for i, (name, _) in enumerate(self.positions):
                    if name in ["R_Virtuel", "R24"]: continue # Ignorer la virtuelle et R24 ici
                    if name in self.simulation_columns: # Vérifier si la colonne existe
                        if name in current_data_row and pd.notna(current_data_row[name]):
                            real_temps_dict[name] = current_data_row[name]
                            valid_data_found = True
                        else:
                            real_temps_dict[name] = np.nan
                    # Gérer R25 spécifiquement si elle n'est pas dans indices_à_garder mais dans positions
                    elif name == "R25" and "R25" in self.simulation_data.columns:
                         if "R25" in current_data_row and pd.notna(current_data_row["R25"]):
                             real_temps_dict["R25"] = current_data_row["R25"]
                             valid_data_found = True
                         else:
                             real_temps_dict["R25"] = np.nan

                if not valid_data_found:
                    print(f"[ERREUR SIMULATION] Aucune donnée valide (hors R24) à l'index CSV {self.simulation_index - 1}.")
                    # Ajouter R24 et la virtuelle avec NaN avant de retourner
                    real_temps_dict["R24"] = np.nan
                    real_temps_dict["R_Virtuel"] = np.nan
                    return real_temps_dict # Retourne le dict avec NaN

                # --- Logique R24 (Moyenne Pondérée) pour Simulation ---
                weighted_sum_r24 = 0.0
                total_weight_r24 = 0.0
                thermistors_low_weight_r24 = ["R20", "R21"]
                thermistors_low_weight_r19 = ["R19"]
                other_real_thermistors_for_r24 = []

                # Identifier les autres thermistances réelles valides (pour R24)
                for name, temp in real_temps_dict.items():
                    # Exclure R25 du calcul de R24 si elle existe et est valide
                    if name != "R25" and name not in thermistors_low_weight_r24 and name not in thermistors_low_weight_r19 and pd.notna(temp):
                         other_real_thermistors_for_r24.append(name)

                # Calculer le poids pour les "autres" thermistances (pour R24)
                weight_per_other_r24 = 0.0
                if other_real_thermistors_for_r24:
                    weight_per_other_r24 = 0.6 / len(other_real_thermistors_for_r24)

                # Calculer la somme pondérée et le poids total (pour R24)
                for name, temp in real_temps_dict.items():
                     # Exclure R25 du calcul de R24
                     if name == "R25": continue
                     if pd.notna(temp):
                         if name in thermistors_low_weight_r24:
                             weight = 0.15
                         elif name in thermistors_low_weight_r19:
                             weight = 0.1    
                         elif name in other_real_thermistors_for_r24:
                             weight = weight_per_other_r24
                         else:
                             continue

                         weighted_sum_r24 += temp * weight
                         total_weight_r24 += weight

                # Assigner la valeur à R24
                if total_weight_r24 > 1e-6:
                    real_temps_dict["R24"] = weighted_sum_r24 / total_weight_r24
                else:
                    real_temps_dict["R24"] = np.nan

            else:
                # Fallback: Génération aléatoire si CSV échoue
                print("[SIMULATION] Données CSV non disponibles, génération de températures aléatoires.")
                temp_gen_dict = {}
                for i, (name, _) in enumerate(self.positions):
                     if name != "R_Virtuel": # Ne pas générer pour la virtuelle initialement
                         temp_gen_dict[name] = np.random.uniform(20.0, 45.0)

                # Calculer R24 à partir des valeurs générées (hors R24 elle-même et R25)
                weighted_sum_r24_gen = 0.0
                total_weight_r24_gen = 0.0
                thermistors_low_weight_r24_gen = ["R19", "R20", "R21"]
                other_real_thermistors_for_r24_gen = []

                for name, temp in temp_gen_dict.items():
                    if name != "R24" and name != "R25" and name not in thermistors_low_weight_r24_gen and pd.notna(temp):
                         other_real_thermistors_for_r24_gen.append(name)

                weight_per_other_gen_r24 = 0.0
                if other_real_thermistors_for_r24_gen:
                    weight_per_other_gen_r24 = 0.7 / len(other_real_thermistors_for_r24_gen)

                for name, temp in temp_gen_dict.items():
                     if name != "R24" and name != "R25" and pd.notna(temp):
                         if name in thermistors_low_weight_r24_gen:
                             weight = 0.1
                         elif name in other_real_thermistors_for_r24_gen:
                             weight = weight_per_other_gen_r24
                         else:
                             continue
                         weighted_sum_r24_gen += temp * weight
                         total_weight_r24_gen += weight

                if total_weight_r24_gen > 1e-6:
                    temp_gen_dict["R24"] = weighted_sum_r24_gen / total_weight_r24_gen
                else:
                    temp_gen_dict["R24"] = np.nan

                real_temps_dict = temp_gen_dict # Assigner le dict généré

        else:
            # --- Logique Lecture Série ---
            data_voltages = self.lire_donnees()
            if data_voltages is None:
                 for i, (name, _) in enumerate(self.positions):
                     real_temps_dict[name] = np.nan
                 return real_temps_dict

            temperatures_raw = {}
            indices_mapping = {
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10,
                # R24 (canal 11) est traitée séparément
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                # "R25": 24 # Si R25 est lue sur canal 24
            }
            coeffs_mapping = {
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10,
                # R24 (coeffs[23]) n'est plus calculée directement ici
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                # "R25": 24 # Si R25 utilise coeffs[24]
            }

            # Calcul initial des températures réelles (SAUF R24)
            for nom_thermistor, canal_index in indices_mapping.items():
                if canal_index == 11: continue # Ignorer canal 11 (R24)

                if canal_index not in data_voltages or pd.isna(data_voltages[canal_index]):
                    print(f"[AVERTISSEMENT] Tension manquante ou invalide pour {nom_thermistor} (canal {canal_index})")
                    temperatures_raw[nom_thermistor] = np.nan
                    continue

                voltage = data_voltages[canal_index]
                coeffs_index = coeffs_mapping.get(nom_thermistor, -1)

                if coeffs_index == -1 or coeffs_index >= len(self.coefficients):
                    print(f"[ERREUR] Index coeff {coeffs_index} invalide pour {nom_thermistor}.")
                    temperatures_raw[nom_thermistor] = np.nan
                    continue

                coeffs = self.coefficients[coeffs_index]
                resistance = self.compute_resistance(voltage)
                temp = self.compute_temperature(resistance, coeffs)
                temperatures_raw[nom_thermistor] = temp

            # Ajouter R25 si elle est lue physiquement
            # if 24 in self.indices_à_garder and 24 in data_voltages:
            #     voltage_r25 = data_voltages[24]
            #     coeffs_r25 = self.coefficients[24] # Assumer coeffs[24] pour R25
            #     resistance_r25 = self.compute_resistance(voltage_r25)
            #     temp_r25 = self.compute_temperature(resistance_r25, coeffs_r25)
            #     temperatures_raw["R25"] = temp_r25
            # elif "R25" in [p[0] for p in self.positions]: # Si R25 existe mais n'est pas lue
            #     temperatures_raw["R25"] = np.nan # Initialiser à NaN

            real_temps_dict = temperatures_raw.copy()

            # --- Logique R24 (Moyenne Pondérée) pour Lecture Série ---
            weighted_sum_r24_real = 0.0
            total_weight_r24_real = 0.0
            thermistors_low_weight_r24_real = ["R19", "R20", "R21"]
            other_real_thermistors_for_r24_real = []

            # Identifier les autres thermistances réelles valides (déjà calculées, hors R25)
            for name, temp in real_temps_dict.items():
                if name != "R25" and name not in thermistors_low_weight_r24_real and pd.notna(temp):
                     other_real_thermistors_for_r24_real.append(name)

            # Calculer le poids pour les "autres" thermistances (pour R24)
            weight_per_other_r24_real = 0.0
            if other_real_thermistors_for_r24_real:
                weight_per_other_r24_real = 0.7 / len(other_real_thermistors_for_r24_real)

            # Calculer la somme pondérée et le poids total (pour R24)
            for name, temp in real_temps_dict.items():
                 # Exclure R25 du calcul de R24
                 if name == "R25": continue
                 if pd.notna(temp):
                     if name in thermistors_low_weight_r24_real:
                         weight = 0.1
                     elif name in other_real_thermistors_for_r24_real:
                         weight = weight_per_other_r24_real
                     else:
                         continue

                     weighted_sum_r24_real += temp * weight
                     total_weight_r24_real += weight

            # Assigner la valeur à R24
            if total_weight_r24_real > 1e-6:
                real_temps_dict["R24"] = weighted_sum_r24_real / total_weight_r24_real
            else:
                real_temps_dict["R24"] = np.nan


        # --- NOUVEAU CALCUL DE LA THERMISTANCE VIRTUELLE (Commun aux deux modes) ---
        weighted_sum_virt = 0.0
        total_weight_virt = 0.0
        thermistors_low_weight_virt = ["R14", "R10", "R9"] # Poids de 0.15 chacune
        other_real_thermistors_for_virt = []

        # Identifier les autres thermistances réelles valides (pour R_Virtuel)
        # Inclut R24 mais exclut R25 et R_Virtuel elle-même
        for name, temp in real_temps_dict.items():
            if name != "R25" and name not in thermistors_low_weight_virt and pd.notna(temp):
                 other_real_thermistors_for_virt.append(name)

        # Calculer le poids pour les "autres" thermistances (pour R_Virtuel)
        weight_per_other_virt = 0.0
        if other_real_thermistors_for_virt:
            # Les autres se partagent 55% du poids total (1.0 - 0.15*3 = 0.55)
            weight_per_other_virt = 0.55 / len(other_real_thermistors_for_virt)

        # Calculer la somme pondérée et le poids total (pour R_Virtuel)
        for name, temp in real_temps_dict.items():
             # Exclure R25 du calcul de R_Virtuel
             if name == "R25": continue
             if pd.notna(temp):
                 if name in thermistors_low_weight_virt:
                     weight = 0.15
                 elif name in other_real_thermistors_for_virt:
                     weight = weight_per_other_virt
                 else:
                     continue # Ignore les autres cas (devrait pas arriver)

                 weighted_sum_virt += temp * weight
                 total_weight_virt += weight

        # Assigner la valeur à R_Virtuel
        if total_weight_virt > 1e-6: # Éviter division par zéro
            virtual_temp = weighted_sum_virt / total_weight_virt
        else:
            virtual_temp = np.nan # Si aucune thermistance valide pour la moyenne

        real_temps_dict["R_Virtuel"] = virtual_temp # Ajouter la virtuelle au dict final

        return real_temps_dict

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        # --- Configuration des subplots (inchangé) ---
        fig.clear()
        ax1 = fig.add_subplot(121) # Subplot pour la heatmap de température
        ax2 = fig.add_subplot(122) # Subplot pour la heatmap de différence

        # --- Collecte des points (inchangé) ---
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

        # --- Calcul Baseline (inchangé) ---
        if not valid_temps_list:
            baseline_temp = 20.0
            print("[AVERTISSEMENT HEATMAP] Aucune donnée valide (hors R25) pour calculs.")
        else:
            avg_temp = np.mean(valid_temps_list)
            baseline_temp = avg_temp - 1.0

        # --- Section Interpolation RBF (inchangé) ---
        r_max = 12.5
        num_edge_points = 12
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [baseline_temp] * num_edge_points
        x_combined = x_all_points + list(edge_x)
        y_combined = y_all_points + list(edge_y)
        t_combined = t_all_points + edge_t

        # Initialisations
        ti_filtered = None
        xi, yi = None, None
        mask = None
        difference_map_masked = None
        filtered_difference_map = None # Pour le calcul du max
        laser_x, laser_y = None, None # Pour la position du laser
        laser_pos_found = False

        if len(x_combined) < 3:
            # ... (gestion erreur comme avant) ...
            print("[ERREUR HEATMAP] Pas assez de points (hors R25) pour l'interpolation RBF.")
            ax1.set_title("Pas assez de données (hors R25) pour RBF")
            ax2.set_title("Différence non calculable")
            # Afficher points sur ax1 si erreur
            for item in thermistor_data_for_plot:
                 if pd.notna(item["temp"]):
                     # ... (code affichage points) ...
                     pass
            ax1.legend()
            return

        try:
            # --- Calcul RBF et Filtre Température (inchangé) ---
            rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
            grid_size = 200
            xi, yi = np.meshgrid(
                np.linspace(-r_max, r_max, grid_size),
                np.linspace(-r_max, r_max, grid_size)
            )
            ti = rbf(xi, yi)
            ti_filtered = gaussian_filter(ti, sigma=1.2) # Filtre sur la température
            mask = xi**2 + yi**2 > r_max**2
            ti_masked = np.ma.array(ti_filtered, mask=mask) # Pour heatmap température

            # --- Calcul de la carte de différence ---
            if self.previous_ti_filtered is not None and self.previous_ti_filtered.shape == ti_filtered.shape:
                difference_map = ti_filtered - self.previous_ti_filtered
                difference_map_masked = np.ma.array(difference_map, mask=mask) # Pour affichage heatmap différence

                # --- NOUVEAU : Filtre Gaussien sur la différence et recherche du maximum ---
                # Appliquer un filtre gaussien à la carte de différence (non masquée)
                # Le sigma peut être ajusté, ici on prend une valeur un peu plus grande
                sigma_diff_filter = 2.0
                filtered_difference_map = gaussian_filter(difference_map, sigma=sigma_diff_filter)

                # Appliquer le masque APRÈS le filtrage pour la recherche du max
                filtered_difference_map_masked = np.ma.array(filtered_difference_map, mask=mask)

                # Trouver l'indice du maximum dans la carte de différence filtrée et masquée
                try:
                    # Utiliser nanargmax pour ignorer les zones masquées si elles sont NaN
                    # Si la zone masquée n'est pas NaN, np.argmax fonctionne sur la partie non masquée
                    if np.ma.is_masked(filtered_difference_map_masked):
                         max_idx_flat = np.nanargmax(filtered_difference_map_masked)
                         max_idx = np.unravel_index(max_idx_flat, filtered_difference_map_masked.shape)
                    else: # Si rien n'est masqué (improbable ici mais pour être sûr)
                         max_idx = np.unravel_index(np.argmax(filtered_difference_map), filtered_difference_map.shape)

                    # Obtenir les coordonnées correspondantes
                    laser_x = xi[max_idx]
                    laser_y = yi[max_idx]
                    laser_pos_found = True
                except ValueError: # Peut arriver si la carte est vide ou entièrement masquée/NaN
                    print("[AVERTISSEMENT LASER] Impossible de trouver le maximum dans la carte de différence.")
                    laser_pos_found = False

            else:
                # Premier frame ou changement de forme
                difference_map_masked = None
                laser_pos_found = False # Pas de différence, pas de position laser basée sur la différence
                if self.previous_ti_filtered is None:
                    print("[INFO DIFF] Calcul de la différence en attente du prochain frame.")
                elif self.previous_ti_filtered.shape != ti_filtered.shape:
                     print("[ERREUR DIFF] Incohérence de forme de grille, réinitialisation.")
                     self.previous_ti_filtered = None

        except Exception as e:
             # ... (gestion erreur RBF comme avant) ...
             print(f"[ERREUR RBF/DIFF/MAX] Échec: {e}")
             ax1.set_title("Erreur Calcul")
             ax2.set_title("Erreur Calcul")
             # Afficher points sur ax1 si erreur
             for item in thermistor_data_for_plot:
                  if pd.notna(item["temp"]):
                      # ... (code affichage points) ...
                      pass
             ax1.legend()
             self.previous_ti_filtered = None
             return

        # --- Affichage Subplot 1 : Heatmap Température ---
        contour1 = ax1.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour1, ax=ax1, label="Température (°C)", shrink=0.6)

        # Affichage des points (réels et virtuel) sur ax1 (inchangé)
        plotted_labels = set()
        for item in thermistor_data_for_plot:
            if pd.notna(item["temp"]):
                is_virtual = item["name"] == "R_Virtuel"
                marker = 's' if is_virtual else 'o'
                color = 'magenta' if is_virtual else 'black'
                label = 'Virtuel' if is_virtual else 'Réelles'
                if label not in plotted_labels:
                    ax1.scatter(item["pos"][0], item["pos"][1], color=color, marker=marker, s=35, label=label)
                    plotted_labels.add(label)
                else:
                    ax1.scatter(item["pos"][0], item["pos"][1], color=color, marker=marker, s=35)
                ax1.annotate(item["name"], item["pos"], textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8, color=color)

        # --- MODIFIÉ : Afficher le point laser (Max Différence Filtrée) sur ax1 ---
        if laser_pos_found:
            # Utiliser 'bx' (croix bleue) pour le différencier
            ax1.plot(laser_x, laser_y, 'bx', markersize=10, label=f'Laser (Max ΔT Filt.) @ ({laser_x:.1f}, {laser_y:.1f})')

        # Configuration ax1
        ax1.set_aspect('equal')
        title_ax1 = f"Heatmap Température (Tps: {elapsed_time:.2f} s)"
        if laser_pos_found:
             title_ax1 += f"\nLaser (Max ΔT Filt.) @ ({laser_x:.1f}, {laser_y:.1f})" # Mise à jour label titre
        ax1.set_title(title_ax1, fontsize=9)
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_xlim(-r_max - 1, r_max + 1)
        ax1.set_ylim(-r_max - 1, r_max + 1)
        ax1.legend(fontsize=7, loc='upper right')

        # --- Affichage Subplot 2 : Heatmap Différence Temporelle ---
        if difference_map_masked is not None:
            cmap_diff = "coolwarm"
            max_abs_diff = np.max(np.abs(difference_map_masked)) if np.any(difference_map_masked.compressed()) else 1.0 # Utiliser .compressed() pour ignorer les masqués
            vmin_diff = -max_abs_diff
            vmax_diff = max_abs_diff

            contour2 = ax2.contourf(xi, yi, difference_map_masked, levels=50, cmap=cmap_diff, vmin=vmin_diff, vmax=vmax_diff)
            fig.colorbar(contour2, ax=ax2, label="Différence Temp. (°C / frame)", shrink=0.6)

            # Optionnel: Afficher les points des thermistances sur ax2
            ax2.scatter(x_all_points, y_all_points, color='black', marker='.', s=10, alpha=0.3)

            # --- MODIFIÉ : Afficher la position du laser (Max Différence Filtrée) sur ax2 ---
            if laser_pos_found:
                ax2.plot(laser_x, laser_y, 'bx', markersize=8, label='Max ΔT Filt.') # Croix bleue

            ax2.set_aspect('equal')
            ax2.set_title("Différence Temporelle Température", fontsize=9)
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_xlim(-r_max - 1, r_max + 1)
            ax2.set_ylim(-r_max - 1, r_max + 1)
            if laser_pos_found:
                ax2.legend(fontsize=7, loc='upper right')
        else:
            # ... (configuration ax2 si différence non calculée) ...
            ax2.set_title("Différence non calculée (attente)")
            ax2.set_aspect('equal')
            # ... (labels et limites) ...


        # --- Mise à jour de la carte précédente (inchangé) ---
        self.previous_ti_filtered = ti_filtered.copy() if ti_filtered is not None else None

        # --- Ajustement final de la mise en page (inchangé) ---
        fig.tight_layout(pad=2.0)






    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        fig = plt.figure(figsize=(7, 6))
        plt.ion()
        fig.show()

        all_data = []
        # --- Définir les headers pour le CSV (inclut R_Virtuel) ---
        headers = [name for name, _ in self.positions] + ["T_ref", "timestamp", "temps_ecoule_s"]
        # Assurer que T_ref, timestamp, temps_ecoule_s sont à la fin
        base_headers = [name for name, _ in self.positions]
        extra_headers = ["T_ref", "timestamp", "temps_ecoule_s"]
        headers = base_headers + extra_headers


        start_time = time.time()
        keep_running = True
        try:
            while keep_running and plt.fignum_exists(fig.number):
                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures() # Récupère le dict avec R_Virtuel calculé

                if data: # data est maintenant le dictionnaire complet
                    if not plt.fignum_exists(fig.number):
                        keep_running = False
                        break

                    os.system("clear") # ou 'cls' sur Windows
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0
                    # --- Afficher toutes les thermistances, y compris la virtuelle ---
                    for name, temp in data.items():
                        display_name = name
                        if name == "R_Virtuel":
                            display_name = "R_Virtuel" # Ou un autre nom si tu préfères
                        if pd.notna(temp):
                            print(f"{display_name:<10} : {temp:6.2f} °C")
                            if name != "R_Virtuel": # Ne compte pas la virtuelle dans les valides réelles
                                valid_temps_count += 1
                        else:
                            print(f"{display_name:<10} :   --   °C (NaN)")
                    # Compter les thermistances réelles attendues
                    real_thermistor_count = len([p for p in self.positions if p[0] != "R_Virtuel"])
                    print(f"({valid_temps_count}/{real_thermistor_count} thermistances réelles valides)")
                    print("=" * 60)

                    # Affichage de la heatmap (utilise le dict complet 'data')
                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # Préparer la ligne pour le CSV
                    ligne = []
                    t_ref_value = data.get("R25", 25.0) # Utiliser R25 comme T_ref si dispo, sinon 25.0
                    if pd.isna(t_ref_value): t_ref_value = 25.0 # Fallback si R25 est NaN

                    for header_name in headers:
                        if header_name == "T_ref":
                            ligne.append(round(t_ref_value, 2)) # Arrondir T_ref
                        elif header_name == "timestamp":
                            ligne.append(datetime.now().isoformat(timespec='seconds'))
                        elif header_name == "temps_ecoule_s":
                            ligne.append(round(elapsed_time, 3))
                        elif header_name in data:
                            temp_value = data[header_name]
                            # Arrondir les températures pour le CSV, gérer NaN
                            ligne.append(round(temp_value, 2) if pd.notna(temp_value) else '')
                        else:
                            ligne.append('') # Laisser vide si header non trouvé dans data (ne devrait pas arriver)
                    all_data.append(ligne)

                else:
                    os.system("clear")
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues.")
                    print("=" * 60)

                if not plt.fignum_exists(fig.number):
                    keep_running = False
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            if plt.fignum_exists(fig.number):
                plt.close(fig)

            if all_data:
                print("💾 Sauvegarde du fichier CSV...")
                desktop_path = Path.home() / "Desktop"
                filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = desktop_path / filename
                try:
                    with open(csv_path, mode='w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers) # Écrit les headers définis plus haut
                        writer.writerows(all_data)
                    print(f"✅ Données sauvegardées dans : {csv_path}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")
            else:
                print("ℹ️ Aucune donnée à sauvegarder.")


if __name__ == "__main__":
    # Mettre simulation=False pour utiliser l'Arduino
    td = TraitementDonnees(simulation=True)
    td.demarrer_acquisition_live(interval=0.05) # Intervalle rapide pour la simulation

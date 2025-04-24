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

    def __init__(self, port="COM4", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        # --- Chargement Coefficients ---
        try:
            # Utiliser un chemin relatif au script pour coeffs_path
            script_dir = Path(__file__).parent
            coeffs_full_path = script_dir.parent / coeffs_path
            self.coefficients = np.load(coeffs_full_path, allow_pickle=True)
            print(f"[INFO] Coefficients chargés depuis : {coeffs_full_path.resolve()}")
        except FileNotFoundError:
             print(f"[ERREUR] Fichier coefficients non trouvé : {coeffs_full_path.resolve()}")
             # Optionnel: Lever l'erreur si les coeffs sont essentiels
             # raise FileNotFoundError(f"Fichier coefficients non trouvé : {coeffs_full_path.resolve()}")
             self.coefficients = None # Ou définir une valeur par défaut / gérer l'absence
        except Exception as e:
             print(f"[ERREUR] Problème chargement coefficients : {e}")
             self.coefficients = None

        # Décalage à appliquer
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas

        self.positions = [
            ("R1", (11 + decalage_x, 0 + decalage_y)), ("R2", (3 + 1, 0 + decalage_y)), ("R3", (-3 + decalage_x, 0 + decalage_y)), ("R4", (-11 + decalage_x, 0 + decalage_y)),
            ("R5", (8 + 1, 2.5 - decalage_y)), ("R6", (0 + decalage_x, 2.5 + decalage_y)), ("R7", (-8 + decalage_x, 2.5 + decalage_y)), ("R8", (8 + 1, 5.5 - decalage_y)),
            ("R9", (0 + decalage_x, 5.5 + decalage_y)), ("R10", (-8 + decalage_x, 5.5 + decalage_y)), ("R11", (4.5 + decalage_x, 8 + decalage_y)), ("R24", (-3.5 + decalage_x, -11.25 + decalage_y)), # Note: R24 est sur le canal 11 physiquement
            ("R13", (4 + decalage_x, 11.25 + decalage_y)), ("R14", (-4 + decalage_x, 11.25 + decalage_y)), ("R15", (8 + 1, -2.5 - decalage_y)), ("R16", (0 + decalage_x, -2.5 + decalage_y)),
            ("R17", (-8 + decalage_x, -2.5 + decalage_y)), ("R18", (8 + 1, -5.5 - decalage_y)), ("R19", (0 + decalage_x, -5.5 + decalage_y)), ("R20", (-8 + decalage_x, -5.5 + decalage_y)),
            ("R21", (4.5 + decalage_x, -8 + decalage_y)), ("R25", (0 + decalage_x, -11.5 + decalage_y)), # R25 est la référence, souvent sur canal 24
            # --- NOUVELLE THERMISTANCE VIRTUELLE ---
            ("R_Virtuel", (-4.9, 7.8))
        ]

        # Indices des canaux à lire sur l'Arduino (0-20 pour R1-R11, R13-R21)
        # Le canal 11 (R24) est lu mais sa valeur est recalculée.
        self.indices_à_garder = list(range(21))
        # Décommenter si R25 est lue physiquement (ex: sur canal 24)
        # self.indices_à_garder.append(24)

        self.simulation_data = None
        self.simulation_index = 0
        # Définir les colonnes attendues pour la simulation
        # Inclut les thermistances réelles (sauf R_Virtuel) et R25 si elle est définie dans positions
        self.simulation_columns = [p[0] for p in self.positions if p[0] != "R_Virtuel"]
        # Note: R24 est incluse ici car elle est dans self.positions, même si calculée différemment

        # --- Initialisations pour les heatmaps et le filtrage ---
        self.previous_ti_filtered = None
        self.position_history = []
        self.history_length = 5
        self.last_valid_raw_pos = None
        self.last_filtered_pos = (None, None)
        self.max_speed_mm_per_interval = 3.0
        self.min_heating_threshold = 0.20
        # --- FIN Initialisations ---

        # --- Bloc Simulation ---
        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                script_dir = Path(__file__).parent
                # Assurez-vous que le nom du fichier est correct
                simulation_file_path = script_dir.parent / "data" / "Test-echelon-976-nm-2025-04-24.csv"

                # Vérifier si le fichier existe
                if not simulation_file_path.is_file():
                     raise FileNotFoundError(f"Fichier simulation non trouvé: {simulation_file_path.resolve()}")

                # --- MODIFICATION ICI : Ajout de nrows=42 ---
                self.simulation_data = pd.read_csv(
                    simulation_file_path,
                    sep=';',      # Séparateur virgule
                    decimal='.',  # Séparateur décimal point
                    nrows=30      # Lire seulement les 42 premières lignes
                    # on_bad_lines='warn' # Optionnel: avertir si des lignes > 42 ont des problèmes
                )
                # --- FIN MODIFICATION ---

                print(f"[SIMULATION] Chargement des {len(self.simulation_data)} premières lignes du fichier CSV : {simulation_file_path.resolve()}")
                print("[DEBUG SIMULATION] Colonnes lues depuis le CSV:", self.simulation_data.columns.tolist())
                print("[DEBUG SIMULATION] Aperçu des 5 premières lignes lues:")
                print(self.simulation_data.head()) # Afficher un aperçu

                # --- Vérification et Conversion des Données ---
                colonnes_lues = self.simulation_data.columns.tolist()
                missing_cols = [col for col in self.simulation_columns if col not in colonnes_lues]
                if missing_cols:
                    print(f"[AVERTISSEMENT SIMULATION] Colonnes attendues mais non trouvées dans le CSV: {missing_cols}")
                    # Ne pas mettre self.simulation_data à None ici, continuer avec les colonnes disponibles

                # Appliquer la conversion numérique aux colonnes attendues qui existent
                colonnes_a_convertir = [col for col in self.simulation_columns if col in colonnes_lues]
                for col in colonnes_a_convertir:
                    # Utiliser .loc pour éviter SettingWithCopyWarning
                    try:
                        self.simulation_data.loc[:, col] = pd.to_numeric(self.simulation_data[col], errors='coerce')
                    except Exception as e_num:
                         print(f"[AVERTISSEMENT SIMULATION] Problème conversion numérique colonne '{col}': {e_num}")

                print(f"[SIMULATION] {len(self.simulation_data)} lignes CSV chargées et traitées.")
                if self.simulation_data.isnull().values.any():
                    print("[AVERTISSEMENT SIMULATION] Le fichier CSV (premières lignes) contient des valeurs non numériques ou manquantes après conversion.")

            except FileNotFoundError as e:
                print(f"[ERREUR SIMULATION] {e}")
                self.simulation_data = None # Mettre à None si fichier non trouvé
            except Exception as e:
                print(f"[ERREUR SIMULATION] Impossible de charger ou lire le fichier CSV : {e}")
                import traceback
                traceback.print_exc() # Afficher la trace complète de l'erreur
                self.simulation_data = None # Mettre à None en cas d'autre erreur
        # --- Fin Bloc Simulation ---

        # --- Bloc Connexion Série ---
        else:
            try:
                # Timeout court pour réactivité
                self.ser = serial.Serial(self.port, 9600, timeout=0.5)
                print(f"[INFO] Port série connecté sur {self.port}")
                # Petite pause pour laisser l'Arduino s'initialiser
                time.sleep(1.5)
                self.ser.reset_input_buffer() # Vider buffer au cas où
            except serial.SerialException as e:
                print(f"[ERREUR] Impossible d'ouvrir le port série {self.port}: {e}")
                self.ser = None
            except Exception as e:
                print(f"[ERREUR] Autre erreur connexion série: {e}")
                self.ser = None
        # --- Fin Bloc Connexion Série ---

    # ... (le reste de vos méthodes : est_connecte, steinhart_hart_temperature, etc.) ...
    def est_connecte(self):
        return self.ser is not None

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

    def compute_temperature(self, resistance, coeffs):
        if self.coefficients is None:
             print("[ERREUR] Coefficients non chargés, impossible de calculer la température.")
             return np.nan
        if resistance == float('inf') or resistance <= 0 or pd.isna(resistance):
            return np.nan
        # S'assurer que coeffs est bien un tuple/liste de 3 éléments
        if not isinstance(coeffs, (list, tuple, np.ndarray)) or len(coeffs) != 3:
             print(f"[ERREUR] Format de coefficients invalide: {coeffs}")
             return np.nan
        A, B, C = coeffs
        kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
        if pd.isna(kelvin):
            return np.nan
        return kelvin - 273.15

    def lire_donnees(self):
        if self.simulation:
            # En simulation, on ne lit pas de l'Arduino, on utilise self.simulation_data
            # Cette fonction pourrait retourner True si les données sont prêtes, ou rien.
            # Pour la compatibilité, on retourne True si les données sont chargées.
            return self.simulation_data is not None and not self.simulation_data.empty

        if self.ser is None:
            print("[ERREUR] Connexion série non établie.")
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        start_time = time.time()
        timeout_sec = 2 # Timeout pour recevoir toutes les données

        while True:
            current_time = time.time()
            if current_time - start_time > timeout_sec:
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                # Retourner None si incomplet est plus sûr
                return None

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    # print(f"[DEBUG RAW] Reçu: '{line}'") # Décommenter pour voir tout ce qui arrive
                    if not line:
                        continue

                    if "Fin du balayage" in line:
                        # print("[DEBUG] Fin du balayage détectée.")
                        break # Sortir de la boucle while

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        canal = int(match.group(1))
                        # Lire seulement les canaux définis dans indices_à_garder
                        if canal in self.indices_à_garder:
                             try:
                                 voltages_dict[canal] = float(match.group(2))
                                 # print(f"[DEBUG] Reçu Canal {canal}: {voltages_dict[canal]} V")
                             except ValueError:
                                 print(f"[AVERTISSEMENT] Impossible de convertir la tension '{match.group(2)}' pour le canal {canal}")
                                 voltages_dict[canal] = np.nan # Marquer comme invalide
                else:
                    # Petite pause pour ne pas saturer le CPU si rien n'est reçu
                    time.sleep(0.01)

            except serial.SerialException as e:
                print(f"[ERREUR] Erreur série pendant la lecture : {e}")
                self.ser = None # Marquer comme déconnecté
                return None
            except Exception as e:
                print(f"[ERREUR] Erreur inattendue pendant la lecture série : {e}")
                # Continuer peut être risqué, retourner None est plus sûr
                return None

        # Vérification après la sortie de boucle (Fin du balayage)
        canaux_attendus = set(self.indices_à_garder)
        canaux_recus = set(voltages_dict.keys())

        if canaux_recus != canaux_attendus:
             canaux_manquants = canaux_attendus - canaux_recus
             print(f"⚠️ Seulement {len(canaux_recus)}/{len(canaux_attendus)} canaux requis reçus. Manquants: {sorted(list(canaux_manquants))}")
             # Retourner None si incomplet
             return None

        # print(f"[DEBUG] Données lues avec succès: {len(voltages_dict)} canaux.")
        return voltages_dict

    def get_temperatures(self):
        real_temps_dict = {} # Dictionnaire pour les températures réelles

        if self.simulation:
            # --- Logique Simulation CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    # Atteint la fin des lignes lues (nrows=42)
                    print(f"[SIMULATION] Fin des {len(self.simulation_data)} lignes CSV lues.")
                    # Option 1: Arrêter (retourner un dict vide ou None pour signaler la fin)
                    # return {}
                    # Option 2: Boucler (recommencer au début)
                    self.simulation_index = 0
                    print("[SIMULATION] Retour au début du fichier CSV.")

                # Lire la ligne actuelle
                current_data_row = self.simulation_data.iloc[self.simulation_index]
                # Incrémenter pour la prochaine lecture
                self.simulation_index += 1
                valid_data_found = False

                # Lire les températures simulées pour les thermistances réelles (SAUF R24 pour l'instant)
                for i, (name, _) in enumerate(self.positions):
                    if name in ["R_Virtuel", "R24"]: continue # Ignorer la virtuelle et R24 ici

                    # Cas général pour les thermistances dans simulation_columns
                    if name in self.simulation_data.columns: # Vérifier si colonne existe
                        if name in current_data_row.index and pd.notna(current_data_row[name]):
                            try:
                                real_temps_dict[name] = float(current_data_row[name])
                                valid_data_found = True
                            except (ValueError, TypeError):
                                real_temps_dict[name] = np.nan
                        else:
                            real_temps_dict[name] = np.nan # Mettre NaN si absent ou déjà NaN

                    # Cas spécifique pour R25 si elle est dans 'positions' mais pas lue directement
                    # et si elle existe dans le CSV
                    elif name == "R25" and "R25" in self.simulation_data.columns:
                         if "R25" in current_data_row.index and pd.notna(current_data_row["R25"]):
                             try:
                                 real_temps_dict["R25"] = float(current_data_row["R25"])
                                 valid_data_found = True # Compte comme donnée valide
                             except (ValueError, TypeError):
                                 real_temps_dict["R25"] = np.nan
                         else:
                             real_temps_dict["R25"] = np.nan

                # Si aucune donnée valide n'a été trouvée (hors R24/Virtuelle)
                if not valid_data_found:
                    print(f"[AVERTISSEMENT SIMULATION] Aucune donnée valide (hors R24/Virtuelle) à l'index CSV {self.simulation_index - 1}.")
                    # Initialiser R24 et R_Virtuel à NaN avant de retourner
                    real_temps_dict["R24"] = np.nan
                    real_temps_dict["R_Virtuel"] = np.nan
                    # Retourner le dict même s'il est plein de NaN
                    return real_temps_dict

                # --- Logique R24 (Moyenne Pondérée) pour Simulation ---
                weighted_sum_r24 = 0.0
                total_weight_r24 = 0.0
                thermistors_r24_weights = {"R19": 0.1, "R20": 0.15, "R21": 0.15} # 40%
                other_thermistors_for_r24 = []

                for name, temp in real_temps_dict.items():
                    if name != "R25" and name not in thermistors_r24_weights and pd.notna(temp):
                         other_thermistors_for_r24.append(name)

                weight_per_other_r24 = 0.0
                if other_thermistors_for_r24:
                    weight_per_other_r24 = 0.6 / len(other_thermistors_for_r24)

                for name, temp in real_temps_dict.items():
                     if name == "R25": continue
                     if pd.notna(temp):
                         if name in thermistors_r24_weights:
                             weight = thermistors_r24_weights[name]
                         elif name in other_thermistors_for_r24:
                             weight = weight_per_other_r24
                         else: continue
                         weighted_sum_r24 += temp * weight
                         total_weight_r24 += weight

                if total_weight_r24 > 1e-6:
                    real_temps_dict["R24"] = weighted_sum_r24 / total_weight_r24
                else:
                    real_temps_dict["R24"] = np.nan

            else:
                # Fallback si CSV échoue ou est vide
                print("[SIMULATION] Données CSV non disponibles ou vides, génération de températures aléatoires.")
                temp_gen_dict = {}
                for i, (name, _) in enumerate(self.positions):
                     if name != "R_Virtuel":
                         temp_gen_dict[name] = np.random.uniform(20.0, 35.0)

                # Calculer R24 à partir des valeurs générées
                weighted_sum_r24_gen = 0.0
                total_weight_r24_gen = 0.0
                thermistors_r24_weights_gen = {"R19": 0.1, "R20": 0.15, "R21": 0.15}
                other_thermistors_for_r24_gen = []

                for name, temp in temp_gen_dict.items():
                    if name != "R24" and name != "R25" and name not in thermistors_r24_weights_gen and pd.notna(temp):
                         other_thermistors_for_r24_gen.append(name)

                weight_per_other_gen_r24 = 0.0
                if other_thermistors_for_r24_gen:
                    weight_per_other_gen_r24 = 0.6 / len(other_thermistors_for_r24_gen)

                for name, temp in temp_gen_dict.items():
                     if name != "R24" and name != "R25" and pd.notna(temp):
                         if name in thermistors_r24_weights_gen: weight = thermistors_r24_weights_gen[name]
                         elif name in other_thermistors_for_r24_gen: weight = weight_per_other_gen_r24
                         else: continue
                         weighted_sum_r24_gen += temp * weight
                         total_weight_r24_gen += weight

                if total_weight_r24_gen > 1e-6: temp_gen_dict["R24"] = weighted_sum_r24_gen / total_weight_r24_gen
                else: temp_gen_dict["R24"] = np.nan

                real_temps_dict = temp_gen_dict

        else:
            # --- Logique Lecture Série ---
            if self.coefficients is None:
                 print("[ERREUR] Coefficients non chargés, impossible de calculer les températures.")
                 # Remplir avec NaN
                 for i, (name, _) in enumerate(self.positions): real_temps_dict[name] = np.nan
                 return real_temps_dict

            data_voltages = self.lire_donnees()
            if data_voltages is None:
                 for i, (name, _) in enumerate(self.positions): real_temps_dict[name] = np.nan
                 return real_temps_dict

            temperatures_raw = {}
            indices_mapping = { # Nom -> Index Canal
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10, # Canal 11 est R24 physiquement
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                # "R25": 24
            }
            coeffs_mapping = { # Nom -> Index Coefficient
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10,
                # R24 (coeffs[23]) sera calculée
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                # "R25": 24
            }

            # Calcul initial des températures réelles (SAUF R24)
            for nom_thermistor, canal_index in indices_mapping.items():
                if canal_index == 11: continue # Ignorer canal 11 (R24)

                if canal_index not in data_voltages or pd.isna(data_voltages[canal_index]):
                    temperatures_raw[nom_thermistor] = np.nan
                    continue

                voltage = data_voltages[canal_index]
                coeffs_index = coeffs_mapping.get(nom_thermistor, -1)

                if coeffs_index == -1 or coeffs_index >= len(self.coefficients):
                    temperatures_raw[nom_thermistor] = np.nan
                    continue

                coeffs = self.coefficients[coeffs_index]
                resistance = self.compute_resistance(voltage)
                temp = self.compute_temperature(resistance, coeffs)
                temperatures_raw[nom_thermistor] = temp

            # Gérer R25 si lue
            # if 24 in self.indices_à_garder and 24 in data_voltages:
            #     if pd.notna(data_voltages[24]):
            #         voltage_r25 = data_voltages[24]
            #         coeffs_r25 = self.coefficients[24]
            #         resistance_r25 = self.compute_resistance(voltage_r25)
            #         temp_r25 = self.compute_temperature(resistance_r25, coeffs_r25)
            #         temperatures_raw["R25"] = temp_r25
            #     else: temperatures_raw["R25"] = np.nan
            # elif "R25" in [p[0] for p in self.positions]: temperatures_raw["R25"] = np.nan

            real_temps_dict = temperatures_raw.copy()

            # --- Logique R24 (Moyenne Pondérée) pour Lecture Série ---
            weighted_sum_r24_real = 0.0
            total_weight_r24_real = 0.0
            thermistors_r24_weights_real = {"R19": 0.1, "R20": 0.15, "R21": 0.15}
            other_thermistors_for_r24_real = []

            for name, temp in real_temps_dict.items():
                if name != "R25" and name not in thermistors_r24_weights_real and pd.notna(temp):
                     other_thermistors_for_r24_real.append(name)

            weight_per_other_r24_real = 0.0
            if other_thermistors_for_r24_real:
                weight_per_other_r24_real = 0.6 / len(other_thermistors_for_r24_real)

            for name, temp in real_temps_dict.items():
                 if name == "R25": continue
                 if pd.notna(temp):
                     if name in thermistors_r24_weights_real: weight = thermistors_r24_weights_real[name]
                     elif name in other_thermistors_for_r24_real: weight = weight_per_other_r24_real
                     else: continue
                     weighted_sum_r24_real += temp * weight
                     total_weight_r24_real += weight

            if total_weight_r24_real > 1e-6: real_temps_dict["R24"] = weighted_sum_r24_real / total_weight_r24_real
            else: real_temps_dict["R24"] = np.nan


        # --- CALCUL DE LA THERMISTANCE VIRTUELLE (Commun aux deux modes) ---
        weighted_sum_virt = 0.0
        total_weight_virt = 0.0
        thermistors_virt_weights = {"R14": 0.15, "R10": 0.15, "R9": 0.15} # 45%
        other_thermistors_for_virt = []

        for name, temp in real_temps_dict.items():
            if name != "R25" and name not in thermistors_virt_weights and pd.notna(temp):
                 other_thermistors_for_virt.append(name)

        weight_per_other_virt = 0.0
        if other_thermistors_for_virt:
            weight_per_other_virt = 0.55 / len(other_thermistors_for_virt)

        for name, temp in real_temps_dict.items():
             if name == "R25": continue
             if pd.notna(temp):
                 if name in thermistors_virt_weights: weight = thermistors_virt_weights[name]
                 elif name in other_thermistors_for_virt: weight = weight_per_other_virt
                 else: continue
                 weighted_sum_virt += temp * weight
                 total_weight_virt += weight

        if total_weight_virt > 1e-6: virtual_temp = weighted_sum_virt / total_weight_virt
        else: virtual_temp = np.nan

        real_temps_dict["R_Virtuel"] = virtual_temp

        return real_temps_dict

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        ax = fig.add_subplot(111) # Un seul subplot pour le gradient

        x_all_points, y_all_points, t_all_points = [], [], []
        valid_temps_list = []
        thermistor_data_for_plot = []

        # 1. Collecter points valides (réels + virtuel) SAUF R25
        for name, pos in self.positions:
            if name == "R25": continue
            temp_val = temperature_dict.get(name, np.nan)
            if pd.notna(temp_val):
                x_all_points.append(pos[0])
                y_all_points.append(pos[1])
                t_all_points.append(temp_val)
                valid_temps_list.append(temp_val)
                thermistor_data_for_plot.append({"name": name, "pos": pos, "temp": temp_val})

        # 2. Calculer baseline (basée sur points valides hors R25)
        if not valid_temps_list:
            baseline_temp = 20.0
            print("[AVERTISSEMENT HEATMAP] Aucune donnée valide (hors R25) pour calculs.")
        else:
            # Utiliser le minimum comme baseline pour mieux voir le chauffage
            baseline_temp = min(valid_temps_list) - 0.5 # Légèrement en dessous du min

        # --- Section Interpolation RBF ---
        r_max = 12.5 # Rayon max de la zone d'intérêt
        num_edge_points = 12 # Points sur le bord pour stabiliser l'interpolation
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [baseline_temp] * num_edge_points # Température du bord = baseline

        # Combiner points réels/virtuels et points de bord
        x_combined = x_all_points + list(edge_x)
        y_combined = y_all_points + list(edge_y)
        t_combined = t_all_points + edge_t

        # Initialisations
        ti_filtered = None
        xi, yi = None, None
        mask = None
        grad_magnitude = None
        grad_magnitude_masked = None
        raw_laser_x, raw_laser_y = None, None
        raw_pos_found_this_frame = False
        final_laser_pos_found = False # Renommé pour clarté

        # Vérifier si assez de points pour RBF
        if len(x_combined) < 3:
            print("[ERREUR HEATMAP] Pas assez de points (hors R25) pour l'interpolation RBF.")
            ax.set_title("Pas assez de données (hors R25) pour RBF/Gradient")
            # Afficher les points disponibles si erreur
            if x_all_points:
                 ax.scatter(x_all_points, y_all_points, c=t_all_points, cmap='plasma', marker='o', s=35)
                 for item in thermistor_data_for_plot:
                     ax.annotate(item["name"], item["pos"], textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)
            return

        try:
            # --- Calcul RBF et Filtre Température ---
            # 'multiquadric' est souvent un bon choix, 'smooth' contrôle le lissage
            rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
            grid_size = 100 # Résolution de la grille (ajustable)
            xi, yi = np.meshgrid(
                np.linspace(-r_max, r_max, grid_size),
                np.linspace(-r_max, r_max, grid_size)
            )
            ti = rbf(xi, yi) # Températures interpolées sur la grille
            # Filtre Gaussien pour lisser la carte de température
            sigma_filter_temp = 1.2 # Sigma du filtre (ajustable)
            ti_filtered = gaussian_filter(ti, sigma=sigma_filter_temp)
            # Masque pour cacher les zones hors du cercle d'intérêt
            mask = xi**2 + yi**2 > r_max**2

            # --- Calcul du Gradient Spatial ---
            # np.gradient calcule le gradient selon les axes y et x
            grad_y, grad_x = np.gradient(ti_filtered)
            # Magnitude du gradient (norme euclidienne)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # Appliquer le masque à la magnitude du gradient pour l'affichage
            grad_magnitude_masked = np.ma.array(grad_magnitude, mask=mask)

            # --- Calcul Position Laser (Max Diff -> Min Grad Local + Filtres) ---
            if self.previous_ti_filtered is not None and self.previous_ti_filtered.shape == ti_filtered.shape:
                # 1. Calculer la différence temporelle
                difference_map = ti_filtered - self.previous_ti_filtered
                # 2. Filtrer la carte de différence pour réduire le bruit
                sigma_diff_filter = 1.5 # Sigma pour le filtre de différence (ajustable)
                filtered_difference_map = gaussian_filter(difference_map, sigma=sigma_diff_filter)
                filtered_difference_map_masked = np.ma.array(filtered_difference_map, mask=mask)

                try:
                    # 3. Trouver le maximum de la différence filtrée (point le plus chaud)
                    # Utiliser nanargmax pour ignorer les NaN potentiels dus au masque
                    max_diff_idx_flat = np.nanargmax(filtered_difference_map_masked.filled(np.nan))
                    max_diff_idx = np.unravel_index(max_diff_idx_flat, filtered_difference_map_masked.shape)
                    max_diff_val = filtered_difference_map_masked[max_diff_idx]

                    # Vérifier si le chauffage est significatif
                    if max_diff_val < self.min_heating_threshold:
                        # print("[INFO LASER] Chauffage insuffisant détecté.")
                        raw_pos_found_this_frame = False
                    else:
                        # 4. Chercher le minimum du gradient autour de ce maximum
                        search_radius_pixels = 20 # Rayon de recherche en pixels (ajustable)
                        rows, cols = np.indices(grad_magnitude.shape)
                        dist_sq_from_max_diff = (rows - max_diff_idx[0])**2 + (cols - max_diff_idx[1])**2
                        in_search_area = dist_sq_from_max_diff <= search_radius_pixels**2

                        # Créer une carte de gradient limitée à la zone de recherche
                        grad_search_map = grad_magnitude.copy()
                        # Mettre NaN hors de la zone ou sous le masque
                        grad_search_map[~in_search_area | mask] = np.nan

                        # Trouver le minimum dans cette zone (position brute du laser)
                        min_grad_idx_flat = np.nanargmin(grad_search_map)

                        # Vérifier si un minimum valide a été trouvé
                        if not np.isnan(grad_search_map.flat[min_grad_idx_flat]):
                            min_grad_idx = np.unravel_index(min_grad_idx_flat, grad_search_map.shape)
                            potential_laser_x = xi[min_grad_idx]
                            potential_laser_y = yi[min_grad_idx]

                            # 5. Vérifier la plausibilité de la position brute
                            # Vérifier si dans un rayon raisonnable du centre
                            distance_from_center = math.sqrt(potential_laser_x**2 + potential_laser_y**2)
                            max_allowed_distance = 9.5 # mm (ajustable)
                            is_within_radius = distance_from_center <= max_allowed_distance

                            # Vérifier si le déplacement est plausible (limite de vitesse)
                            is_plausible_move = True
                            if self.last_valid_raw_pos is not None and is_within_radius:
                                prev_x, prev_y = self.last_valid_raw_pos
                                dist_moved_sq = (potential_laser_x - prev_x)**2 + (potential_laser_y - prev_y)**2
                                if dist_moved_sq > self.max_speed_mm_per_interval**2:
                                    is_plausible_move = False
                                    # print(f"[INFO LASER] Position brute ({potential_laser_x:.1f}, {potential_laser_y:.1f}) rejetée : déplacement trop rapide.")

                            # Si la position est plausible, la garder comme position brute
                            if is_within_radius and is_plausible_move:
                                raw_laser_x = potential_laser_x
                                raw_laser_y = potential_laser_y
                                raw_pos_found_this_frame = True
                                self.last_valid_raw_pos = (raw_laser_x, raw_laser_y) # Mémoriser pour prochain frame
                            else:
                                raw_pos_found_this_frame = False
                                # if not is_within_radius:
                                #      print(f"[INFO LASER] Position brute ({potential_laser_x:.1f}, {potential_laser_y:.1f}) ignorée car trop éloignée (> {max_allowed_distance} mm).")
                        else:
                            # print("[AVERTISSEMENT LASER] Aucun minimum de gradient valide trouvé dans la zone de recherche (chauffage détecté).")
                            raw_pos_found_this_frame = False
                except (ValueError, IndexError):
                    # print("[AVERTISSEMENT LASER] Impossible de trouver/évaluer le maximum de la différence temporelle.")
                    raw_pos_found_this_frame = False
            else:
                # Pas de carte précédente ou forme différente
                raw_pos_found_this_frame = False
                self.last_valid_raw_pos = None # Réinitialiser si pas de différence calculable
                if self.previous_ti_filtered is None:
                    print("[INFO LASER] Attente du prochain frame pour calculer la position.")
                elif self.previous_ti_filtered.shape != ti_filtered.shape:
                     print("[ERREUR LASER] Incohérence de forme de grille, réinitialisation.")
                     self.previous_ti_filtered = None # Forcer recalcul au prochain frame

            # --- Filtrage Temporel (Médiane Mobile) ---
            # Ajouter la position brute trouvée (ou non) à l'historique
            if raw_pos_found_this_frame:
                self.position_history.append((raw_laser_x, raw_laser_y))
            # Garder seulement les N dernières positions
            self.position_history = self.position_history[-self.history_length:]

            # Calculer la médiane si assez de points, sinon moyenne, sinon rien
            filtered_laser_x, filtered_laser_y = None, None
            if len(self.position_history) > 0:
                valid_x = [p[0] for p in self.position_history]
                valid_y = [p[1] for p in self.position_history]
                # Médiane si au moins 3 points (plus robuste aux outliers)
                if len(valid_x) >= 3:
                    filtered_laser_x = np.median(valid_x)
                    filtered_laser_y = np.median(valid_y)
                # Moyenne si 1 ou 2 points
                elif len(valid_x) > 0:
                     filtered_laser_x = np.mean(valid_x)
                     filtered_laser_y = np.mean(valid_y)

            # Mettre à jour la position filtrée finale
            if filtered_laser_x is not None:
                self.last_filtered_pos = (filtered_laser_x, filtered_laser_y)
                final_laser_pos_found = True # On a une position filtrée à afficher
            else:
                # Garder la dernière position filtrée connue si aucune nouvelle n'est calculée
                # self.last_filtered_pos reste inchangée
                final_laser_pos_found = False # Pas de nouvelle position filtrée ce frame

        except Exception as e:
             print(f"[ERREUR RBF/GRADIENT/LASER] Échec: {e}")
             import traceback
             traceback.print_exc()
             ax.set_title("Erreur Calcul Gradient/Laser")
             # Réinitialiser les états en cas d'erreur grave
             self.previous_ti_filtered = None
             self.last_valid_raw_pos = None
             self.position_history = []
             self.last_filtered_pos = (None, None)
             final_laser_pos_found = False
             return # Sortir de la fonction d'affichage

        # --- Mise à jour de la carte précédente pour le prochain calcul de différence ---
        if ti_filtered is not None:
            self.previous_ti_filtered = ti_filtered.copy()

        # --- Affichage Subplot UNIQUE : Heatmap Magnitude du Gradient ---
        if grad_magnitude_masked is not None:
            # Utiliser 'viridis' ou 'plasma' pour le gradient
            contour = ax.contourf(xi, yi, grad_magnitude_masked, levels=50, cmap="viridis")
            fig.colorbar(contour, ax=ax, label="Magnitude Gradient Temp. (°C/mm)", shrink=0.8)
            # Optionnel: Afficher les points des thermistances (hors R25) pour référence
            ax.scatter(x_all_points, y_all_points, color='white', marker='.', s=10, alpha=0.5, label='Thermistances')

            # --- Afficher la position du laser FILTRÉE ---
            plot_x, plot_y = self.last_filtered_pos # Utiliser la dernière position filtrée
            if plot_x is not None: # Vérifier si une position filtrée existe
                # Label pour la légende
                label_laser = f'Laser (Médiane {len(self.position_history)}/{self.history_length}) @ ({plot_x:.1f}, {plot_y:.1f})'
                # Afficher en croix rouge ('rx')
                ax.plot(plot_x, plot_y, 'rx', markersize=10, label=label_laser)

            # Configuration de l'axe unique
            ax.set_aspect('equal')
            ax.set_title(f"Gradient Température (Tps: {elapsed_time:.2f} s)", fontsize=10)
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)
            # Afficher la légende (inclut thermistances et laser si trouvé)
            ax.legend(fontsize=8, loc='upper right')
        else:
            # Cas où le gradient n'a pas pu être calculé
            ax.set_title("Gradient non calculé")
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)

        # --- Ajustement final de la mise en page ---
        try:
            fig.tight_layout(pad=2.5)
        except Exception as e_layout:
             print(f"[AVERTISSEMENT] Erreur lors de tight_layout: {e_layout}")


    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        fig = plt.figure(figsize=(8, 7)) # Taille ajustée pour un seul plot
        plt.ion() # Mode interactif
        fig.show()

        all_data = [] # Pour stocker les données à sauvegarder
        # Définir les headers pour le CSV (inclut R_Virtuel et position laser)
        base_headers = [name for name, _ in self.positions] # R1..R25, R_Virtuel
        extra_headers = ["T_ref", "timestamp", "temps_ecoule_s", "laser_x_filtre", "laser_y_filtre"]
        headers = base_headers + extra_headers

        start_time = time.time()
        keep_running = True
        try:
            while keep_running:
                # Vérifier si la fenêtre est toujours ouverte
                if not plt.fignum_exists(fig.number):
                    print("\nFenêtre graphique fermée. Arrêt de l'acquisition.")
                    keep_running = False
                    break

                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures() # Récupère le dict avec R_Virtuel calculé

                if data: # Si des données valides sont reçues/calculées
                    # Effacer la console (optionnel, peut être bruyant)
                    # os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0
                    # Afficher toutes les thermistances, y compris la virtuelle
                    for name, temp in data.items():
                        display_name = name
                        if pd.notna(temp):
                            print(f"{display_name:<10} : {temp:6.2f} °C")
                            # Compter les valides réelles (hors R25 et R_Virtuel)
                            if name not in ["R_Virtuel", "R25"]:
                                valid_temps_count += 1
                        else:
                            print(f"{display_name:<10} :   --   °C (NaN)")
                    # Compter les thermistances réelles attendues (hors R25 et R_Virtuel)
                    real_thermistor_count = len([p for p in self.positions if p[0] not in ["R_Virtuel", "R25"]])
                    print(f"({valid_temps_count}/{real_thermistor_count} thermistances réelles (hors R25) valides)")

                    # Afficher la position laser filtrée
                    laser_x, laser_y = self.last_filtered_pos
                    if laser_x is not None:
                         print("-" * 60)
                         print(f"🎯 Position Laser (filtrée): ({laser_x:.1f}, {laser_y:.1f}) mm")
                    print("=" * 60)

                    # Affichage de la heatmap (utilise le dict complet 'data')
                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # Préparer la ligne pour le CSV
                    ligne = []
                    # Utiliser R25 comme T_ref si dispo, sinon 25.0
                    t_ref_value = data.get("R25", 25.0)
                    if pd.isna(t_ref_value): t_ref_value = 25.0 # Fallback si R25 est NaN

                    for header_name in headers:
                        if header_name == "T_ref":
                            ligne.append(round(t_ref_value, 2))
                        elif header_name == "timestamp":
                            ligne.append(datetime.now().isoformat(timespec='seconds'))
                        elif header_name == "temps_ecoule_s":
                            ligne.append(round(elapsed_time, 3))
                        elif header_name == "laser_x_filtre":
                            lx, _ = self.last_filtered_pos
                            ligne.append(round(lx, 2) if lx is not None else '')
                        elif header_name == "laser_y_filtre":
                            _, ly = self.last_filtered_pos
                            ligne.append(round(ly, 2) if ly is not None else '')
                        elif header_name in data:
                            temp_value = data[header_name]
                            # Arrondir les températures pour le CSV, gérer NaN
                            ligne.append(round(temp_value, 2) if pd.notna(temp_value) else '')
                        else:
                            ligne.append('') # Laisser vide si header non trouvé (ne devrait pas arriver)
                    all_data.append(ligne)

                else: # Si get_temperatures retourne None ou un dict vide
                    # os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues. Attente...")
                    print("=" * 60)

                # Pause pour respecter l'intervalle demandé
                # Calcule le temps restant avant la prochaine itération
                time_spent = time.time() - current_time
                sleep_time = max(0, interval - time_spent)
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            if plt.fignum_exists(fig.number):
                plt.close(fig) # Fermer la fenêtre graphique

            # Sauvegarde du fichier CSV si des données ont été collectées
            if all_data:
                print("💾 Sauvegarde du fichier CSV...")
                # Sauvegarder sur le Bureau pour accès facile
                desktop_path = Path.home() / "Desktop"
                # Nom de fichier avec date et heure
                filename = f"acquisition_gradient_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = desktop_path / filename
                try:
                    # Écrire le fichier CSV
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers) # Écrit les headers définis plus haut
                        writer.writerows(all_data) # Écrit toutes les lignes collectées
                    print(f"✅ Données sauvegardées dans : {csv_path}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")
            else:
                print("ℹ️ Aucune donnée collectée à sauvegarder.")


if __name__ == "__main__":
    # Mettre simulation=False pour utiliser l'Arduino
    # Mettre simulation=True pour utiliser le fichier CSV
    td = TraitementDonnees(simulation=True)
    # Intervalle de mise à jour (en secondes)
    td.demarrer_acquisition_live(interval=0.1) # Intervalle rapide pour la simulation

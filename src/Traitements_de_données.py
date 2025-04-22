
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


class TraitementDonnees:
    VREF = 3.003
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem101",path = "data/", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.path = path
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)
        self.puissance = 0
        self.data_photodiodes = [0,0,0,0,0,0]

        self.correction_matrices = [pd.read_csv(self.path + f"matrice_corr_diode_{i}.csv", sep=',', decimal='.').values for i in range(6)]
        self.photodiode_ratios_450 = [pd.read_csv(self.path + "ratios_photodiodes_450.csv", sep=';', decimal=',')[col].values
                                for col in pd.read_csv(self.path + "ratios_photodiodes_450.csv", sep=';', decimal=',').columns]
        self.photodiode_ratios_976 = [pd.read_csv(self.path + "ratios_photodiodes_976.csv", sep=';', decimal=',')[col].values
                                for col in pd.read_csv(self.path + "ratios_photodiodes_976.csv", sep=';', decimal=',').columns]
        self.photodiode_ratios_1976 = pd.read_csv(self.path + "ratios_photodiodes_1976.csv", sep=';', decimal=',').values
        self.photodiode_tensions_450 = [pd.read_csv(self.path + "tensions_photodiodes_450.csv", sep=';', decimal=',')[col].values
                                    for col in pd.read_csv(self.path + "tensions_photodiodes_450.csv", sep=';', decimal=',').columns]
        self.photodiode_tensions_976 = [pd.read_csv(self.path + "tensions_photodiodes_976.csv", sep=';', decimal=',')[col].values
                                    for col in pd.read_csv(self.path + "tensions_photodiodes_976.csv", sep=';', decimal=',').columns]

        # Décalage à appliquer
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas

        self.tension_photodidodes = [0,0,0,0,0,0]

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

        self.photodiodes = ["PD25","PD26","PD27","PD28","PD29","PD30"]

        self.indices_à_garder = list(range(31)) # R1-R11, R13-R21 (R24 est sur canal 11)
        # self.indices_à_garder.append(24) # Si R25 est lue
        # self.indices_photodiodes = list(range(25, 31))

        self.simulation_data = None
        self.simulation_index = 150
        self.simulation_columns = [p[0] for i, p in enumerate(self.positions) if p[0] != "R_Virtuel" and i in self.indices_à_garder]
        if "R25" in [p[0] for p in self.positions] and 24 not in self.indices_à_garder:
             if any(p[0] == "R25" for p in self.positions):
                 self.simulation_columns.append("R25")

        self.simulation_columns += self.photodiodes

        # print(self.simulation_columns)

        # --- AJOUT : Pour stocker la carte de température précédente ---
        self.previous_ti_filtered = None
        # --- FIN AJOUT ---
        # --- AJOUT : Pour stocker la carte de température précédente ---
        self.previous_ti_filtered = None
        # --- FIN AJOUT ---

        # --- AJOUT : Pour le filtrage de position ---
        self.position_history = [] # Historique pour la médiane mobile
        self.history_length = 5    # Nombre de positions à garder (ajustable)
        self.last_valid_raw_pos = None # Dernière position brute jugée valide (pour limite vitesse)
        self.last_filtered_pos = (None, None) # Dernière position filtrée (pour affichage)
        self.max_speed_mm_per_interval = 3.0 # Max déplacement en mm entre frames (ajustable)
        self.min_heating_threshold = 0.05
        # --- FIN AJOUT ---
        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                script_dir = Path(__file__).parent
                simulation_file_path = script_dir.parent / "data" / "Hauteur 3.csv"
                self.simulation_data = pd.read_csv(simulation_file_path)
                #self.simulation_data = pd.read_csv(simulation_file_path, sep = ';', decimal = ',')
                print(f"[SIMULATION] Chargement du fichier CSV : {simulation_file_path.resolve()}")

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
        if resistance == float('inf') or resistance <= 0 or pd.isna(resistance):
            return np.nan
        A, B, C = coeffs
        kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
        if pd.isna(kelvin):
            return np.nan
        return kelvin - 273.15

    def lire_donnees(self):
        if self.simulation:
            return self.simulation_data is not None and not self.simulation_data.empty

        if self.ser is None:
            print("[ERREUR] Connexion série non établie.")
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        photodiodes_dict = {}
        start_time = time.time()
        timeout_sec = 2 # Augmenté légèrement pour la robustesse

        while True:
            current_time = time.time()
            if current_time - start_time > timeout_sec:
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                # Retourner les données partielles ou None ? Ici on retourne partiel si on a quelque chose.
                return voltages_dict if voltages_dict else None

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    print(f"[DEBUG RAW] Reçu: '{line}'")
                    if not line:
                        continue

                    if "Fin du balayage" in line:
                        # print("[DEBUG] Fin du balayage détectée.") # Debug
                        break # Sortir de la boucle while interne

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        canal = int(match.group(1))
                        if canal in self.indices_à_garder:
                            try:
                                 voltages_dict[canal] = float(match.group(2))
                                 # print(f"[DEBUG] Reçu Canal {canal}: {voltages_dict[canal]} V") # Debug
                            except ValueError:
                                 print(f"[AVERTISSEMENT] Impossible de convertir la tension '{match.group(2)}' pour le canal {canal}")
                                 voltages_dict[canal] = np.nan

                        # if canal in self.indices_photodiodes:
                        #      try:
                        #          photodiodes_dict[canal] = float(match.group(2))
                        #          # print(f"[DEBUG] Reçu Canal {canal}: {voltages_dict[canal]} V") # Debug
                        #      except ValueError:
                        #          print(f"[AVERTISSEMENT] Impossible de convertir la tension '{match.group(2)}' pour le canal {canal}")
                        #          photodiodes_dict[canal] = np.nan
                else:
                    # Petite pause pour ne pas saturer le CPU si rien n'est reçu
                    time.sleep(0.01)

            except serial.SerialException as e:
                print(f"Erreur série pendant la lecture : {e}")
                self.ser = None # Marquer comme déconnecté
                return None
            except Exception as e:
                print(f"Erreur inattendue pendant la lecture série : {e}")
                # Continuer la boucle peut être risqué, mais on essaie
                continue

        # Vérification après la sortie de boucle (Fin du balayage ou timeout)
        canaux_attendus = set(self.indices_à_garder)
        canaux_recus = set(voltages_dict.keys())

        if canaux_recus != canaux_attendus:
             canaux_manquants = canaux_attendus - canaux_recus
             print(f"⚠️ Seulement {len(canaux_recus)}/{len(canaux_attendus)} canaux requis reçus. Manquants: {sorted(list(canaux_manquants))}")
             # Option: Remplir les manquants avec NaN si on veut quand même continuer
             # for canal_manquant in canaux_manquants:
             #     voltages_dict[canal_manquant] = np.nan
             # return voltages_dict # Retourner données partielles + NaN
             return None # Préférable de retourner None si incomplet

        # print(f"[DEBUG] Données lues avec succès: {len(voltages_dict)} canaux.") # Debug

        data_phot = []

        for k, v in photodiodes_dict.items():
            data_phot.append(v)
            # print(v)

        # light_type, wavelength, power = self.get_wavelength()

        # print(f"Laser {light_type}, longueur d'onde de {wavelength:.0f} nm et puissance estimée de {power:.2f} W\n")

        # self.data_photodiodes = data_phot

        # print(voltages_dict)

        return voltages_dict

    def get_temperatures(self):
        real_temps_dict = {} # Dictionnaire pour les températures réelles
        real_tension_dict = {} # Dictionnaire pour les tensions réelles

        if self.simulation:
            # --- Logique Simulation CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0 # Retour au début
                    print("[SIMULATION] Fin du fichier CSV atteinte, retour au début.")

                # Lire la ligne actuelle
                current_data_row = self.simulation_data.iloc[self.simulation_index]
                # Incrémenter pour la prochaine lecture (peut être ajusté)
                self.simulation_index += 2 # Lire chaque ligne
                valid_data_found = False

                # Lire les températures simulées pour les thermistances réelles (SAUF R24 pour l'instant)
                for i, (name, _) in enumerate(self.positions):
                    if name in ["R_Virtuel", "R24"]: continue # Ignorer la virtuelle et R24 ici

                    # Cas général pour les thermistances dans simulation_columns
                    if name in self.simulation_columns:
                        if name in current_data_row and pd.notna(current_data_row[name]):
                            real_temps_dict[name] = current_data_row[name]
                            valid_data_found = True
                        else:
                            real_temps_dict[name] = np.nan # Mettre NaN si absent ou non numérique

                    # Cas spécifique pour R25 si elle est dans 'positions' mais pas lue directement
                    # et si elle existe dans le CSV
                    elif name == "R25" and "R25" in self.simulation_data.columns:
                         if "R25" in current_data_row and pd.notna(current_data_row["R25"]):
                             real_temps_dict["R25"] = current_data_row["R25"]
                             valid_data_found = True # Compte comme donnée valide
                         else:
                             real_temps_dict["R25"] = np.nan

                for i, name in enumerate(self.photodiodes):
                    if name in self.simulation_columns:
                        if name in current_data_row and pd.notna(current_data_row[name]):
                            real_temps_dict[name] = current_data_row[name]
                            valid_data_found = True
                        else:
                            real_temps_dict[name] = 0 # Mettre NaN si absent ou non numérique


                # Si aucune donnée valide n'a été trouvée (hors R24/Virtuelle)
                if not valid_data_found:
                    print(f"[AVERTISSEMENT SIMULATION] Aucune donnée valide (hors R24/Virtuelle) à l'index CSV {self.simulation_index - 1}.")
                    # Initialiser R24 et R_Virtuel à NaN avant de retourner
                    real_temps_dict["R24"] = np.nan
                    real_temps_dict["R_Virtuel"] = np.nan
                    return real_temps_dict # Retourne le dict avec potentiellement que des NaN

                # --- Logique R24 (Moyenne Pondérée) pour Simulation ---
                # (Identique à avant, utilise real_temps_dict rempli ci-dessus)
                weighted_sum_r24 = 0.0
                total_weight_r24 = 0.0
                # --- MODIFIÉ : Poids ajustés pour R24 ---
                thermistors_r24_weights = {"R19": 0.1, "R20": 0.15, "R21": 0.15} # 40%
                other_thermistors_for_r24 = []

                # Identifier les autres thermistances réelles valides (pour R24)
                for name, temp in real_temps_dict.items():
                    # Exclure R25 et celles avec poids spécifique, et vérifier validité
                    if name != "R25" and name not in thermistors_r24_weights and pd.notna(temp):
                         other_thermistors_for_r24.append(name)
                    if name != "R25" and name not in thermistors_r24_weights and pd.notna(temp):
                         other_thermistors_for_r24.append(name)

                for name, tension in real_tension_dict.items():
                    pass

                # Calculer le poids pour les "autres" thermistances (pour R24)
                weight_per_other_r24 = 0.0
                if other_thermistors_for_r24:
                    # Les autres se partagent 60% du poids
                    weight_per_other_r24 = 0.6 / len(other_thermistors_for_r24)

                # Calculer la somme pondérée et le poids total (pour R24)
                for name, temp in real_temps_dict.items():
                     if name == "R25": continue # Exclure R25
                     if pd.notna(temp):
                         if name in thermistors_r24_weights:
                             weight = thermistors_r24_weights[name]
                         elif name in other_thermistors_for_r24:
                             weight = weight_per_other_r24
                         else:
                             continue # Ignorer si non pertinent pour R24

                         weighted_sum_r24 += temp * weight
                         total_weight_r24 += weight

                # Assigner la valeur à R24
                if total_weight_r24 > 1e-6: # Éviter division par zéro
                    real_temps_dict["R24"] = weighted_sum_r24 / total_weight_r24
                else:
                    real_temps_dict["R24"] = np.nan # Si aucun contributeur valide

            else:
                # Fallback si CSV échoue ou est vide
                print("[SIMULATION] Données CSV non disponibles ou vides, génération de températures aléatoires.")
                temp_gen_dict = {}
                for i, (name, _) in enumerate(self.positions):
                     if name != "R_Virtuel": # Ne pas générer pour la virtuelle initialement
                         temp_gen_dict[name] = np.random.uniform(20.0, 35.0) # Plage réaliste

                # Calculer R24 à partir des valeurs générées (logique similaire à ci-dessus)
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
                         if name in thermistors_r24_weights_gen:
                             weight = thermistors_r24_weights_gen[name]
                         elif name in other_thermistors_for_r24_gen:
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
                 # Si lire_donnees retourne None (erreur ou incomplet), remplir avec NaN
                for i, (name, _) in enumerate(self.positions):
                     # Initialiser toutes les positions (y compris R_Virtuel) à NaN
                    real_temps_dict[name] = np.nan
                return real_temps_dict # Retourner le dict rempli de NaN

            temperatures_raw = {}
            # Mapping Nom -> Index Canal (pour lecture voltage)
            # R24 n'est pas listée ici car elle n'est pas lue directement via son nom
            indices_mapping = {
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10, # Canal 11 est pour R24 physiquement
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                "R25": 24 # Décommenter si R25 est lue sur le canal 24
            }
            # Mapping Nom -> Index Coefficient (pour calcul température)
            coeffs_mapping = {
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10,
                # R24 (coeffs[23]) sera calculée par moyenne pondérée, pas directement ici
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                "R25": 24 # Décommenter si R25 utilise coeffs[24]
            }

            # Calcul initial des températures réelles (SAUF R24)
            for nom_thermistor, canal_index in indices_mapping.items():
                # Le canal 11 est physiquement connecté à R24, on l'ignore ici
                # car R24 est calculée par moyenne pondérée plus tard.
                if canal_index == 11: continue

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

            # Gérer R25 si elle est lue physiquement
            # if 24 in self.indices_à_garder and 24 in data_voltages:
            #     if pd.notna(data_voltages[24]):
            #         voltage_r25 = data_voltages[24]
            #         coeffs_r25 = self.coefficients[24] # Assumer coeffs[24] pour R25
            #         resistance_r25 = self.compute_resistance(voltage_r25)
            #         temp_r25 = self.compute_temperature(resistance_r25, coeffs_r25)
            #         temperatures_raw["R25"] = temp_r25
            #     else:
            #         temperatures_raw["R25"] = np.nan
            # elif "R25" in [p[0] for p in self.positions]: # Si R25 existe mais n'est pas lue
            #     temperatures_raw["R25"] = np.nan # Initialiser à NaN

            real_temps_dict = temperatures_raw.copy() # Copier les températures calculées

            # --- Logique R24 (Moyenne Pondérée) pour Lecture Série ---
            # (Identique à la logique de simulation, mais utilise les températures réelles calculées)
            weighted_sum_r24_real = 0.0
            total_weight_r24_real = 0.0
            thermistors_r24_weights_real = {"R19": 0.1, "R20": 0.15, "R21": 0.15} # 40%
            other_thermistors_for_r24_real = []

            # Identifier les autres thermistances réelles valides (déjà calculées, hors R25)
            for name, temp in real_temps_dict.items():
                if name != "R25" and name not in thermistors_r24_weights_real and pd.notna(temp):
                     other_thermistors_for_r24_real.append(name)

            # Calculer le poids pour les "autres" thermistances (pour R24)
            weight_per_other_r24_real = 0.0
            if other_thermistors_for_r24_real:
                # Les autres se partagent 60% du poids
                weight_per_other_r24_real = 0.6 / len(other_thermistors_for_r24_real)

            # Calculer la somme pondérée et le poids total (pour R24)
            for name, temp in real_temps_dict.items():
                 if name == "R25": continue # Exclure R25
                 if pd.notna(temp):
                     if name in thermistors_r24_weights_real:
                         weight = thermistors_r24_weights_real[name]
                     elif name in other_thermistors_for_r24_real:
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


        # --- CALCUL DE LA THERMISTANCE VIRTUELLE (Commun aux deux modes) ---
        # Utilise le `real_temps_dict` qui contient maintenant R1-R11, R13-R21, R24 (calculée), et R25 (si lue/simulée)
        weighted_sum_virt = 0.0
        total_weight_virt = 0.0
        # --- MODIFIÉ : Poids ajustés pour R_Virtuel ---
        thermistors_virt_weights = {"R14": 0.15, "R10": 0.15, "R9": 0.15} # 45%
        other_thermistors_for_virt = []

        # Identifier les autres thermistances réelles valides (pour R_Virtuel)
        # Inclut R24 mais exclut R25 et R_Virtuel elle-même
        for name, temp in real_temps_dict.items():
            # Exclure R25, celles avec poids spécifique, et vérifier validité
            if name != "R25" and name not in thermistors_virt_weights and pd.notna(temp):
                 other_thermistors_for_virt.append(name)

        # Calculer le poids pour les "autres" thermistances (pour R_Virtuel)
        weight_per_other_virt = 0.0
        if other_thermistors_for_virt:
            # Les autres se partagent 55% du poids total (1.0 - 0.45 = 0.55)
            weight_per_other_virt = 0.55 / len(other_thermistors_for_virt)

        # Calculer la somme pondérée et le poids total (pour R_Virtuel)
        for name, temp in real_temps_dict.items():
             if name == "R25": continue # Exclure R25
             if pd.notna(temp):
                 if name in thermistors_virt_weights:
                     weight = thermistors_virt_weights[name]
                 elif name in other_thermistors_for_virt:
                     weight = weight_per_other_virt
                 else:
                     continue # Ignore les autres cas

                 weighted_sum_virt += temp * weight
                 total_weight_virt += weight

        # Assigner la valeur à R_Virtuel
        if total_weight_virt > 1e-6: # Éviter division par zéro
            virtual_temp = weighted_sum_virt / total_weight_virt
        else:
            virtual_temp = np.nan # Si aucune thermistance valide pour la moyenne

        real_temps_dict["R_Virtuel"] = virtual_temp # Ajouter la virtuelle au dict final

        if not self.simulation:
            for i, name in enumerate(self.photodiodes):
                real_temps_dict[name] = data_voltages[i + 25]

        return real_temps_dict

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        # --- MODIFIÉ : Un seul subplot pour le gradient ---
        ax = fig.add_subplot(111) # Heatmap Magnitude Gradient

        x_all_points, y_all_points, t_all_points = [], [], []
        valid_temps_list = []
        thermistor_data_for_plot = [] # Gardé pour le calcul de baseline et RBF

        # 1. Collecter points valides (réels + virtuel) SAUF R25 (INCHANGÉ)
        for name, pos in self.positions:
            if name == "R25": continue
            temp_val = temperature_dict.get(name, np.nan)
            if pd.notna(temp_val):
                x_all_points.append(pos[0])
                y_all_points.append(pos[1])
                t_all_points.append(temp_val)
                valid_temps_list.append(temp_val)
                # Garder les données pour RBF même si on n'affiche pas les points sur ce graphe
                thermistor_data_for_plot.append({"name": name, "pos": pos, "temp": temp_val})

        # 2. Calculer baseline (basée sur points valides hors R25) (INCHANGÉ)
        if not valid_temps_list:
            baseline_temp = 20.0
            print("[AVERTISSEMENT HEATMAP] Aucune donnée valide (hors R25) pour calculs.")
        else:
            baseline_temp = min(valid_temps_list) - 0.5

        # --- Section Interpolation RBF (INCHANGÉ) ---
        r_max = 12.5
        num_edge_points = 12
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [baseline_temp] * num_edge_points

        x_combined = x_all_points + list(edge_x)
        y_combined = y_all_points + list(edge_y)
        t_combined = t_all_points + edge_t

        # Initialisations (INCHANGÉ)
        ti_filtered = None
        xi, yi = None, None
        mask = None
        grad_magnitude = None
        grad_magnitude_masked = None
        raw_laser_x, raw_laser_y = None, None
        raw_pos_found_this_frame = False
        final_laser_pos_found = False

        if len(x_combined) < 3:
            print("[ERREUR HEATMAP] Pas assez de points (hors R25) pour l'interpolation RBF.")
            # --- MODIFIÉ : Titre pour le seul axe ---
            ax.set_title("Pas assez de données (hors R25) pour RBF/Gradient")
            # Optionnel: Afficher les points sur cet axe si erreur
            # for item in thermistor_data_for_plot: ...
            return

        try:
            # --- Calcul RBF et Filtre Température (INCHANGÉ) ---
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
            # ti_masked = np.ma.array(ti_filtered, mask=mask) # Plus nécessaire pour l'affichage direct

            # --- Calcul du Gradient Spatial (INCHANGÉ) ---
            grad_y, grad_x = np.gradient(ti_filtered)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_magnitude_masked = np.ma.array(grad_magnitude, mask=mask)

            # --- Calcul Position Laser (Max Diff -> Min Grad Local + Filtres) (INCHANGÉ) ---
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
                                    # print(f"[INFO LASER] Position brute ({potential_laser_x:.1f}, {potential_laser_y:.1f}) rejetée : déplacement trop rapide.")

                            if is_within_radius and is_plausible_move:
                                raw_laser_x = potential_laser_x
                                raw_laser_y = potential_laser_y
                                raw_pos_found_this_frame = True
                                self.last_valid_raw_pos = (raw_laser_x, raw_laser_y)
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
                raw_pos_found_this_frame = False
                self.last_valid_raw_pos = None
                if self.previous_ti_filtered is None:
                    print("[INFO LASER] Attente du prochain frame pour calculer la position.")
                elif self.previous_ti_filtered.shape != ti_filtered.shape:
                     print("[ERREUR LASER] Incohérence de forme de grille, réinitialisation.")
                     self.previous_ti_filtered = None

            # --- Filtrage Temporel (Médiane Mobile) (INCHANGÉ) ---
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
             print(f"[ERREUR RBF/GRADIENT/LASER] Échec: {e}")
             # --- MODIFIÉ : Titre pour le seul axe ---
             ax.set_title("Erreur Calcul Gradient/Laser")
             # Optionnel: Afficher points si erreur
             # for item in thermistor_data_for_plot: ...
             self.previous_ti_filtered = None
             self.last_valid_raw_pos = None
             self.position_history = []
             self.last_filtered_pos = (None, None)
             final_laser_pos_found = False
             return

        # --- Mise à jour de la carte précédente (INCHANGÉ) ---
        if ti_filtered is not None:
            self.previous_ti_filtered = ti_filtered.copy()

        # --- Affichage Subplot UNIQUE : Heatmap Magnitude du Gradient ---
        if grad_magnitude_masked is not None:
            contour = ax.contourf(xi, yi, grad_magnitude_masked, levels=50, cmap="viridis")
            fig.colorbar(contour, ax=ax, label="Magnitude Gradient Temp. (°C/mm)", shrink=0.8) # Ajusté shrink
            # Optionnel: Afficher les points des thermistances (hors R25)
            ax.scatter(x_all_points, y_all_points, color='white', marker='.', s=10, alpha=0.5, label='Thermistances')

            # --- MODIFIÉ : Afficher la position du laser FILTRÉE et ajouter à la légende ---
            plot_x, plot_y = self.last_filtered_pos
            if final_laser_pos_found:
                # Le label contient déjà les coordonnées
                label_laser = f'Laser (Médiane {len(self.position_history)}/{self.history_length}) @ ({plot_x:.1f}, {plot_y:.1f})'
                ax.plot(plot_x, plot_y, 'rx', markersize=10, label=label_laser) # Croix rouge

            # Configuration de l'axe unique
            ax.set_aspect('equal')
            ax.set_title(f"Gradient Température (Tps: {elapsed_time:.2f} s)", fontsize=10) # Ajusté fontsize
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)
            # --- MODIFIÉ : Afficher la légende (inclut maintenant le laser si trouvé) ---
            ax.legend(fontsize=8, loc='upper right') # Ajusté fontsize
        else:
            # --- MODIFIÉ : Titre pour le seul axe ---
            ax.set_title("Gradient non calculé")
            ax.set_aspect('equal')
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)

        # --- Ajustement final de la mise en page ---
        fig.tight_layout(pad=2.5) # Ajusté pad




    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        fig = plt.figure(figsize=(12, 6))
        plt.ion()
        fig.show()

        all_data = []
        base_headers = [name for name, _ in self.positions]
        extra_headers = ["timestamp", "temps_ecoule_s"]
        headers = base_headers[:-1] + self.photodiodes.copy() + extra_headers

        start_time = time.time()
        keep_running = True
        try:
            while keep_running:
                if not plt.fignum_exists(fig.number):
                    print("\nFenêtre graphique fermée. Arrêt de l'acquisition.")
                    keep_running = False
                    break

                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures()

                # print(data)

                if data:
                    #os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0

                    for i in range(6):
                        self.tension_photodidodes[i] = data[self.photodiodes[i]]

                    for name, temp in data.items():
                        display_name = name
                        if pd.notna(temp):
                            if name in self.photodiodes:
                                print(f"{display_name:<10} : {temp:6.3f}  V")
                            elif name == "R_Virtuel":
                                pass
                            else:
                                print(f"{display_name:<10} : {temp:6.2f} °C")
                            if name != "R_Virtuel" and name != "R25":
                                valid_temps_count += 1
                        else:
                            print(f"{display_name:<10} :   --   °C (NaN)")
                    # real_thermistor_count = len([p for p in self.positions if p[0] not in ["R_Virtuel", "R25"]])
                    # print(f"({valid_temps_count}/{real_thermistor_count} thermistances réelles (hors R25) valides)")
                    print("-" * 60)
                    light_type, wavelength, self.puissance = self.get_wavelength()
                    print(f"Laser {light_type}, longueur d'onde de {wavelength:.0f} nm et puissance estimée de {self.puissance:.2f} W")
                    print("=" * 60)

                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    ligne = []
                    t_ref_value = data.get("R25", 25.0)
                    if pd.isna(t_ref_value): t_ref_value = 25.0

                    for header_name in headers:
                        if header_name == "T_ref":
                            ligne.append(round(t_ref_value, 2))
                        elif header_name == "timestamp":
                            ligne.append(datetime.now().isoformat(timespec='seconds'))
                        elif header_name == "temps_ecoule_s":
                            ligne.append(round(elapsed_time, 3))
                        elif header_name in data:
                            temp_value = data[header_name]
                            ligne.append(round(temp_value, 2) if pd.notna(temp_value) else '')
                        else:
                            ligne.append('')
                    all_data.append(ligne)

                else:
                    #os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues. Attente...")
                    print("=" * 60)

                time.sleep(max(0, interval - (time.time() - current_time)))

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            if plt.fignum_exists(fig.number):
                plt.close(fig)

            if all_data:
                print("Sauvegarde du fichier CSV...")
                desktop_path = Path.home() / "Desktop"
                filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = desktop_path / filename
                try:
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers)
                        writer.writerows(all_data)
                    print(f"✅ Données sauvegardées dans : {csv_path}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")
            else:
                print("ℹAucune donnée collectée à sauvegarder.")

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

    def get_visible_wavelength(self, V_corr, threshold=0.1):
        V_corr[-2] = 0
        ratios_corr = np.divide(V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0)
        ratio_ids_corr = [self.indexes(self.photodiode_ratios_450[i], ratio, threshold) for i, ratio in enumerate(ratios_corr)]
        if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
            return np.array([])
        return reduce(np.intersect1d, ratio_ids_corr)

    def get_NIR_wavelength(self, V_corr, threshold=0.1):
        ratios_corr = np.divide(V_corr[1:], V_corr[:-1], out=np.zeros_like(V_corr[1:]), where=V_corr[:-1] != 0)
        ratio_ids_corr = [self.indexes(self.photodiode_ratios_976[i], ratio, threshold) for i, ratio in enumerate(ratios_corr)]
        if not ratio_ids_corr or any(len(ids) == 0 for ids in ratio_ids_corr):
            return np.array([])
        return reduce(np.intersect1d, ratio_ids_corr)

    def get_IR_wavelength(self, V_corr, puissance, threshold):
        ratio = V_corr / puissance
        return self.indexes(self.photodiode_ratios_1976, ratio, threshold)

    def get_VIS_power(self, wavelength, V_corr):
        V_corr[-2] = 0
        V_corr[-1] = 0
        V_ratio = [10 * V_corr[i] / self.photodiode_tensions_450[i][int(wavelength) - 200] for i in range(6)
                if self.photodiode_tensions_450[i][int(wavelength) - 200] != 0 and V_corr[i] != 0]
        return np.mean(V_ratio)

    def get_NIR_power(self, wavelength, V_corr):
        V_ratio = [10 * V_corr[i] / self.photodiode_tensions_976[i][int(wavelength) - 200] for i in range(6)
                if self.photodiode_tensions_976[i][int(wavelength) - 200] != 0 and V_corr[i] != 0]
        return np.mean(V_ratio)

    def get_wavelength(self, threshold=0.1, threshold_mult=1.25):
        if self.last_valid_raw_pos is None:
            y, x = (0, 0)
        else:
            y, x = self.last_valid_raw_pos

        pos = self.id_pos((x, y))

        V_photodiodes = self.data_photodiodes

        # print(V_photodiodes)

        # V_corr = np.array([self.tension_photodidodes * self.correction_matrices[i][pos] for i, V in enumerate(V_photodiodes)])
        # index_max = np.argmax(V_corr)

        V_corr = V_photodiodes
        index_max = np.argmax(V_corr)

        # print(V_corr)

        # if all(V < 0.01 for V in V_corr):
            # return "inconnu", 0, self.puissance
        if index_max == 0:
            return "UV", 0, self.puissance
        elif index_max == 1:
            self.wavelength = np.mean(self.precise_wavelength(self.get_visible_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "VIS", self.wavelength, self.get_VIS_power(self.wavelength, V_corr)
        elif index_max == 5:
            self.wavelength = np.mean(self.precise_wavelength(self.get_IR_wavelength, V_corr[-1], self.puissance, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "IR", self.wavelength, self.puissance
        else:
            self.wavelength = np.mean(self.precise_wavelength(self.get_NIR_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "NIR", self.wavelength, self.get_NIR_power(self.wavelength, V_corr)


if __name__ == "__main__":
    td = TraitementDonnees(simulation=False)
    td.demarrer_acquisition_live(interval=0.1)
    puissance_estimee = 0.75  # en Watts (exemple arbitraire)

    # type_lumiere, lambda_nm, puissance_corrigee = get_wavelength(
    #     position=position,
    #     V_photodiodes=V_photodiodes,
    #     puissance=puissance_estimee
    # )

    # print(f"Résultat : {type_lumiere} | λ = {lambda_nm:.1f} nm | Puissance corrigée = {puissance_corrigee:.2f} W")

    
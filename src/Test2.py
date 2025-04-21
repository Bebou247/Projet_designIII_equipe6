# -*- coding: utf-8 -*-
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
        self.simulation_index = 0
        # Noms des colonnes pour la simulation (basés sur les *vraies* thermistances)
        self.simulation_columns = [self.positions[i][0] for i, (nom, _) in enumerate(self.positions) if nom != "R_Virtuel" and i in self.indices_à_garder]
        # Ajouter R25 si elle est simulée depuis le CSV
        if "R25" in [p[0] for p in self.positions] and 24 not in self.indices_à_garder: # Si R25 existe mais n'est pas lue directement
             if any(p[0] == "R25" for p in self.positions): # Vérifie si R25 est dans la liste
                 self.simulation_columns.append("R25")


        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                script_dir = Path(__file__).parent
                # --- MODIFIÉ : Chemin vers le fichier CSV ---
                # Assurez-vous que ce chemin est correct pour votre fichier de simulation
                simulation_file_path = script_dir.parent / "data" / "Hauteur 4.csv"
                # --- FIN MODIFICATION ---
                self.simulation_data = pd.read_csv(simulation_file_path)
                print(f"[SIMULATION] Chargement du fichier CSV : {simulation_file_path.resolve()}")

                # Vérification des colonnes nécessaires pour la simulation
                missing_cols = [col for col in self.simulation_columns if col not in self.simulation_data.columns]
                if missing_cols:
                    print(f"[ERREUR SIMULATION] Colonnes manquantes dans {simulation_file_path.name}: {missing_cols}")
                    self.simulation_data = None
                else:
                    # Convertir les colonnes pertinentes en numérique, gérant les erreurs
                    for col in self.simulation_columns:
                        # Utiliser le séparateur décimal approprié si nécessaire (ex: decimal=',')
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
        return voltages_dict

    def get_temperatures(self):
        real_temps_dict = {} # Dictionnaire pour les températures réelles

        if self.simulation:
            # --- Logique Simulation CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0 # Retour au début
                    print("[SIMULATION] Fin du fichier CSV atteinte, retour au début.")

                # Lire la ligne actuelle
                current_data_row = self.simulation_data.iloc[self.simulation_index]
                # Incrémenter pour la prochaine lecture (peut être ajusté)
                self.simulation_index += 5 # Lire chaque ligne
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
                # "R25": 24 # Décommenter si R25 est lue sur le canal 24
            }
            # Mapping Nom -> Index Coefficient (pour calcul température)
            coeffs_mapping = {
                "R1": 0, "R2": 1, "R3": 2, "R4": 3, "R5": 4, "R6": 5, "R7": 6,
                "R8": 7, "R9": 8, "R10": 9, "R11": 10,
                # R24 (coeffs[23]) sera calculée par moyenne pondérée, pas directement ici
                "R13": 12, "R14": 13, "R15": 14, "R16": 15, "R17": 16, "R18": 17,
                "R19": 18, "R20": 19, "R21": 20,
                # "R25": 24 # Décommenter si R25 utilise coeffs[24]
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

        return real_temps_dict


    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        ax1 = fig.add_subplot(121) # Heatmap Température
        ax2 = fig.add_subplot(122) # Heatmap Magnitude Gradient

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
            avg_temp = np.mean(valid_temps_list)
            # Ajustement baseline : peut être plus subtil, ex: min ou percentile
            baseline_temp = min(valid_temps_list) - 0.5 # Un peu en dessous du minimum mesuré

        # --- Section Interpolation RBF ---
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
        grad_magnitude_masked = None
        laser_x, laser_y = None, None
        laser_pos_found = False

        if len(x_combined) < 3: # Besoin d'au moins 3 points pour RBF
            print("[ERREUR HEATMAP] Pas assez de points (hors R25) pour l'interpolation RBF.")
            ax1.set_title("Pas assez de données (hors R25) pour RBF")
            ax2.set_title("Gradient non calculable")
            # Afficher points disponibles sur ax1
            for item in thermistor_data_for_plot:
                 if pd.notna(item["temp"]):
                     is_virtual = item["name"] == "R_Virtuel"
                     marker = 's' if is_virtual else 'o'
                     color = 'magenta' if is_virtual else 'black'
                     ax1.scatter(item["pos"][0], item["pos"][1], color=color, marker=marker, s=35)
                     ax1.annotate(item["name"], item["pos"], textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8, color=color)
            # Pas de légende nécessaire ici car pas assez de points pour la heatmap
            return

        try:
            rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
            grid_size = 200 # Taille de la grille pour l'interpolation
            xi, yi = np.meshgrid(
                np.linspace(-r_max, r_max, grid_size),
                np.linspace(-r_max, r_max, grid_size)
            )
            ti = rbf(xi, yi)
            # Appliquer un filtre Gaussien pour lisser la carte de température
            sigma_filter_temp = 1.2 # Ajustable
            ti_filtered = gaussian_filter(ti, sigma=sigma_filter_temp)
            mask = xi**2 + yi**2 > r_max**2 # Masque circulaire
            ti_masked = np.ma.array(ti_filtered, mask=mask) # Pour heatmap température

            # --- Calcul du Gradient ---
            grad_y, grad_x = np.gradient(ti_filtered) # Calculé sur la carte filtrée
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            grad_magnitude_masked = np.ma.array(grad_magnitude, mask=mask) # Pour heatmap gradient

            # --- Calcul Position Laser (Minimum du Gradient) ---
            try:
                # S'assurer qu'il y a des données valides non masquées
                if grad_magnitude_masked.count() > 0:
                    # Remplacer les valeurs masquées par NaN pour utiliser nanargmin
                    grad_filled_nan = grad_magnitude_masked.filled(np.nan)
                    min_idx_flat = np.nanargmin(grad_filled_nan)
                    min_idx = np.unravel_index(min_idx_flat, grad_magnitude_masked.shape)

                    # Obtenir les coordonnées correspondantes
                    laser_x = xi[min_idx]
                    laser_y = yi[min_idx]
                    laser_pos_found = True
                else:
                     print("[AVERTISSEMENT LASER] Impossible de trouver le minimum du gradient (carte vide ou entièrement masquée).")
                     laser_pos_found = False

            except ValueError: # Peut arriver si nanargmin échoue (ex: tout est NaN)
                print("[AVERTISSEMENT LASER] Impossible de trouver le minimum du gradient (ValueError).")
                laser_pos_found = False
            # --- FIN Calcul Position Laser ---

        except Exception as e:
             print(f"[ERREUR RBF/GRADIENT] Échec: {e}")
             ax1.set_title("Erreur RBF/Gradient")
             ax2.set_title("Erreur RBF/Gradient")
             # Afficher points sur ax1 si erreur
             for item in thermistor_data_for_plot:
                  if pd.notna(item["temp"]):
                      is_virtual = item["name"] == "R_Virtuel"
                      marker = 's' if is_virtual else 'o'
                      color = 'magenta' if is_virtual else 'black'
                      ax1.scatter(item["pos"][0], item["pos"][1], color=color, marker=marker, s=35)
                      ax1.annotate(item["name"], item["pos"], textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8, color=color)
             return

        # --- Affichage Subplot 1 : Heatmap Température ---
        contour1 = ax1.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour1, ax=ax1, label="Température (°C)", shrink=0.6)

        # Affichage des points (réels et virtuel) sur ax1
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

        # Afficher le point laser (Minimum Gradient) sur ax1
        if laser_pos_found:
            # Utiliser 'bx' (croix bleue) pour le différencier
            ax1.plot(laser_x, laser_y, 'bx', markersize=10, label=f'Laser (Min Grad) @ ({laser_x:.1f}, {laser_y:.1f})')

        # Configuration ax1
        ax1.set_aspect('equal')
        title_ax1 = f"Heatmap Température (Tps: {elapsed_time:.2f} s)"
        if laser_pos_found:
             title_ax1 += f"\nLaser (Min Grad) @ ({laser_x:.1f}, {laser_y:.1f})"
        ax1.set_title(title_ax1, fontsize=9)
        ax1.set_xlabel("X (mm)")
        ax1.set_ylabel("Y (mm)")
        ax1.set_xlim(-r_max - 1, r_max + 1)
        ax1.set_ylim(-r_max - 1, r_max + 1)
        ax1.legend(fontsize=7, loc='upper right')

        # --- Affichage Subplot 2 : Heatmap Magnitude du Gradient ---
        if grad_magnitude_masked is not None:
            # Utiliser une colormap différente, ex: 'viridis'
            contour2 = ax2.contourf(xi, yi, grad_magnitude_masked, levels=50, cmap="viridis")
            fig.colorbar(contour2, ax=ax2, label="Magnitude Gradient Temp. (°C/mm)", shrink=0.6)

            # Optionnel: Afficher les points des thermistances sur ax2
            ax2.scatter(x_all_points, y_all_points, color='white', marker='.', s=10, alpha=0.5)

            # Afficher la position du laser (Minimum Gradient) sur ax2
            if laser_pos_found:
                # Utiliser 'rx' (croix rouge) pour contraster avec viridis
                ax2.plot(laser_x, laser_y, 'rx', markersize=8, label='Min Gradient')

            ax2.set_aspect('equal')
            ax2.set_title("Gradient Température", fontsize=9)
            ax2.set_xlabel("X (mm)")
            ax2.set_ylabel("Y (mm)")
            ax2.set_xlim(-r_max - 1, r_max + 1)
            ax2.set_ylim(-r_max - 1, r_max + 1)
            if laser_pos_found:
                ax2.legend(fontsize=7, loc='upper right')
        else:
            ax2.set_title("Gradient non calculé")

        # --- Ajustement final de la mise en page ---
        fig.tight_layout(pad=2.0) # Espace entre les plots


    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        # --- MODIFIÉ : Taille figure pour deux plots ---
        fig = plt.figure(figsize=(12, 6)) # Plus large pour deux subplots
        # --- FIN MODIFICATION ---
        plt.ion() # Mode interactif
        fig.show()

        all_data = []
        # Headers CSV (inclut R_Virtuel et les nouvelles colonnes)
        base_headers = [name for name, _ in self.positions]
        extra_headers = ["T_ref", "timestamp", "temps_ecoule_s"]
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
                data = self.get_temperatures() # Récupère le dict complet

                if data: # Si des données valides sont retournées
                    # Nettoyer la console (optionnel)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0
                    # Afficher toutes les thermistances (réelles + virtuelle)
                    for name, temp in data.items():
                        display_name = name
                        if pd.notna(temp):
                            print(f"{display_name:<10} : {temp:6.2f} °C")
                            # Compter seulement les *vraies* thermistances valides
                            if name != "R_Virtuel" and name != "R25": # Exclure virtuelle et référence du compte
                                valid_temps_count += 1
                        else:
                            print(f"{display_name:<10} :   --   °C (NaN)")
                    # Compter les thermistances réelles attendues (hors R25 et virtuelle)
                    real_thermistor_count = len([p for p in self.positions if p[0] not in ["R_Virtuel", "R25"]])
                    print(f"({valid_temps_count}/{real_thermistor_count} thermistances réelles (hors R25) valides)")
                    print("=" * 60)

                    # Affichage des heatmaps (utilise le dict complet 'data')
                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events() # Important pour l'affichage interactif

                    # Préparer la ligne pour le CSV
                    ligne = []
                    # Utiliser R25 comme T_ref si dispo, sinon une valeur par défaut (ex: 25.0)
                    t_ref_value = data.get("R25", 25.0)
                    if pd.isna(t_ref_value): t_ref_value = 25.0 # Fallback si R25 est NaN

                    for header_name in headers:
                        if header_name == "T_ref":
                            ligne.append(round(t_ref_value, 2))
                        elif header_name == "timestamp":
                            ligne.append(datetime.now().isoformat(timespec='seconds'))
                        elif header_name == "temps_ecoule_s":
                            ligne.append(round(elapsed_time, 3))
                        elif header_name in data:
                            temp_value = data[header_name]
                            # Arrondir les températures, gérer NaN
                            ligne.append(round(temp_value, 2) if pd.notna(temp_value) else '')
                        else:
                            # Si un header n'est pas dans data (ne devrait pas arriver avec R_Virtuel)
                            ligne.append('')
                    all_data.append(ligne)

                else:
                    # Si get_temperatures retourne None (erreur lecture série ou données incomplètes)
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues. Attente...")
                    print("=" * 60)
                    # Optionnel: Afficher une heatmap vide ou un message sur le graphe
                    # fig.clear()
                    # ax = fig.add_subplot(111)
                    # ax.text(0.5, 0.5, 'Données non disponibles', ha='center', va='center')
                    # fig.canvas.draw()
                    # fig.canvas.flush_events()

                # Pause pour respecter l'intervalle demandé
                time.sleep(max(0, interval - (time.time() - current_time)))


        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            # Fermer la figure matplotlib si elle existe encore
            if plt.fignum_exists(fig.number):
                plt.close(fig)

            # Sauvegarde CSV
            if all_data:
                print("💾 Sauvegarde du fichier CSV...")
                # Sauvegarder sur le bureau
                desktop_path = Path.home() / "Desktop"
                # Nom de fichier avec date et heure
                filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                csv_path = desktop_path / filename
                try:
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers) # Écrire les headers
                        writer.writerows(all_data) # Écrire toutes les données collectées
                    print(f"✅ Données sauvegardées dans : {csv_path}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")
            else:
                print("ℹ️ Aucune donnée collectée à sauvegarder.")


if __name__ == "__main__":
    # Mettre simulation=False pour utiliser l'Arduino
    # Mettre simulation=True pour utiliser le fichier CSV défini dans __init__
    td = TraitementDonnees(simulation=True)

    # Ajuster l'intervalle si nécessaire (en secondes)
    # Un intervalle trop court (< 0.1s) peut être difficile à tenir en temps réel
    # avec les calculs et l'affichage.
    td.demarrer_acquisition_live(interval=0.1) # Intervalle de 100ms

import serial
import numpy as np
# Supprimé Rbf car nous ne l'utilisons plus directement pour l'interpolation principale
# from scipy.interpolate import Rbf
import re
import time
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
import pandas as pd
from pathlib import Path
# Supprimé gaussian_filter si non utilisé ailleurs
# from scipy.ndimage import gaussian_filter
# NOUVELLE IMPORTATION pour l'ajustement de courbe (Levenberg-Marquardt)
from scipy.optimize import curve_fit
# Importer OptimizeWarning pour gérer les avertissements de curve_fit
from scipy.optimize import OptimizeWarning
import warnings # Pour filtrer les avertissements si nécessaire

# --- Définition de la fonction Gaussienne 2D ---
# 'xy_data' est un tableau (2, N) où N est le nombre de points.
# x = xy_data[0], y = xy_data[1]
def gaussian_2d(xy_data, A, x0, y0, sigma_x, sigma_y, k):
    x, y = xy_data
    # Assurer que sigma_x et sigma_y sont positifs pour éviter les erreurs mathématiques
    sigma_x = np.abs(sigma_x) + 2 # Ajout d'epsilon pour éviter la division par zéro
    sigma_y = np.abs(sigma_y) + 2
    exponent = -(((x - x0)**2 / (2 * sigma_x**2)) + ((y - y0)**2 / (2 * sigma_y**2)))
    return A * np.exp(exponent) + k
# ---------------------------------------------

class TraitementDonnees:
    VREF = 3.003
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        try:
            self.coefficients = np.load(coeffs_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"[ERREUR] Fichier coefficients non trouvé: {coeffs_path}")
            # Gérer l'erreur comme nécessaire, par exemple:
            self.coefficients = None # Ou lever une exception

        # 🔁 R24 à l’ancienne position de R24 (canal 11), R12 supprimée
        self.positions = [
            ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
            ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
            ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R24", (-4, -11.25)), # Position R24
            ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
            ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
            ("R21", (4.5, -8))
        ]
        # Canaux 0 à 20 utilisés pour les thermistances R1-R11, R13-R21, R24(sur canal 11)
        self.indices_à_garder = list(range(21)) # 0 à 20 inclus
        self.simulation_data = None
        self.simulation_index = 0
        self.simulation_columns = [self.positions[i][0] for i in self.indices_à_garder]

        # --- Mapping Nom -> Index dans self.positions (utile pour les coefficients) ---
        # Note: Cet index correspond à l'index dans la liste self.positions originale (0-20)
        self.name_to_position_index = {name: idx for idx, (name, pos) in enumerate(self.positions)}
       
        self.name_to_coeffs_index = {}
        for name, pos_idx in self.name_to_position_index.items():
            if name == "R24":
                self.name_to_coeffs_index[name] = 23
            elif name == "R19":
                self.name_to_coeffs_index[name] = 18
            elif name == "R20":
                self.name_to_coeffs_index[name] = 19
            elif name == "R16":
                self.name_to_coeffs_index[name] = 15
            else:
                # Assurez-vous que l'index de position est valide pour les coefficients
                # Si R12 était présent avant, les indices pourraient être décalés.
                # Ici, on suppose que l'index de position (0-10, 12-20) correspond
                # directement aux indices de coefficients (0-10, 12-20).
                # Vérifiez la structure de votre fichier coefficients.npy !
                self.name_to_coeffs_index[name] = pos_idx # Hypothèse à vérifier

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                script_dir = Path(__file__).parent
                simulation_file_path = script_dir.parent / "data" / "Hauteur 3.csv"
                self.simulation_data = pd.read_csv(simulation_file_path)
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
        # Vérifie aussi si les coefficients sont chargés
        coeffs_loaded = self.coefficients is not None
        if not coeffs_loaded:
            print("[ERREUR] Coefficients non chargés.")
        # En simulation, on considère connecté si les données sont chargées
        if self.simulation:
            sim_data_ok = self.simulation_data is not None
            return sim_data_ok and coeffs_loaded
        # En mode réel, on vérifie la connexion série
        else:
            serial_ok = self.ser is not None
            return serial_ok and coeffs_loaded


    def steinhart_hart_temperature(self, R, A, B, C):
        # Ajout de gestion pour R <= 0
        if R <= 0:
            # print(f"[AVERTISSEMENT] Résistance non positive ({R}), impossible de calculer log.")
            return np.nan # Retourne NaN si la résistance n'est pas valide
        log_R = np.log(R)
        # Vérifier si A, B, C sont valides (non-NaN, finis)
        if not all(np.isfinite([A, B, C])):
            # print(f"[AVERTISSEMENT] Coefficients Steinhart-Hart invalides: A={A}, B={B}, C={C}")
            return np.nan
        # Calculer l'inverse de la température en Kelvin
        inv_T = A + B * log_R + C * (log_R**3)
        # Éviter la division par zéro ou des valeurs très proches de zéro
        if abs(inv_T) < 1e-9:
            # print(f"[AVERTISSEMENT] Dénominateur proche de zéro dans Steinhart-Hart (inv_T={inv_T}).")
            return np.nan # Ou une autre valeur indiquant une erreur/impossibilité
        return 1.0 / inv_T # Température en Kelvin


    def compute_resistance(self, voltage):
        # Gérer les tensions hors limites ou invalides
        if not np.isfinite(voltage) or voltage < 0:
            # print(f"[AVERTISSEMENT] Tension invalide reçue: {voltage}")
            return np.nan # Retourne NaN pour tension invalide
        if voltage >= self.VREF:
            # print(f"[AVERTISSEMENT] Tension ({voltage}V) >= VREF ({self.VREF}V), résistance infinie.")
            return float('inf') # Résistance infinie
        if voltage == 0:
             # Éviter la division par zéro si V=0 (résistance nulle)
             return 0.0
        denominator = self.VREF - voltage
        # Vérifier si le dénominateur est trop proche de zéro (arrive si voltage est très proche de VREF)
        if abs(denominator) < 1e-9:
            # print(f"[AVERTISSEMENT] Dénominateur proche de zéro dans compute_resistance (V={voltage}, VREF={self.VREF}).")
            return float('inf') # Considérer comme résistance infinie
        return self.R_FIXED * (voltage / denominator)


    def compute_temperature(self, resistance, coeffs):
        # Gérer résistance infinie ou NaN avant d'appeler Steinhart-Hart
        if not np.isfinite(resistance) or resistance == float('inf'):
            # print(f"[AVERTISSEMENT] Résistance infinie ou NaN ({resistance}), température non calculable.")
            return np.nan
        if resistance < 0:
            # print(f"[AVERTISSEMENT] Résistance négative ({resistance}), température non calculable.")
            return np.nan

        # Vérifier si les coefficients sont valides
        if coeffs is None or len(coeffs) != 3 or not all(isinstance(c, (int, float)) for c in coeffs):
             print(f"[ERREUR] Coefficients invalides fournis: {coeffs}")
             return np.nan
        A, B, C = coeffs

        try:
            kelvin = self.steinhart_hart_temperature(resistance, A, B, C)
            # Vérifier si le résultat de Steinhart-Hart est valide
            if not np.isfinite(kelvin) or kelvin <= 0:
                # print(f"[AVERTISSEMENT] Température Kelvin invalide ({kelvin}) calculée pour R={resistance}.")
                return np.nan
            celsius = kelvin - 273.15
            # Optionnel: Ajouter une vérification de plage raisonnable pour Celsius
            # if not (-50 < celsius < 200): # Exemple de plage
            #     print(f"[AVERTISSEMENT] Température Celsius hors plage raisonnable: {celsius}°C")
            #     return np.nan
            return celsius
        except (ValueError, OverflowError, TypeError) as e:
            # Capturer les erreurs potentielles (ex: log de nombre négatif si non géré avant)
            print(f"[ERREUR] Erreur mathématique dans compute_temperature pour R={resistance}: {e}")
            return np.nan


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

        # --- Utilisation d'une map pour suivre les canaux reçus ---
        canaux_recus = {canal: False for canal in self.indices_à_garder}
        canaux_attendus = len(self.indices_à_garder)
        canaux_comptes = 0

        while True:
            current_time = time.time()
            if current_time - start_time > timeout_sec:
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s).")
                break # Sortir si timeout

            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line:
                        continue

                    # print(f"Ligne reçue: {line}") # Debug

                    if "Fin du balayage" in line:
                        # print("Fin du balayage détectée.") # Debug
                        break # Sortir si fin de balayage

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        try:
                            canal = int(match.group(1))
                            voltage = float(match.group(2))
                            # Vérifier si le canal est attendu et non déjà reçu
                            if canal in canaux_recus and not canaux_recus[canal]:
                                voltages_dict[canal] = voltage
                                canaux_recus[canal] = True
                                canaux_comptes += 1
                                # print(f"Canal {canal} lu: {voltage} V ({canaux_comptes}/{canaux_attendus})") # Debug
                                # Sortir tôt si tous les canaux attendus sont reçus
                                if canaux_comptes == canaux_attendus:
                                    # print("Tous les canaux attendus reçus.") # Debug
                                    # Lire jusqu'à "Fin du balayage" pour vider le buffer ? Ou sortir ?
                                    # Pour l'instant, on sort directement.
                                    break
                        except ValueError:
                            print(f"[AVERTISSEMENT] Impossible de convertir les données série: {match.groups()}")
                            continue # Ignorer cette ligne

                else:
                    time.sleep(0.01) # Petite pause

            except serial.SerialException as e:
                print(f"Erreur série pendant la lecture : {e}")
                self.ser.close() # Fermer le port en cas d'erreur
                self.ser = None
                return None
            except Exception as e:
                print(f"Erreur inattendue pendant la lecture série : {e}")
                # Peut-être préférable de retourner None ici aussi
                return None

        # --- Vérification finale ---
        canaux_manquants = [c for c, recu in canaux_recus.items() if not recu]
        if canaux_manquants:
            print(f"⚠️ Données incomplètes. Canaux manquants: {canaux_manquants}")
            return None # Retourner None si incomplet

        # print("Données complètes reçues.") # Debug
        return voltages_dict


    def get_temperatures(self):
        # Vérifier si les coefficients sont chargés
        if self.coefficients is None:
             print("[ERREUR CRITIQUE] Coefficients non chargés. Impossible de calculer les températures.")
             return None

        if self.simulation:
            # --- Logique Simulation CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                # Vérifier si l'index dépasse la taille avant de lire
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0 # Retourner au début
                    print("[SIMULATION] Fin du fichier CSV atteinte, retour au début.")
                    # S'assurer que l'index 0 est valide après réinitialisation
                    if self.simulation_index >= len(self.simulation_data):
                         print("[ERREUR SIMULATION] Fichier CSV vide ou index invalide après réinitialisation.")
                         return {self.positions[i][0]: np.nan for i in self.indices_à_garder}


                current_data_row = self.simulation_data.iloc[self.simulation_index]
                # --- MODIFICATION ICI: Incrémenter de 10 ---
                previous_index = self.simulation_index # Garder l'index lu pour le message d'erreur
                self.simulation_index += 10 # Lire toutes les 10 lignes

                temperature_dict = {}
                temps_for_r24_average_sim = []
                valid_data_found = False

                # 1. Lecture initiale depuis CSV
                for pos_idx in self.indices_à_garder: # Utiliser l'index de position
                    thermistor_name = self.positions[pos_idx][0]
                    if thermistor_name in current_data_row and pd.notna(current_data_row[thermistor_name]):
                        temp_value = current_data_row[thermistor_name]
                        temperature_dict[thermistor_name] = temp_value
                        # Exclure R24 du calcul de sa propre moyenne
                        if thermistor_name != "R24":
                            temps_for_r24_average_sim.append(temp_value)
                        valid_data_found = True
                    else:
                        # Mettre NaN si la donnée est manquante ou invalide dans le CSV
                        temperature_dict[thermistor_name] = np.nan
                        # print(f"[AVERTISSEMENT SIMULATION] Donnée manquante/NaN pour {thermistor_name} à l'index {previous_index}")


                if not valid_data_found:
                    # --- MODIFICATION ICI: Message d'erreur ajusté ---
                    print(f"[ERREUR SIMULATION] Aucune donnée valide trouvée à l'index CSV {previous_index}.")
                    # Retourner un dictionnaire avec des NaN pour éviter les erreurs en aval
                    return {self.positions[i][0]: np.nan for i in self.indices_à_garder}


                # 2. Calcul et assignation de la moyenne pour R24 (si R24 existe dans le dict)
                if "R24" in temperature_dict:
                    if temps_for_r24_average_sim: # S'il y a des températures valides pour calculer la moyenne
                        average_temp_sim = np.mean(temps_for_r24_average_sim)
                        temperature_dict["R24"] = average_temp_sim
                    else:
                        # Si R24 est présent mais aucune autre temp valide, mettre R24 à NaN
                        temperature_dict["R24"] = np.nan
                # Si R24 n'était pas dans les colonnes CSV, il aura déjà NaN

                # --- Logique de Boost R24 (Simulation) ---
                # (Le reste de la logique de boost reste inchangé)
                r19_temp_sim = temperature_dict.get("R19", np.nan)
                r20_temp_sim = temperature_dict.get("R20", np.nan)
                r16_temp_sim = temperature_dict.get("R16", np.nan)
                r24_temp_sim = temperature_dict.get("R24", np.nan) # Utilise la valeur moyennée si calculée

                temps_for_check_sim = [
                    temp for name, temp in temperature_dict.items()
                    if pd.notna(temp) and name not in ["R19", "R20", "R24"]
                ]

                if temps_for_check_sim:
                    average_for_check_sim = np.mean(temps_for_check_sim)

                    # Condition 1 (Boost 1.08)
                    condition_1_met_sim = (
                        pd.notna(r19_temp_sim) and pd.notna(r20_temp_sim) and
                        r19_temp_sim > (average_for_check_sim * 1.15) and
                        r20_temp_sim > (average_for_check_sim * 1.15)
                    )
                    if condition_1_met_sim and pd.notna(r24_temp_sim):
                        boosted_r24_temp_sim = r24_temp_sim * 1.08
                        temperature_dict["R24"] = boosted_r24_temp_sim
                        r24_temp_sim = boosted_r24_temp_sim

                    # Condition 2 (Boost 1.20)
                    condition_2_met_sim = (
                        pd.notna(r19_temp_sim) and pd.notna(r16_temp_sim) and
                        r19_temp_sim >= (average_for_check_sim * 1.20) and
                        r16_temp_sim <= (average_for_check_sim * 1.18)
                    )
                    if condition_2_met_sim and pd.notna(r24_temp_sim):
                        boosted_r24_temp_2_sim = r24_temp_sim * 1.20
                        temperature_dict["R24"] = boosted_r24_temp_2_sim

                # Retourner le dictionnaire final pour la simulation
                return temperature_dict

            else:
                # Fallback si CSV non chargé
                print("[AVERTISSEMENT SIMULATION] Données CSV non disponibles, retour de NaN.")
                return {self.positions[i][0]: np.nan for i in self.indices_à_garder}

        # --- Logique pour le mode non-simulation (lecture série) ---
        # (Le reste de la fonction pour le mode non-simulation reste inchangé)
        data_voltages = self.lire_donnees()
        if data_voltages is None:
            print("[AVERTISSEMENT] Aucune donnée de tension complète reçue du port série.")
            return {self.positions[i][0]: np.nan for i in self.indices_à_garder}

        temperatures_raw = {}
        temps_for_r24_average_real = []

        # 1. Calcul initial des températures
        for canal, voltage in data_voltages.items():
            nom_thermistor = None
            pos_idx_thermistor = -1
            for idx in self.indices_à_garder:
                if idx == canal:
                    nom_thermistor = self.positions[idx][0]
                    pos_idx_thermistor = idx
                    break

            if nom_thermistor is None:
                 print(f"[ERREUR] Impossible de trouver le nom pour le canal {canal}.")
                 continue

            coeffs_index = self.name_to_coeffs_index.get(nom_thermistor)

            if coeffs_index is None or coeffs_index >= len(self.coefficients):
                print(f"[ERREUR] Index de coefficients ({coeffs_index}) invalide ou hors limites pour {nom_thermistor} (canal {canal}).")
                temperatures_raw[nom_thermistor] = np.nan
                continue

            coeffs = self.coefficients[coeffs_index]
            if coeffs is None:
                 print(f"[ERREUR] Coefficients non trouvés pour {nom_thermistor} (index {coeffs_index}).")
                 temperatures_raw[nom_thermistor] = np.nan
                 continue

            resistance = self.compute_resistance(voltage)
            temp = self.compute_temperature(resistance, coeffs)
            temperatures_raw[nom_thermistor] = temp

            if pd.notna(temp) and nom_thermistor != "R24":
                temps_for_r24_average_real.append(temp)

        final_temperatures = {self.positions[i][0]: np.nan for i in self.indices_à_garder}
        final_temperatures.update(temperatures_raw)

        # 2. Calcul et assignation de la moyenne pour R24
        if "R24" in final_temperatures:
            if temps_for_r24_average_real:
                average_temp_real = np.mean(temps_for_r24_average_real)
                final_temperatures["R24"] = average_temp_real
            else:
                final_temperatures["R24"] = np.nan

        # --- LOGIQUE POUR BOOSTER R24 (NON-SIMULATION) ---
        r19_temp_real = final_temperatures.get("R19", np.nan)
        r20_temp_real = final_temperatures.get("R20", np.nan)
        r16_temp_real = final_temperatures.get("R16", np.nan)
        r24_temp_real = final_temperatures.get("R24", np.nan)

        temps_for_check_real = [
            temp for name, temp in final_temperatures.items()
            if pd.notna(temp) and name not in ["R19", "R20", "R24"]
        ]

        if temps_for_check_real:
            average_for_check_real = np.mean(temps_for_check_real)

            # Condition 1 (Boost 1.08)
            condition_1_met_real = (
                pd.notna(r19_temp_real) and pd.notna(r20_temp_real) and
                r19_temp_real > (average_for_check_real * 1.15) and
                r20_temp_real > (average_for_check_real * 1.15)
            )
            if condition_1_met_real and pd.notna(r24_temp_real):
                boosted_r24_temp = r24_temp_real * 1.08
                final_temperatures["R24"] = boosted_r24_temp
                r24_temp_real = boosted_r24_temp

            # Condition 2 (Boost 1.20)
            condition_2_met_real = (
                pd.notna(r19_temp_real) and pd.notna(r16_temp_real) and
                r19_temp_real >= (average_for_check_real * 1.20) and
                r16_temp_real <= (average_for_check_real * 1.18)
            )
            if condition_2_met_real and pd.notna(r24_temp_real):
                boosted_r24_temp_2 = r24_temp_real * 1.20
                final_temperatures["R24"] = boosted_r24_temp_2

        # Retourner le dictionnaire final
        return final_temperatures
   


    
    # --- MODIFICATION MAJEURE ICI ---
    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        ax = fig.add_subplot(111)
        r_max = 12.5 # Rayon max pour l'affichage et le masque

        # --- 1. Préparation des données valides ---
        x_orig, y_orig, t_orig = [], [], []
        valid_temps_list = []
        names_orig = [] # Garder les noms pour l'annotation

        for i in self.indices_à_garder:
            name, pos = self.positions[i]
            temp_val = temperature_dict.get(name, np.nan)
            if pd.notna(temp_val):
                x_orig.append(pos[0])
                y_orig.append(pos[1])
                t_orig.append(temp_val)
                valid_temps_list.append(temp_val)
                names_orig.append(name)

        # Convertir en arrays numpy pour curve_fit et calculs
        x_orig = np.array(x_orig)
        y_orig = np.array(y_orig)
        t_orig = np.array(t_orig)

        # --- 2. Vérifier s'il y a assez de données pour l'ajustement ---
        if len(t_orig) < 6:
            print(f"[ERREUR HEATMAP] Pas assez de points valides ({len(t_orig)} < 6) pour l'ajustement Gaussien.")
            ax.set_title("Pas assez de données pour l'ajustement")
            if len(t_orig) > 0:
                ax.scatter(x_orig, y_orig, c=t_orig, cmap="plasma", marker='o', s=35, label='Thermistances (Données insuffisantes)')
                for i, name in enumerate(names_orig):
                     ax.annotate(name, (x_orig[i], y_orig[i]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)
            ax.set_xlim(-r_max - 1, r_max + 1)
            ax.set_ylim(-r_max - 1, r_max + 1)
            ax.set_aspect('equal')
            if len(t_orig) > 0: # Afficher la légende seulement si des points sont affichés
                ax.legend(fontsize=8)
            fig.tight_layout()
            # --- MODIFICATION ICI: Retourner (None, False) ---
            return None, False # <--- MODIFIÉ: Assure un retour de tuple

        # --- 3. Estimations initiales (p0) pour curve_fit ---
        # (Le reste de cette section est inchangé...)
        k0 = np.min(t_orig) if len(valid_temps_list) > 0 else 20.0
        max_temp = np.max(t_orig)
        A0 = max_temp - k0 if max_temp > k0 else 10.0
        idx_max = np.argmax(t_orig)
        x0_guess = x_orig[idx_max]
        y0_guess = y_orig[idx_max]
        sigma_guess = r_max / 4.0
        p0 = [A0, x0_guess, y0_guess, sigma_guess, sigma_guess, k0]

        # --- 4. Préparer les données pour curve_fit ---
        xy_data_orig = np.vstack((x_orig.ravel(), y_orig.ravel()))

        # --- 5. Ajustement avec Levenberg-Marquardt (curve_fit) ---
        popt = None
        pcov = None
        fit_successful = False
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                bounds = ([-np.inf, -r_max, -r_max, 1e-6, 1e-6, -np.inf], # Bornes inf (sigmas > 0 + epsilon)
                          [np.inf, r_max, r_max, r_max*2, r_max*2, np.inf]) # Bornes sup
                # --- MODIFICATION ICI: Utiliser method='trf' avec les bornes ---
                popt, pcov = curve_fit(gaussian_2d, xy_data_orig, t_orig.ravel(), p0=p0, method='trf', maxfev=5000, bounds=bounds) # <--- MODIFIÉ
            fit_successful = True
            # print(f"Paramètres ajustés (A, x0, y0, sx, sy, k): {popt}")
        except RuntimeError as e:
            print(f"[AVERTISSEMENT HEATMAP] Échec de l'ajustement initial ('trf' avec bornes): {e}")
            # Tentative SANS bornes avec 'lm' (qui est fait pour ça)
            try:
                 with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    # --- Utiliser 'lm' SANS bornes ---
                    popt, pcov = curve_fit(gaussian_2d, xy_data_orig, t_orig.ravel(), p0=p0, method='lm', maxfev=5000) # <--- 'lm' sans bornes
                 fit_successful = True
                 print("[INFO HEATMAP] Ajustement réussi sans bornes ('lm') après échec initial.")
            except RuntimeError as e2:
                 print(f"[ERREUR HEATMAP] Échec de l'ajustement LM même sans bornes: {e2}")
                 fit_successful = False
            except ValueError as e_val: # Capturer aussi ValueError ici
                 print(f"[ERREUR HEATMAP] Erreur de valeur pendant l'ajustement sans bornes ('lm'): {e_val}")
                 fit_successful = False
        except ValueError as e:
             # Cette erreur peut aussi survenir avec 'trf' si p0 est hors bornes ou données invalides
             print(f"[ERREUR HEATMAP] Erreur de valeur pendant l'ajustement initial ('trf'): {e}")
             fit_successful = False

        # --- 6. Génération de la grille et calcul de la heatmap ajustée ---
        # (Inchangé...)
        grid_size = 100
        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )
        xy_data_grid = np.vstack((xi.ravel(), yi.ravel()))

        if fit_successful and popt is not None:
            ti_fit = gaussian_2d(xy_data_grid, *popt)
            ti_reshaped = ti_fit.reshape(grid_size, grid_size)
            A_fit, x0_fit, y0_fit, sx_fit, sy_fit, k_fit = popt
            # S'assurer que les sigmas sont positifs pour l'affichage (ils devraient l'être à cause des bornes/abs dans gaussian_2d)
            sx_fit, sy_fit = np.abs(sx_fit), np.abs(sy_fit)
            title_suffix = f"\nFit: A={A_fit:.1f}, Ctr=({x0_fit:.1f},{y0_fit:.1f}), Sig=({sx_fit:.1f},{sy_fit:.1f}), k={k_fit:.1f}"
            laser_pos_label = f'Centre Gaussien @ ({x0_fit:.1f}, {y0_fit:.1f})'
            laser_x, laser_y = x0_fit, y0_fit
            laser_pos_found = True
        else:
            print("[AVERTISSEMENT HEATMAP] Affichage basé sur l'estimation initiale car l'ajustement a échoué.")
            # Utiliser p0 pour l'affichage si l'ajustement a échoué
            ti_initial_guess = gaussian_2d(xy_data_grid, *p0)
            ti_reshaped = ti_initial_guess.reshape(grid_size, grid_size)
            title_suffix = "\n(Échec de l'ajustement Gaussien)"
            laser_pos_label = 'Centre (Échec Fit)'
            laser_x, laser_y = None, None
            laser_pos_found = False
            # Utiliser les valeurs de p0 pour vmin/vmax si le fit échoue
            A0_disp, _, _, _, _, k0_disp = p0
            vmin_fail = k0_disp - 1
            vmax_fail = k0_disp + A0_disp + 1


        # Masque pour la zone circulaire
        mask = xi**2 + yi**2 > r_max**2
        ti_masked = np.ma.array(ti_reshaped, mask=mask)

        # --- 7. Affichage ---
        # Définir vmin/vmax basé sur les données ajustées OU initiales
        if fit_successful and popt is not None:
            vmin = np.min(ti_masked) if ti_masked.count() > 0 else k_fit - 1
            vmax = np.max(ti_masked) if ti_masked.count() > 0 else k_fit + A_fit + 1
        else:
            # Utiliser les valeurs calculées à partir de p0 si le fit a échoué
            vmin = vmin_fail
            vmax = vmax_fail

        # Assurer vmin < vmax
        if vmin >= vmax:
            vmin = vmax - 1 # Ajustement simple pour éviter l'erreur dans contourf

        levels = np.linspace(vmin, vmax, 101)

        contour = ax.contourf(xi, yi, ti_masked, levels=levels, cmap="plasma", vmin=vmin, vmax=vmax, extend='both') # extend='both' gère les valeurs hors limites
        try: # Ajout d'un try/except pour la colorbar qui peut échouer si vmin=vmax
            fig.colorbar(contour, ax=ax, label="Température Estimée (°C)")
        except ValueError as cbar_err:
            print(f"[AVERTISSEMENT HEATMAP] Impossible d'afficher la colorbar: {cbar_err}")

        ax.scatter(x_orig, y_orig, color='black', marker='o', s=25, label='Thermistances')

        # (Annotations et affichage du centre inchangés...)
        for i, name in enumerate(names_orig):
            ax.annotate(name, (x_orig[i], y_orig[i]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)

        if laser_pos_found and laser_x is not None and laser_y is not None:
             if -r_max <= laser_x <= r_max and -r_max <= laser_y <= r_max:
                ax.plot(laser_x, laser_y, 'go', markersize=10, label=laser_pos_label)
             else:
                print(f"[AVERTISSEMENT HEATMAP] Centre gaussien ({laser_x:.1f}, {laser_y:.1f}) hors limites d'affichage.")


        # Configuration de l'axe
        ax.set_aspect('equal')
        title_ax = f"Heatmap Gaussienne 2D (Tps: {elapsed_time:.2f} s)" + title_suffix
        ax.set_title(title_ax, fontsize=9)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max - 1, r_max + 1)
        ax.set_ylim(-r_max - 1, r_max + 1)
        ax.legend(fontsize=8)

        fig.tight_layout()

        # --- 8. Retourner les paramètres et le succès ---
        # --- MODIFICATION ICI: Retourner (None, False) si échec ---
        if fit_successful and popt is not None:
            return popt, True
        else:
            return None, False # <--- MODIFIÉ: Assure un retour de tuple même si fit échoue
    


    def demarrer_acquisition_live(self, interval=0.2):
        # --- Vérification initiale améliorée ---
        if not self.est_connecte():
             # est_connecte gère maintenant la simulation et les coefficients
             print("❌ Problème de connexion ou de configuration (Arduino/Coefficients/Données Simu). Arrêt.")
             return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        fig = plt.figure(figsize=(7, 6))
        plt.ion()
        # --- Gestion d'erreur pour fig.show() ---
        try:
            fig.show()
            # Petite pause pour laisser la fenêtre apparaître
            plt.pause(0.1)
            if not plt.fignum_exists(fig.number):
                 print("❌ Impossible d'afficher la fenêtre Matplotlib. Vérifiez votre backend graphique.")
                 plt.ioff() # Désactiver le mode interactif
                 return
        except Exception as e:
            print(f"❌ Erreur lors de l'affichage de la fenêtre Matplotlib: {e}")
            plt.ioff()
            return


        all_data = []
        # Les headers dépendent des thermistances réellement utilisées (indices_à_garder)
        headers = [self.positions[i][0] for i in self.indices_à_garder] + ["T_ref", "timestamp", "temps_ecoule_s"]
        # Ajouter les paramètres gaussiens au CSV si l'ajustement réussit
        gaussian_headers = ["Gauss_A", "Gauss_x0", "Gauss_y0", "Gauss_sx", "Gauss_sy", "Gauss_k", "Fit_Success"]
        full_headers = headers + gaussian_headers


        start_time = time.time()
        keep_running = True
        try:
            while keep_running and plt.fignum_exists(fig.number):
                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures() # Récupère le dict {nom: temp}

                # --- Initialiser les données gaussiennes pour le CSV ---
                gaussian_data_row = {hdr: '' for hdr in gaussian_headers} # Valeurs par défaut vides

                if data:
                    # --- Vérification fenêtre avant affichage ---
                    if not plt.fignum_exists(fig.number):
                        keep_running = False
                        break

                    # --- Nettoyage et affichage console ---
                    os.system('cls' if os.name == 'nt' else 'clear') # Multi-plateforme
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées (°C)")
                    print("-" * 60)
                    valid_temps_count = 0
                    # Trier par nom pour un affichage cohérent? Optionnel.
                    # sorted_data = dict(sorted(data.items()))
                    for name, temp in data.items(): # Utiliser data directement
                        if pd.notna(temp):
                            print(f"{name:<6} : {temp:6.2f}")
                            valid_temps_count += 1
                        else:
                            print(f"{name:<6} :   --   (NaN)")
                    print(f"({valid_temps_count}/{len(self.indices_à_garder)} thermistances valides)")
                    print("=" * 60)

                    # --- Affichage de la heatmap (qui fait maintenant l'ajustement) ---
                    # La fonction `afficher_heatmap_dans_figure` a été modifiée pour faire le fit
                    # et renvoyer les paramètres si succès. Modifions-la pour qu'elle renvoie les params.

                    # --- Modification nécessaire: `afficher_heatmap_dans_figure` doit renvoyer les params ---
                    # Pour l'instant, on suppose qu'elle ne le fait pas et on recalcule ici (pas idéal)
                    # ou on modifie `afficher_heatmap_dans_figure` (préférable).

                    # --- Option 1: Recalculer ici (redondant) ---
                    # fit_params, fit_success_flag = self.calculer_fit_gaussien(data) # Fonction à créer

                    # --- Option 2: Modifier `afficher_heatmap_dans_figure` pour renvoyer les params ---
                    # C'est ce qu'on va supposer pour la sauvegarde CSV.
                    # On va AJOUTER un retour à `afficher_heatmap_dans_figure`

                    # Appel modifié (supposant que la fonction renvoie les params et un flag de succès)
                    # Note: Il faut modifier `afficher_heatmap_dans_figure` pour ajouter ce `return`
                    # Voir modification suggérée à la fin de la fonction `afficher_heatmap_dans_figure`
                    fit_params, fit_success_flag = self.afficher_heatmap_dans_figure(data, fig, elapsed_time)

                    if fit_success_flag and fit_params is not None:
                        gaussian_data_row["Gauss_A"] = round(fit_params[0], 3)
                        gaussian_data_row["Gauss_x0"] = round(fit_params[1], 3)
                        gaussian_data_row["Gauss_y0"] = round(fit_params[2], 3)
                        gaussian_data_row["Gauss_sx"] = round(fit_params[3], 3)
                        gaussian_data_row["Gauss_sy"] = round(fit_params[4], 3)
                        gaussian_data_row["Gauss_k"] = round(fit_params[5], 3)
                        gaussian_data_row["Fit_Success"] = 1 # Succès
                        print("Ajustement Gaussien réussi.")
                    else:
                        gaussian_data_row["Fit_Success"] = 0 # Échec
                        print("Échec de l'ajustement Gaussien.")
                    print("=" * 60)


                    # --- Mise à jour de la figure ---
                    try:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    except Exception as e:
                         print(f"⚠️ Erreur lors de la mise à jour de la figure: {e}")
                         # Peut arriver si la fenêtre est fermée très rapidement
                         keep_running = False
                         break


                    # --- Préparer la ligne pour le CSV ---
                    ligne_dict = {}
                    # Ajouter les températures
                    for header_name in headers: # Headers sans les params gaussiens
                        if header_name == "T_ref":
                            # Calculer T_ref comme la moyenne des valides? Ou utiliser 'k' du fit?
                            # Utilisons 'k' si le fit a réussi, sinon une valeur fixe ou la moyenne
                            t_ref_val = gaussian_data_row["Gauss_k"] if fit_success_flag else 25.0 # Ou np.mean(valid_temps)
                            ligne_dict[header_name] = t_ref_val if pd.notna(t_ref_val) else ''
                        elif header_name == "timestamp":
                            ligne_dict[header_name] = datetime.now().isoformat(timespec='milliseconds') # Plus de précision
                        elif header_name == "temps_ecoule_s":
                            ligne_dict[header_name] = round(elapsed_time, 3)
                        elif header_name in data:
                            temp_value = data[header_name]
                            # Remplacer NaN par chaîne vide pour CSV
                            ligne_dict[header_name] = temp_value if pd.notna(temp_value) else ''
                        else:
                            ligne_dict[header_name] = '' # Cas où un header n'est pas dans data (ne devrait pas arriver)

                    # Ajouter les données gaussiennes
                    ligne_dict.update(gaussian_data_row)

                    # Convertir le dictionnaire en liste dans l'ordre de full_headers
                    ligne_csv = [ligne_dict.get(h, '') for h in full_headers]
                    all_data.append(ligne_csv)

                else:
                    # Affichage si données incomplètes
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues cette itération.")
                    print("=" * 60)
                    # Optionnel: Afficher une heatmap vide ou précédente?
                    # Pour l'instant, on ne met pas à jour la heatmap si pas de données.

                # --- Vérification fenêtre après pause ---
                # Mettre la pause *avant* la vérification pour laisser le temps de fermer
                time.sleep(interval)
                if not plt.fignum_exists(fig.number):
                    keep_running = False
                    # print("Fenêtre fermée détectée après pause.") # Debug
                    break

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        except Exception as e:
             # Capturer d'autres erreurs potentielles dans la boucle principale
             print(f"\n❌ Erreur inattendue pendant l'acquisition: {e}")
             import traceback
             traceback.print_exc() # Imprimer la trace pour le débogage
             keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            plt.ioff() # Désactiver le mode interactif
            if plt.fignum_exists(fig.number):
                plt.close(fig)
                print("Fenêtre Matplotlib fermée.")

            if all_data:
                print("💾 Sauvegarde du fichier CSV...")
                # Utiliser un sous-dossier 'output' relatif au script?
                script_dir = Path(__file__).parent
                output_dir = script_dir.parent / "output"
                output_dir.mkdir(parents=True, exist_ok=True) # Crée le dossier s'il n'existe pas

                # Nom de fichier plus informatif
                sim_mode = "_SIMULATION" if self.simulation else "_REAL"
                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"acquisition{sim_mode}_{timestamp_str}.csv"
                csv_path = output_dir / filename

                try:
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(full_headers) # Utiliser les headers complets
                        writer.writerows(all_data)
                    print(f"✅ Données sauvegardées dans : {csv_path.resolve()}")
                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")
            else:
                print("ℹ️ Aucune donnée collectée à sauvegarder.")


# --- AJOUTER CE RETURN à la fin de `afficher_heatmap_dans_figure` ---
# Juste avant la fin de la fonction, après fig.tight_layout():
#
#         # --- 8. Retourner les paramètres et le succès ---
#         if fit_successful and popt is not None:
#             return popt, True
#         else:
#             return p0, False # Retourner p0 ou None si échec? p0 est peut-être plus utile.
#
# --- FIN AJOUT ---


if __name__ == "__main__":
    # Choisir True pour simulation, False pour connexion série réelle
    MODE_SIMULATION = True

    # Créer l'instance
    td = TraitementDonnees(simulation=MODE_SIMULATION)

    # Démarrer l'acquisition seulement si l'initialisation a réussi
    # (connexion série ou chargement CSV/coeffs OK)
    if td.est_connecte() or MODE_SIMULATION: # En simu, on tente même si le CSV a échoué au début
         # Intervalle de rafraîchissement (secondes)
         # Attention: Un intervalle trop court (< temps de lecture série/calcul)
         # peut causer des problèmes. 0.05s est très rapide.
         # Augmenter à 0.2s ou 0.5s pourrait être plus stable.
         REFRESH_INTERVAL = 0.2
         td.demarrer_acquisition_live(interval=REFRESH_INTERVAL)
    else:
         print("❌ Initialisation échouée. Impossible de démarrer l'acquisition.")


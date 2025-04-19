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
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        # Décalage à appliquer
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas

        # 🔁 R24 à l’ancienne position de R24 (canal 11), R12 supprimée
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
    traitement = TraitementDonnees(simulation=True)
    traitement.demarrer_acquisition_live(interval=1.0, utiliser_bords=True)
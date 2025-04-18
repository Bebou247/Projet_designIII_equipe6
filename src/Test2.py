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

class TraitementDonnees:
    VREF = 3.003
    R_FIXED = 4700

        # Dans la classe TraitementDonnees
    def __init__(self, port="/dev/cu.usbmodem14201", coeffs_path="data/raw/coefficients.npy", simulation=False):
        self.port = port
        self.simulation = simulation
        self.coefficients = np.load(coeffs_path, allow_pickle=True)

        # 🔁 R24 à l’ancienne position de R24 (canal 11), R12 supprimée
        self.positions = [
            ("R1", (11, 0)), ("R2", (3, 0)), ("R3", (-3, 0)), ("R4", (-11, 0)),
            ("R5", (8, 2.5)), ("R6", (0, 2.5)), ("R7", (-8, 2.5)), ("R8", (8, 5.5)),
            ("R9", (0, 5.5)), ("R10", (-8, 5.5)), ("R11", (4.5, 8)), ("R24", (-3.5, -11.25)),
            ("R13", (4, 11.25)), ("R14", (-4, 11.25)), ("R15", (8, -2.5)), ("R16", (0, -2.5)),
            ("R17", (-8, -2.5)), ("R18", (8, -5.5)), ("R19", (0, -5.5)), ("R20", (-8, -5.5)),
            ("R21", (4.5, -8))
        ]
        # J'ai changé tempo la pos de la R24 (-3.5, -11.25) Canaux 0 à 20 utilisés pour les thermistances R1-R11, R13-R21, R24(sur canal 11)
        self.indices_à_garder = list(range(21))
        self.simulation_data = None
        self.simulation_index =  0
        # Te permet de décider à quel rang tu commences
        # Noms des colonnes attendues dans le CSV (basés sur self.positions et self.indices_à_garder)
        self.simulation_columns = [self.positions[i][0] for i in self.indices_à_garder]

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                # Chemin vers le fichier CSV relatif au script Test.py
                script_dir = Path(__file__).parent
                # Te permet de choisir quel fichier prendre
                simulation_file_path = script_dir.parent / "data" / "Hauteur 4.csv"
                # Lecture du CSV, essayez différents séparateurs si nécessaire (ex: sep=';')
                self.simulation_data = pd.read_csv(simulation_file_path) # Adaptez le séparateur si besoin: sep=';'
                print(f"[SIMULATION] Chargement du fichier CSV : {simulation_file_path.resolve()}")

                # Vérification des colonnes nécessaires
                missing_cols = [col for col in self.simulation_columns if col not in self.simulation_data.columns]
                if missing_cols:
                    print(f"[ERREUR SIMULATION] Colonnes manquantes dans {simulation_file_path.name}: {missing_cols}")
                    self.simulation_data = None # Invalider les données
                else:
                    # Conversion des colonnes requises en numérique, gère les erreurs
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
            # Logique originale pour la connexion série
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

    # Dans la classe TraitementDonnees
    def lire_donnees(self):
        if self.simulation:
            # En mode simulation CSV, cette fonction signale juste si les données sont prêtes.
            # get_temperatures lira directement depuis self.simulation_data.
            if self.simulation_data is not None and not self.simulation_data.empty:
                return True # Signal que les données de simulation CSV sont prêtes
            else:
                # Si le CSV n'a pas pu être chargé, on retourne None
                # print("[AVERTISSEMENT SIMULATION] Aucune donnée de simulation CSV disponible.")
                return None

        # --- Code original pour lire depuis le port série ---
        if self.ser is None:
            print("[ERREUR] Connexion série non établie.")
            return None

        self.ser.reset_input_buffer()
        voltages_dict = {}
        start_time = time.time()
        timeout_sec = 2 # Augmenté légèrement pour être sûr

        while True:
            current_time = time.time()
            if current_time - start_time > timeout_sec:
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                # Retourne ce qui a été lu jusqu'à présent ou None si rien
                return voltages_dict if voltages_dict else None

            try:
                # Vérifier s'il y a des données à lire pour éviter de bloquer sur readline()
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode(errors='ignore').strip()
                    if not line:
                        # Peut arriver si timeout court ou fin de ligne partielle
                        continue

                    # print(f"Ligne reçue: {line}") # Debug

                    if "Fin du balayage" in line:
                        # print("Fin du balayage détectée.") # Debug
                        break # Sortir de la boucle while interne

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    if match:
                        canal = int(match.group(1))
                        if canal in self.indices_à_garder:
                            voltages_dict[canal] = float(match.group(2))
                            # print(f"Canal {canal} lu: {voltages_dict[canal]} V") # Debug

                else:
                    # Pas de données immédiatement disponibles, petite pause pour ne pas surcharger le CPU
                    time.sleep(0.01)

            except serial.SerialException as e:
                print(f"Erreur série pendant la lecture : {e}")
                self.ser = None # Marquer comme déconnecté
                return None
            except Exception as e:
                print(f"Erreur inattendue pendant la lecture série : {e}")
                # Continue d'essayer de lire ? Ou retourner None ?
                continue # On essaie de continuer

        # Vérification après la sortie de boucle (Fin du balayage ou timeout)
        if len(voltages_dict) != len(self.indices_à_garder):
            print(f"⚠️ Seulement {len(voltages_dict)}/{len(self.indices_à_garder)} canaux requis reçus.")
            # Décider si retourner les données partielles ou None
            return None # Préférable de retourner None si incomplet

        # print("Données complètes reçues.") # Debug
        return voltages_dict


    # Dans la classe TraitementDonnees
    # Dans la classe TraitementDonnees (fichier Test.py)

    # Dans la classe TraitementDonnees (fichier Test.py)

    def get_temperatures(self):
        if self.simulation:
            # --- Logique pour la simulation basée sur CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0
                    print("[SIMULATION] Fin du fichier CSV atteinte, retour au début.")

                current_data_row = self.simulation_data.iloc[self.simulation_index]
                self.simulation_index += 10

                temperature_dict = {}
                temps_for_r24_average_sim = []
                valid_data_found = False

                # 1. Lecture initiale des températures depuis le CSV
                for i in self.indices_à_garder:
                    thermistor_name = self.positions[i][0]
                    if thermistor_name in current_data_row and pd.notna(current_data_row[thermistor_name]):
                        temp_value = current_data_row[thermistor_name]
                        temperature_dict[thermistor_name] = temp_value
                        if thermistor_name != "R24":
                            temps_for_r24_average_sim.append(temp_value)
                        valid_data_found = True
                    else:
                        temperature_dict[thermistor_name] = np.nan

                if not valid_data_found:
                    print(f"[ERREUR SIMULATION] Aucune donnée valide à l'index CSV {self.simulation_index - 10}.")
                    return None

                # 2. Calcul et assignation de la moyenne pour R24
                if temps_for_r24_average_sim:
                    average_temp_sim = np.mean(temps_for_r24_average_sim)
                    temperature_dict["R24"] = average_temp_sim
                elif "R24" in temperature_dict:
                    temperature_dict["R24"] = np.nan

                # --- LOGIQUE POUR BOOSTER R24 (SIMULATION) ---
                # 3. Calcul de la moyenne SANS R19, R20, R24 pour les conditions
                temps_for_check_sim = []
                r19_temp_sim = temperature_dict.get("R19", np.nan)
                r20_temp_sim = temperature_dict.get("R20", np.nan)
                r16_temp_sim = temperature_dict.get("R16", np.nan) # << NOUVEAU: Obtenir R16
                r24_temp_sim = temperature_dict.get("R24", np.nan)
                r3_temp_sim = temperature_dict.get("R3", np.nan)
                r18_temp_sim = temperature_dict.get("R18", np.nan)
                r21_temp_sim = temperature_dict.get("R21", np.nan)

                for name, temp in temperature_dict.items():
                    if pd.notna(temp) and name not in ["R19"]: # Exclut aussi R16 implicitement si R16 est dans la liste
                        temps_for_check_sim.append(temp)

                # 4. Vérifier les conditions et appliquer les boosts si nécessaire
                if temps_for_check_sim: # S'il y a d'autres températures valides
                    average_for_check_sim = np.mean(temps_for_check_sim)

                    # --- Condition 1 (existante, boost 1.08) ---
                    condition_1_met = (
                        pd.notna(r19_temp_sim) and
                        pd.notna(r20_temp_sim) and
                        r19_temp_sim > (average_for_check_sim * 1.12) and
                        r20_temp_sim > (average_for_check_sim * 1.08) and
                        r3_temp_sim > (average_for_check_sim * 1.05)
                    )

                    if condition_1_met and pd.notna(r24_temp_sim):
                        boosted_r24_temp = r24_temp_sim * 1.10
                        temperature_dict["R24"] = boosted_r24_temp
                        r24_temp_sim = boosted_r24_temp # Mettre à jour pour la condition suivante
                        # print(f"[SIM BOOST 1 R24] Boost 1.08 appliqué")

                    # --- Condition 2 (NOUVELLE, boost 1.20) ---
                    condition_2_met = (
                        pd.notna(r19_temp_sim) and
                        pd.notna(r16_temp_sim) and
                        r19_temp_sim >= (average_for_check_sim * 1.21) and # R19 >= 120%
                        r16_temp_sim <= (average_for_check_sim * 1.10) and
                        r18_temp_sim <= (average_for_check_sim * 1.02)# R16 <= 118%
                    )

                    if condition_2_met and pd.notna(r24_temp_sim):
                        boosted_r24_temp_2 = r24_temp_sim * 1.13 # Applique le boost 1.20 (écrase le 1.08 si besoin)
                        temperature_dict["R24"] = boosted_r24_temp_2
                        # print(f"[SIM BOOST 2 R24] Boost 1.20 appliqué")
                        # print(f"    Cond 2: R19={r19_temp_sim:.2f} (>= {average_for_check_sim*1.20:.2f}), R16={r16_temp_sim:.2f} (<= {average_for_check_sim*1.18:.2f})")
                    
                    condition_3_met = (
                        pd.notna(r18_temp_sim) and
                        pd.notna(r21_temp_sim) and
                        r18_temp_sim >= (average_for_check_sim * 1.12) and
                        r21_temp_sim >= (average_for_check_sim * 1.12)
                    )

                    if condition_3_met and pd.notna(r18_temp_sim) and pd.notna(r21_temp_sim ):
                        boosted_r21_temp = r21_temp_sim * 1.05
                        boosted_r18_temp = r18_temp_sim * 1.15
                        temperature_dict["R21"] = boosted_r21_temp
                        temperature_dict["R18"] = boosted_r18_temp

                # --- FIN LOGIQUE POUR BOOSTER R24 ---

                return temperature_dict
            else:
                # Fallback: Génération aléatoire (inchangé)
                print("[SIMULATION] Données CSV non disponibles, génération de températures aléatoires.")
                random_temps = {self.positions[i][0]: np.random.uniform(20.0, 45.0)
                                for i in self.indices_à_garder}
                return random_temps

        # --- Logique pour le mode non-simulation (lecture série) ---
        data_voltages = self.lire_donnees()
        if data_voltages is None:
            return None

        temperatures_raw = []
        noms = []
        indices_r24 = -1
        indices_r19 = -1
        indices_r20 = -1
        indices_r16 = -1 # << NOUVEAU: Index pour R16

        # 1. Calcul initial des températures
        for idx, i in enumerate(self.indices_à_garder):
            nom_thermistor = self.positions[i][0]
            noms.append(nom_thermistor)

            # Marquer les index
            if i == 11: indices_r24 = idx
            if nom_thermistor == "R19": indices_r19 = idx
            if nom_thermistor == "R20": indices_r20 = idx
            if nom_thermistor == "R16": indices_r16 = idx # << NOUVEAU: Trouver R16

            if i not in data_voltages:
                print(f"[AVERTISSEMENT] Tension manquante pour le canal {i} ({nom_thermistor})")
                temperatures_raw.append(np.nan)
                continue

            voltage = data_voltages[i]

            # Sélectionner les bons coefficients (vérifiez toujours ces indices)
            if i == 11: coeffs_index = 23
            elif nom_thermistor == "R19": coeffs_index = 18 # Supposition
            elif nom_thermistor == "R20": coeffs_index = 19 # Supposition
            elif nom_thermistor == "R16": coeffs_index = 15 # Supposition: R16 est sur canal 15 -> coeffs[15]
            else: coeffs_index = i

            if coeffs_index < len(self.coefficients):
                coeffs = self.coefficients[coeffs_index]
            else:
                print(f"[ERREUR] Index coeff {coeffs_index} hors limites pour {nom_thermistor} (canal {i}).")
                temperatures_raw.append(np.nan)
                continue

            # Calculer résistance et température
            resistance = self.compute_resistance(voltage)
            if resistance == float('inf') or resistance <= 0:
                temp = np.nan
            else:
                try:
                    temp = self.compute_temperature(resistance, coeffs)
                except (ValueError, OverflowError):
                    temp = np.nan
            temperatures_raw.append(temp)

        # 2. Calcul et assignation de la moyenne pour R24
        temps_for_r24_average_real = []
        for idx, temp in enumerate(temperatures_raw):
            if pd.notna(temp) and idx != indices_r24:
                temps_for_r24_average_real.append(temp)

        final_temperatures = list(temperatures_raw) # Copie pour modification

        if temps_for_r24_average_real and indices_r24 != -1:
            average_temp_real = np.mean(temps_for_r24_average_real)
            final_temperatures[indices_r24] = average_temp_real
        elif indices_r24 != -1:
            final_temperatures[indices_r24] = np.nan

        # --- LOGIQUE POUR BOOSTER R24 (NON-SIMULATION) ---
        # 3. Calcul de la moyenne SANS R19, R20, R24 pour les conditions
        temps_for_check_real = []
        r19_temp_real = np.nan
        r20_temp_real = np.nan
        r16_temp_real = np.nan # << NOUVEAU
        r24_temp_real = np.nan

        if indices_r19 != -1: r19_temp_real = final_temperatures[indices_r19]
        if indices_r20 != -1: r20_temp_real = final_temperatures[indices_r20]
        if indices_r16 != -1: r16_temp_real = final_temperatures[indices_r16] # << NOUVEAU
        if indices_r24 != -1: r24_temp_real = final_temperatures[indices_r24]

        # Indices à exclure de la moyenne de vérification
        indices_to_exclude = [idx for idx in [indices_r19, indices_r20, indices_r24] if idx != -1]

        for idx, temp in enumerate(final_temperatures):
            if pd.notna(temp) and idx not in indices_to_exclude:
                temps_for_check_real.append(temp)

        # 4. Vérifier les conditions et appliquer les boosts si nécessaire
        if temps_for_check_real: # S'il y a d'autres températures valides
            average_for_check_real = np.mean(temps_for_check_real)

            # --- Condition 1 (existante, boost 1.08) ---
            # Note: J'ai corrigé la condition ici pour utiliser 1.15 comme dans la partie simulation
            condition_1_met = (
                pd.notna(r19_temp_real) and
                pd.notna(r20_temp_real) and
                r19_temp_real > (average_for_check_real * 1.15) and
                r20_temp_real > (average_for_check_real * 1.15)
            )

            if condition_1_met and pd.notna(r24_temp_real) and indices_r24 != -1:
                boosted_r24_temp = r24_temp_real * 1.08
                final_temperatures[indices_r24] = boosted_r24_temp
                r24_temp_real = boosted_r24_temp # Mettre à jour pour la condition suivante
                # print(f"[BOOST 1 R24] Boost 1.08 appliqué")


            # --- Condition 2 (NOUVELLE, boost 1.20) ---
            condition_2_met = (
                pd.notna(r19_temp_real) and
                pd.notna(r16_temp_real) and
                r19_temp_real >= (average_for_check_real * 1.20) and # R19 >= 120%
                r16_temp_real <= (average_for_check_real * 1.18)    # R16 <= 118%
            )

            if condition_2_met and pd.notna(r24_temp_real) and indices_r24 != -1:
                boosted_r24_temp_2 = r24_temp_real * 1.20 # Applique le boost 1.20
                final_temperatures[indices_r24] = boosted_r24_temp_2
                # print(f"[BOOST 2 R24] Boost 1.20 appliqué")
                # print(f"    Cond 2: R19={r19_temp_real:.2f} (>= {average_for_check_real*1.20:.2f}), R16={r16_temp_real:.2f} (<= {average_for_check_real*1.18:.2f})")


        # --- FIN LOGIQUE POUR BOOSTER R24 ---

        if len(noms) != len(final_temperatures):
            print("[ERREUR CRITIQUE] Discordance noms/températures finales.")
            return None

        # Retourner le dictionnaire final
        return dict(zip(noms, final_temperatures))






    # Dans la classe TraitementDonnees (fichier Test.py)

    # Dans la classe TraitementDonnees (fichier Test2.py)

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear() # Efface toute la figure précédente

        # --- Revenir à un seul axe ---
        ax = fig.add_subplot(111)

        # --- Préparation des données (commune) ---
        x_orig, y_orig, t_orig = [], [], []
        valid_temps_list = []

        # 1. Extraire les données valides
        for i in self.indices_à_garder:
            name, pos = self.positions[i]
            temp_val = temperature_dict.get(name, np.nan)
            if pd.notna(temp_val):
                x_orig.append(pos[0])
                y_orig.append(pos[1])
                t_orig.append(temp_val)
                valid_temps_list.append(temp_val)

        # 2. Calculer la température de référence/bord
        can_calculate_laser_pos = False
        if not valid_temps_list:
            baseline_temp = 20.0
            print("[AVERTISSEMENT] Aucune donnée valide pour calculs.")
        else:
            avg_temp = np.mean(valid_temps_list)
            baseline_temp = avg_temp - 1.0 # Référence pour barycentre et bords
            can_calculate_laser_pos = True

        # 3. Calcul du barycentre pondéré
        laser_x, laser_y = None, None
        laser_pos_found = False
        if can_calculate_laser_pos and len(x_orig) > 0:
            total_weight = 0.0
            weighted_x_sum = 0.0
            weighted_y_sum = 0.0
            for i in range(len(x_orig)):
                delta_temp = t_orig[i] - baseline_temp
                weight = max(0, delta_temp)**2
                if weight > 1e-6:
                    weighted_x_sum += weight * x_orig[i]
                    weighted_y_sum += weight * y_orig[i]
                    total_weight += weight
            if total_weight > 1e-6:
                laser_x = weighted_x_sum / total_weight
                laser_y = weighted_y_sum / total_weight
                laser_pos_found = True

        # 4. Interpolation RBF
        r_max = 12.5
        num_edge_points = 12
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [baseline_temp] * num_edge_points

        x_combined = x_orig + list(edge_x)
        y_combined = y_orig + list(edge_y)
        t_combined = t_orig + edge_t

        if len(x_combined) < 3:
            print("[ERREUR HEATMAP] Pas assez de points pour l'interpolation.")
            ax.set_title("Pas assez de données")
            if x_orig:
                ax.scatter(x_orig, y_orig, color='black', marker='o', s=25)
            return

        rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
        grid_size = 200
        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )
        ti = rbf(xi, yi) # Données interpolées brutes
        mask = xi**2 + yi**2 > r_max**2
        ti_masked = np.ma.array(ti, mask=mask) # Données masquées pour affichage

        # --- 5. Suppression du calcul du filtre Gaussien et du max filtré ---
        # (Les lignes correspondantes ont été enlevées)

        # --- 6. Affichage sur l'axe unique (ax) : Original + Barycentre ---
        contour = ax.contourf(xi, yi, ti_masked, levels=100, cmap="plasma")
        fig.colorbar(contour, ax=ax, label="Température (°C)")
        ax.scatter(x_orig, y_orig, color='black', marker='o', s=25, label='Thermistances')

        # Annotations des points réels sur ax
        for i in range(len(x_orig)):
            original_index_in_positions = -1
            for k in self.indices_à_garder:
                # Comparaison prudente des coordonnées (gestion des flottants)
                if np.isclose(self.positions[k][1][0], x_orig[i]) and np.isclose(self.positions[k][1][1], y_orig[i]):
                    original_index_in_positions = k
                    break
            if original_index_in_positions != -1:
                name = self.positions[original_index_in_positions][0]
                ax.annotate(name, (x_orig[i], y_orig[i]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)

        # Afficher le point barycentre (vert) sur ax
        if laser_pos_found:
            ax.plot(laser_x, laser_y, 'go', markersize=10, label=f'Laser (Bary) @ ({laser_x:.1f}, {laser_y:.1f})')

        # Configuration ax
        ax.set_aspect('equal')
        title_ax = f"Heatmap (Tps: {elapsed_time:.2f} s)"
        if laser_pos_found:
            title_ax += f"\nLaser @ ({laser_x:.1f}, {laser_y:.1f})"
        ax.set_title(title_ax, fontsize=10)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max - 1, r_max + 1)
        ax.set_ylim(-r_max - 1, r_max + 1)
        ax.legend(fontsize=8)

        # --- 7. Suppression de l'affichage sur le deuxième axe (ax2) ---
        # (Les lignes correspondantes ont été enlevées)

        # --- Finalisation ---
        fig.tight_layout() 

# --- Assure-toi que le reste de ta classe et l'appel à cette fonction restent corrects ---



    # Dans la classe TraitementDonnees (fichier Test.py)

    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Ctrl+C pour arrêter)")
        # --- MODIFICATION ICI: Figure plus large ---
        fig = plt.figure(figsize=(12, 6)) # Augmenter la largeur (ex: 12 pouces)
        # -----------------------------------------
        plt.ion()
        fig.show()

        all_data = []
        # Utiliser les noms de colonnes définis dans __init__ pour le CSV si en mode simulation
        if self.simulation:
             # Assure-toi que les headers correspondent bien aux données que tu veux sauvegarder
             headers = self.simulation_columns + ["T_ref", "timestamp", "temps_ecoule_s"]
        else:
             headers = [self.positions[i][0] for i in self.indices_à_garder] + ["T_ref", "timestamp", "temps_ecoule_s"]


        start_time = time.time()
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures()

                if data:
                    os.system("clear") # ou 'cls' sur Windows
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0
                    for name, temp in data.items():
                        if pd.notna(temp):
                             print(f"{name:<6} : {temp:6.2f} °C")
                             valid_temps_count += 1
                        else:
                             print(f"{name:<6} :   --   °C (NaN)")
                    print(f"({valid_temps_count}/{len(self.indices_à_garder)} thermistances valides)")
                    print("=" * 60)

                    # L'appel reste le même, la fonction interne gère les 2 plots
                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # Préparer la ligne pour le CSV
                    ligne = []
                    for header_name in headers:
                        if header_name == "T_ref":
                            ligne.append(25.0) # Valeur T_ref fixe
                        elif header_name == "timestamp":
                            ligne.append(datetime.now().isoformat(timespec='seconds'))
                        elif header_name == "temps_ecoule_s":
                            ligne.append(round(elapsed_time, 3))
                        elif header_name in data:
                            temp_value = data[header_name]
                            ligne.append(temp_value if pd.notna(temp_value) else '') # NaN -> ''
                        else:
                            # Gérer le cas où un header n'est pas dans les données (devrait moins arriver maintenant)
                            ligne.append('')
                    all_data.append(ligne)


                else:
                    # Affichage si données incomplètes
                    os.system("clear") # ou 'cls' sur Windows
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues.")
                    print("=" * 60)


                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée. Sauvegarde du fichier CSV...")
            # ... (code de sauvegarde CSV inchangé) ...
            desktop_path = Path.home() / "Desktop"
            filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = desktop_path / filename

            try:
                with open(csv_path, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers) # Utiliser les headers définis au début
                    writer.writerows(all_data)
                print(f"✅ Données sauvegardées dans : {csv_path}")
            except Exception as e:
                print(f"❌ Erreur lors de la sauvegarde du CSV : {e}")


# --- Le reste du fichier reste inchangé ---



if __name__ == "__main__":
    td = TraitementDonnees(simulation=True)
    td.demarrer_acquisition_live(interval=0.05)


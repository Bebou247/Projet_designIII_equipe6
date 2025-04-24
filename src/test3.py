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
    VREF = 3.003
    R_FIXED = 4700

    def __init__(self, port="/dev/cu.usbmodem101",path = "data/", coeffs_path="data/raw/coefficients.npy", simulation=False, fichier_simulation=None):
        self.path = path
        self.port = port
        self.simulation = simulation
        self.fichier_simulation = fichier_simulation # Stocker le chemin fourni

        # Utiliser un chemin absolu basé sur l'emplacement du script est plus robuste
        base_path = Path(__file__).parent.parent # Chemin vers le dossier racine du projet

        try:
            coeffs_full_path = base_path / coeffs_path
            self.coefficients = np.load(coeffs_full_path, allow_pickle=True)
            print(f"[INFO] Coefficients chargés depuis : {coeffs_full_path}")
        except FileNotFoundError:
             print(f"[ERREUR] Fichier coefficients non trouvé : {coeffs_full_path}")
             raise # Arrêter si les coefficients sont essentiels
        except Exception as e:
             print(f"[ERREUR] Problème chargement coefficients : {e}")
             raise

        self.puissance = 0
        self.data_photodiodes = [0,0,0,0,0,0]
        # Initialiser les historiques comme des listes vides
        self.puissance_hist = []
        self.puissance_hist_2 = []
        self.puissance_P = []
        self.puissance_I = []
        self.puissance_D = []
        self.puissance_DD = []
        self.delta_T_hist = [] # Important pour estimate_power_from_row
        self.time_test = [] # Garder si utilisé ailleurs

        # Chargement des données photodiode (utiliser base_path)
        try:
            data_path_full = base_path / self.path
            self.correction_matrices = [pd.read_csv(data_path_full / f"matrice_corr_diode_{i}.csv", sep=',', decimal='.').values for i in range(6)]
            # Lire une seule fois pour les colonnes et les données
            df_vis_ratios = pd.read_csv(data_path_full / "ratios_photodiodes_VIS.csv", sep=';', decimal=',')
            self.photodiode_ratios_VIS = [df_vis_ratios[col].values for col in df_vis_ratios.columns]
            df_nir_ratios = pd.read_csv(data_path_full / "ratios_photodiodes_NIR.csv", sep=';', decimal=',')
            self.photodiode_ratios_NIR = [df_nir_ratios[col].values for col in df_nir_ratios.columns]
            self.photodiode_ratios_IR = pd.read_csv(data_path_full / "ratios_photodiodes_IR.csv", sep=';', decimal=',').values
            df_vis_tensions = pd.read_csv(data_path_full / "tensions_photodiodes_VIS.csv", sep=';', decimal=',')
            self.photodiode_tensions_VIS = [df_vis_tensions[col].values for col in df_vis_tensions.columns]
            df_nir_tensions = pd.read_csv(data_path_full / "tensions_photodiodes_NIR.csv", sep=';', decimal=',')
            self.photodiode_tensions_NIR = [df_nir_tensions[col].values for col in df_nir_tensions.columns]
            print("[INFO] Données photodiode chargées.")
        except FileNotFoundError as e:
            print(f"[ERREUR] Fichier photodiode manquant : {e}")
            raise
        except Exception as e:
            print(f"[ERREUR] Problème chargement données photodiode : {e}")
            raise

        # Décalage à appliquer
        decalage_x = -0.4  # vers la gauche
        decalage_y = -0.2  # légèrement plus bas

        self.tension_photodidodes = [0,0,0,0,0,0] # Renommer peut-être en self.tensions_photodiodes ?

        self.positions = [
            ("R1", (11 + decalage_x, 0 + decalage_y)), ("R2", (3 + 1, 0 + decalage_y)), ("R3", (-3 + decalage_x, 0 + decalage_y)), ("R4", (-11 + decalage_x, 0 + decalage_y)),
            ("R5", (8 + 1, 2.5 - decalage_y)), ("R6", (0 + decalage_x, 2.5 + decalage_y)), ("R7", (-8 + decalage_x, 2.5 + decalage_y)), ("R8", (8 + 1, 5.5 - decalage_y)),
            ("R9", (0 + decalage_x, 5.5 + decalage_y)), ("R10", (-8 + decalage_x, 5.5 + decalage_y)), ("R11", (4.5 + decalage_x, 8 + decalage_y)), ("R24", (-3.5 + decalage_x, -11.25 + decalage_y)), # Note: R24 est sur le canal 11 physiquement
            ("R13", (4 + decalage_x, 11.25 + decalage_y)), ("R14", (-4 + decalage_x, 11.25 + decalage_y)), ("R15", (8 + 1, -2.5 - decalage_y)), ("R16", (0 + decalage_x, -2.5 + decalage_y)),
            ("R17", (-8 + decalage_x, -2.5 + decalage_y)), ("R18", (8 + 1, -5.5 - decalage_y)), ("R19", (0 + decalage_x, -5.5 + decalage_y)), ("R20", (-8 + decalage_x, -5.5 + decalage_y)),
            ("R21", (4.5 + decalage_x, -8 + decalage_y)), ("R25", (0 + decalage_x, -11.5 + decalage_y)), # R25 est la référence, souvent sur canal 24
            ("R_Virtuel", (-4.9, 7.8))
        ]

        self.photodiodes = ["PD25","PD26","PD27","PD28","PD29","PD30"]

        # Indices des canaux à lire sur l'Arduino (0-20 pour R1-R21, 24 pour R25, 25-30 pour PDs)
        # Le canal 11 (R24) est lu mais sa valeur est recalculée.
        self.indices_à_garder = list(range(21)) + [24] + list(range(25, 31))

        self.simulation_data = None
        self.simulation_index = 0
        # Colonnes attendues dans le fichier de simulation (basé sur l'en-tête du fichier)
        self.simulation_columns = [
            "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10",
            "R11", "R24", "R13", "R14", "R15", "R16", "R17", "R18", "R19", "R20",
            "R21", "R25", "R_virtuel", # Correspond à ton en-tête
            "PD25", "PD26", "PD27", "PD28", "PD29", "PD30",
            "timestamp"
        ]

        self.previous_ti_filtered = None
        self.position_history = []
        self.history_length = 5
        self.last_valid_raw_pos = None
        self.last_filtered_pos = (None, None)
        self.max_speed_mm_per_interval = 3.0
        self.min_heating_threshold = 0.05
        # self.fichier_simulation est déjà défini plus haut

        # --- Bloc de chargement Simulation ---
        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")

            try:
                # Déterminer le chemin du fichier de simulation
                if self.fichier_simulation:
                    # Si un chemin est fourni, on essaie de l'utiliser
                    simulation_file_path = Path(self.fichier_simulation)
                    print(f"[INFO] Utilisation du fichier de simulation fourni : {simulation_file_path}")
                else:
                    # Sinon, on prend un fichier par défaut (AJUSTEZ CE NOM SI NÉCESSAIRE)
                    default_filename = "Test-echelon-976-nm-2025-04-24.csv" # Exemple
                    simulation_file_path = base_path / "data" / default_filename
                    print(f"[INFO] Utilisation du fichier de simulation par défaut : {simulation_file_path}")

                # Vérifier si le fichier existe avant de tenter de le lire
                if not simulation_file_path.is_file():
                     raise FileNotFoundError(f"Fichier simulation non trouvé: {simulation_file_path.resolve()}")

                # --- Lecture du CSV ---
                # Assurez-vous que sep=';' correspond bien à votre fichier.
                # decimal='.' est généralement correct si les chiffres utilisent un point.
                # encoding='utf-8' est souvent un bon choix.
                # header=0 est la valeur par défaut, utilise la première ligne comme en-tête
                self.simulation_data = pd.read_csv(
                    simulation_file_path,
                    sep=';',
                    decimal='.',
                    encoding='utf-8', # Ajout pour plus de robustesse
                    # parse_dates=['timestamp'] # On le fait manuellement après pour plus de contrôle
                )
                # --- Fin Lecture ---

                print(f"[INFO] Fichier de simulation '{simulation_file_path.resolve()}' chargé.")

                # --- DEBUG: Afficher les colonnes lues et les premières lignes ---
                print("[DEBUG INIT] Colonnes lues par pandas:", self.simulation_data.columns.tolist())
                print("[DEBUG INIT] Aperçu des 5 premières lignes lues:")
                print(self.simulation_data.head())
                # --- FIN DEBUG ---

                # Vérifier si les colonnes attendues sont présentes (basé sur self.simulation_columns)
                colonnes_lues = self.simulation_data.columns.tolist()
                colonnes_manquantes = [col for col in self.simulation_columns if col not in colonnes_lues]
                if colonnes_manquantes:
                    print(f"[AVERTISSEMENT SIMULATION] Colonnes attendues mais non trouvées dans le CSV: {colonnes_manquantes}")
                    # Vous pourriez vouloir arrêter ici ou continuer avec les colonnes disponibles

                # Convertir la colonne timestamp en datetime
                if 'timestamp' in self.simulation_data.columns:
                     try:
                         # Essayer de convertir en datetime (pandas est assez flexible)
                         # Essayer différents formats si nécessaire
                         try:
                             self.simulation_data['timestamp'] = pd.to_datetime(self.simulation_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
                         except ValueError:
                             try:
                                 # Essayer un format plus général si le premier échoue
                                 self.simulation_data['timestamp'] = pd.to_datetime(self.simulation_data['timestamp'])
                             except Exception as e_time_inner:
                                 raise ValueError(f"Impossible de convertir la colonne timestamp avec les formats essayés: {e_time_inner}")

                         if self.simulation_data['timestamp'].isnull().any():
                             print("[AVERTISSEMENT SIMULATION] Certaines valeurs de timestamp n'ont pas pu être converties.")
                     except Exception as e_time:
                         print(f"[ERREUR INIT] Impossible de convertir la colonne timestamp: {e_time}")
                         # Optionnel: Remplacer par NaT ou arrêter
                         self.simulation_data['timestamp'] = pd.NaT
                else:
                     print("[AVERTISSEMENT SIMULATION] Colonne 'timestamp' non trouvée. Un dt fixe sera utilisé.")
                     # Pas besoin de créer une colonne si elle manque, la logique dans get_temperatures gère dt

                # Nettoyage et validation des autres colonnes (convertir en numérique)
                colonnes_numeriques_attendues = [col for col in self.simulation_columns if col != 'timestamp']
                for col in colonnes_numeriques_attendues:
                    if col in self.simulation_data.columns:
                        # Utilise .loc pour éviter SettingWithCopyWarning
                        try:
                            # Tenter la conversion, mettre NaN si échec
                            self.simulation_data.loc[:, col] = pd.to_numeric(self.simulation_data[col], errors='coerce')
                        except Exception as e_num:
                             print(f"[AVERTISSEMENT SIMULATION] Problème conversion numérique colonne '{col}': {e_num}")
                    # else: # La colonne n'a pas été trouvée, déjà signalé plus haut

                if self.simulation_data.isnull().values.any():
                    print("[AVERTISSEMENT SIMULATION] Le fichier CSV contient des valeurs non numériques ou manquantes après conversion.")

                self.simulation_index = 0
                # Initialiser last_timestamp pour le calcul de dt dans get_temperatures
                # Utiliser NaT (Not a Time) de pandas pour une meilleure compatibilité
                self.last_timestamp = pd.NaT
                print(f"[INFO] Données chargées : {self.simulation_data.shape[0]} lignes, {self.simulation_data.shape[1]} colonnes.")

            except FileNotFoundError as e:
                print(f"[ERREUR] {e}")
                self.simulation_data = None # Assurer que c'est None si échec
            except Exception as e:
                print(f"[ERREUR] Échec du chargement ou traitement du fichier CSV : {e}")
                import traceback
                traceback.print_exc() # Donne plus de détails sur l'erreur
                self.simulation_data = None # Assurer que c'est None si échec
        # --- Fin du bloc Simulation ---

        # --- Bloc Connexion Série ---
        else:
            try:
                self.ser = serial.Serial(self.port, 9600, timeout=0.2) # Timeout court pour réactivité
                print(f"[INFO] Port série connecté sur {self.port}")
                # Petite pause pour laisser l'Arduino s'initialiser si nécessaire
                time.sleep(1.5)
                self.ser.reset_input_buffer() # Vider buffer au cas où
            except serial.SerialException as e:
                print(f"[ERREUR] Impossible d'ouvrir le port série {self.port}: {e}")
                self.ser = None
            except Exception as e:
                print(f"[ERREUR] Autre erreur connexion série: {e}")
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
            # print(self.ser.readline().decode(errors='ignore').strip())
            current_time = time.time()

            if current_time - start_time > timeout_sec:
                print(f"⚠️ Temps de lecture dépassé ({timeout_sec}s), données incomplètes.")
                # Retourner les données partielles ou None ? Ici on retourne partiel si on a quelque chose.
                return voltages_dict if voltages_dict else None
            
            # print(self.ser.in_waiting)

            try:
                # line = self.ser.readline().decode(errors='ignore').strip()
                if self.ser.in_waiting >= 0:
                # if "=== DEBUT BALAYAGE ===" in line:
                    # print("Ça marche")
                    line = self.ser.readline().decode(errors='ignore').strip()
                    # print(f"[DEBUG RAW] Reçu: '{line}'")
                    if not line:
                        continue

                    if "Fin du balayage" in line:
                        # print("[DEBUG] Fin du balayage détectée.") # Debug
                        break # Sortir de la boucle while interne

                    match = re.search(r"Canal (\d+): ([\d.]+) V", line)
                    # print(match)
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
                    # print("Ça marche pas")
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

        data_phot = [0,0,0,0,0,0]

        # print(voltages_dict)

        for i in range(25, 31):
            data_phot[i-25] = voltages_dict[i]
            # print(v)

        # print(data_phot)

        # light_type, wavelength, power = self.get_wavelength()

        # print(f"Laser {light_type}, longueur d'onde de {wavelength:.0f} nm et puissance estimée de {power:.2f} W\n")

        self.data_photodiodes = data_phot

        # print(voltages_dict)

        return voltages_dict

    def get_temperatures(self):
        """
        Récupère les températures des thermistances.
        En mode simulation, lit les données depuis le fichier CSV chargé.
        En mode série, communique avec l'Arduino.
        Calcule également la température virtuelle R_Virtuel.
        """
        real_temps_dict = {} # Dictionnaire pour les températures réelles
        real_tension_dict = {} # Dictionnaire pour les tensions réelles

        if self.simulation:
            if self.simulation_data is not None and not self.simulation_data.empty:
                # Vérifier si l'index est toujours dans les limites du DataFrame
                if self.simulation_index >= len(self.simulation_data):
                    print("[INFO SIMULATION] Fin des données de simulation atteinte.")
                    # Optionnel : Arrêter la mise à jour ou boucler
                    # self.after_cancel(self.update_id) # Arrêter si update_id est stocké
                    # Ou réinitialiser l'index pour boucler :
                    # self.simulation_index = 0
                    # Pour l'instant, retournons les dernières valeurs valides ou NaN
                    # Récupérer les dernières valeurs connues si elles existent
                    last_temps = getattr(self, "last_real_temps_dict", {})
                    last_tensions = getattr(self, "last_real_tension_dict", {})
                    return last_temps, last_tensions # Retourne les dernières connues

                # Calcul du pas de temps dt pour estimate_power_from_row
                current_timestamp_str = self.simulation_data.iloc[self.simulation_index]['timestamp']
                try:
                    # Essayer de parser avec le format attendu
                    current_timestamp = pd.to_datetime(current_timestamp_str, format='%Y-%m-%d %H:%M:%S')
                except ValueError:
                    try:
                       # Essayer avec un format ISO 8601 plus général si le premier échoue
                       current_timestamp = pd.to_datetime(current_timestamp_str)
                    except ValueError:
                       print(f"[ERREUR SIMULATION] Format de timestamp invalide : {current_timestamp_str}")
                       # Gérer l'erreur: utiliser dt=0 ou une valeur par défaut, ou arrêter
                       current_timestamp = pd.NaT # Not a Time

                if pd.notna(current_timestamp) and hasattr(self, 'last_timestamp') and pd.notna(self.last_timestamp):
                    dt = (current_timestamp - self.last_timestamp).total_seconds()
                    # Gérer les cas où dt serait 0 ou négatif (données mal ordonnées?)
                    if dt <= 0:
                        # print(f"[AVERTISSEMENT SIMULATION] dt <= 0 ({dt}s). Utilisation de dt=0.1s par défaut.")
                        dt = 0.1 # Ou une autre petite valeur positive
                else:
                    # print("[INFO SIMULATION] Premier point ou timestamp invalide, dt mis à 0.1s par défaut.")
                    dt = 0.1 # Valeur par défaut pour le premier point ou si erreur

                self.last_timestamp = current_timestamp # Mettre à jour pour la prochaine itération

                # Lire la ligne actuelle
                current_data_row = self.simulation_data.iloc[self.simulation_index]

                # --- Appel estimate_power_from_row (déjà corrigé) ---
                # S'assurer que dt est bien un float positif
                if isinstance(dt, (int, float)) and dt > 0:
                    self.estimate_power_from_row(current_data_row, float(dt))
                else:
                    # print(f"[AVERTISSEMENT SIMULATION] dt invalide ({dt}), saut de estimate_power_from_row.")
                    pass # Ne pas appeler si dt n'est pas valide

                # --- Incrémenter index (déjà là) ---
                # S'assurer que l'incrément est correct (normalement 1 pour lire ligne par ligne)
                self.simulation_index += 1
                valid_data_found = False

                # --- DÉBOGAGE : Afficher la ligne lue ---
                print(f"\n[DEBUG SIMULATION] Index {self.simulation_index - 1}: Ligne brute lue:")
                print(current_data_row)
                # --- FIN DÉBOGAGE ---

                # Lire les températures simulées pour les thermistances réelles (SAUF R24 pour l'instant)
                for i, (name, _) in enumerate(self.positions):
                    if name in ["R_Virtuel", "R24"]: continue # Ignorer la virtuelle et R24 ici

                    # Cas général pour les thermistances dans simulation_columns
                    if name in self.simulation_columns:
                        if name in current_data_row.index and pd.notna(current_data_row[name]):
                            try:
                                temp_value = float(current_data_row[name]) # Essayer de convertir en float
                                real_temps_dict[name] = temp_value # Stocker
                                valid_data_found = True
                                # --- DÉBOGAGE ---
                                print(f"[DEBUG SIMULATION] Lu {name}: {temp_value}")
                                # --- FIN DÉBOGAGE ---
                            except (ValueError, TypeError):
                                real_temps_dict[name] = np.nan # Mettre NaN si la conversion échoue
                                print(f"[DEBUG SIMULATION] Lu {name}: NaN (conversion float échouée pour '{current_data_row[name]}')")
                        else:
                            real_temps_dict[name] = np.nan # Mettre NaN si absent ou déjà NaN
                            # --- DÉBOGAGE ---
                            print(f"[DEBUG SIMULATION] Lu {name}: NaN (colonne absente ou valeur NaN dans CSV)")
                            # --- FIN DÉBOGAGE ---

                    # Cas spécifique pour R25 (souvent présente même si pas dans self.positions par défaut)
                    elif name == "R25" and "R25" in current_data_row.index:
                         if pd.notna(current_data_row["R25"]):
                             try:
                                 temp_value_r25 = float(current_data_row["R25"])
                                 real_temps_dict["R25"] = temp_value_r25
                                 valid_data_found = True
                                 # --- DÉBOGAGE ---
                                 print(f"[DEBUG SIMULATION] Lu R25: {temp_value_r25}")
                                 # --- FIN DÉBOGAGE ---
                             except (ValueError, TypeError):
                                 real_temps_dict["R25"] = np.nan
                                 print(f"[DEBUG SIMULATION] Lu R25: NaN (conversion float échouée pour '{current_data_row['R25']}')")
                         else:
                             real_temps_dict["R25"] = np.nan
                             # --- DÉBOGAGE ---
                             print(f"[DEBUG SIMULATION] Lu R25: NaN (colonne absente ou valeur NaN dans CSV)")
                             # --- FIN DÉBOGAGE ---

                # Lire les tensions des photodiodes simulées
                for i, name in enumerate(self.photodiodes):
                    if name in current_data_row.index and pd.notna(current_data_row[name]):
                        try:
                            tension_value = float(current_data_row[name])
                            real_tension_dict[name] = tension_value
                            # print(f"[DEBUG SIMULATION] Lu {name}: {tension_value}") # Décommenter si besoin
                        except (ValueError, TypeError):
                            real_tension_dict[name] = np.nan
                            print(f"[DEBUG SIMULATION] Lu {name}: NaN (conversion float échouée pour '{current_data_row[name]}')")
                    else:
                        real_tension_dict[name] = np.nan
                        # print(f"[DEBUG SIMULATION] Lu {name}: NaN (colonne absente ou valeur NaN dans CSV)") # Décommenter si besoin

                # --- DÉBOGAGE : Afficher le dict AVANT calculs R24/Virtuel ---
                print("[DEBUG SIMULATION] real_temps_dict AVANT calculs R24/Virtuel:")
                print(real_temps_dict)
                # --- FIN DÉBOGAGE ---

                if not valid_data_found:
                    print("[AVERTISSEMENT SIMULATION] Aucune donnée de température valide trouvée dans cette ligne.")
                    # Remplir avec NaN pour éviter les erreurs si le dict est vide
                    for i, (name, _) in enumerate(self.positions):
                         if name not in real_temps_dict: real_temps_dict[name] = np.nan
                    for name in self.photodiodes:
                         if name not in real_tension_dict: real_tension_dict[name] = np.nan
                    # On peut quand même essayer de calculer R24/Virtuel, ils seront NaN si les entrées sont NaN

                # --- Logique R24 (Moyenne Pondérée) ---
                # Assurer que les clés existent avant de tenter la moyenne
                r12_temp = real_temps_dict.get("R12", np.nan)
                r13_temp = real_temps_dict.get("R13", np.nan)
                r14_temp = real_temps_dict.get("R14", np.nan)
                r15_temp = real_temps_dict.get("R15", np.nan)

                # Utiliser np.nanmean pour ignorer les NaN dans le calcul de la moyenne
                # Note: Ceci est une moyenne simple, pas pondérée. Ajuster si besoin.
                temps_for_r24 = [r12_temp, r13_temp, r14_temp, r15_temp]
                # Filtrer les NaN explicitement avant la moyenne si np.nanmean n'est pas souhaité
                valid_temps_r24 = [t for t in temps_for_r24 if pd.notna(t)]
                if valid_temps_r24:
                    real_temps_dict["R24"] = np.mean(valid_temps_r24)
                    # print(f"[DEBUG SIMULATION] R24 calculé: {real_temps_dict['R24']:.2f}")
                else:
                    real_temps_dict["R24"] = np.nan
                    # print("[DEBUG SIMULATION] R24 calculé: NaN (pas de données valides)")

            else:
                print("[ERREUR SIMULATION] Pas de données de simulation chargées ou DataFrame vide.")
                # Retourner des dictionnaires vides ou avec NaN
                for i, (name, _) in enumerate(self.positions):
                     real_temps_dict[name] = np.nan
                for name in self.photodiodes:
                     real_tension_dict[name] = np.nan
                # Assurer que R24 et R_Virtuel existent aussi comme clés
                real_temps_dict["R24"] = np.nan
                real_temps_dict["R_Virtuel"] = np.nan
                return real_temps_dict, real_tension_dict

        # --- Mode Série (Communication Arduino) ---
        else:
            # ... (Code existant pour le mode série) ...
            # Lire les données depuis Arduino
            data_line = self.read_serial_data()
            if data_line:
                parts = data_line.split(',')
                num_parts = len(parts)
                num_expected = len(self.positions) -1 + len(self.photodiodes) # -1 car R_Virtuel n'est pas envoyé

                if num_parts == num_expected:
                    try:
                        # Lire les températures réelles (R1 à R21 + R25)
                        temp_index = 0
                        for i, (name, _) in enumerate(self.positions):
                             if name != "R_Virtuel" and name != "R24": # Ignorer virtuelle et R24 ici
                                 real_temps_dict[name] = float(parts[temp_index])
                                 temp_index += 1

                        # Lire les tensions des photodiodes (PD25 à PD30)
                        tension_index = temp_index # Continue après les températures
                        for i, name in enumerate(self.photodiodes):
                             real_tension_dict[name] = float(parts[tension_index])
                             tension_index += 1

                        # --- Logique R24 (Moyenne Pondérée) ---
                        # (Même logique que pour la simulation, basée sur les valeurs lues)
                        r12_temp = real_temps_dict.get("R12", np.nan)
                        r13_temp = real_temps_dict.get("R13", np.nan)
                        r14_temp = real_temps_dict.get("R14", np.nan)
                        r15_temp = real_temps_dict.get("R15", np.nan)
                        temps_for_r24 = [r12_temp, r13_temp, r14_temp, r15_temp]
                        valid_temps_r24 = [t for t in temps_for_r24 if pd.notna(t)]
                        if valid_temps_r24:
                            real_temps_dict["R24"] = np.mean(valid_temps_r24)
                        else:
                            real_temps_dict["R24"] = np.nan

                    except (ValueError, IndexError) as e:
                        print(f"[ERREUR SERIE] Erreur parsing données série: {e} - Ligne: {data_line}")
                        # Remplir avec NaN en cas d'erreur
                        for i, (name, _) in enumerate(self.positions): real_temps_dict[name] = np.nan
                        for name in self.photodiodes: real_tension_dict[name] = np.nan
                        real_temps_dict["R24"] = np.nan # Assurer que R24 existe
                else:
                    print(f"[ERREUR SERIE] Nombre de parties incorrect ({num_parts} reçu, {num_expected} attendu). Ligne: {data_line}")
                    # Remplir avec NaN
                    for i, (name, _) in enumerate(self.positions): real_temps_dict[name] = np.nan
                    for name in self.photodiodes: real_tension_dict[name] = np.nan
                    real_temps_dict["R24"] = np.nan # Assurer que R24 existe
            else:
                # Pas de données lues du port série
                print("[AVERTISSEMENT SERIE] Aucune donnée lue du port série.")
                # Remplir avec NaN
                for i, (name, _) in enumerate(self.positions): real_temps_dict[name] = np.nan
                for name in self.photodiodes: real_tension_dict[name] = np.nan
                real_temps_dict["R24"] = np.nan # Assurer que R24 existe

        # --- Calcul de R_Virtuel (commun aux deux modes) ---
        # S'assurer que les clés existent avant de tenter la moyenne
        r7_temp = real_temps_dict.get("R7", np.nan)
        r8_temp = real_temps_dict.get("R8", np.nan)
        r9_temp = real_temps_dict.get("R9", np.nan)
        r10_temp = real_temps_dict.get("R10", np.nan)

        # Utiliser np.nanmean pour ignorer les NaN dans le calcul de la moyenne
        temps_for_virtuel = [r7_temp, r8_temp, r9_temp, r10_temp]
        # Filtrer les NaN explicitement avant la moyenne si np.nanmean n'est pas souhaité
        valid_temps_virtuel = [t for t in temps_for_virtuel if pd.notna(t)]
        if valid_temps_virtuel:
            real_temps_dict["R_Virtuel"] = np.mean(valid_temps_virtuel)
            # print(f"[DEBUG] R_Virtuel calculé: {real_temps_dict['R_Virtuel']:.2f}")
        else:
            real_temps_dict["R_Virtuel"] = np.nan
            # print("[DEBUG] R_Virtuel calculé: NaN (pas de données valides)")

        # Stocker les dernières valeurs valides (utile si la simulation s'arrête)
        self.last_real_temps_dict = real_temps_dict.copy()
        self.last_real_tension_dict = real_tension_dict.copy()

        # --- DÉBOGAGE : Afficher le dict FINAL retourné ---
        print("\n[DEBUG] real_temps_dict FINAL retourné:")
        print(real_temps_dict)
        # --- FIN DÉBOGAGE ---

        return real_temps_dict, real_tension_dict



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


    def demarrer_acquisition_live(self, interval=0.3):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Fermez la fenêtre pour arrêter ou Ctrl+C)")
        fig = plt.figure(figsize=(12, 6))
        plt.ion()
        fig.show()

        all_data_to_save = [] # Renommé pour clarté
        # Définir les headers pour le CSV
        base_headers = [name for name, _ in self.positions] # Inclut R_Virtuel
        photodiode_headers = self.photodiodes.copy()
        extra_headers = ["timestamp", "temps_ecoule_s", "puissance_estimee_W", "lambda_estimee_nm", "type_lumiere"]
        # Ordre: Thermistances (y compris R_Virtuel), Photodiodes, Extras
        headers = base_headers + photodiode_headers + extra_headers

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
                self.time_test.append(elapsed_time) # Utilisé pour calculer dt en mode série

                # --- CORRECTION : Récupérer le tuple retourné ---
                temp_data, tension_data = self.get_temperatures()
                # Utiliser temp_data pour l'affichage et la heatmap
                data = temp_data
                # --- FIN CORRECTION ---

                # Mettre à jour self.data_photodiodes (déjà fait dans get_temperatures, mais redondance ok)
                self.data_photodiodes = [tension_data.get(name, 0) for name in self.photodiodes]

                if data: # Vérifie si temp_data n'est pas None ou vide
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    valid_temps_count = 0

                    # Affichage des températures (utilise 'data' qui est temp_data)
                    for name, temp in data.items():
                        display_name = name
                        if pd.notna(temp):
                            if name in self.photodiodes: # Ne devrait plus être dans 'data'
                                pass # Ignorer ici, affiché plus bas
                            elif name == "R_Virtuel":
                                print(f"{display_name:<10} : {temp:6.2f} °C (Virtuelle)")
                            else:
                                print(f"{display_name:<10} : {temp:6.2f} °C")
                                if name != "R25": # Ne compte pas R25 comme valide pour le décompte
                                    valid_temps_count += 1
                        else:
                             print(f"{display_name:<10} :   --   °C (NaN)")

                    print("-" * 60)
                    print("Tensions Photodiodes")
                    print("-" * 60)
                    # Affichage des tensions (utilise 'tension_data')
                    for name, tension in tension_data.items():
                         if pd.notna(tension):
                             print(f"{name:<10} : {tension:6.3f}  V")
                         else:
                             print(f"{name:<10} :   --    V (NaN)")

                    print("-" * 60)
                    # Calcul longueur d'onde et puissance (utilise self.data_photodiodes et self.puissance)
                    light_type, wavelength, power_photo = self.get_wavelength() # Renommé power_photo pour éviter conflit
                    # Afficher la puissance estimée par PID (self.puissance) et celle par photodiodes (power_photo)
                    print(f"Laser {light_type}, λ estimé: {wavelength:.0f} nm")
                    print(f"Puissance (PID Temp): {self.puissance:.2f} W | Puissance (Photodiodes): {power_photo:.2f} W")
                    print("=" * 60)

                    # Affichage Heatmap (utilise 'data' qui est temp_data)
                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    # Préparation de la ligne pour sauvegarde CSV
                    ligne_dict = {}
                    # Ajouter les températures
                    for header_name in base_headers:
                        ligne_dict[header_name] = data.get(header_name, np.nan) # Utilise get pour éviter KeyError
                    # Ajouter les tensions photodiodes
                    for header_name in photodiode_headers:
                        ligne_dict[header_name] = tension_data.get(header_name, np.nan)
                    # Ajouter les extras
                    ligne_dict["timestamp"] = datetime.now().isoformat(timespec='seconds')
                    ligne_dict["temps_ecoule_s"] = elapsed_time
                    ligne_dict["puissance_estimee_W"] = self.puissance # Sauvegarde puissance PID
                    ligne_dict["lambda_estimee_nm"] = wavelength
                    ligne_dict["type_lumiere"] = light_type

                    # Convertir en liste ordonnée selon les headers
                    ligne_pour_csv = []
                    for h in headers:
                        value = ligne_dict.get(h, '') # Valeur par défaut si clé manque
                        # Arrondir si c'est un nombre
                        if isinstance(value, (int, float)) and pd.notna(value):
                            if h == "temps_ecoule_s":
                                ligne_pour_csv.append(round(value, 3))
                            elif h.startswith("PD"):
                                ligne_pour_csv.append(round(value, 3)) # Plus de précision pour tensions
                            else:
                                ligne_pour_csv.append(round(value, 2))
                        elif pd.isna(value):
                             ligne_pour_csv.append('') # Laisser vide si NaN
                        else:
                             ligne_pour_csv.append(value) # Garder tel quel (ex: timestamp, type_lumiere)

                    all_data_to_save.append(ligne_pour_csv)
                    self.all_data = all_data_to_save # Garder la compatibilité si self.all_data est utilisé ailleurs

                else: # Si get_temperatures a retourné (None, None) ou ({}, {})
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("⚠️ Données incomplètes ou non reçues. Attente...")
                    print("=" * 60)

                # Pause
                time.sleep(max(0, interval - (time.time() - current_time)))

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée par Ctrl+C.")
            keep_running = False
        finally:
            print("\n🛑 Fin de l'acquisition.")
            if plt.fignum_exists(fig.number):
                plt.close(fig)

            # Sauvegarde CSV (utilise all_data_to_save et headers définis au début)
            if all_data_to_save:
                print("Sauvegarde du fichier CSV...")
                desktop_path = Path.home() / "Desktop"
                # Nom de fichier principal
                filename_base = f"acquisition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                csv_path = desktop_path / f"{filename_base}.csv"

                try:
                    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow(headers) # Écrit les headers
                        writer.writerows(all_data_to_save) # Écrit les données collectées
                    print(f"✅ Données principales sauvegardées dans : {csv_path}")

                    # Sauvegarde optionnelle des historiques PID (si existent)
                    pid_histories = {
                        "puissance_PID_brute": getattr(self, "puissance_hist", []),
                        "puissance_PID_lissee": getattr(self, "puissance_hist_2", []),
                        "PID_P": getattr(self, "puissance_P", []),
                        "PID_I": getattr(self, "puissance_I", []),
                        "PID_D": getattr(self, "puissance_D", []),
                        "PID_DD": getattr(self, "puissance_DD", [])
                    }
                    for name, history in pid_histories.items():
                        if history: # Sauvegarde seulement si l'historique n'est pas vide
                             pid_csv_path = desktop_path / f"{filename_base}_{name}.csv"
                             try:
                                 with open(pid_csv_path, mode='w', newline='', encoding='utf-8') as f_pid:
                                     writer_pid = csv.writer(f_pid)
                                     writer_pid.writerow([name]) # Simple header
                                     writer_pid.writerow(history) # Écrit l'historique sur une ligne
                                 print(f"✅ Historique {name} sauvegardé dans : {pid_csv_path}")
                             except Exception as e_pid:
                                 print(f"❌ Erreur sauvegarde historique {name} : {e_pid}")

                except Exception as e:
                    print(f"❌ Erreur lors de la sauvegarde du CSV principal : {e}")
            else:
                print("ℹ️ Aucune donnée collectée à sauvegarder.")


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

        print(V_photodiodes)


        for i, V in enumerate(V_photodiodes):
            if V < 0.05:
                V_photodiodes[i] = 0
            # print(self.correction_matrices[i][pos])

        # print(V_photodiodes)

        V_corr = np.array([V * self.correction_matrices[i][pos] for i, V in enumerate(V_photodiodes)])
        index_max = np.argmax(V_corr)

        # V_corr = V_photodiodes
        # index_max = np.argmax(V_corr)

        # print(V_corr)

        if all(V < 0.1 for V in V_corr):
            return "inconnu", 0, 0
        elif index_max == 0:
            return "UV", 358, V_corr[0]/0.04
        elif index_max == 1:
            self.wavelength = np.mean(self.precise_wavelength(self.get_VIS_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "VIS", self.wavelength, self.get_VIS_power(self.wavelength, V_corr)
        elif index_max == 5:
            #self.estimate_power_from_row()
            self.wavelength = np.mean(self.precise_wavelength(self.get_IR_wavelength, V_corr[-1], self.puissance, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "IR", self.wavelength, self.get_IR_power()
        else:
            self.wavelength = np.mean(self.precise_wavelength(self.get_NIR_wavelength, V_corr, threshold=threshold, threshold_mult=threshold_mult)) + 200
            return "NIR", self.wavelength, self.get_NIR_power(self.wavelength, V_corr)



    def estimate_power_from_row(self, row, dt):
        """
        Estime la puissance thermique à partir d'une ligne de données (DataFrame row)
        et du pas de temps dt.
        """
        try:
            # 1. Extraction robuste des températures pertinentes
            # Sélectionne les noms des colonnes des thermistances (sauf R25 et R_Virtuel)
            temp_cols_names = [p[0] for p in self.positions if p[0].startswith('R') and p[0] not in ["R25", "R_Virtuel"]]

            # Vérifie quelles colonnes existent réellement dans la ligne 'row'
            valid_temp_cols = [col for col in temp_cols_names if col in row.index]

            if not valid_temp_cols:
                print("[ERREUR estimate_power] Aucune colonne de température valide trouvée dans la ligne.")
                temperatures = np.array([np.nan]) # Gérer le cas vide
            else:
                # Extrait les valeurs, convertit en float, et obtient un array NumPy
                # Utilise .values pour obtenir un array NumPy, plus sûr pour les opérations suivantes
                temperatures = row[valid_temp_cols].astype(float).values

            # 2. Calcul de T_max et delta_T avec gestion des NaN
            # Utilise np.nanmax pour ignorer les NaN lors de la recherche du max
            T_max = np.nanmax(temperatures) if temperatures.size > 0 else np.nan

            if pd.isna(T_max):
                print("[AVERTISSEMENT estimate_power] T_max est NaN, impossible de calculer delta_T.")
                delta_T = np.nan
            else:
                # Utilise la valeur de R25 de la ligne actuelle si elle existe et est valide, sinon 25.0
                T_ref = float(row["R25"]) if "R25" in row.index and pd.notna(row["R25"]) else 25.0
                delta_T = T_max - T_ref
                # print(f"[DEBUG estimate_power] T_max={T_max:.2f}, T_ref={T_ref:.2f}, delta_T={delta_T:.2f}") # Debug

            # 3. Vérification de delta_T avant les calculs PID
            if pd.isna(delta_T):
                print("[AVERTISSEMENT estimate_power] delta_T est NaN, saut du calcul PID.")
                # Optionnel: mettre la puissance à 0 ou garder la précédente
                # self.puissance = 0 # ou laisser tel quel
                return # Sortir de la fonction si delta_T est NaN

            # 4. Calculs PID (avec initialisation et gestion de l'historique)
            # Gains PID (à ajuster expérimentalement)
            kp = 0.56
            ki = -0.012
            kd = 12.0 # Mettre en float pour éviter les problèmes de type
            kdd = -4.0 # Mettre en float
            bias = -0.5

            # Initialisation de l'historique si nécessaire
            if not hasattr(self, "delta_T_hist"):
                self.delta_T_hist = []
            self.delta_T_hist.append(delta_T)

            # Limiter la taille de l'historique pour éviter une consommation mémoire excessive
            max_hist_len = 100
            if len(self.delta_T_hist) > max_hist_len:
                self.delta_T_hist.pop(0)

            # S'assurer qu'on a assez de points pour les calculs de gradient/filtrage
            if len(self.delta_T_hist) < 5: # Besoin d'au moins quelques points
                 print("[INFO estimate_power] Pas assez de données historiques pour le calcul PID complet.")
                 self.puissance = 0 # Ou une autre valeur par défaut
                 return

            # Filtrage et calculs différentiels/intégraux
            delta_T_array = np.array(self.delta_T_hist, dtype=float) # Assurer le type float

            # Filtre Savitzky-Golay (plus robuste que la moyenne mobile simple)
            # window_length doit être impair et <= len(delta_T_array)
            window_length = min(len(delta_T_array) // 2 * 2 + 1, 15) # Ex: max 15, mais s'adapte si moins de points
            polyorder = 3
            try:
                 # Utiliser mode='interp' pour gérer les bords
                 delta_T_filt = savgol_filter(delta_T_array, window_length, polyorder, mode='interp')
            except ValueError:
                 # Fallback si savgol échoue (ex: pas assez de points malgré la vérification)
                 print("[AVERTISSEMENT estimate_power] Échec du filtre Savitzky-Golay, utilisation des données brutes.")
                 delta_T_filt = delta_T_array

            # Calcul des dérivées et de l'intégrale sur les données filtrées
            # Utiliser np.gradient avec dt pour une meilleure précision
            d_delta_T_dt = np.gradient(delta_T_filt, dt)
            dd_delta_T_dt = np.gradient(d_delta_T_dt, dt)
            integral = np.cumsum(delta_T_filt) * dt # Intégrale cumulative

            # Calcul des termes PID (utiliser les dernières valeurs des tableaux)
            P = kp * delta_T_filt[-1]
            I = ki * integral[-1]
            D = kd * d_delta_T_dt[-1]
            DD = kdd * dd_delta_T_dt[-1]

            # Calcul de la puissance estimée
            estimated_power_pid = P + I + D + DD + bias
            # S'assurer que la puissance n'est pas négative
            estimated_power_pid = max(0, estimated_power_pid)

            # 5. Lissage final et mise à jour de self.puissance
            # Stocker l'estimation actuelle pour le lissage
            if not hasattr(self, "puissance_hist"):
                 self.puissance_hist = [] # Initialiser si besoin
            self.puissance_hist.append(estimated_power_pid)
            if len(self.puissance_hist) > max_hist_len: # Limiter aussi cet historique
                 self.puissance_hist.pop(0)

            # Lisser sur les N dernières estimations (ex: 10)
            smoothing_window = 10
            if len(self.puissance_hist) >= smoothing_window:
                puissance_lissée = np.mean(self.puissance_hist[-smoothing_window:])
            elif len(self.puissance_hist) > 0:
                 puissance_lissée = np.mean(self.puissance_hist) # Moyenne sur ce qu'on a
            else:
                 puissance_lissée = 0 # Si aucun historique

            # Mettre à jour la puissance de l'objet, en appliquant un seuil minimal si désiré
            min_power_threshold = 0.5
            if puissance_lissée > min_power_threshold:
                self.puissance = puissance_lissée
            else:
                self.puissance = 0

            # Stockage pour debug (si nécessaire, peut être commenté/supprimé)
            if not hasattr(self, "puissance_P"): self.puissance_P = []
            if not hasattr(self, "puissance_I"): self.puissance_I = []
            if not hasattr(self, "puissance_D"): self.puissance_D = []
            if not hasattr(self, "puissance_DD"): self.puissance_DD = []
            if not hasattr(self, "puissance_hist_2"): self.puissance_hist_2 = []

            self.puissance_P.append(P)
            self.puissance_I.append(I)
            self.puissance_D.append(D)
            self.puissance_DD.append(DD)
            self.puissance_hist_2.append(self.puissance) # Historique de la puissance finale lissée

            # print(f"[DEBUG estimate_power] P={P:.2f}, I={I:.2f}, D={D:.2f}, DD={DD:.2f}, Bias={bias:.2f} -> Est={estimated_power_pid:.2f}, Lissé={self.puissance:.2f}") # Debug

        except Exception as e:
            # Afficher une trace plus détaillée pour l'erreur
            import traceback
            print(f"[ERREUR] Estimation puissance échouée : {e}")
            traceback.print_exc() # Très utile pour voir la ligne exacte de l'erreur
            self.puissance = 0 # Mettre à 0 en cas d'erreur grave
            # Réinitialiser les historiques en cas d'erreur peut être une bonne idée
            if hasattr(self, "delta_T_hist"): self.delta_T_hist = []
            if hasattr(self, "puissance_hist"): self.puissance_hist = []




    
if __name__ == "__main__":
    td = TraitementDonnees(simulation=True)
    td.demarrer_acquisition_live(interval=0.1)
    #td.estimate_laser_power_from_csv()

    # type_lumiere, lambda_nm, puissance_corrigee = get_wavelength(
    #     position=position,
    #     V_photodiodes=V_photodiodes,
    #     puissance=puissance_estimee
    # )

    # print(f"Résultat : {type_lumiere} | λ = {lambda_nm:.1f} nm | Puissance corrigée = {puissance_corrigee:.2f} W")

    
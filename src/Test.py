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
        # Canaux 0 à 20 utilisés pour les thermistances R1-R11, R13-R21, R24(sur canal 11)
        self.indices_à_garder = list(range(21))
        self.simulation_data = None
        self.simulation_index = 0  # Te permet de décider à quel rang tu commences
        # Noms des colonnes attendues dans le CSV (basés sur self.positions et self.indices_à_garder)
        self.simulation_columns = [self.positions[i][0] for i in self.indices_à_garder]

        if self.simulation:
            self.ser = None
            print("[SIMULATION] Mode simulation activé.")
            try:
                # Chemin vers le fichier CSV relatif au script Test.py
                script_dir = Path(__file__).parent
                # Te permet de choisir quel fichier prendre
                simulation_file_path = script_dir.parent / "data" / "Hauteur 6.csv"
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
    def get_temperatures(self):
        if self.simulation:
            # --- Logique pour la simulation basée sur CSV ---
            if self.simulation_data is not None and not self.simulation_data.empty:
                if self.simulation_index >= len(self.simulation_data):
                    self.simulation_index = 0 # Recommencer au début du fichier

                # Récupérer la ligne actuelle du DataFrame
                current_data_row = self.simulation_data.iloc[self.simulation_index]
                self.simulation_index += 1

                # Créer le dictionnaire de températures directement depuis la ligne CSV
                temperature_dict = {}
                valid_data_found = False
                for i in self.indices_à_garder:
                    thermistor_name = self.positions[i][0]
                    if thermistor_name in current_data_row and pd.notna(current_data_row[thermistor_name]):
                        temperature_dict[thermistor_name] = current_data_row[thermistor_name]
                        valid_data_found = True
                    else:
                        # Gérer les données manquantes ou NaN pour ce thermistor dans cette ligne
                        # print(f"[AVERTISSEMENT SIMULATION] Donnée manquante/NaN pour {thermistor_name} à l'index CSV {self.simulation_index-1}")
                        temperature_dict[thermistor_name] = np.nan # Utiliser NaN pour indiquer l'absence de donnée valide

                if not valid_data_found:
                    print(f"[ERREUR SIMULATION] Aucune donnée de température valide trouvée à l'index CSV {self.simulation_index-1}.")
                    return None # Retourner None si la ligne entière est invalide

                return temperature_dict
            else:
                # --- Fallback: Si le CSV n'est pas chargé, générer des températures aléatoires ---
                print("[SIMULATION] Données CSV non disponibles, génération de températures aléatoires.")
                # Génère des températures aléatoires dans une plage plausible
                random_temps = {self.positions[i][0]: np.random.uniform(20.0, 45.0)
                                for i in self.indices_à_garder}
                return random_temps

        # --- Logique originale pour le mode non-simulation (lecture série) ---
        data_voltages = self.lire_donnees() # Lire les tensions depuis le port série
        if data_voltages is None:
            # lire_donnees a déjà affiché une erreur si nécessaire
            return None

        temperatures = []
        noms = [] # Garder une trace des noms dans le bon ordre

        for i in self.indices_à_garder:
            nom_thermistor = self.positions[i][0]
            noms.append(nom_thermistor)

            if i not in data_voltages:
                print(f"[AVERTISSEMENT] Tension manquante pour le canal {i} ({nom_thermistor})")
                temperatures.append(np.nan) # Ajouter NaN si la tension manque
                continue # Passer au canal suivant

            voltage = data_voltages[i]

            # Sélectionner les bons coefficients
            # Rappel: R24 (nom) est sur le canal 11 (index i) et utilise les coeffs[23]
            if i == 11: # Canal 11 correspond à R24 dans self.positions
                if 23 < len(self.coefficients):
                    coeffs = self.coefficients[23]
                else:
                    print(f"[ERREUR] Index de coefficient 23 hors limites pour R24 (canal 11).")
                    temperatures.append(np.nan)
                    continue
            else: # Pour tous les autres canaux dans indices_à_garder
                if i < len(self.coefficients):
                    coeffs = self.coefficients[i]
                else:
                    print(f"[ERREUR] Index de coefficient {i} hors limites pour {nom_thermistor}.")
                    temperatures.append(np.nan)
                    continue

            # Calculer résistance et température
            resistance = self.compute_resistance(voltage)
            if resistance == float('inf') or resistance <= 0: # Gérer résistance invalide
                # print(f"[AVERTISSEMENT] Résistance invalide ({resistance:.2f} Ω) calculée pour {nom_thermistor} (canal {i}) à partir de {voltage:.3f} V.")
                temp = np.nan
            else:
                try:
                    temp = self.compute_temperature(resistance, coeffs)
                except ValueError: # np.log peut échouer si R est <= 0
                    # print(f"[AVERTISSEMENT] Erreur de calcul de température pour {nom_thermistor} (R={resistance:.2f} Ω).")
                    temp = np.nan

            temperatures.append(temp)

        # Créer le dictionnaire final en associant les noms et les températures calculées
        # S'assurer que le nombre de noms et de températures correspond
        if len(noms) != len(temperatures):
            print("[ERREUR CRITIQUE] Discordance entre noms et températures calculées.")
            return None

        return dict(zip(noms, temperatures))



    # Dans la classe TraitementDonnees (fichier Test.py)

    def afficher_heatmap_dans_figure(self, temperature_dict, fig, elapsed_time):
        fig.clear()
        ax = fig.add_subplot(111)

        x_orig, y_orig, t_orig = [], [], []
        valid_temps_list = [] # Pour calculer la moyenne

        # 1. Extraire les données valides des thermistances réelles
        for i in self.indices_à_garder:
            name, pos = self.positions[i]
            temp_val = temperature_dict.get(name, np.nan) # Utiliser .get pour plus de sûreté
            if pd.notna(temp_val): # Ignorer les NaN pour l'interpolation et la moyenne
                x_orig.append(pos[0])
                y_orig.append(pos[1])
                t_orig.append(temp_val)
                valid_temps_list.append(temp_val)
            # else: # Optionnel: si tu veux afficher les points NaN différemment
            #     ax.scatter(pos[0], pos[1], color='gray', marker='x', s=30, label='NaN' if name == self.positions[self.indices_à_garder[0]][0] else "") # Marquer les NaN

        # 2. Calculer la température cible pour les bords
        if not valid_temps_list:
            print("[AVERTISSEMENT] Aucune donnée de température valide pour calculer la moyenne des bords.")
            # Comportement par défaut : utiliser une température fixe ou ne pas ajouter de bords
            target_edge_temp = 20.0 # Exemple: température par défaut pour les bords
            # Ou tu pourrais choisir de ne pas ajouter de points de bord et retourner
            # ax.set_title("Données invalides pour la heatmap")
            # return
        else:
            avg_temp = np.mean(valid_temps_list)
            target_edge_temp = avg_temp - 1.0
            print(f"[INFO HEATMAP] Température moyenne: {avg_temp:.2f}°C, Température des bords: {target_edge_temp:.2f}°C")


        # 3. Définir les points virtuels sur le périmètre
        r_max = 12.5 # Le rayon de ta carte thermique
        num_edge_points = 12 # Plus de points pour une meilleure contrainte (ajustable)
        edge_angles = np.linspace(0, 2 * np.pi, num_edge_points, endpoint=False)
        edge_x = r_max * np.cos(edge_angles)
        edge_y = r_max * np.sin(edge_angles)
        edge_t = [target_edge_temp] * num_edge_points # Tous les points de bord ont la température cible

        # 4. Combiner les points réels et les points de bord
        x_combined = x_orig + list(edge_x)
        y_combined = y_orig + list(edge_y)
        t_combined = t_orig + edge_t

        # 5. Vérifier s'il y a assez de points pour l'interpolation
        # Rbf a besoin d'au moins N+1 points en N dimensions (ici 2D, donc au moins 3 points)
        if len(x_combined) < 3:
            print("[ERREUR HEATMAP] Pas assez de points valides (réels + bords) pour l'interpolation.")
            ax.set_title("Pas assez de données pour la heatmap")
            ax.set_xlabel("X (mm)")
            ax.set_ylabel("Y (mm)")
            # Afficher les points originaux s'il y en a
            if x_orig:
                ax.scatter(x_orig, y_orig, color='black', marker='o', s=25)
            return # Quitter la fonction si pas assez de points

        # 6. Créer l'interpolation RBF avec les données combinées
        # smooth=0.5 est une valeur de départ, tu peux l'ajuster pour plus ou moins de lissage
        # epsilon pourrait aussi être ajusté selon la fonction RBF choisie
        rbf = Rbf(x_combined, y_combined, t_combined, function='multiquadric', smooth=0.5)
        grid_size = 200

        # 7. Définir la grille et masquer l'extérieur du cercle
        # Utiliser r_max pour la grille et le masque pour correspondre aux points de bord
        xi, yi = np.meshgrid(
            np.linspace(-r_max, r_max, grid_size),
            np.linspace(-r_max, r_max, grid_size)
        )

        ti = rbf(xi, yi)
        mask = xi**2 + yi**2 > r_max**2 # Masque basé sur le rayon où les points de bord ont été placés
        ti_masked = np.ma.array(ti, mask=mask)
            # 8. Appliquer un filtre Gaussien aux données interpolées (non masquées)
        #    Le paramètre sigma contrôle le lissage. Ajuste-le si nécessaire (ex: 1, 2, 3...).
        #    mode='nearest' gère les bords de manière raisonnable.
        sigma_filtre = 2 # A ajuster selon le niveau de lissage souhaité
        ti_filtered = gaussian_filter(ti, sigma=sigma_filtre, mode='nearest')

        # 9. Appliquer le masque aux données filtrées pour chercher le max DANS le cercle
        ti_filtered_masked = np.ma.array(ti_filtered, mask=mask)

        # 10. Trouver l'index du maximum dans la grille filtrée et masquée
        try:
            # argmax sur l'array masqué retourne l'index plat du max non masqué
            max_idx_flat = np.argmax(ti_filtered_masked)
            # Convertir l'index plat en indices 2D
            max_idx_2d = np.unravel_index(max_idx_flat, ti.shape)
            # Récupérer les coordonnées (x, y) correspondantes depuis la grille
            max_x = xi[max_idx_2d]
            max_y = yi[max_idx_2d]
            max_temp_val = ti_filtered_masked[max_idx_2d] # Température max (lissée)
            point_max_trouve = True
            print(f"[INFO LASER] Point max détecté (après filtre) à ({max_x:.2f}, {max_y:.2f}) avec T={max_temp_val:.2f}°C")
        except (ValueError, IndexError):
            # Gérer le cas où toutes les valeurs sont masquées ou autres erreurs
            print("[AVERTISSEMENT] Impossible de trouver le point maximum sur la grille filtrée.")
            point_max_trouve = False
        # 11. Afficher la heatmap et les points originaux
        contour = ax.contourf(xi, yi, ti_masked, levels=100, cmap="plasma") # levels=100 pour un dégradé lisse
        fig.colorbar(contour, ax=ax, label="Température (°C)")
        ax.scatter(x_orig, y_orig, color='black', marker='o', s=25, label='Thermistances') # Afficher seulement les points réels
        
        # 12 Annoter seulement les points réels
        for i in range(len(x_orig)):
            # Trouver le nom correspondant à x_orig[i], y_orig[i] peut être un peu complexe
            # On peut le faire en retrouvant l'index original ou en cherchant par position
            # Ici, on suppose que l'ordre de x_orig correspond à l'ordre des thermistances valides
            # C'est plus simple si on itère sur les positions originales et on vérifie la validité
            original_index_in_positions = -1
            for k in self.indices_à_garder:
                if self.positions[k][1] == (x_orig[i], y_orig[i]):
                    original_index_in_positions = k
                    break
            if original_index_in_positions != -1:
                name = self.positions[original_index_in_positions][0]
                ax.annotate(name, (x_orig[i], y_orig[i]), textcoords="offset points", xytext=(4, 4), ha='left', fontsize=8)
        # 13. Si un point maximum a été trouvé, l'afficher sur le graphique en vert
        if point_max_trouve:
            # Utiliser un cercle vert ('go') ou une étoile verte ('g*')
            ax.plot(max_x, max_y, 'go', markersize=10, label=f'Laser estimé @ ({max_x:.1f}, {max_y:.1f})')

        ax.set_aspect('equal')
        ax.set_title(f"Map de chaleur (Tps: {elapsed_time:.2f} s) - Bords ajustés")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_xlim(-r_max - 1, r_max + 1) # Légère marge pour la visualisation
        ax.set_ylim(-r_max - 1, r_max + 1)
        # ax.legend() # Décommenter si tu veux une légende pour les points NaN/réels
        fig.tight_layout()

# --- Assure-toi que le reste de ta classe et l'appel à cette fonction restent corrects ---



    def demarrer_acquisition_live(self, interval=0.2):
        if not self.est_connecte() and not self.simulation:
            print("Arduino non connecté.")
            return

        print("🚀 Acquisition live en cours... (Ctrl+C pour arrêter)")
        fig = plt.figure(figsize=(6, 6))
        plt.ion()
        fig.show()

        all_data = []
        headers = [self.positions[i][0] for i in self.indices_à_garder] + ["T_ref", "timestamp"]
        start_time = time.time()
        try:
            while True:
                current_time = time.time()
                elapsed_time = current_time - start_time
                data = self.get_temperatures()

                if data:
                    os.system("clear")
                    print("=" * 60)
                    print(f"⏱️ Temps écoulé: {elapsed_time:.2f} secondes")
                    print("-" * 60)
                    print("Températures mesurées")
                    print("-" * 60)
                    for name, temp in data.items():
                        print(f"{name:<6} : {temp:6.2f} °C")
                    print("=" * 60)

                    self.afficher_heatmap_dans_figure(data, fig, elapsed_time)
                    fig.canvas.draw()
                    fig.canvas.flush_events()

                    ligne = [data[name] for name in data]
                    ligne.append(25.0)
                    ligne.append(datetime.now().isoformat(timespec='seconds'))
                    all_data.append(ligne)

                else:
                    print("⚠️ Données incomplètes ou non reçues.")

                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Acquisition stoppée. Sauvegarde du fichier CSV...")

            desktop_path = Path.home() / "Desktop"
            filename = f"acquisition_thermistances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = desktop_path / filename

            with open(csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(all_data)

            print(f"✅ Données sauvegardées dans : {csv_path}")


if __name__ == "__main__":
    td = TraitementDonnees(simulation=True)
    td.demarrer_acquisition_live(interval=0.05)


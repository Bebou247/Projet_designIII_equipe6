from mytk import *
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Traitement_données import *
import numpy as np
import csv
from datetime import datetime, timedelta
import threading
import time
import os

class InterfaceGraphique(App):
    def __init__(self):
        super().__init__(name="Acquisition Laser", geometry="1400x1000")
        
        self.acquisition_active = False
        self.fichier_csv = None
        self.writer_csv = None
        self.donnees = {
            'temps': [],
            'puissances': [],
            'temperatures': [],
            'positions': [],
            'longueurs_onde': []
        }

        # Initialisation de la gestion des simulations
        self.dossier_essais = "Essais"
        if not os.path.exists(self.dossier_essais):
            os.makedirs(self.dossier_essais)
        
        self.main_frame = ttk.Frame(self.window.widget)
        self.main_frame.pack(expand=True, fill='both', padx=20, pady=20)
        self.main_frame.grid_columnconfigure(0, weight=4)
        self.main_frame.grid_columnconfigure(1, weight=6)
        self.main_frame.grid_rowconfigure(0, weight=4)
        self.main_frame.grid_rowconfigure(1, weight=6)

        self.creer_panneau_controle()
        self.creer_visualisation()

    def creer_panneau_controle(self):
        frame = ttk.Frame(self.main_frame, padding=15)
        frame.grid(row=0, column=0, rowspan=2, sticky='nsew')
        
        ttk.Label(frame, text="Contrôle Acquisition", font=('Arial', 14, 'bold')).pack(pady=10)
        self.btn_play = ttk.Button(frame, text="▶ Démarrer", command=self.toggle_acquisition)
        self.btn_play.pack(pady=5, fill='x')
        
        ttk.Button(frame, text="Sélectionner simulation", command=self.choisir_fichier_simulation).pack(pady=5)
        self.label_fichier = ttk.Label(frame, text="Aucun fichier sélectionné")
        self.label_fichier.pack(pady=2)

        ttk.Separator(frame).pack(fill='x', pady=10)
        self.labels = {
            'duree': ttk.Label(frame, text="Durée: 00:00:00"),
            'echantillons': ttk.Label(frame, text="Échantillons: 0"),
            'puissance': ttk.Label(frame, text="Puissance: 0.0 W"),
            'puissance_moy': ttk.Label(frame, text="Puissance moy: 0.0 W"),
            'temperature': ttk.Label(frame, text="Temp. max: 0.0°C"),
            'longueur_onde': ttk.Label(frame, text="Longueur d'onde: 0 nm"),
            'centre': ttk.Label(frame, text="Centre: (0.00, 0.00) mm")
        }
        for lbl in self.labels.values():
            lbl.pack(pady=2, anchor='w')

    def creer_visualisation(self):
        visual_frame = ttk.Frame(self.main_frame)
        visual_frame.grid(row=0, column=1, rowspan=2, sticky='nsew')
        
        visual_frame.grid_rowconfigure(0, weight=5)
        visual_frame.grid_rowconfigure(1, weight=5)
        visual_frame.grid_columnconfigure(0, weight=1)

        self.fig_heat = plt.Figure(figsize=(10, 6))
        self.ax_heat = self.fig_heat.add_subplot(111)
        self.canvas_heat = FigureCanvasTkAgg(self.fig_heat, visual_frame)
        self.canvas_heat.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        self.fig_puiss = plt.Figure(figsize=(10, 4))
        self.ax_puiss = self.fig_puiss.add_subplot(111)
        self.line_puiss, = self.ax_puiss.plot([], [], 'b-')
        self.ax_puiss.set_ylim(0, 10)
        self.ax_puiss.set_title("Puissance en temps réel")
        self.ax_puiss.grid(True)
        self.canvas_puiss = FigureCanvasTkAgg(self.fig_puiss, visual_frame)
        self.canvas_puiss.get_tk_widget().grid(row=1, column=0, sticky='nsew')

    def toggle_acquisition(self):
        self.acquisition_active = not self.acquisition_active
        
        if self.acquisition_active:
            self.btn_play.config(text="⏹ Stopper")
            self.demarrer_acquisition()
        else:
            self.btn_play.config(text="▶ Démarrer")
            self.arreter_acquisition()

    def demarrer_acquisition(self):
        # Créer un dossier pour l'essai actuel (essaie_1, essaie_2, etc.)
        essais_existants = [f for f in os.listdir(self.dossier_essais) if f.startswith("essai_")]
        num_essai = len(essais_existants) + 1
        dossier_essai = os.path.join(self.dossier_essais, f"essai_{num_essai}")
        os.makedirs(dossier_essai)

        # Créer un fichier CSV dans le dossier de l'essai
        self.fichier_csv = os.path.join(dossier_essai, f"donnees_{num_essai}.csv")
        self.fichier = open(self.fichier_csv, 'w', newline='')
        self.writer_csv = csv.writer(self.fichier, delimiter=';')
        self.writer_csv.writerow([
            'Timestamp', 'Position X (mm)', 'Position Y (mm)', 
            'Puissance (W)', 'Puissance moyenne (W)', 'Température max (°C)',
            'Longueur d\'onde (nm)'
        ])

        # Lancement de l'acquisition
        thread = threading.Thread(target=self.boucle_acquisition, daemon=True)
        thread.start()

    def boucle_acquisition(self):
        start_time = datetime.now()
        puissance_moyenne = 5.0  # Puissance moyenne de 5W
        
        while self.acquisition_active:
            temps_actuel = datetime.now()
            
            # Simulation de la puissance fluctuante autour de 5W
            puissance = np.random.normal(puissance_moyenne, 0.2)  # Écart-type de 0.2W
            puissance = max(0, min(10, puissance))  # Limite entre 0 et 10W
            
            # Simulation des autres paramètres
            adc_values = generate_realistic_adc()
            temperatures = [adc_to_temperature(adc) for adc in adc_values]
            Zi, x0, y0, im = create_or_update_heatmap(temperatures, self.ax_heat)
            longueur_onde = estimate_laser_wavelength(temperatures)
            
            self.donnees['temps'].append(temps_actuel)
            self.donnees['puissances'].append(puissance)
            self.donnees['temperatures'].append(np.max(temperatures))
            self.donnees['positions'].append((x0, y0))
            self.donnees['longueurs_onde'].append(longueur_onde)
            
            puissance_moyenne_10s = np.mean(self.donnees['puissances'][-10:])
            
            self.writer_csv.writerow([
                temps_actuel.strftime("%Y-%m-%d %H:%M:%S"),
                x0,
                y0,
                puissance,
                puissance_moyenne_10s,
                np.max(temperatures),
                longueur_onde
            ])
            
            self.mettre_a_jour_interface(temps_actuel - start_time, puissance, puissance_moyenne_10s, x0, y0, im)
            
            # Attendre jusqu'à la prochaine seconde
            prochaine_seconde = (temps_actuel + timedelta(seconds=1)).replace(microsecond=0)
            temps_attente = (prochaine_seconde - datetime.now()).total_seconds()
            if temps_attente > 0:
                time.sleep(temps_attente)

    def mettre_a_jour_interface(self, duree, puissance, puissance_moyenne, x0, y0, im):
        self.canvas_heat.draw()
        
        self.line_puiss.set_data(
            range(len(self.donnees['puissances'])), 
            self.donnees['puissances']
        )
        self.ax_puiss.relim()
        self.ax_puiss.autoscale_view(True, True, True)
        self.canvas_puiss.draw()
        
        self.labels['duree'].config(text=f"Durée: {str(duree).split('.')[0]}")
        self.labels['echantillons'].config(text=f"Échantillons: {len(self.donnees['puissances'])}")
        self.labels['puissance'].config(text=f"Puissance: {puissance:.2f} W")
        self.labels['puissance_moy'].config(text=f"Puissance moy: {puissance_moyenne:.2f} W")
        self.labels['temperature'].config(text=f"Temp. max: {np.max(self.donnees['temperatures']):.1f}°C")
        self.labels['longueur_onde'].config(text=f"Longueur d'onde: {self.donnees['longueurs_onde'][-1]:.0f} nm")
        self.labels['centre'].config(text=f"Centre: ({x0:.2f}, {y0:.2f}) mm")

        # Mise à jour de la colorbar
        if not hasattr(self, 'colorbar'):
            self.colorbar = self.fig_heat.colorbar(im, ax=self.ax_heat)
        else:
            self.colorbar.update_normal(im)

    def choisir_fichier_simulation(self):
        # Ouvrir une boîte de dialogue pour sélectionner un fichier
        chemin_fichier = filedialog.askopenfilename(title="Sélectionner un fichier de simulation", filetypes=[("Fichiers CSV", "*.csv"), ("Tous les fichiers", "*.*")])
        
        if chemin_fichier:
            self.label_fichier.config(text=f"Fichier sélectionné: {chemin_fichier}")
            # Charger et afficher les données du fichier sélectionné
            self.charger_simulation(chemin_fichier)

    def charger_simulation(self, chemin_fichier):
        # Charger le fichier CSV et afficher les données
        self.donnees_simulation = {'temps': [], 'puissances': [], 'temperatures': [], 'positions': [], 'longueurs_onde': []}
        
        with open(chemin_fichier, 'r') as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                self.donnees_simulation['temps'].append(row['Timestamp'])
                self.donnees_simulation['puissances'].append(float(row['Puissance (W)']))
                self.donnees_simulation['temperatures'].append(float(row['Température max (°C)']))
                self.donnees_simulation['positions'].append((float(row['Position X (mm)']), float(row['Position Y (mm)'])))
                self.donnees_simulation['longueurs_onde'].append(float(row['Longueur d\'onde (nm)']))
        
        # Afficher les graphes associés
        self.mettre_a_jour_interface_simulation()

    def mettre_a_jour_interface_simulation(self):
        # Utiliser les données du fichier pour mettre à jour les graphes
        temps = self.donnees_simulation['temps']
        puissances = self.donnees_simulation['puissances']
        self.line_puiss.set_data(range(len(puissances)), puissances)
        self.ax_puiss.relim()
        self.ax_puiss.autoscale_view(True, True, True)
        self.canvas_puiss.draw()

        # Update heatmap simulation (impossible à faire sans plus de données, mais à adapter)
        self.canvas_heat.draw()

    def arreter_acquisition(self):
        if self.fichier_csv:
            self.fichier.close()
            
        self.donnees = {k: [] for k in self.donnees}  # Réinitialisation

if __name__ == "__main__":
    app = InterfaceGraphique()
    app.mainloop()

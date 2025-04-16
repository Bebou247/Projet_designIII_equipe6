from mytk import App
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from Traitements_de_données import *
import time
import csv
from pathlib import Path
import pandas as pd

class MyApp(App):
    def __init__(self):
        super().__init__(name="Interface puissance-mètre")
        self.window.widget.attributes("-fullscreen", True)
        self.window.widget.bind("<Escape>", lambda e: self.window.widget.attributes("-fullscreen", False))

        try:
            self.td = TraitementDonnees(simulation=False)
            if not self.td.est_connecte():
                raise Exception("Arduino non détecté")
        except:
            print("[INFO] Lancement en mode simulation")
            self.td = TraitementDonnees(simulation=True)

        self.running = False
        self.donnees_enregistrées = []

        self.frame = ttk.Frame(self.window.widget, padding=10)
        self.frame.pack(expand=True, fill='both')
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=3)

        self.frame_gauche = ttk.Frame(self.frame)
        self.frame_gauche.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        self.frame_puissance = ttk.LabelFrame(self.frame_gauche, text=" Puissance estimée (W)")
        self.frame_puissance.pack(fill="x", pady=5)
        self.label_puissance = ttk.Label(self.frame_puissance, text="-- W", font=("Helvetica", 16, "bold"))
        self.label_puissance.pack()

        self.frame_lambda = ttk.LabelFrame(self.frame_gauche, text=" Longueur d’onde estimée (nm)")
        self.frame_lambda.pack(fill="x", pady=5)
        self.label_lambda = ttk.Label(self.frame_lambda, text="-- nm", font=("Helvetica", 16, "bold"))
        self.label_lambda.pack()

        self.frame_logs = ttk.LabelFrame(self.frame_gauche, text="📜 Logs système")
        self.frame_logs.pack(fill="both", expand=True, pady=10)
        self.text_logs = tk.Text(self.frame_logs, height=15, font=("Courier", 10))
        self.text_logs.pack(fill="both", expand=True)

        self.frame_droite = ttk.Frame(self.frame)
        self.frame_droite.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.frame_droite.rowconfigure(0, weight=1)
        self.frame_droite.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_droite)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        boutons_frame = ttk.Frame(self.frame_droite)
        boutons_frame.grid(row=1, column=0, pady=10)
        ttk.Button(boutons_frame, text="▶ Démarrer Live", command=self.demarrer_live).pack(side="left", padx=10)
        ttk.Button(boutons_frame, text="⏹ Arrêter", command=self.arreter_live).pack(side="left", padx=10)

        self.label_etat = ttk.Label(self.frame_droite, text="", foreground="red")
        self.label_etat.grid(row=2, column=0, pady=5)

        self.maj_etat_connection()
        self.check_connection_loop()

    def log(self, message):
        horodatage = time.strftime("[%H:%M:%S]")
        self.text_logs.insert(tk.END, f"{horodatage} {message}\n")
        self.text_logs.see(tk.END)

    def maj_etat_connection(self):
        if not self.td.est_connecte():
            self.label_etat.config(text="Arduino non connecté", foreground="red")
            self.log("Arduino non connecté =-(")
        else:
            self.label_etat.config(text=" Prêt", foreground="green")
            self.log(" Arduino connecté")

    def demarrer_live(self):
        if not self.td.est_connecte() and not self.td.simulation:
            self.label_etat.config(text="Aucun Arduino détecté.", foreground="red")
            self.log("❌ Échec : Arduino non détecté.")
            return

        if not self.running:
            self.running = True
            self.label_etat.config(text="Lecture en cours.", foreground="blue")
            self.log("▶ Acquisition démarrée")
            self.donnees_enregistrées.clear()
            self.mettre_a_jour_interface()

    def arreter_live(self):
        self.running = False
        self.label_etat.config(text="⏹ Lecture arrêtée.", foreground="gray")
        self.log("⏹ Acquisition arrêtée")

        if self.donnees_enregistrées:
            noms = [self.td.positions[i][0] if i != 24 else "R25" for i in self.td.indices_à_garder]
            photodiode_headers = [f"PD{i}" for i in self.td.canaux_photodiodes]
            headers = noms + photodiode_headers + ["Puissance (W)", "Longueur d'onde (nm)", "Timestamp"]

            desktop = Path.home() / "Desktop"
            filename = f"donnees_thermistances_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            chemin_fichier = desktop / filename

            try:
                with open(chemin_fichier, mode='w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows(self.donnees_enregistrées)

                self.log(f"📁 Données sauvegardées : {chemin_fichier}")
            except Exception as e:
                self.log(f"⚠️ Erreur de sauvegarde : {e}")

    def mettre_a_jour_interface(self):
        if self.running and (self.td.est_connecte() or self.td.simulation):
            data_raw = self.td.lire_donnees()
            if data_raw:
                temp_dict = self.td.get_temperatures(data_raw)
                if temp_dict:
                    self.td.afficher_heatmap_dans_figure(temp_dict, self.fig)
                    self.canvas.draw()

                    puissance = self.td.estimate_laser_power(25, max(temp_dict.values()), 3, (0, 0))
                    lambda_nm = self.td.estimate_laser_wavelength() if hasattr(self.td, 'estimate_laser_wavelength') else 0.0

                    self.label_puissance.config(text=f"{puissance:.2f} W")
                    self.label_lambda.config(text=f"{lambda_nm:.1f} nm")

                    ligne = [temp_dict.get(pos[0], "--") for pos in self.td.positions[:21]]
                    ligne += [data_raw.get(i, "--") for i in self.td.canaux_photodiodes]
                    ligne += [puissance, lambda_nm, time.strftime('%Y-%m-%d %H:%M:%S')]
                    self.donnees_enregistrées.append(ligne)

            self.window.widget.after(200, self.mettre_a_jour_interface)
        elif self.running:
            self.label_etat.config(text="Arduino déconnecté.", foreground="red")
            self.log("❌ Arduino déconnecté")

    def check_connection_loop(self):
        if not self.td.est_connecte() and not self.td.simulation:
            try:
                self.td = TraitementDonnees(simulation=False)
                if self.td.est_connecte():
                    self.label_etat.config(text="✅ Arduino reconnecté", foreground="green")
                    self.log("🔌 Arduino reconnecté")
            except Exception:
                pass
        self.window.widget.after(3000, self.check_connection_loop)

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()




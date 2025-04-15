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
        super().__init__(name="Interface puissance-mètre", geometry="900x600+100+100")

        try:
            self.td = TraitementDonnees(simulation=False)
            if not self.td.est_connecte():
                raise Exception("Arduino non détecté")
        except:
            print("[INFO] Lancement en mode simulation")
            self.td = TraitementDonnees(simulation=True)

        self.running = False
        self.donnees_enregistrées = []  # Liste pour stocker les données

        # Conteneur général
        self.frame = ttk.Frame(self.window.widget, padding=10)
        self.frame.pack(expand=True, fill='both')
        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=3)

        # Colonne gauche : données et logs
        self.frame_gauche = ttk.Frame(self.frame)
        self.frame_gauche.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.frame_puissance = ttk.LabelFrame(self.frame_gauche, text="Puissance estimée (W)")
        self.frame_puissance.pack(fill="x", pady=5)
        self.label_puissance = ttk.Label(self.frame_puissance, text="-- W", font=("Helvetica", 12))
        self.label_puissance.pack()

        self.frame_lambda = ttk.LabelFrame(self.frame_gauche, text="🌈 Longueur d’onde estimée (nm)")
        self.frame_lambda.pack(fill="x", pady=5)
        self.label_lambda = ttk.Label(self.frame_lambda, text="-- nm", font=("Helvetica", 12))
        self.label_lambda.pack()

        self.frame_logs = ttk.LabelFrame(self.frame_gauche, text="📜 Logs")
        self.frame_logs.pack(fill="both", expand=True, pady=5)
        self.text_logs = tk.Text(self.frame_logs, height=10)
        self.text_logs.pack(fill="both", expand=True)

        # Colonne droite : heatmap
        self.frame_droite = ttk.Frame(self.frame)
        self.frame_droite.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_droite)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        boutons_frame = ttk.Frame(self.frame_droite)
        boutons_frame.pack(pady=10)
        bouton_start = ttk.Button(boutons_frame, text="▶ Démarrer Live", command=self.demarrer_live)
        bouton_start.pack(side="left", padx=10)
        bouton_stop = ttk.Button(boutons_frame, text="⏹ Arrêter", command=self.arreter_live)
        bouton_stop.pack(side="left", padx=10)

        self.label_etat = ttk.Label(self.frame_droite, text="", foreground="red")
        self.label_etat.pack(pady=5)

        self.maj_etat_connection()
        self.check_connection_loop()

    def maj_etat_connection(self):
        if not self.td.est_connecte():
            self.label_etat.config(text="Arduino non connecté", foreground="red")
        else:
            self.label_etat.config(text="✅ Prêt", foreground="green")

    def demarrer_live(self):
        if not self.td.est_connecte() and not self.td.simulation:
            self.label_etat.config(text="Aucun Arduino détecté.", foreground="red")
            return

        if not self.running:
            self.running = True
            self.label_etat.config(text="Lecture en cours.", foreground="blue")
            self.donnees_enregistrées.clear()
            self.mettre_a_jour_interface()

    def arreter_live(self):
        self.running = False
        self.label_etat.config(text="⏹ Lecture arrêtée.", foreground="gray")

        if self.donnees_enregistrées:
            headers = [pos[0] for pos in self.td.positions[:21]] + ["Puissance (W)", "Longueur d'onde (nm)", "Timestamp"]
            desktop = Path.home() / "Desktop"
            filename = f"donnees_thermistances_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            chemin_fichier = desktop / filename

            with open(chemin_fichier, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
                writer.writerows(self.donnees_enregistrées)

            print(f"Données sauvegardées dans : {chemin_fichier}")
            self.text_logs.insert(tk.END, f"\n📁 Données sauvegardées : {chemin_fichier}\n")
            self.text_logs.see(tk.END)

    def mettre_a_jour_interface(self):
        if self.running and (self.td.est_connecte() or self.td.simulation):
            temperature_dict = self.td.get_temperatures()
            if temperature_dict:
                self.td.afficher_heatmap_dans_figure(temperature_dict, self.fig)

                self.canvas.draw()
                self.fig.canvas.draw_idle() 

                puissance = self.td.estimer_puissance()
                self.label_puissance.config(text=f"{puissance:.2f} W")

                lambda_nm = self.td.estimate_laser_wavelength()
                self.label_lambda.config(text=f"{lambda_nm:.1f} nm")

                t_max = max(temperature_dict.values())
                self.text_logs.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] T max: {t_max:.2f}°C | P: {puissance:.2f} W | λ: {lambda_nm:.1f} nm\n")
                self.text_logs.see(tk.END)

                ligne = [temperature_dict.get(pos[0], "--") for pos in self.td.positions[:21]]
                ligne += [puissance, lambda_nm, time.strftime('%Y-%m-%d %H:%M:%S')]
                self.donnees_enregistrées.append(ligne)

            # Relance la mise à jour dans 200 ms (0.2 sec)
            self.window.widget.after(200, self.mettre_a_jour_interface)

        elif self.running:
            self.label_etat.config(text="Arduino déconnecté.", foreground="red")

    def check_connection_loop(self):
        if not self.td.est_connecte() and not self.td.simulation:
            try:
                self.td = TraitementDonnees(simulation=False)
                if self.td.est_connecte():
                    print("[INFO] Arduino reconnecté !")
                    self.label_etat.config(text="✅ Arduino reconnecté", foreground="green")
            except Exception:
                pass
        self.window.widget.after(3000, self.check_connection_loop)

if __name__ == "__main__":
    app = MyApp()
    app.mainloop()



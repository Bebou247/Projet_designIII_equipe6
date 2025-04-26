from mytk import App
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from Traitements_de_données import *
import time
import csv
from pathlib import Path
import pandas as pd
from PIL import ImageTk, Image


class MyApp(App):
    def __init__(self):
        super().__init__(name="Interface puissance-mètre")
        self.window.widget.attributes("-fullscreen", True)
        self.window.widget.bind("<Escape>", lambda e: self.window.widget.attributes("-fullscreen", False))

        self.dossier_sauvegarde = Path("data/saves")
        self.dossier_sauvegarde.mkdir(parents=True, exist_ok=True)

        self.td = None
        self.simulation_mode = True
        self.fichier_simulation = None
        self.start_time = None
        self.running = False
        self.simulation_completee = False
        self.donnees_enregistrées = []
        self.premiere_lecture = True
        self.puissance_display = []
        self.temps_display = []

        self.build_interface()

        if self.arduino_disponible():
            self.simulation_mode = False
            self.td = TraitementDonnees(simulation=False, path="data/")
            self.log_mode = "Arduino détecté. Mode acquisition live."
        else:
            self.simulation_mode = True
            self.log_mode = "Arduino non détecté. Mode simulation."
            self.label_etat.config(text=self.log_mode, foreground="orange")
            self.after(100, self.choisir_csv_interface)

        self.maj_etat_connexion()
        self.check_connection_loop()
        self.log(self.log_mode)

    def log(self, message):
        horodatage = time.strftime("[%H:%M:%S]")
        self.text_logs.config(state="normal")
        self.text_logs.insert(tk.END, f"{horodatage} {message}\n")
        self.text_logs.config(state="disabled")
        self.text_logs.see(tk.END)

    def arduino_disponible(self):
        try:
            test = TraitementDonnees(simulation=False, path="data/")
            return test.est_connecte()
        except:
            return False

    def maj_etat_connexion(self):
        if not self.td or not self.td.est_connecte():
            self.label_etat.config(text="Arduino non connecté", foreground="red")
        else:
            self.label_etat.config(text=" Prêt", foreground="green")

    def choisir_csv_interface(self):
        fichier = filedialog.askopenfilename(initialdir=self.dossier_sauvegarde, title="Choisir un fichier CSV", filetypes=[("Fichiers CSV", "*.csv")])
        if fichier:
            self.rejouer_simulation(fichier)
        else:
            self.label_etat.config(text="Aucun fichier sélectionné.", foreground="red")
            self.log("Aucun fichier sélectionné.")

    def build_interface(self):
        self.frame = ttk.Frame(self.window.widget, padding=10)
        self.frame.pack(expand=True, fill='both')

        self.frame.columnconfigure(0, weight=1)
        self.frame.columnconfigure(1, weight=3)
        self.frame.rowconfigure(0, weight=1)
        self.frame.rowconfigure(1, weight=1)

        self.frame_h_gauche = ttk.Frame(self.frame)
        self.frame_h_gauche.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.frame_h_droite = ttk.Frame(self.frame)
        self.frame_h_droite.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.frame_b_gauche = ttk.Frame(self.frame)
        self.frame_b_gauche.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.frame_b_droite = ttk.Frame(self.frame)
        self.frame_b_droite.grid(row=1, column=1, sticky="nsew", padx=10, pady=10)

        boutons_frame = ttk.Frame(self.frame_b_droite)
        boutons_frame.pack(pady=10, side="left")

        self.label_etat = ttk.Label(boutons_frame, text="", font=("Helvetica", 18, "bold"), width=20, anchor="center")
        self.label_etat.pack(pady=10)
        self.bouton_start = ttk.Button(boutons_frame, text="▶ Commencer", command=self.reprendre_simulation, width=16)
        self.bouton_start.pack(pady=10)
        self.bouton_stop = ttk.Button(boutons_frame, text="⏹ Arrêter", command=self.arreter_live, width=16)
        self.bouton_stop.pack(pady=10)
        self.bouton_csv = ttk.Button(boutons_frame, text="📂 Charger CSV", command=self.choisir_csv_interface, width=16)
        self.bouton_csv.pack(pady=10)

        self.frame_puissance = ttk.LabelFrame(self.frame_b_gauche, text="Puissance estimée (W)")
        self.frame_puissance.pack(fill="x", padx=5, pady=5)
        self.label_puissance = ttk.Label(self.frame_puissance, text="-- W", font=("Helvetica", 32, "bold"))
        self.label_puissance.pack()

        self.frame_lambda = ttk.LabelFrame(self.frame_b_gauche, text="Longueur d’onde estimée (nm)")
        self.frame_lambda.pack(fill="x", padx=5, pady=5)
        self.label_lambda = ttk.Label(self.frame_lambda, text="-- nm", font=("Helvetica", 32, "bold"))
        self.label_lambda.pack()

        self.frame_logs = ttk.LabelFrame(self.frame_b_gauche, text="Logs système")
        self.frame_logs.pack(fill="both", padx=5, pady=10)
        self.text_logs = tk.Text(self.frame_logs, height=12, font=("Courier", 16), state="disabled")
        self.text_logs.pack(fill="both")

        self.fig = Figure(figsize=(6.2, 4.3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_h_droite)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.fig_2 = Figure(figsize=(8.3, 4.3), dpi=100)
        self.ax_2 = self.fig_2.add_subplot(111)
        self.canvas_2 = FigureCanvasTkAgg(self.fig_2, master=self.frame_h_gauche)
        self.canvas_2.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.img_laser = ImageTk.PhotoImage(Image.open("data/Laser_Flow_Squad.jpg").resize((320, 320)))
        self.label_laser = ttk.Label(self.frame_b_droite, image=self.img_laser)
        self.label_laser.pack(side="right", padx=(0, 50))

    def reprendre_simulation(self):
        if not self.running:
            if self.simulation_mode and self.simulation_completee:
                self.simulation_completee = False
                self.td.simulation_index = 0
            if self.premiere_lecture:
                self.bouton_start.config(text="▶ Continuer")
                self.premiere_lecture = False
            self.log("▶ Reprise de la simulation")
            self.after(100, self.demarrer_live)

    def demarrer_live(self):
        if not self.running:
            if not self.td:
                self.td = TraitementDonnees(simulation=self.simulation_mode, path="data/")
            self.running = True
            self.start_time = time.time()
            self.label_etat.config(text="Lecture en cours.", foreground="cyan")
            self.bouton_start.state(["disabled"])
            self.bouton_csv.state(["disabled"])
            self.log("▶ Acquisition démarrée")
            self.donnees_enregistrées.clear()
            self.mettre_a_jour_interface()

    def mettre_a_jour_interface(self):
        if self.running and (self.td.est_connecte() or self.td.simulation):
            data = self.td.get_temperatures()
            if data:
                elapsed_time = time.time() - self.start_time
                self.td.afficher_heatmap_dans_figure(data, self.fig, elapsed_time)
                self.canvas.draw()

                light_type, lambda_nm, puissance = self.td.get_wavelength()
                self.label_puissance.config(text=f"{puissance:.2f} W")
                self.label_lambda.config(text=f"{lambda_nm:.1f} nm")

                self.puissance_display.append(puissance)
                self.temps_display.append(elapsed_time)

                self.ax_2.clear()
                self.ax_2.set_xlabel("Temps écoulé (s)")
                self.ax_2.set_ylabel("Puissance (W)")
                self.ax_2.set_title("Évolution de la puissance")
                if len(self.temps_display) >= 2:
                    self.ax_2.plot(self.temps_display, self.puissance_display, label="Puissance mesurée (W)")
                    self.ax_2.legend()
                self.canvas_2.draw()

                row = {k: v for k, v in data.items()}
                row["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self.donnees_enregistrées.append(row)

            if self.td.simulation and self.td.simulation_index >= len(self.td.simulation_data):
                self.simulation_completee = True
                self.arreter_live()
                return

            self.window.widget.after(200, self.mettre_a_jour_interface)

    def arreter_live(self):
        if self.running:
            self.running = False
            self.simulation_completee = self.simulation_mode
            self.label_etat.config(text="⏹ Lecture arrêtée.", foreground="gray")
            self.bouton_start.state(["!disabled"])
            self.bouton_csv.state(["!disabled"])
            self.log("⏹ Acquisition arrêtée")

            if not self.fichier_simulation and self.donnees_enregistrées:
                if messagebox.askyesno("Sauvegarde", "Voulez-vous sauvegarder les données de cette session ?"):
                    filename = f"acquisition_{time.strftime('%Y%m%d_%H%M%S')}.csv"
                    filepath = self.dossier_sauvegarde / filename
                    try:
                        with open(filepath, mode='w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(self.donnees_enregistrées[0].keys())
                            for row in self.donnees_enregistrées:
                                writer.writerow(row.values())
                        self.log(f"Données sauvegardées : {filepath}")
                    except Exception as e:
                        self.log(f"Erreur lors de la sauvegarde : {e}")

    def rejouer_simulation(self, fichier=None):
        if fichier:
            self.simulation_mode = True
            self.fichier_simulation = fichier
            self.td = TraitementDonnees(simulation=True, path="data/", fichier_simulation=fichier)
            self.td.simulation_index = 0
            self.simulation_completee = False
            self.label_etat.config(text="Mode simulation.", foreground="orange")
            self.log(f"Lectures du fichier : {fichier}")
            self.premiere_lecture = True
            self.bouton_start.config(text="▶ Commencer")
            self.demarrer_live()

    def check_connection_loop(self):
        if not self.td or not self.td.est_connecte():
            if self.arduino_disponible():
                self.td = TraitementDonnees(simulation=False, path="data/")
                self.simulation_mode = False
                self.label_etat.config(text="Arduino reconnecté", foreground="green")
                self.log("Arduino reconnecté")
        self.window.widget.after(3000, self.check_connection_loop)


if __name__ == "__main__":
    app = MyApp()
    app.mainloop()

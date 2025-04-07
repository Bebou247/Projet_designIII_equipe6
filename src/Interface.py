from mytk import App
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from Thermistances_finale import TraitementDonnees


class MyTestApp(App):
    def __init__(self):
        super().__init__(name="Heatmap Live", geometry="600x500+100+100")

        self.td = TraitementDonnees(simulation=False)
        self.running = False

        # Interface
        self.frame = ttk.Frame(self.window.widget, padding=10)
        self.frame.pack(expand=True, fill='both')

        # Zone graphique
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas.get_tk_widget().pack(expand=True, fill='both')

        # Boutons
        bouton_start = ttk.Button(self.frame, text="Démarrer Live", command=self.demarrer_live)
        bouton_start.pack(side="left", padx=10, pady=10)

        bouton_stop = ttk.Button(self.frame, text="Arrêter", command=self.arreter_live)
        bouton_stop.pack(side="left", padx=10, pady=10)

        # Message d'état
        self.label_etat = ttk.Label(self.frame, text="", foreground="red")
        self.label_etat.pack(side="left", padx=20)

        self.maj_etat_connection()
        self.check_connection_loop()  # ← vérifie régulièrement la reconnexion

    def maj_etat_connection(self):
        if not self.td.est_connecte():
            self.label_etat.config(text="❌ Arduino non connecté", foreground="red")
        else:
            self.label_etat.config(text="✅ Prêt", foreground="green")

    def demarrer_live(self):
        if not self.td.est_connecte():
            self.label_etat.config(text="❌ Aucun Arduino détecté.", foreground="red")
            return

        if not self.running:
            self.running = True
            self.label_etat.config(text="📡 Lecture en cours...", foreground="blue")
            self.mettre_a_jour_heatmap()

    def arreter_live(self):
        self.running = False
        self.label_etat.config(text="⏹ Lecture arrêtée.", foreground="gray")

    def mettre_a_jour_heatmap(self):
        if self.running and self.td.est_connecte():
            temp = self.td.get_temperatures()
            if temp:
                self.td.afficher_heatmap_dans_figure(temp, self.fig)
                self.canvas.draw()
            self.window.widget.after(1000, self.mettre_a_jour_heatmap)
        elif self.running:
            self.label_etat.config(text="❌ Arduino déconnecté.", foreground="red")

    def check_connection_loop(self):
        # Vérifie toutes les 3 secondes si l’Arduino est revenu
        if not self.td.est_connecte():
            try:
                self.td = TraitementDonnees(simulation=False)
                if self.td.est_connecte():
                    print("[INFO] Arduino reconnecté !")
                    self.label_etat.config(text="✅ Arduino reconnecté", foreground="green")
            except Exception as e:
                pass
        self.window.widget.after(3000, self.check_connection_loop)


if __name__ == "__main__":
    app = MyTestApp()
    app.mainloop()

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt

# Charger les données CSV
file_path = "Line2.csv"  # Assurez-vous que le chemin est correct
data = pd.read_csv(file_path, delimiter=";", header=None)

# Conversion des données
wavelengths_original = data[0].str.replace(',', '.').astype(float)
responsivity_relative = data[1].str.replace(',', '.').astype(float)

# Créer une fonction d'interpolation
interp_func = interpolate.interp1d(wavelengths_original, responsivity_relative, kind='linear', fill_value="extrapolate")

# Créer un tableau de longueurs d'onde de 350 à 1120 nm
wavelength_range = np.arange(350, 1121, 1)

# Calculer les valeurs interpolées
interpolated_responsivity_relative = interp_func(wavelength_range)

# Paramètres de la photodiode
diameter = 3e-3  # 3 mm en mètres
area = np.pi * (diameter/2)**2  # en m²

# Données du tableau
Isc = 15e-6  # 15 μA
Ee = 10  # 1 mW/cm² = 10 W/m²

# Calcul de la responsivité à 950 nm
P_incident = Ee * area  # Puissance incidente en W
R_950nm = Isc / P_incident  # Responsivité à 950 nm en A/W

# Trouver l'index correspondant à 950 nm
index_950nm = np.argmin(np.abs(wavelength_range - 950))

# Normaliser et convertir en A/W
responsivity_AW = interpolated_responsivity_relative * (R_950nm / interpolated_responsivity_relative[index_950nm])

# Créer un DataFrame pour les résultats
result_data = pd.DataFrame({
    'Wavelength (nm)': wavelength_range,
    'Responsivity (A/W)': responsivity_AW
})

# Afficher les premières lignes des résultats
print(result_data.head())

# Sauvegarder les résultats dans un fichier CSV
result_data.to_csv("responsivity_AW.csv", index=False)

# Créer un graphique
plt.figure(figsize=(10, 6))
plt.plot(wavelength_range, responsivity_AW)
plt.xlabel('Longueur d\'onde (nm)')
plt.ylabel('Responsivité (A/W)')
plt.title('Responsivité spectrale de la photodiode')
plt.grid(True)
plt.savefig('responsivity_curve.png')
plt.show()

print(f"Responsivité maximale : {responsivity_AW.max():.4f} A/W")
print(f"Longueur d'onde au pic de responsivité : {wavelength_range[responsivity_AW.argmax()]} nm")

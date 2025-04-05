import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d  

# Charger le fichier CSV en ignorant les deux premières lignes
file_path = r"Plaque gravee-1_abs.csv"
df = pd.read_csv(file_path, skiprows=2)

# Renommer les colonnes pour plus de clarté
df.columns = ["Longueur_donde_refl", "Reflectance", "Longueur_donde_abs", "Absorbance", "Unused"]

# Supprimer la colonne inutilisée
df = df.drop(columns=["Unused"])

# Conversion explicite des colonnes en float pour éviter les erreurs de type
df["Longueur_donde_refl"] = pd.to_numeric(df["Longueur_donde_refl"], errors='coerce')
df["Reflectance"] = pd.to_numeric(df["Reflectance"], errors='coerce')
df["Longueur_donde_abs"] = pd.to_numeric(df["Longueur_donde_abs"], errors='coerce')
df["Absorbance"] = pd.to_numeric(df["Absorbance"], errors='coerce')

# Supprimer les lignes contenant des NaN résultant de la conversion
df = df.dropna()

# Conversion de l'absorbance en pourcentage : Absorbance (%) = (1 - 10^(-Absorbance)) * 100
df["Absorbance"] = (1 - 10 ** (-df["Absorbance"])) * 100

# Créer un objet d'interpolation pour l'absorbance en fonction de la longueur d'onde
interpolate_abs = interp1d(df["Longueur_donde_abs"], df["Absorbance"], kind='linear', fill_value="extrapolate")

# Liste pour stocker les absorbances à différentes longueurs d'onde
absorbances_list = []

def get_absorbance_at_wavelength(longueur_donde):

    absorbance_donnee = interpolate_abs(longueur_donde).item() 
    print(f"L'absorbance à {longueur_donde} nm est {absorbance_donnee:.2f}%")
    
    # Ajouter à la liste des absorbances
    absorbances_list.append(absorbance_donnee)
    
    return absorbance_donnee


longueur_donde_donnee = 2000  # Partie à retravailler avec Elias pour obtenir la longueur d'onde donnée selon les photodiodes


# Tracer les courbes avec l'absorbance en pourcentage
plt.figure(figsize=(10, 5))
plt.scatter(longueur_donde_donnee, get_absorbance_at_wavelength(longueur_donde_donnee), label='Yep'  ,color="green")
plt.plot(df["Longueur_donde_refl"], df["Reflectance"], label="Réflectance (%)", color="blue")
plt.plot(df["Longueur_donde_abs"], df["Absorbance"], label="Absorbance (%)", color="red")

# Afficher le graphique
plt.xlabel("Longueur d'onde (nm)")
plt.ylabel("Pourcentage (%)")
plt.title("Réflectance et Absorbance (en %) en fonction de la longueur d'onde")
plt.legend()
plt.grid(True)

# Afficher le graphique
plt.show()


# Afficher la liste des absorbances
print(f"Liste des absorbances : {absorbances_list}")
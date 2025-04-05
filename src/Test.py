import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Données de la thermistance
R_25 = 10000  # Résistance à 25°C en ohms
beta = 3892   # Coefficient β en Kelvin
temp_range = np.linspace(-55, 300, 500)

# Génération des résistances avec le modèle β
resistance = R_25 * np.exp(beta * (1 / (temp_range + 273.15) - 1 / 298.15))

# Conversion de la température en Kelvin
temp_kelvin = temp_range + 273.15

# Fonction Steinhart-Hart

def steinhart_hart(R, A, B, C):
    return 1 / (A + B * np.log(R) + C * (np.log(R))**3)

# Ajustement des coefficients avec curve_fit
initial_guess = [1e-3, 1e-4, 1e-7]
params, _ = curve_fit(steinhart_hart, resistance, temp_kelvin, p0=initial_guess)
A_fit, B_fit, C_fit = params

# Affichage des coefficients
print(f"Coefficients Steinhart-Hart :")
print(f"A = {A_fit:.6e}")
print(f"B = {B_fit:.6e}")
print(f"C = {C_fit:.6e}")



# Tracé des courbes et de l'erreur
plt.figure(figsize=(10, 6))
plt.plot(temp_range, resistance, label='Modèle β', color='g')
plt.plot(temp_range, R_25 * np.exp(beta * (1 / (steinhart_hart(resistance, A_fit, B_fit, C_fit)) - 1 / 298.15)), '--', label='Fit Steinhart-Hart', color='r')
plt.xlabel("Température (°C)")
plt.ylabel("Résistance (Ω)")
plt.yscale("log")
plt.title("Courbe R-T avec ajustement Steinhart-Hart")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()



# Supposons que P_cond, P_conv, P_rad soient déjà calculées à partir de tes données de température

# Coefficient de réflexion (exemple, à ajuster en fonction de la surface de la plaque)
R = 0.2  # 20% de la lumière est réfléchie

# Calcul de la puissance laser totale en tenant compte de la réflexion
P_laser_total = (P_cond + P_conv + P_rad) / (1 - R)

print(f"Puissance laser totale : {P_laser_total} W")

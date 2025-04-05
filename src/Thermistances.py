import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Données complètes
temps_C = np.array([20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 40, 45, 50, 55, 60])

resistances_kOhm = np.array([
    [11.71, 11.75, 11.8, 11.6, 11.81, 11.78, 11.73, 11.43, 11.85, 11.61, 11.75, 11.7, 11.7, 11.85, 11.5, 11.73, 11.52, 11.56, 11.68, 11.45, 11.7, 11.67, 11.5, 11.7, 11.6, 11.46, 11.9, 11.6],
    [11.35, 11.35, 11.35, 11.1, 11.32, 11.36, 11.33, 11.05, 11.38, 11.23, 11.35, 11.23, 11.38, 11.42, 11.1, 11.3, 11.15, 11.18, 11.32, 11.06, 11.27, 11.28, 11.16, 11.3, 11.23, 11.08, 11.46, 11.33],
    [10.9, 10.9, 10.9, 10.7, 10.91, 10.97, 10.88, 10.7, 10.92, 10.85, 10.91, 10.82, 10.98, 10.98, 10.7, 10.88, 10.75, 10.83, 10.87, 10.64, 10.79, 10.97, 10.77, 10.97, 10.8, 10.78, 10.86, 10.97],
    [10.51, 10.53, 10.55, 10.42, 10.56, 10.58, 10.45, 10.42, 10.51, 10.4, 10.52, 10.45, 10.5, 10.48, 10.36, 10.44, 10.38, 10.43, 10.43, 10.31, 10.39, 10.47, 10.38, 10.5, 10.38, 10.37, 10.5, 10.42],
    [10.1, 10.13, 10.13, 9.84, 10.15, 10.17, 10.14, 9.89, 10.16, 10.15, 10.23, 10.15, 10.05, 10.12, 9.95, 10.04, 10.02, 10, 10.1, 9.92, 10.1, 10.12, 10.01, 10.11, 9.97, 10, 10.13, 10.08],
    [9.8, 9.82, 9.77, 9.61, 9.82, 9.82, 9.79, 9.62, 9.8, 9.69, 9.88, 9.87, 9.8, 9.75, 9.53, 9.69, 9.65, 9.6, 9.73, 9.51, 9.71, 9.74, 9.6, 9.72, 9.53, 9.58, 9.71, 9.75],
    [9.48, 9.49, 9.43, 9.3, 9.44, 9.48, 9.41, 9.3, 9.48, 9.33, 9.47, 9.44, 9.5, 9.39, 9.22, 9.29, 9.28, 9.2, 9.35, 9.15, 9.4, 9.35, 9.27, 9.37, 9.29, 9.2, 9.43, 9.32],
    [9.17, 9.15, 9.13, 8.95, 9.14, 9.16, 9.05, 9.03, 9.14, 9.01, 9.3, 9.19, 9.18, 9.07, 8.87, 8.96, 8.95, 8.82, 9.1, 8.81, 9.07, 9, 8.94, 9.06, 9.02, 8.9, 9.21, 9.06],
    [8.87, 8.82, 8.8, 8.67, 8.82, 8.83, 8.76, 8.7, 8.8, 8.69, 8.97, 8.9, 8.88, 8.72, 8.57, 8.64, 8.59, 8.51, 8.74, 8.5, 8.75, 8.68, 8.61, 8.73, 8.71, 8.55, 8.92, 8.74],
    [8.54, 8.52, 8.47, 8.42, 8.5, 8.5, 8.42, 8.39, 8.5, 8.38, 8.32, 8.6, 8.6, 8.42, 8.26, 8.31, 8.26, 8.19, 8.44, 8.18, 8.43, 8.37, 8.31, 8.41, 8.4, 8.23, 8.65, 8.44],
    [8.25, 8.23, 8.2, 8.14, 8.2, 8.24, 8.11, 8.12, 8.22, 8.12, 8.01, 8.32, 8.31, 8.12, 8, 8.05, 7.99, 7.88, 8.14, 7.88, 8.15, 8.09, 8.01, 8.13, 8.09, 7.94, 8.35, 8.16],
    [7.68, 7.62, 7.61, 7.57, 7.67, 7.7, 7.56, 7.61, 7.65, 7.53, 7.53, 7.6, 7.79, 7.58, 7.46, 7.49, 7.41, 7.27, 7.62, 7.34, 7.62, 7.52, 7.46, 7.58, 7.55, 7.36, 7.92, 7.63],
    [6.92, 6.91, 6.87, 6.85, 6.97, 6.95, 6.77, 6.88, 6.89, 6.76, 6.67, 6.95, 7.12, 6.82, 6.62, 6.73, 6.61, 6.5, 6.85, 6.56, 6.81, 6.75, 6.72, 6.83, 6.77, 6.6, 7.15, 6.75],
    [5.82, 5.87, 5.83, 5.79, 5.87, 5.86, 5.7, 5.84, 5.78, 5.68, 5.62, 5.85, 5.98, 5.73, 5.2, 5.36, 5.5, 5.39, 5.78, 5.5, 5.75, 5.7, 5.65, 5.77, 5.7, 5.59, 6.05, 5.79],
    [4.9, 4.95, 4.92, 4.93, 4.96, 4.97, 4.8, 4.97, 4.86, 4.77, 4.72, 4.98, 5.1, 4.88, 4.3, 4.6, 4.59, 4.58, 4.87, 4.62, 4.83, 4.8, 4.78, 4.88, 4.78, 4.71, 5.14, 4.86],
    [4.05, 4.16, 4.17, 4.16, 4.04, 4.19, 4.03, 4.21, 4.2, 4.02, 4.02, 4.28, 4.32, 4.15, 3.56, 3.86, 3.87, 3.79, 4.14, 3.92, 4.09, 4.08, 4.12, 4.14, 4.06, 3.99, 4.4, 4.1],
    [3.55, 3.55, 3.51, 3.55, 3.43, 3.62, 3.41, 3.6, 3.58, 3.41, 3.44, 3.62, 3.62, 3.5, 3.1, 3.35, 3.27, 3.21, 3.51, 3.34, 3.45, 3.45, 3.56, 3.5, 3.43, 3.44, 3.8, 3.46],
    [3, 3.04, 3.04, 3.01, 2.95, 3.08, 2.94, 3.03, 3.06, 2.89, 2.93, 3.05, 3.07, 3, 2.67, 2.97, 2.76, 2.69, 2.99, 2.83, 2.92, 2.92, 2.95, 3.02, 2.98, 2.93, 3.3, 2.93]
])

# Fonctions (comme précédemment)
def steinhart_hart_matrix(temps_C, resistances_kOhm):
    temps_K = temps_C + 273.15
    resistances_Ohm = resistances_kOhm * 1000
    ln_R = np.log(resistances_Ohm)
    
    indices = [0, len(temps_K)//2, -1]
    T_sample = temps_K[indices]
    ln_R_sample = ln_R[indices]
    
    X = np.column_stack((np.ones(3), ln_R_sample, ln_R_sample**3))
    Y = 1 / T_sample
    
    return np.linalg.solve(X, Y)

def steinhart_hart(R, A, B, C):
    return 1 / (A + B * np.log(R) + C * (np.log(R))**3)

def steinhart_hart_least_squares(temps_C, resistances_kOhm):
    temps_K = temps_C + 273.15
    resistances_Ohm = resistances_kOhm * 1000
    
    popt, _ = curve_fit(steinhart_hart, resistances_Ohm, temps_K, 
                        p0=[1e-3, 2e-4, 1e-7], maxfev=5000)
    
    return popt

# Modèle beta du fabricant
R_25 = 10000  # Résistance à 25°C en ohms
beta = 3892   # Coefficient β en Kelvin

def beta_model(R, R_25, beta):
    return 1 / ((1 / 298.15) + (1 / beta) * np.log(R / R_25))

# Calcul des coefficients et tracé des courbes
R_range = np.linspace(2000, 12000, 1000)  # Plage de résistances en Ohms

# Graphique original comparant les méthodes et le modèle β
plt.figure(figsize=(20, 15))
for i in range(resistances_kOhm.shape[1]):
    A_matrix, B_matrix, C_matrix = steinhart_hart_matrix(temps_C, resistances_kOhm[:, i])
    A_ls, B_ls, C_ls = steinhart_hart_least_squares(temps_C, resistances_kOhm[:, i])
    
    T_matrix = steinhart_hart(R_range, A_matrix, B_matrix, C_matrix) - 273.15
    T_ls = steinhart_hart(R_range, A_ls, B_ls, C_ls) - 273.15
    T_beta = beta_model(R_range, R_25, beta) - 273.15
    
    plt.subplot(5, 6, i+1)
    plt.plot(R_range/1000, T_matrix, label='Matricielle', color='blue')
    plt.plot(R_range/1000, T_ls, label='Moindres carrés', color='red', linestyle='--')
    plt.plot(R_range/1000, T_beta, label='Modèle β', color='green', linestyle=':')
    plt.scatter(resistances_kOhm[:, i], temps_C, color='black', label='Données')
    
    plt.title(f'R{i+1}')
    plt.xlabel('Résistance (kΩ)')
    plt.ylabel('Température (°C)')
    plt.legend()
    plt.grid(True)

plt.tight_layout()
#plt.show()

# Graphique pour la méthode des moindres carrés
plt.figure(figsize=(12, 8))
for i in range(resistances_kOhm.shape[1]):
    A_ls, B_ls, C_ls = steinhart_hart_least_squares(temps_C, resistances_kOhm[:, i])
    T_ls = steinhart_hart(R_range, A_ls, B_ls, C_ls) - 273.15
    plt.plot(R_range/1000, T_ls, label=f'R{i+1}')

plt.title('Comparaison des courbes - Méthode des moindres carrés')
plt.xlabel('Résistance (kΩ)')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True)
#plt.show()

# Graphique pour la méthode matricielle
plt.figure(figsize=(12, 8))
for i in range(resistances_kOhm.shape[1]):
    A_matrix, B_matrix, C_matrix = steinhart_hart_matrix(temps_C, resistances_kOhm[:, i])
    T_matrix = steinhart_hart(R_range, A_matrix, B_matrix, C_matrix) - 273.15
    plt.plot(R_range/1000, T_matrix, label=f'R{i+1}')

plt.title('Comparaison des courbes - Méthode matricielle')
plt.xlabel('Résistance (kΩ)')
plt.ylabel('Température (°C)')
plt.legend()
plt.grid(True)
#plt.show()

coefficients = []

for i in range(resistances_kOhm.shape[1]):
    A_ls, B_ls, C_ls = steinhart_hart_least_squares(temps_C, resistances_kOhm[:, i])
    coefficients.append([A_ls, B_ls, C_ls])

coefficients = np.array(coefficients)
np.save("coefficients.npy", coefficients) 


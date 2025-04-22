import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.signal import savgol_filter

def estimate_laser_power(temp_initial, temp_measured, time):
    delta_t = temp_measured - temp_initial
    K = 0.8411
    tau = 0.9987
    coeff = 0.9999
    
    denominator = K * (1 - np.exp(-time/tau))
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    estimated_power = delta_t / denominator
    return estimated_power


temps = np.linspace(0, 10, 6)  # 100 points de 0 à 10 secondes
temp_initiale = 25  # température initiale en °C
temp_heatmap = np.array([30,35,43,32,36,45])  

puissances_estimees = estimate_laser_power(temp_initiale, temp_heatmap, temps)
print(puissances_estimees)

# Données du CSV
df = pd.read_csv('C:/Users/emile/Desktop/fichierspython/Design 3/Centre_echelons_test.csv')
temps  = df.iloc[:, 0:21].values
ref    = df['R25'].values
max_temps = temps.max(axis=1)
delta_T   = max_temps - ref
dt = 0.5836

# Filtre - dérivée - intégrale
delta_T_filt = savgol_filter(delta_T, 51, 3)
d_delta_T_dt = np.gradient(delta_T_filt, dt)
integral     = np.cumsum(delta_T_filt) * dt

# Échelon de puissance pour comparer ---
duration         = 60
samples_per_step = int(round(duration / dt))
power_steps      = np.array([0, 2.5, 5, 7.5, 10, 7.5, 5, 2.5, 0])
power_sequence   = np.repeat(power_steps, samples_per_step)
n = len(delta_T)
power_vec = np.zeros(n)
power_vec[:len(power_sequence)] = power_sequence[:n]

time = np.arange(n) * dt

# Coefficients du PID
init_kp = 0.294
init_kd = 12.3
init_ki = 0.00026
init_b  = -0.167  # intercept

# Calcul de la puissance estimée ---
est = init_kp * delta_T_filt + init_kd * d_delta_T_dt - init_ki * integral + init_b
est = np.clip(est, 0, None)


# Lissage de la puissance
seuil = 0.1  # seuil de détection de saut en W
edges = np.where(np.abs(np.diff(est)) > seuil)[0] + 1
bounds = np.concatenate(([0], edges, [len(est)]))

est_liss = np.empty_like(est)
for start, end in zip(bounds[:-1], bounds[1:]):
    segment = est[start:end]
    mean_val = segment.mean()
    est_liss[start:end] = mean_val 


# Plot
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(left=0.1, bottom=0.3)
l_real, = ax.plot(time, power_vec, label='Puissance réelle')
l_est,  = ax.plot(time, est,       label='Puissance estimée')
l_est_liss,  = ax.plot(time, est_liss,       label='Puissance estimée applatie')
ax.set_xlabel('Temps (s)')
ax.set_ylabel('Puissance (W)')
ax.legend()
ax.set_xlim(40, 500)
ax.set_yticks([2.5, 5.0, 7.5, 10.0])
ax.grid(True, which='both', axis='y')

plt.show()

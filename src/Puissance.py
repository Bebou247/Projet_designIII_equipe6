import numpy as np

def estimate_laser_power(temp_initial, temp_measured, time):
    delta_t = temp_measured - temp_initial
    K = 0.8411
    tau = 0.9987
    
    denominator = K * (1 - np.exp(-time/tau))
    denominator = np.where(denominator < 1e-10, 1e-10, denominator)
    
    estimated_power = delta_t / denominator
    return estimated_power


temps = np.linspace(0, 10, 6)  # 100 points de 0 à 10 secondes
temp_initiale = 25  # température initiale en °C
temp_heatmap = np.array([30,35,43,32,36,45])  

puissances_estimees = estimate_laser_power(temp_initiale, temp_heatmap, temps)
print(puissances_estimees)

# Vous pouvez maintenant tracer ou analyser puissances_estimees

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf
from scipy.optimize import curve_fit
from matplotlib.patches import Ellipse

coefficients = np.load("coefficients.npy")
print(coefficients)  # Affiche les coefficients


# Paramètres
R25 = 10000   # Résistance à 25°C (10kΩ)
beta = 3950    # Coefficient beta standard
V_ref = 3.3     # Tension d'alimentation
adc_resolution = 1023
R_fixed = 4700   # Résistance ajustée
plate_size = 25.0  # Diamètre total de la plaque en mm

# Positions des thermistances (coordonnées en mm)
thermistor_positions = [
    (-3, 0), (-11, 0), (3, 0), (11, 0), (0, -2.5), (8, -2.5),
    (-8, -2.5), (0, -5.5), (8, -5.5), (-8, -5.5), (4.5, -8), (-4.5, -8),
    (3.5, -11.25), (-3.5, -11.25), (0, 2.5), (8, 2.5), (-8, 2.5),
    (0, 5.5), (8, 5.5), (-8, 5.5), (4.5, 8), (-4.5, 8), (4, 11.25), (-4, 11.25)
]

positions_array = np.array(thermistor_positions)
x, y = positions_array[:, 0], positions_array[:, 1]

def a4dc_to_temperature(adc_value):
    """Conversion ADC vers température avec contrôle des limites"""
    V_out = adc_value * V_ref / adc_resolution
    V_out = np.clip(V_out, 0.1, V_ref-0.1)
    R_therm = R_fixed * (V_ref - V_out) / V_out
    steinhart = 1/(25 + 273.15) + (1/beta)*np.log(R_therm/R25)
    return (1/steinhart) - 273.15


def gaussian_2d(xy, amplitude, x0_norm, y0_norm, sigma_x_norm, sigma_y_norm, offset):
    """Gaussienne 2D avec coordonnées normalisées"""
    x, y = xy
    return offset + amplitude * np.exp(
        -(((x-x0_norm)/sigma_x_norm)**2 + ((y-y0_norm)/sigma_y_norm)**2) / 2)

def estimate_laser_wavelength(temperatures):
    """Estimation simulée de la longueur d'onde basée sur la température"""
    max_temp = np.max(temperatures)
    base_wavelength = 1064  # Longueur d'onde typique d'un laser Nd:YAG en nm
    wavelength = base_wavelength - 0.1 * (max_temp - 25)  # Variation fictive avec la température
    return np.clip(wavelength, 1000, 1100)

def estimate_laser_power(temperatures):
    """Estimation simulée de la puissance basée sur la température maximale"""
    max_temp = np.max(temperatures)
    # Relation fictive entre température et puissance
    power = 0.5 * (max_temp - 25)  # 0.5W par degré au-dessus de 25°C
    return np.clip(power, 0, 10)  # Limite la puissance entre 0 et 10W

def create_or_update_heatmap(temperatures, ax=None):
    # Interpolation haute résolution
    xi = yi = np.linspace(-12.5, 12.5, 1000)
    Xi, Yi = np.meshgrid(xi, yi)
    rbf = Rbf(x, y, temperatures, function='thin_plate', smooth=0.5)
    Zi = rbf(Xi, Yi)

    # Trouver le point central 
    max_temp_index = np.unravel_index(np.argmax(Zi), Zi.shape)
    x0 = xi[max_temp_index[1]]
    y0 = yi[max_temp_index[0]]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        ax.clear()

    # Dessin de la heatmap
    im = ax.imshow(Zi, extent=[-12.5,12.5,-12.5,12.5], 
                  origin='lower', cmap='inferno', vmin=25, vmax=45)
    
    # Marquer le point central
    ax.plot(x0, y0, 'wo', markersize=10, markeredgecolor='k')

    ax.set_title("Distribution thermique du faisceau laser")
    ax.set_xlabel("Position X (mm)")
    ax.set_ylabel("Position Y (mm)")

    return Zi, x0, y0, im

# Le code pour le calcul du diamètre du faisceau est commenté ci-dessous:
"""
    # Normalisation des coordonnées pour l'ajustement
    x_flat, y_flat = Xi.flatten(), Yi.flatten()
    x_norm = (x_flat + 12.5) / 25.0
    y_norm = (y_flat + 12.5) / 25.0

    # Ajustement avec contraintes réalistes
    initial_guess = [np.max(Zi), 0.5, 0.5, 0.15, 0.15, np.min(Zi)]
    bounds = (
        [0, 0.4, 0.4, 0.05, 0.05, 0],
        [np.inf, 0.6, 0.6, 0.3, 0.3, np.inf]
    )

    popt, _ = curve_fit(gaussian_2d, (x_norm, y_norm), Zi.flatten(), p0=initial_guess, bounds=bounds)

    # Conversion des paramètres normalisés en mm
    amplitude, x0_norm, y0_norm, sigma_x_norm, sigma_y_norm, offset = popt
    x0 = x0_norm * 25.0 - 12.5
    y0 = y0_norm * 25.0 - 12.5
    sigma_x = sigma_x_norm * 12.5  # Conversion du sigma normalisé
    sigma_y = sigma_y_norm * 12.5

    # Calcul du FWHM réaliste
    fwhm_x = 2 * sigma_x * np.sqrt(2 * np.log(2))
    fwhm_y = 2 * sigma_y * np.sqrt(2 * np.log(2))
    fwhm_avg = (fwhm_x + fwhm_y) / 2

    # Dessin de l'ellipse
    ellipse = Ellipse((x0, y0), width=fwhm_x, height=fwhm_y,
                     angle=0, fill=False, color='cyan', linewidth=2, linestyle='--')
    ax.add_patch(ellipse)
"""

if __name__ == "__main__":
    # Test des fonctions
    adc_values = generate_realistic_adc()
    temperatures = [adc_to_temperature(adc) for adc in adc_values]
    
    fig, ax = plt.subplots()
    Zi, x0, y0 = create_or_update_heatmap(temperatures, ax)
    
    wavelength = estimate_laser_wavelength(temperatures)
    power = estimate_laser_power(temperatures)
    
    plt.colorbar(ax.images[0], label='Température (°C)')
    
    print(f"Centre du faisceau: ({x0:.2f} mm, {y0:.2f} mm)")
    print(f"Longueur d'onde estimée: {wavelength:.0f} nm")
    print(f"Puissance estimée: {power:.1f} W")
    
    plt.show()

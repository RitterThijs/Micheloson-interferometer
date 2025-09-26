#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: thijsritter
"""

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# data
i = np.array([0, 2, 4, 6, 8, 10])  # graden
N = np.array([0, 3, 12, 29, 41, 58])

# Constanten
d = 0.003      
lam = 532e-9   

def N_model(i_deg, n):
    i_rad = np.deg2rad(i_deg)
    # Veilige sqrt
    sqrt_term = np.sqrt(np.clip(n**2 - np.sin(i_rad)**2, 1e-12, None))
    cos_term = np.cos(i_rad)
    return (2*d/lam) * (sqrt_term + (np.sin(i_rad)**2 - 1)/cos_term + (1 - n))

# Fit uitvoeren
popt, pcov = curve_fit(N_model, i, N, p0=[1.5], bounds=[1.0, 2.0])
n_fit = popt[0]
print(f"Geschatte n: {n_fit:.4f}")

# Plotten
i_fit = np.linspace(0, 10, 200)
plt.scatter(i, N, color='red', label='Meetpunten')
plt.plot(i_fit, N_model(i_fit, n_fit), label=f'Fit: n={n_fit:.4f}')
plt.xlabel("i (graden)")
plt.ylabel("N")
plt.legend()
plt.show()

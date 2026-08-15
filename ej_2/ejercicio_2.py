"""
Actividad 2: Simulación de Procesos Estocásticos
X(t) = A * cos(2*pi*f0*t + Phi), con Phi ~ Uniforme[0, 2*pi]
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------------
# Parámetros del proceso
# ------------------------------------------------------------------
A = 2           # amplitud fija
f0 = 5          # frecuencia fija (Hz)
n_realizaciones = 50
t = np.linspace(0, 3, 300)  # 300 puntos entre 0 y 3 segundos

# ------------------------------------------------------------------
# 1. Generar las 50 realizaciones del proceso
# ------------------------------------------------------------------
# Una fase aleatoria por realización (fija en el tiempo dentro de cada una)
phi = np.random.uniform(0, 2 * np.pi, n_realizaciones)

# Matriz de realizaciones: filas = realizaciones, columnas = tiempo
X = np.zeros((n_realizaciones, len(t)))
for i in range(n_realizaciones):
    X[i, :] = A * np.cos(2 * np.pi * f0 * t + phi[i])

# ------------------------------------------------------------------
# 2. Graficar las primeras 10 realizaciones
# ------------------------------------------------------------------
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.plot(t, X[i, :], alpha=0.8, linewidth=1)
plt.title('Primeras 10 realizaciones de X(t) = A·cos(2πf₀t + Φ)')
plt.xlabel('t [s]')
plt.ylabel('X(t)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('realizaciones_proceso.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# 3. Media empírica: promedio sobre las realizaciones, para cada t
# ------------------------------------------------------------------
media_empirica = np.mean(X, axis=0)  # promedio a lo largo del eje de realizaciones

# Media teórica: E[X(t)] = A * E[cos(2*pi*f0*t + Phi)] = 0 para todo t,
# porque Phi ~ Uniforme[0,2pi] hace que el coseno se "reparta" simétricamente.
media_teorica = np.zeros_like(t)

# ------------------------------------------------------------------
# 4. Graficar media empírica vs teórica
# ------------------------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, media_empirica, label='Media empírica μ̂(t)', color='steelblue')
plt.plot(t, media_teorica, label='Media teórica μ(t) = 0', color='red',
         linestyle='--', linewidth=2)
plt.title('Media empírica vs. media teórica del proceso X(t)')
plt.xlabel('t [s]')
plt.ylabel('Media')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('media_empirica_vs_teorica.png', dpi=150)
plt.show()

print(f"Media empírica promedio (sobre todo t): {np.mean(media_empirica):.4f}")
print(f"Máximo |desvío| respecto de la media teórica: {np.max(np.abs(media_empirica - media_teorica)):.4f}")
print("Media teórica: 0 para todo t (Φ uniforme hace que el proceso sea estacionario en media)")
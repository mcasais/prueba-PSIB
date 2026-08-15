"""
Actividad 4: Estacionariedad en Señales Biomédicas
ECG(t) = Señal_base(t) + Ruido(t) + Deriva(t)
"""

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ------------------------------------------------------------------
# 1. Parámetros de simulación
# ------------------------------------------------------------------
fs = 250            # frecuencia de muestreo (Hz)
duracion = 10        # segundos
t = np.arange(0, duracion, 1 / fs)  # vector de tiempo

# ------------------------------------------------------------------
# Señal_base(t): suma de 4 senoidales con frecuencias de rango medio
# (dentro del ancho de banda típico de un ECG, ~0.5-40 Hz)
# ------------------------------------------------------------------
frecuencias = [2, 5, 10, 18]     # Hz
amplitudes = [1.0, 0.6, 0.4, 0.2]

señal_base = np.zeros_like(t)
for f, amp in zip(frecuencias, amplitudes):
    señal_base += amp * np.sin(2 * np.pi * f * t)

# ------------------------------------------------------------------
# Ruido(t): ruido blanco gaussiano
# ------------------------------------------------------------------
sigma_ruido = 0.1
ruido = np.random.normal(0, sigma_ruido, len(t))

# ------------------------------------------------------------------
# Deriva(t): componente de baja frecuencia (baseline wander)
# ------------------------------------------------------------------
f_deriva = 0.05  # Hz -> período de 20s, mucho más lento que la señal base
amp_deriva = 0.8
deriva = amp_deriva * np.sin(2 * np.pi * f_deriva * t)

# ------------------------------------------------------------------
# Señal ECG simulada completa
# ------------------------------------------------------------------
ecg = señal_base + ruido + deriva

# ------------------------------------------------------------------
# 2. Dividir en ventanas de 2 segundos
# ------------------------------------------------------------------
dur_ventana = 2  # segundos
n_muestras_ventana = int(dur_ventana * fs)
n_ventanas = len(ecg) // n_muestras_ventana

medias_ventana = np.zeros(n_ventanas)
varianzas_ventana = np.zeros(n_ventanas)
t_centro_ventana = np.zeros(n_ventanas)

for i in range(n_ventanas):
    ini = i * n_muestras_ventana
    fin = ini + n_muestras_ventana
    ventana = ecg[ini:fin]
    medias_ventana[i] = np.mean(ventana)
    varianzas_ventana[i] = np.var(ventana)
    t_centro_ventana[i] = t[ini] + dur_ventana / 2

# ------------------------------------------------------------------
# 4. Gráficos
# ------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 10))

# Señal completa
axes[0].plot(t, ecg, linewidth=0.7, color='steelblue')
axes[0].set_title('Señal ECG simulada completa')
axes[0].set_xlabel('t [s]')
axes[0].set_ylabel('Amplitud')
axes[0].grid(alpha=0.3)

# Media y varianza por ventana vs tiempo
ax1 = axes[1]
ax1.plot(t_centro_ventana, medias_ventana, 'o-', color='darkorange', label='Media')
ax1.set_xlabel('t [s]')
ax1.set_ylabel('Media', color='darkorange')
ax1.tick_params(axis='y', labelcolor='darkorange')
ax1.set_title('Media y varianza por ventana (2 s) vs. tiempo')
ax1.grid(alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(t_centro_ventana, varianzas_ventana, 's-', color='seagreen', label='Varianza')
ax2.set_ylabel('Varianza', color='seagreen')
ax2.tick_params(axis='y', labelcolor='seagreen')

# Histograma de la señal completa
axes[2].hist(ecg, bins=40, color='mediumpurple', edgecolor='black', alpha=0.8)
axes[2].set_title('Histograma de la señal ECG completa')
axes[2].set_xlabel('Amplitud')
axes[2].set_ylabel('Frecuencia')
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('ecg_estacionariedad.png', dpi=150)
plt.show()

# ------------------------------------------------------------------
# 5. Evaluación de estacionariedad
# ------------------------------------------------------------------
print("--- Medias por ventana ---")
for i, m in enumerate(medias_ventana):
    print(f"Ventana {i+1} ({t_centro_ventana[i]-1:.0f}-{t_centro_ventana[i]+1:.0f}s): media = {m:.4f}")

print("\n--- Varianzas por ventana ---")
for i, v in enumerate(varianzas_ventana):
    print(f"Ventana {i+1} ({t_centro_ventana[i]-1:.0f}-{t_centro_ventana[i]+1:.0f}s): varianza = {v:.4f}")

print(f"\nDesvío estándar de las medias entre ventanas: {np.std(medias_ventana):.4f}")
print(f"Desvío estándar de las varianzas entre ventanas: {np.std(varianzas_ventana):.4f}")

print("""
--- Conclusión sobre estacionariedad ---
La media NO es constante a lo largo del tiempo: varía de ventana en ventana
porque la Deriva(t) (0.05 Hz, período de 20s) desplaza el nivel de base de
la señal de forma lenta y sistemática dentro de los 10s simulados.
Como la media cambia con t, el proceso ECG(t) simulado NO es estacionario
en sentido estricto (ni siquiera en sentido amplio/débil, ya que ese
requiere media constante). La varianza, en cambio, se mantiene bastante
más estable entre ventanas porque el ruido y la señal base tienen
potencia aproximadamente constante en el tiempo; el componente que rompe
la estacionariedad es principalmente la deriva de baja frecuencia.
""")
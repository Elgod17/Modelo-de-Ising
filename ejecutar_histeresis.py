import numpy as np
import random
import matplotlib.pyplot as plt
#from numba import njit

import cadena_z2 as z2
import celda_z3 as z3
import celda_z4 as z4
import cubo_z8 as z8



# simulaciones_paralelas.py
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import cadena_z2 as z2  # impor

# === CONFIGURACIÓN GLOBAL ===
OUTPUT_DIR = "resultados"  # carpeta donde se guardarán los resultados
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FUNCIÓN AUXILIAR: ejecuta una simulación y guarda su resultado ===
def ejecutar_simulacion(id_sim, T, Hmin, Hmax, dH, J, mu, pasos, n):
    print(f"[Simulación {id_sim}] Iniciando con T={T}, H ∈ [{Hmin}, {Hmax}]")

    # Ejecuta tu función del modelo de Ising
    H, M = z2.m_vs_h_paramagneto_2(
        0, 300, Hmin, Hmax, dH, J, mu, T, n
    )

    # Guarda resultados
    filename = os.path.join(OUTPUT_DIR, f"simulacion_{id_sim}_T{T:.2f}.npz")
    np.savez(filename, H=H, M=M, T=T, Hmin=Hmin, Hmax=Hmax)
    print(f"[Simulación {id_sim}] Guardado en {filename}")

    return filename


# === BLOQUE PRINCIPAL ===
if __name__ == "__main__":
    # Parámetros de las 4 simulaciones
    simulaciones = [
        {"id": 1, "T": 1.0, "Hmin": -100, "Hmax": 100, "dH": 0.01, "J": 1, "mu": 0.5, "pasos": 15, "n": 1},
        {"id": 2, "T": 2.0, "Hmin": -100, "Hmax": 100, "dH": 0.01, "J": 1, "mu": 0.5, "pasos": 15, "n": 1},
        {"id": 3, "T": 3.0, "Hmin": -100, "Hmax": 100, "dH": 0.01, "J": 1, "mu": 0.5, "pasos": 15, "n": 1},
        {"id": 4, "T": 4.0, "Hmin": -100, "Hmax": 100, "dH": 0.01, "J": 1, "mu": 0.5, "pasos": 15, "n": 1},
    ]

    # Detecta núcleos disponibles
    num_cores = os.cpu_count()
    max_workers = min(len(simulaciones), max(1, num_cores - 4))
    print(f"🧠 Usando {max_workers} de {num_cores} núcleos disponibles...\n")

    # Ejecuta en paralelo
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                ejecutar_simulacion,
                s["id"], s["T"], s["Hmin"], s["Hmax"],
                s["dH"], s["J"], s["mu"], s["pasos"], s["n"]
            )
            for s in simulaciones
        ]

        # Espera a que terminen todas y muestra los archivos guardados
        for f in futures:
            print("✅ Resultado guardado en:", f.result())

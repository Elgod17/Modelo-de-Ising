import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import glob
import os
import re
import matplotlib.pyplot as plt

# Ruta a tu carpeta
ruta = "C:/Users/Asus/modelo_ising/mh_param/"

# Buscar todos los CSV
archivos = glob.glob(os.path.join(ruta, "param_q*.csv"))

# Expresión regular para extraer q, T, z del nombre
patron = re.compile(r"q([0-9.]+)T([0-9.]+)_z([0-9.]+)")

# Cargar y clasificar
data = []
for f in archivos:
    match = patron.search(f)
    if match:
        q, T, z = map(float, match.groups())
        arr = np.loadtxt(f, delimiter=",", skiprows=1)
        data.append({"q": q, "T": T, "z": z, "array": arr, "nombre": os.path.basename(f)})

print(f"✅ Se cargaron {len(data)} archivos.")





def plotear(data, q=None, T=None, z=None):
    plt.figure()

    for d in data:
        if (q is None or d["q"] == q) and (T is None or d["T"] == T) and (z is None or d["z"] == z):
            arr = d["array"]
            plt.plot(arr[:,0], arr[:,1], label=f"q={d['q']}, T={d['T']}, z={d['z']}")

    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"Filtro: q={q}, T={T}, z={z}")
    plt.show()


plotear(data, q=0, T=None, z=None)
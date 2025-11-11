import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
#from numba import njit
import csv

import cadena_z2 as z2
import celda_z3 as z3
import celda_z4 as z4
import cubo_z8 as z8

N = 20
l2 = N
l3 = int(np.sqrt(1+4*N)-1)
l4 = int(np.sqrt(2*N))
l8 = int((2*N)**(1/3))

import concurrent.futures

# Parámetros de ejemplo para cada corrida
#histeresis(q,l,Binicial,Bfinal,deltab,J,mu,T,n)
simulaciones = [
    (0, l8, -250, 250, 0.01, 1, 0.5, 15, 1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
    (0.5, l8, -210, 210, 0.01, 1, 0.5, 15, 1),
    (0.8, l8, -140, 140, 0.01, 1, 0.5, 15, 1)
]
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
#        futures = {executor.submit(z4.histeresis_4, *args): args[0] for args in simulaciones}


import concurrent.futures

if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(z8.histeresis_8, *args): args[0] for args in simulaciones}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                result = future.result()  # Esto fuerza a que la función termine
                print(f"Simulación con q={q} terminada")
            except Exception as e:
                print(f"Simulación con q={q} falló: {e}")



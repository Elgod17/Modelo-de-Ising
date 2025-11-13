import numpy as np
import glob
from scipy.optimize import curve_fit
import os
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import pandas as pd

param_q05T4_z2 = pd.read_csv("C:/Users/Asus/modelo_ising/mh_param/param_q0.5T4_z2.csv")
data = param_q05T4_z2.iloc[0:len(param_q05T4_z2):4000,0:3]

def lectura(ruta_archivo,nombre):
    datos = pd.read_csv(ruta_archivo)
    data = datos.iloc[0:len(datos):4000,0:3]
    return data.to_csv(nombre,index=False)

lista_mh_param = [
    'param_q0.5T15_z2.csv',
    'param_q0.5T15_z3.csv',
    'param_q0.5T15_z4.csv',
    'param_q0.5T15_z8.csv',
    'param_q0.5T4_z2.csv',
    'param_q0.5T4_z3.csv',
    'param_q0.5T4_z4.csv',
    'param_q0.5T4_z8.csv',
    'param_q0.5T9_z2.csv',
    'param_q0.5T9_z3.csv',
    'param_q0.5T9_z4.csv',
    'param_q0.5T9_z8.csv',
    'param_q0.8T15_z2.csv',
    'param_q0.8T15_z3.csv',
    'param_q0.8T15_z4.csv',
    'param_q0.8T15_z8.csv',
    'param_q0.8T4_z2.csv',
    'param_q0.8T4_z3.csv',
    'param_q0.8T4_z4.csv',
    'param_q0.8T4_z8.csv',
    'param_q0.8T9_z2.csv',
    'param_q0.8T9_z3.csv',
    'param_q0.8T9_z4.csv',
    'param_q0.8T9_z8.csv',
    'param_q0T15_z2.csv',
    'param_q0T15_z3.csv',
    'param_q0T15_z4.csv',
    'param_q0T15_z8.csv',
    'param_q0T4_z2.csv',
    'param_q0T4_z3.csv',
    'param_q0T4_z4.csv',
    'param_q0T4_z8.csv',
    'param_q0T9_z2.csv',
    'param_q0T9_z3.csv',
    'param_q0T9_z4.csv',
    'param_q0T9_z8.csv',
]

ruta = 'C:/Users/Asus/modelo_ising/mh_param/'
datos_lista = list()

for i in range(len(lista_mh_param)):
    ruta_completa = ruta + lista_mh_param[i]
    datos = lectura(ruta_completa,lista_mh_param[i])

    


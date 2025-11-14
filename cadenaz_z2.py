import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
from numba import njit
import concurrent.futures
from matplotlib.colors import ListedColormap
#################################
#           Cadena Z = 2
##################################

############################
# Definir cadena
############################
def generador_decadenas(l, p_menosuno, p_masuno, p_cero):
    """
    Genera una cadena de espines con posibles vacíos embebidos entre ceros.

    Descripción original:
        Se asignan valores de los spines aleatoriamente. p_menosuno es la probabilidad
        de que tenga spin negativo. p_masuno de que sea positivo y p_cero = q es la
        probabilidad de que el nodo esté vacío.

    Parámetros:
        l (int): longitud de la cadena.
        p_menosuno (float): probabilidad de que el spin sea -1.
        p_masuno (float): probabilidad de que el spin sea +1.
        p_cero (float): probabilidad de que el nodo esté vacío.

    Retorna:
        tuple: (cadena, cadena_embebida)
            cadena (np.ndarray): cadena generada de espines.
            cadena_embebida (np.ndarray): cadena con ceros en los bordes para cálculos.
    """
    cadena = np.ones(l)
    cadena_embebida = []

    for i in range(0, l):
        cadena[i] = np.random.choice([-1, 1, 0], p=[p_menosuno, p_masuno, p_cero])

    cero = np.array([0])
    cadena_embebida = np.concatenate((cero, cadena, cero))

    return cadena, cadena_embebida


############################
# Función de energía
############################
@njit
def Energy_2(l, celda, celda_embebida0, jota, be, muu):
    """
    Calcula la energía total de la cadena considerando interacción y campo externo.

    Descripción original:
        Se calcula la energía de interacción y la energía de Zeeman
        para una cadena de espines.

    Parámetros:
        l (int): longitud de la cadena.
        celda (np.ndarray): cadena de espines actual.
        celda_embebida0 (np.ndarray): cadena embebida en ceros para bordes.
        jota (float): constante de interacción entre espines vecinos.
        be (float): campo magnético externo.
        muu (float): momento magnético de cada spin.

    Retorna:
        tuple: (Energia, M)
            Energia (float): energía total de la cadena.
            M (float): magnetización total.
    """
    Energia_interaccion = 0

    if jota != 0:
        for j in range(0, l):
            if celda[j] != 0:
                Energia_interaccion += celda_embebida0[j] * celda_embebida0[j + 1]
                Energia_interaccion = -jota * Energia_interaccion

    Energia_interaccion = -jota * Energia_interaccion
    M = muu * np.sum(celda)
    Energia_zeeman = -be * M
    Energia = Energia_interaccion + Energia_zeeman

    return Energia, M


############################
# Flip spin
############################
@njit
def flip_spin_2(l, cadena, cadena_embebida, J, B, mu, T):
    """
    Genera nuevas configuraciones de los spines a partir de la configuración anterior.

    Descripción original:
        l es la longitud de la cadena, cadena es la cadena generada,
        J es el parámetro de interacción, B es la magnitud del campo magnético externo,
        mu el momento magnético y T la temperatura.

    Parámetros:
        l (int): longitud de la cadena.
        cadena (np.ndarray): configuración actual de los espines.
        cadena_embebida (np.ndarray): cadena con bordes embebidos en ceros.
        J (float): constante de interacción entre espines vecinos.
        B (float): campo magnético externo.
        mu (float): momento magnético del spin.
        T (float): temperatura del sistema.

    Retorna:
        list: [nueva_cadena, nueva_cadena_embebida].
    """
    k = 1  # constante de Boltzmann para aumentar probabilidad de salir de mínimo local
    w = 0

    while w == 0:
        filasarray = np.arange(0, l, 1)
        iarb = np.random.choice(filasarray)

        if cadena[iarb] != 0:
            w = 1
            e = B * mu * cadena_embebida[iarb + 1] + J * cadena_embebida[iarb + 1] * (
                cadena_embebida[iarb + 2] + cadena_embebida[iarb]
            )
            Eactual = Energy_2(l, cadena, cadena_embebida, J, B, mu)[0]
            r = random.uniform(0, 1)

            if Eactual >= Eactual + 2 * e or r < np.exp(-2 * e / (k * T)):
                cadena_embebida[iarb + 1] = -cadena_embebida[iarb + 1]
                cadena[iarb] = -cadena[iarb]

    return [cadena, cadena_embebida]


############################
# Evolución temporal
############################
def evolucionar_2(l, cadena, cadena_embebida, J, B, mu, T, n):
    """
    Evoluciona la cadena n pasos usando el método de flip de spins.

    Descripción original:
        n es el número de pasos de evolución de la red,
        l es la longitud de la cadena, cadena es la cadena generada,
        J es el parámetro de interacción, B es la magnitud del campo magnético externo,
        mu el momento magnético y T la temperatura.

    Parámetros:
        l (int): longitud de la cadena.
        celda (np.ndarray): cadena de espines inicial.
        celda_embebida (np.ndarray): cadena embebida en ceros.
        ensamble (list): lista de configuraciones de la cadena.
        ensamble_embebido (list): lista de configuraciones embebidas.
        J (float): constante de interacción.
        B (float): campo magnético externo.
        mu (float): momento magnético.
        T (float): temperatura.
        n (int): número de pasos de evolución.

    Retorna:
        tuple: (Energia, M)
            Energia (float): energía de la última configuración.
            M (float): magnetización de la última configuración.
    """
    for j in range(0, n):
        flip_spin_2(l, cadena, cadena_embebida, J, B, mu, T).copy()
 

    Cantidades = Energy_2(l, cadena, cadena_embebida, J, B, mu)
    return Cantidades[0], Cantidades[1]




def m_vs_T_ferro_2(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  energias = []
  magnetizaciones = []
  ensamble = []
  
  ## q = 1-probabilidad asignar spin menos uno
  cadena, cadena_embebida = generador_decadenas(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero


  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  evolucionar_2(l,cadena, cadena_embebida,J,H,mu,Tinicial,f)
  u = len(Ts)
  for j in range(u-1):
      energia_, magnetizacion_ = evolucionar_2(l,cadena,cadena_embebida,J,H,mu,Ts[j],n)
      magnetizaciones.append(magnetizacion_)
      if j == 0 or j == int(u/4) or j == int(u/2) or j == int(3*u/4):
          a = cadena.copy()
          ensamble.append(a)


  # Crear un colormap personalizado: -1 (rojo), 0 (negro), 1 (azul)
  colors = ['red', 'black', 'blue']
  cmap = ListedColormap(colors)

  # Crear figura con subplots de 2 filas x 3 columnas
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  axes = axes.flatten()  # Aplanar para iterar fácilmente

  # Graficar cada matriz del ensamble
  for i, matriz in enumerate(ensamble):
      im = axes[i].imshow(matriz, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
      axes[i].set_title(f'Matriz {i+1}')
      axes[i].axis('off')  # Opcional: quitar ejes

  # Añadir barra de color común
  cbar = fig.colorbar(im, ax=axes, orientation='horizontal', 
                       pad=0.05, shrink=0.8, ticks=[-1, 0, 1])
  cbar.ax.set_xticklabels(['-1 (Rojo)', '0 (Negro)', '1 (Azul)'])

  plt.tight_layout()
  plt.show()

  return ensamble


  
  #til = "mtferro_q"+str(q)+"_z2.csv"
  #with open(til, "w", newline="") as f:
  #  writer = csv.writer(f)
  #  writer.writerow(["T", "M", "Mn"])  # encabezados opcionales
  #  for i in range(len(Ts)):
  #      writer.writerow([Ts[i], magnetizaciones[i], m_normalizado[i]])
  #plt.figure()
  #plt.plot(Ts, m_normalizado, 'o', ms=2)
  #plt.title(til)
  #plt.show()
  #return Ts, magnetizaciones

#m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f)
N = 5000
l8 = N
#m_vs_T_ferro(0,0,l8,0.1,60,0.1,1,0.5,2,400,1)



simulaciones = [
    (0,l8,0.1,60,0.1,1,0.5,2,100,1),
    (0.5,l8,0.1,60,0.1,1,0.5,2,100,1),
    (0.8,l8,0.1,60,1,0.1,0.5,2,100,1)
    
    #(0.5,l8,0.1,60,0.1,1,0.5,2,600,1),
    #(0.8,l8,0.1,60,0.1,1,0.5,2,600,1)  
]


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(m_vs_T_ferro_2, *args): args[0] for args in simulaciones}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                result = future.result()  # Esto fuerza a que la función termine
                print(f"Simulación con q={q} terminada")
            except Exception as e:
                print(f"Simulación con q={q} falló: {e}")


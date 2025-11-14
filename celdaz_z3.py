import numpy as np
import random  
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import csv
from numba import njit
import concurrent.futures
from matplotlib.colors import ListedColormap
executor = concurrent.futures.ProcessPoolExecutor()
executor.shutdown(wait=True, cancel_futures=True)

import gc
gc.collect()
################################################################################## Z = 3
############################################################################################## GENERAR CELDA DE l*l HEXAGONOS
def largo(l):
    longitud = (l * 3) - (l-2)
    return longitud
##################################################################################
def generador_deceldas(l, p_menosuno, p_masuno, p_cero):
  longitud = largo(l) # defino para llenar la red
  celda = [] # celda vacia para armar la red
  celda_embebida = []
  celdarray = np.array([])
  enteros = np.arange(0,longitud,1) # creo un arreglo de ceros
  pares = np.arange(0,longitud,2)
  impares = np.arange(1, longitud, 2)

  # crear estructura de un bloque
  fila1 = np.ones(longitud)
  fila2 = np.ones(longitud)
  for k in impares: # tomar igual a cero las posiciones con numeracion impar
    fila1[k] = 0
  for j in impares: # // // // // // // // // par
    fila2[j-1] = 0
  cero = np.array([0])
  fila_1 = np.delete(fila1, len(fila1)-1)
  fila_2 = np.delete(fila2, len(fila2)-1)
  # se crea la red

  for i in range(0,l):
    celda.append(fila_2)
    celda.append(fila_1)
    celda.append(fila_1)
    celda.append(fila_2)

  # convierte la celda a array para hacer operaciones más facilmente
  celda = np.array(celda)


  # se le asigna aleatoriamente un valor de -1/2 o de 1/2 a los nodos con valor distinto de cero
  for i in range(0,len(celda)):
    for j in range(0,len(fila1)-1):
      if celda[i,j] !=0:
        celda[i,j] = np.random.choice([-1,1,0], p = [p_menosuno,p_masuno,p_cero])    ## aqui se asigna -1 con un 90% de probabilidad y 1 copn un 10 %
  # crear celda embebida
  cero = np.array([0])
  celda_embebida.append(np.zeros(longitud+1))  ## agregar fila inicial de ceros
  for i in range(0,len(celda)):
    celda_embebida.append(np.concatenate((cero,celda[i],cero)))

  celda_embebida.append(np.zeros(longitud+1))  ## agregar fila final de ceros
  celda_embebida = np.array(celda_embebida)

  return celda, celda_embebida

##################################################################################  CALCULAR ENERGIA
@njit
def Energy(celda, celda_embebida0, jota, be, muu):
  Energia_interaccion = 0

  if jota != 0:
    for j in range(0,len(celda)-1):    ## recorrer cada nodo con posicion j,k
      for k in range(0,len(celda[0])-1):
        if celda[j,k] != 0:
          Energia_interaccion += celda_embebida0[j+1,k+1] * (celda_embebida0[j+1+1,k-1+1] + celda_embebida0[j+1+1,k+1+1] + celda_embebida0[j+1+1,k+1]) # se suma la energia de interaccion del spin j,k con los espines más cercanos que estan en la siguiente fila, que es la fila j+1
    Energia_interaccion = - jota * Energia_interaccion


  M = muu * np.sum(celda)
  Energia_zeeman = -be * M
  Energia = Energia_interaccion + Energia_zeeman
  return Energia, M
################################################################################## algoritmo de metropolis
@njit
def flip_spin(celda, celda_embebida,J,B,mu,T):
  #k = 1.380649 * 1e-23  # constante de Bolsman
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  w = 0
  while w == 0:
    filasarray = np.arange(0,len(celda)-1,1)
    columnasarray = np.arange(0,len(celda[0])-1,1)
    ## seleccionar i,j aleatoriamente
    iarb = np.random.choice(filasarray)
    jarb = np.random.choice(columnasarray)
    # la posicion i,j en celda se corresponde con la posicion i+1,j+1 en la celda embebida
    if celda_embebida[iarb+1,jarb+1] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*celda_embebida[iarb+1,jarb+1] + J * celda_embebida[iarb+1,jarb+1] * (celda_embebida[iarb+2,jarb+2] + celda_embebida[iarb+2,jarb+1] + celda_embebida[iarb+2,jarb] + celda_embebida[iarb,jarb+2] + celda_embebida[iarb,jarb+1] + celda_embebida[iarb,jarb] )
      Eactual = Energy(celda,celda_embebida, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        celda_embebida[iarb+1,jarb+1] = -celda_embebida[iarb+1,jarb+1]
        celda[iarb,jarb] = -celda[iarb,jarb]
      # si la nueva energia no es menor entonces
  return [celda, celda_embebida]



###################################################################### generador de la evolucion de la celda


def evolucionar_celda(l,celda, celda_embebida, ensamble, ensamble_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
  ## crear copy para guardar configuraciones en disitntos espacios de memoria
  a = celda.copy()
  b = celda_embebida.copy()
  ## generar evolucion de n pasos en la configuracion de los espines

  for j in range(0,n):
    ab = flip_spin(a, b, J,B,mu,T).copy()
    a,b = ab[0], ab[1]
    ensamble.append(a)
    ensamble_embebido.append(b)

  Cantidades = Energy(ensamble[-1],ensamble_embebido[-1], J, B, mu)

  return Cantidades[0], Cantidades[1]




def m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_deceldas(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_celda(l,celda, celda_embebida, ensamble, ensamble_embebido,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_celda(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  # Gráfica M vs T
  plt.figure()
  plt.plot(Ts,magnetizaciones,'o',ms = 2)
  plt.title("M(t)")
  plt.xlabel("T")
  plt.ylabel("M")
  plt.legend()
  plt.grid(True)
  plt.show()

  # Visualización de las 6 matrices del ensamble
  # Seleccionar 6 matrices equiespaciadas del ensamble
  indices = np.linspace(0, len(ensamble)-1, 6, dtype=int)
  matrices_seleccionadas = [ensamble[i] for i in indices]
  
  # Crear colormap personalizado: -1 (rojo), 0 (negro), 1 (azul)
  colors = ['red', 'black', 'blue']
  cmap = ListedColormap(colors)
  
  # Crear figura con subplots de 2 filas x 3 columnas
  fig, axes = plt.subplots(2, 3, figsize=(12, 8))
  axes = axes.flatten()
  titulo = 'Evolución para z = 3, N = 2000 y q='+str(q)
  fig.suptitle(titulo, fontsize=16, fontweight='bold')
  # Graficar cada matriz seleccionada
  for i, (idx, matriz) in enumerate(zip(indices, matrices_seleccionadas)):
      im = axes[i].imshow(matriz, cmap=cmap, vmin=-1, vmax=1, aspect='auto')
      axes[i].set_title(f'Paso {idx}' if idx > 0 else f'Inicial (T={Tinicial:.2f})')
      axes[i].axis('off')
  

  
  plt.tight_layout()
  plt.show()
  
  return energias, magnetizaciones, ensamble

N = 2000
l8 = int(np.sqrt(1+4*N)-1)
#m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f)
simulaciones = [
    (0, l8,   0.1, 60, 0.1, 1, 0.5, 15, 600,1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
    (0.5, l8, 0.1, 60, 0.1, 1, 0.5, 15, 600,1),
    (0.8, l8, 0.1, 60, 0.1, 1, 0.5, 15, 600,1)
]


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(m_vs_T_ferro, *args): args[0] for args in simulaciones}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                result = future.result()  # Esto fuerza a que la función termine
                print(f"Simulación con q={q} terminada")
            except Exception as e:
                print(f"Simulación con q={q} falló: {e}")
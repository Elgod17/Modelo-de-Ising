import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
from numba import njit
import concurrent.futures
################################## Cadena Z=2
################## definir cadena
def generador_decadenas(l, p_menosuno, p_masuno, p_cero):
  cadena = np.ones(l) # celda vacia para armar la red
  cadena_embebida = []

  #### aqui se asignan los valores de los spines aleatoriamente. p_menosuno es la probabilidad de que tenga spin negativo. p_masuno de que sea positivo y p_cero = q es la probabilidad de que el nodo este vacio
  for i in range(0,l):
      cadena[i] = np.random.choice([-1,1,0], p = [p_menosuno,p_masuno,p_cero])    ## aqui se asigna -1 con un 90% de probabilidad y 1 copn un 10 %


  ## se crea un array con los espines rodeados por ceros para facilitar el calculo en el borde
  cero = np.array([0])
  cadena_embebida = np.concatenate((cero,cadena,cero))

  return cadena, cadena_embebida


##################### funcion de energia
@njit
def Energy_2(l,celda, celda_embebida0, jota, be, muu):
  Energia_interaccion = 0
  if jota != 0:
    for j in range(0,l):    ## recorrer cada nodo con posicion j,k
      if celda[j] != 0:
        Energia_interaccion += celda_embebida0[j] * celda_embebida0[j+1]
        Energia_interaccion = - jota * Energia_interaccion

  Energia_interaccion = - jota * Energia_interaccion
  M = muu * np.sum(celda)  ## aqui se calcula la magnetizacion
  Energia_zeeman = -be * M
  Energia = Energia_interaccion + Energia_zeeman
  return Energia, M




@njit
def flip_spin_2(l,cadena, cadena_embebida,J,B,mu,T):
  "l es la longitud de la cadena, cadena es la cadena generada, J es el parametro de interaccion, B es la magnitud del campo magnetico externo, mu el momento magnetico y T la temperatura"
  #k = 1.380649 * 1e-23  # constante de Boltzmann. Se toma k = 1 para aumentar la probabilidad de salir del minimno local de energia
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  # w es un contador que aumenta bajo la condicion de que el nodo sea no vacio
  w = 0
  ## crear copias de las redes para guardar las nuevas configuraciones en lugares distintos de la memoria del computador
  cadena2 = cadena.copy()
  cadena_embebida2 = cadena_embebida.copy()
  ## vamos a seleccionar un nodo i,j para cambiarle el sentido del spin. Desde luego, esto solo tiene sentido si el nodo i,j es distinto de cero.
  ## Asi que se introduce en el algoritmo la condicion de que solo se ejecute si el i,j seleccionado aleatoriamente es distinto de cero
  while w == 0:
    filasarray = np.arange(0,l,1)
    ## seleccionar i
    iarb = np.random.choice(filasarray)
    if cadena[iarb] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*cadena_embebida[iarb+1] + J * cadena_embebida[iarb+1] * (cadena_embebida[iarb+2] + cadena_embebida[iarb] )
      Eactual = Energy_2(l,cadena,cadena_embebida, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        cadena_embebida2[iarb+1] = -cadena_embebida[iarb+1]
        cadena2[iarb] = -cadena[iarb]

  return [cadena2, cadena_embebida2]




def evolucionar_2(l,celda, celda_embebida, ensamble, ensamble_embebido, J,B,mu,T,n):   ##############
  "n es el numero de pasos de evolucion de la red,l es la longitud de la cadena, cadena es la cadena generada, J es el parametro de interaccion, B es la magnitud del campo magnetico externo, mu el momento magnetico y T la temperatura"

  ## crear copy para guardar configuraciones en disitntos espacios de memoria
  a = celda.copy()
  b = celda_embebida.copy()
  ## generar evolucion de n pasos en la configuracion de los espines
  ## pone a evolucionar la celda n pasos
  for j in range(0,n):
    ab = flip_spin_2(l,a, b, J,B,mu,T).copy()
    a,b = ab[0], ab[1]
    ensamble.append(a)
    ensamble_embebido.append(b)

  # se calcula la energia y la magnetizacion de la ultima configuracion
  Cantidades = Energy_2(l,ensamble[-1],ensamble_embebido[-1], J, B, mu)

  return Cantidades[0], Cantidades[1]




def m_vs_h_paramagneto_2(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  "q es la dilucion magnetica, l la longitud de la cadena, el campo magnetico H barre valores desde Hinicial hasa Hfinal en pasos de longitud deltaH, J es el parametro de interaccion, mu el moemnto magnetico, T la temperatura y n el numero de pasos de evolucion paara cada valor del campo magnetico H"
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cadena, cadena_embebida = generador_decadenas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(cadena)
  ensamble_embebido.append(cadena_embebida)
  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  lenh = len(Hs)
  evolucionar_2(l,cadena, cadena_embebida, ensamble, ensamble_embebido,J,Hinicial,mu,T,n)

  # calcular energia y magnetizacion para cada campo magnetico
  for h in Hs:
      energia_, magnetizacion_ = evolucionar_2(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  return Hs, magnetizaciones, energias



def energia_relajacion_2(q,p,l, J,H,mu,T,n,f):
  "q es la disolucion magnetica, p la probabilidad de que sea spin negativo, l la longitud de la cadena, J la energia de inter"
  cadena, cadena_embebida = generador_decadenas(l,p,1-p-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno
  energia_relajacion = []
  ensamble = []
  ensamble_embebido = []
  ensamble.append(cadena)
  ensamble_embebido.append(cadena_embebida)
  for j in range(0,f):
    energia_relajacion.append(evolucionar_2(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,T,n)[0])
  return [np.array(energia_relajacion), ensamble, ensamble_embebido]


def histeresis_2(q,l,Binicial,Bfinal,deltab,J,mu,T,n):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  celda, celda_embebida = generador_decadenas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero
  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_2(l,celda, celda_embebida, ensamble, ensamble_embebido,J,Binicial,mu,T,n)
  for b in Bs:
      energia_, magnetizacion_ = evolucionar_2(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,b,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)
  a = "Histeresis con z=2 y q =" +str(q)
  plt.figure()
  plt.plot(B_aumentando,magnetizaciones[0:len(B_aumentando)],'o',label = "B aumentando",ms = 0.5)
  plt.plot(B_disminuyendo,magnetizaciones[len(B_aumentando):lenb],'o',label = "B disminuyendo",ms = 0.5)
  plt.title(a)
  plt.xlabel("H[T]")
  plt.ylabel("M[J/T]")
  plt.legend()
  plt.grid(True)
  plt.show()





def m_vs_T_ferro_2(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cadena, cadena_embebida = generador_decadenas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(cadena)
  ensamble_embebido.append(cadena_embebida)
  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_2(l,cadena, cadena_embebida, ensamble, ensamble_embebido,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_2(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  print(len(Ts))
  print(len(magnetizaciones))

  return Ts, magnetizaciones, energias



N = 10000
l8 = N
simulaciones = [
    (0., l8, -250, 250, 0.01, 1, 0.5, 15, 1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
    (0.5, l8, -210, 210,0.01, 1, 0.5, 15, 1),
    (0.8, l8, -140, 140, 0.01, 1, 0.5, 15, 1)
]


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(histeresis_2, *args): args[0] for args in simulaciones}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                result = future.result()  # Esto fuerza a que la función termine
                print(f"Simulación con q={q} terminada")
            except Exception as e:
                print(f"Simulación con q={q} falló: {e}")
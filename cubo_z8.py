import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import csv
from numba import njit
import concurrent.futures
####################################################### Z = 8


def generador_8(l, p_menosuno, p_masuno, p_cero):
  cubo = np.indices((l, l, l)).sum(axis=0) % 2

  for i in range(0,l):
    for j in range(0,l):
      for k in range(0,l):
        if cubo[i,j,k] != 0:
          cubo[i,j,k] = np.random.choice([-1,1,0], p = [p_menosuno,p_masuno,p_cero])

  # crear un cubo más grande, con un "marco" de ceros
  L = l + 2
  cubo_embebido = np.zeros((L, L, L), dtype=int)

  # insertar el cubo de unos en el centro
  cubo_embebido[1:-1, 1:-1, 1:-1] = cubo

  return cubo, cubo_embebido
##################################################################################  FERROMAGNETICO
@njit
def Energy_8(l,cubo, cubo_embebido, jota, be, muu):
  Energia_interaccion = 0
  for i in range(0,l):
    for j in range(0,l):
      for k in range(0,l):
        if cubo[i,j,k] != 0:
          Energia_interaccion += cubo_embebido[i+1,j+1,k+1]*(cubo_embebido[i+2,j+2,k+2]+cubo_embebido[i+2,j,k]+cubo_embebido[i+2,j+2,k]+cubo_embebido[i+2,j,k+2]) # se suma la energia de interaccion del spin j,k con los espines más cercanos que estan en la siguiente fila, que es la fila j+1


  Energia_interaccion = - jota * Energia_interaccion
  M = muu * np.sum(cubo)
  Energia_zeeman = -be * M
  Energia = Energia_interaccion + Energia_zeeman
  return Energia, M
##################################################################################
@njit
def flip_spin_8(l,cubo, cubo_embebido,J,B,mu,T):
  #k = 1.380649 * 1e-23  # constante de Bolsman
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  w = 0

  while w == 0:
    filasarray = np.arange(0,l,1)
    ## seleccionar i,j aleatoriamente
    iarb = np.random.choice(filasarray)
    jarb = np.random.choice(filasarray)
    karb = np.random.choice(filasarray)
    # la posicion i,j en celda se corresponde con la posicion i+1,j+1 en la celda embebida
    if cubo[iarb,jarb,karb] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*cubo_embebido[iarb+1,jarb+1,karb+1] + J * cubo_embebido[iarb+1,jarb+1,karb+1] * (cubo_embebido[iarb+2,jarb+2,karb+2]+cubo_embebido[iarb+2,jarb,karb]+cubo_embebido[iarb+2,jarb+2,karb]+cubo_embebido[iarb+2,jarb,karb+2] + cubo_embebido[iarb,jarb+2,karb+2]+cubo_embebido[iarb,jarb,karb]+cubo_embebido[iarb,jarb+2,karb]+cubo_embebido[iarb,jarb,karb+2])
      Eactual = Energy_8(l,cubo,cubo_embebido, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        cubo_embebido[iarb+1,jarb+1] = -cubo_embebido[iarb+1,jarb+1]
        cubo[iarb,jarb] = -cubo[iarb,jarb]
      # si la nueva energia no es menor entonces
  return [cubo, cubo_embebido]


def evolucionar_8_copy(l,cubo, cubo_embebido, ensamble, ensamble_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
  ## crear copy para guardar configuraciones en disitntos espacios de memoria
  a = cubo.copy()
  b = cubo_embebido.copy()
  ## generar evolucion de n pasos en la configuracion de los espines

  for j in range(0,n):
    ab = flip_spin_8(l,a, b, J,B,mu,T).copy()
    a,b = ab[0], ab[1]
    ensamble.append(a)
    ensamble_embebido.append(b)

  Cantidades = Energy_8(l,ensamble[-1],ensamble_embebido[-1], J, B, mu)

  return Cantidades[0], Cantidades[1]

def evolucionar_8(l,cubo, cubo_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
  for j in range(0,n):
    flip_spin_8(l,cubo,cubo_embebido, J,B,mu,T)


  Cantidades = Energy_8(l,cubo,cubo_embebido, J, B, mu)

  return Cantidades[0], Cantidades[1]
################################################################################################

def m_vs_h_paramagneto_8(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cubo, cubo_embebido = generador_8(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero


  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  evolucionar_8(l,cubo, cubo_embebido,J,Hinicial,mu,T,n)

  for h in Hs:
      energia_, magnetizacion_ = evolucionar_8(l,cubo,cubo_embebido,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  til = "param_q"+str(q)+"T"+str(T)+"_z8.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "M"])  # encabezados opcionales
    for i in range(len(Hs)):
        writer.writerow([Hs[i], magnetizaciones[i]])


  plt.figure()
  plt.plot(Hs, magnetizaciones, 'o', ms=2)
  plt.title(til)
  plt.show()  
  return Hs, magnetizaciones





#N = 1000
#l8 = int((2*N)**(1/3))
##m_vs_h_paramagneto(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n)
#modH = 120
#simulaciones = [
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 10, 1), 
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 5, 1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 10, 1), 
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 5, 1),
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 10, 1), 
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 5, 1)
#    
#]
#
#
#
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=9) as executor:
#        futures = {executor.submit(m_vs_h_paramagneto_8, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
#####################################################################################################
#





################################################################################################



def histeresis_8(q,l,Binicial,Bfinal,deltab,J,mu,T,n):
  energias = []
  magnetizaciones = []
  cubo, cubo_embebido = generador_8(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_8(l,cubo, cubo_embebido,J,Binicial,mu,T,n)
  for b in Bs:
      energia_, magnetizacion_ = evolucionar_8(l,cubo,cubo_embebido,J,b,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)
  
  til = "histeresis_q"+str(q)+"_z8.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "M"])  # encabezados opcionales
    for i in range(len(Bs)):
        writer.writerow([Bs[i], magnetizaciones[i]])

  plt.figure()
  plt.plot(Bs, magnetizaciones, 'o', ms=2)
  plt.xlabel("H")
  plt.ylabel("M")
  plt.title(til)
  plt.grid()
  plt.show()

  return Bs, magnetizaciones




def m_vs_T_ferro_8(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cubo, cubo_embebido = generador_8(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(cubo)
  ensamble_embebido.append(cubo_embebido)
  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_8(l,cubo, cubo_embebido,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_8(l,cubo,cubo_embebido,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  m_normalizado = np.array(magnetizaciones)/magnetizaciones[0]

  til = "mtferro_q"+str(q)+"_z8.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["T", "M", "Mn"])  # encabezados opcionales
    for i in range(len(Ts)):
        writer.writerow([Ts[i], magnetizaciones[i], m_normalizado[i]])
  plt.figure()
  plt.plot(Ts, m_normalizado, 'o', ms=2)
  plt.title(til)
  plt.show()
  return Ts, magnetizaciones

##m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f)
#N = 4000
#l8 = int((2*N)**(1/3))
#
#
#
#
#simulaciones = [
#    (0,l8,0.1,60,0.1,1,0.5,2,600,1),  
#    (0.5,l8,0.1,60,0.1,1,0.5,2,600,1),
#    (0.8,l8,0.1,60,0.1,1,0.5,2,600,1)  
#]
#
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
#        futures = {executor.submit(m_vs_T_ferro_8, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
#
#
#
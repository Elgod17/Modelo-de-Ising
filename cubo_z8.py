import numpy as np
import random
import matplotlib.pyplot as plt
#from numba import njit
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
#@njit
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
#@njit
def flip_spin_8(l,cubo, cubo_embebido,J,B,mu,T):
  #k = 1.380649 * 1e-23  # constante de Bolsman
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  w = 0
  ## crear copias de las redes para guardar las nuevas configuraciones en lugares distintos de la memoria del computador
  cubo2 = cubo.copy()
  cubo_embebido2 = cubo_embebido.copy()
  ## vamos a seleccionar un nodo i,j para cambiarle el sentido del spin. Desde luego, esto solo tiene sentido si el nodo i,j es distinto de cero.
  ## Asi que se introduce en el algoritmo la condicion de que solo se ejecute si el i,j seleccionado aleatoriamente es distinto de cero
  while w == 0:
    filasarray = np.arange(0,l,1)
    ## seleccionar i,j aleatoriamente
    iarb = random.choice(filasarray)
    jarb = random.choice(filasarray)
    karb = random.choice(filasarray)
    # la posicion i,j en celda se corresponde con la posicion i+1,j+1 en la celda embebida
    if cubo[iarb,jarb,karb] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*cubo_embebido[iarb+1,jarb+1,karb+1] + J * cubo_embebido[iarb+1,jarb+1,karb+1] * (cubo_embebido[iarb+2,jarb+2,karb+2]+cubo_embebido[iarb+2,jarb,karb]+cubo_embebido[iarb+2,jarb+2,karb]+cubo_embebido[iarb+2,jarb,karb+2] + cubo_embebido[iarb,jarb+2,karb+2]+cubo_embebido[iarb,jarb,karb]+cubo_embebido[iarb,jarb+2,karb]+cubo_embebido[iarb,jarb,karb+2])
      Eactual = Energy_8(l,cubo,cubo_embebido, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        cubo_embebido2[iarb+1,jarb+1] = -cubo_embebido[iarb+1,jarb+1]
        cubo2[iarb,jarb] = -cubo[iarb,jarb]
      # si la nueva energia no es menor entonces
  return [cubo2, cubo_embebido2]


def evolucionar_8(l,cubo, cubo_embebido, ensamble, ensamble_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
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


################################################################################################

def m_vs_h_paramagneto_8(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cubo, cubo_embebido = generador_8(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(cubo)
  ensamble_embebido.append(cubo_embebido)
  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  evolucionar_8(l,cubo, cubo_embebido, ensamble, ensamble_embebido,J,Hinicial,mu,T,n)

  for h in Hs:
      energia_, magnetizacion_ = evolucionar_8(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  return Hs, magnetizaciones, energias

####################################################################################################
def energia_relajacion_8(q,p,l, J,H,mu,T,f):
  "q es la disolucion magnetica, p la probabilidad de que sea spin negativo, l la longitud de la cadena, J la energia de interaccion, H el campo externo, mu el moemnto magnetico, T la temperatura, f el numero de pasos monte carlo"
  cubo, cubo_embebido = generador_8(l,p,1-p-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno
  energia_relajacion = []
  ensamble = []
  ensamble_embebido = []
  ensamble.append(cubo)
  ensamble_embebido.append(cubo_embebido)
  for j in range(0,f):
    energia_relajacion.append(evolucionar_8(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,T,1)[0])
  return [np.array(energia_relajacion), ensamble, ensamble_embebido]









################################################################################################



def histeresis_8(q,l,Binicial,Bfinal,deltab,J,mu,T,n):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  cubo, cubo_embebido = generador_8(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero
  ensamble.append(cubo)
  ensamble_embebido.append(cubo_embebido)
  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_8(l,cubo, cubo_embebido, ensamble, ensamble_embebido,J,Binicial,mu,T,n)
  for b in Bs:
      energia_, magnetizacion_ = evolucionar_8(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,b,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)
  a = "Histeresis con q =" +str(q)
  plt.figure()
  plt.plot(B_aumentando,magnetizaciones[0:len(B_aumentando)],'o',label = "B aumentando",ms = 0.5)
  plt.plot(B_disminuyendo,magnetizaciones[len(B_aumentando):lenb],'o',label = "B disminuyendo",ms = 0.5)
  plt.title(a)
  plt.xlabel("H")
  plt.ylabel("M")
  plt.legend()
  plt.show()

  return Bs, magnetizaciones




def m_vs_T_ferro_4(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cubo, cubo_embebido = generador_8(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(cubo)
  ensamble_embebido.append(cubo_embebido)
  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_8(l,cubo, cubo_embebido, ensamble, ensamble_embebido,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_8(l,ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)


  return Ts, magnetizaciones, energias, ensamble


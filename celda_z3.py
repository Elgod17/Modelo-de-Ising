import numpy as np
import random  
import matplotlib.pyplot as plt
#from numba import njit
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
#@njit
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
#@njit
def flip_spin(celda, celda_embebida,J,B,mu,T):
  #k = 1.380649 * 1e-23  # constante de Bolsman
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  w = 0
  ## crear copias de las redes para guardar las nuevas configuraciones en lugares distintos de la memoria del computador
  celda2 = celda.copy()
  celda_embebida2 = celda_embebida.copy()
  ## vamos a seleccionar un nodo i,j para cambiarle el sentido del spin. Desde luego, esto solo tiene sentido si el nodo i,j es distinto de cero.
  ## Asi que se introduce en el algoritmo la condicion de que solo se ejecute si el i,j seleccionado aleatoriamente es distinto de cero
  while w == 0:
    filasarray = np.arange(0,len(celda)-1,1)
    columnasarray = np.arange(0,len(celda[0])-1,1)
    ## seleccionar i,j aleatoriamente
    iarb = random.choice(filasarray)
    jarb = random.choice(columnasarray)
    # la posicion i,j en celda se corresponde con la posicion i+1,j+1 en la celda embebida
    if celda_embebida[iarb+1,jarb+1] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*celda_embebida[iarb+1,jarb+1] + J * celda_embebida[iarb+1,jarb+1] * (celda_embebida[iarb+2,jarb+2] + celda_embebida[iarb+2,jarb+1] + celda_embebida[iarb+2,jarb] + celda_embebida[iarb,jarb+2] + celda_embebida[iarb,jarb+1] + celda_embebida[iarb,jarb] )
      Eactual = Energy(celda,celda_embebida, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        celda_embebida2[iarb+1,jarb+1] = -celda_embebida[iarb+1,jarb+1]
        celda2[iarb,jarb] = -celda[iarb,jarb]
      # si la nueva energia no es menor entonces
  return [celda2, celda_embebida2]



###################################################################### generador de la evolucion de la celda

def evolucionar_celda(celda, celda_embebida, ensamble, ensamble_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
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


########################################################################################
def m_vs_h_paramagneto(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_deceldas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  lenh = len(Hs)

  for h in Hs:
      energia_, magnetizacion_ = evolucionar_celda(ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  return Hs, magnetizaciones, energias

#####################################################################################
def energia_relajacion(q,p,l, J,H,mu,T,f):
  "q es la disolucion magnetica, p la probabilidad de que sea spin negativo, l la longitud de la cadena, J la energia de interaccion, H el campo externo, mu el moemnto magnetico, T la temperatura, f el numero de pasos monte carlo"
  celda, celda_embebida = generador_deceldas(l,p,1-p-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno
  energia_relajacion = []
  ensamble = []
  ensamble_embebido = []
  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  for j in range(0,f):
    energia_relajacion.append(evolucionar_celda(ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,T,1)[0])
  return [np.array(energia_relajacion), ensamble, ensamble_embebido]

#####################################################################################
### Hacer histeresis. En el proceso de histeresis, la magnetizacion inicial debe de ser completamente negativa o completamente positiva
def histeresis(q,l,Binicial,Bfinal,deltab,J,mu,T,n):

  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_deceldas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_celda(celda, celda_embebida, ensamble, ensamble_embebido,J,Binicial,mu,T,n)

  for b in Bs:
      energia_, magnetizacion_ = evolucionar_celda(ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,b,mu,T,n)
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

###################################################################################
def m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  ensamble = [] ## aqui se guardan snapshots
  ensamble_embebido = [] ## // // // // embebidos
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_deceldas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  ensamble.append(celda)
  ensamble_embebido.append(celda_embebida)
  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_celda(celda, celda_embebida, ensamble, ensamble_embebido,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_celda(ensamble[-1], ensamble_embebido[-1], ensamble, ensamble_embebido,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  return Ts, magnetizaciones, energias, ensamble
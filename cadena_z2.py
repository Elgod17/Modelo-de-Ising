import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from numba import njit
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
#@njit
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
#@njit
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

    cadena2 = cadena.copy()
    cadena_embebida2 = cadena_embebida.copy()

    while w == 0:
        filasarray = np.arange(0, l, 1)
        iarb = random.choice(filasarray)

        if cadena[iarb] != 0:
            w = 1
            e = B * mu * cadena_embebida[iarb + 1] + J * cadena_embebida[iarb + 1] * (
                cadena_embebida[iarb + 2] + cadena_embebida[iarb]
            )
            Eactual = Energy_2(l, cadena, cadena_embebida, J, B, mu)[0]
            r = random.uniform(0, 1)

            if Eactual >= Eactual + 2 * e or r < np.exp(-2 * e / (k * T)):
                cadena_embebida2[iarb + 1] = -cadena_embebida[iarb + 1]
                cadena2[iarb] = -cadena[iarb]

    return [cadena2, cadena_embebida2]


############################
# Evolución temporal
############################
def evolucionar_2(l, celda, celda_embebida, ensamble, ensamble_embebido, J, B, mu, T, n):
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
    a = celda.copy()
    b = celda_embebida.copy()

    for j in range(0, n):
        ab = flip_spin_2(l, a, b, J, B, mu, T).copy()
        a, b = ab[0], ab[1]
        ensamble.append(a)
        ensamble_embebido.append(b)

    Cantidades = Energy_2(l, ensamble[-1], ensamble_embebido[-1], J, B, mu)
    return Cantidades[0], Cantidades[1]




############################
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

  return Hs, magnetizaciones



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
  return np.array(energia_relajacion)


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

  return Ts, magnetizaciones, energias, ensamble







#histeresis_2(0,1000,-300,300,0.05,1,0.5,15,1)
#Ts3, magnetizaciones3, energias3, ensamble3 = m_vs_T_ferro_2(0,6000,1,70,1,1,0.1,10,900,30000)
#plt.plot(Ts3,magnetizaciones3,'o',ms = 2, label = "60")   
#plt.legend()
#plt.grid(True)
#plt.show()
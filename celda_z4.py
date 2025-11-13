import numpy as np
import random  
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
import csv
from numba import njit
import concurrent.futures
########################################### Z = 4     cuadrado

def generador_4(l, p_menosuno, p_masuno, p_cero):
  celda = [] # celda vacia para armar la red
  celda_embebida = []
  celdarray = np.array([])

  # crear estructura de un bloque
  fila = np.ones(l)
  for i in range(0,l):
    celda.append(fila)
  # convierte la celda a array para hacer operaciones más facilmente
  celda = np.array(celda)


  # se le asigna aleatoriamente un valor de -1/2 o de 1/2 a los nodos con valor distinto de cero
  for i in range(0,l):
    for j in range(0,l):
      celda[i,j] = np.random.choice([-1,1,0], p = [p_menosuno,p_masuno,p_cero])
  # crear celda embebida
  cero = np.array([0])
  celda_embebida.append(np.zeros(l+2))
  for i in range(0,len(celda)):
    celda_embebida.append(np.concatenate((cero,celda[i],cero)))
  celda_embebida.append(np.zeros(l+2))
  celda_embebida = np.array(celda_embebida)

  return celda, celda_embebida
##################################################################################  FERROMAGNETICO
@njit
def Energy_4(l,celda, celda_embebida0, jota, be, muu):
  Energia_interaccion = 0
  if jota != 0:
    for j in range(0,l):    ## recorrer cada nodo con posicion j,k
      for k in range(0,l):
        if celda[j,k] != 0:
          Energia_interaccion += celda_embebida0[j+1,k+1] * (celda_embebida0[j+1+1,k+1] + celda_embebida0[j+1,k+1+1]) # se suma la energia de interaccion del spin j,k con los espines más cercanos que estan en la siguiente fila, que es la fila j+1
    Energia_interaccion = - jota * Energia_interaccion

  M = muu * np.sum(celda)
  Energia_zeeman = -be * M
  Energia = Energia_interaccion + Energia_zeeman
  return Energia, M
##################################################################################
@njit
def flip_spin_4(l,celda, celda_embebida,J,B,mu,T):
  #k = 1.380649 * 1e-23  # constante de Bolsman
  k=1
  ## este algoritmo genera nuevas configuraciones de los spines a partir de la configuracion anterior
  w = 0
  ## crear copias de las redes para guardar las nuevas configuraciones en lugares distintos de la memoria del computador
  ## vamos a seleccionar un nodo i,j para cambiarle el sentido del spin. Desde luego, esto solo tiene sentido si el nodo i,j es distinto de cero.
  ## Asi que se introduce en el algoritmo la condicion de que solo se ejecute si el i,j seleccionado aleatoriamente es distinto de cero
  while w == 0:
    filasarray = np.arange(0,l,1)
    ## seleccionar i,j aleatoriamente
    iarb = np.random.choice(filasarray)
    jarb = np.random.choice(filasarray)
    # la posicion i,j en celda se corresponde con la posicion i+1,j+1 en la celda embebida
    if celda[iarb,jarb] != 0:
      w = 1
      # calcular la nueva energia del spin invertido
      e = B*mu*celda_embebida[iarb+1,jarb+1] + J * celda_embebida[iarb+1,jarb+1] * (celda_embebida[iarb+1+1,jarb+1] + celda_embebida[iarb,jarb+1] + celda_embebida[iarb+1,jarb+1+1] + celda_embebida[iarb+1,jarb])
      Eactual = Energy_4(l,celda,celda_embebida, J,B,mu)[0]
      r = random.uniform(0,1)
      if Eactual >= Eactual + 2 * e or r < np.exp(-2*e/(k*T)):   ## aceptar nueva  nueva configuracion de la red si la nueva energia es menor a la energia total anterior
        celda_embebida[iarb+1,jarb+1] = -celda_embebida[iarb+1,jarb+1]
        celda[iarb,jarb] = -celda[iarb,jarb]
      # si la nueva energia no es menor entonces
  return [celda, celda_embebida]




def evolucionar_4(l,celda, celda_embebida, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria

  for j in range(0,n):
    flip_spin_4(l,celda, celda_embebida, J,B,mu,T)
  Cantidades = Energy_4(l,celda,celda_embebida, J, B, mu)
  return Cantidades[0], Cantidades[1]


def evolucionar_4_copy(l,celda, celda_embebida, ensamble, ensamble_embebido, J,B,mu,T,n):   ############### esta funcion funciona para cualquier geometria
  ## crear copy para guardar configuraciones en disitntos espacios de memoria
  a = celda.copy()
  b = celda_embebida.copy()
  ## generar evolucion de n pasos en la configuracion de los espines

  for j in range(0,n):
    ab = flip_spin_4(l,a, b, J,B,mu,T).copy()
    a,b = ab[0], ab[1]
    ensamble.append(a)
    ensamble_embebido.append(b)

  Cantidades = Energy_4(l,ensamble[-1],ensamble_embebido[-1], J, B, mu)

  return Cantidades[0], Cantidades[1]


################################################################################################

def m_vs_h_paramagneto_4(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_4(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  evolucionar_4(l,celda, celda_embebida,J,Hinicial,mu,T,n)

  for h in Hs:
      energia_, magnetizacion_ = evolucionar_4(l,celda,celda_embebida,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  m_normalizado = np.array(magnetizaciones)/abs(magnetizaciones[0])
  til = "param_q"+str(q)+"T"+str(T)+"_z4.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "M", "Mn"])  # encabezados opcionales
    for i in range(len(Hs)):
        writer.writerow([Hs[i], magnetizaciones[i], m_normalizado[i]])

  #plt.figure()
  #plt.plot(Hs, magnetizaciones, 'o', ms=2)
  #plt.title(til)
  #plt.grid()
  #plt.show()


  return Hs, magnetizaciones

#N = 2000
#l8 = int(np.sqrt(2*N))
###m_vs_h_paramagneto(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n)
#modH = 120
#simulaciones = [
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 9, 1), 
#    (0, l8, -modH, modH, 0.0001, 1, 0.5, 4, 1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 9, 1), 
#    (0.5, l8, -modH, modH, 0.0001, 1, 0.5, 4, 1),
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 15, 1),
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 9, 1), 
#    (0.8, l8, -modH, modH, 0.0001, 1, 0.5, 4, 1)
#    
#]
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#        futures = {executor.submit(m_vs_h_paramagneto_4, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
#################################################################################################
#
#
#
def histeresis_4(q,l,Binicial,Bfinal,deltab,J,mu,T,n):

  energias = []
  magnetizaciones = []
  celda, celda_embebida = generador_4(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_4(l,celda, celda_embebida,J,Binicial,mu,T,n)
  for b in Bs:
      energia_, magnetizacion_ = evolucionar_4(l,celda,celda_embebida,J,b,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)
  
  til = "histeresis_q"+str(q)+"_z4.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "M"])  # encabezados opcionales
    for i in range(len(Bs)):
        writer.writerow([Bs[i], magnetizaciones[i]])

  plt.figure()
  plt.plot(Bs, magnetizaciones, 'o', ms=2)
  plt.title(til)
  plt.grid()
  plt.show()

  return Bs, magnetizaciones




def m_vs_T_ferro_4(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  celda, celda_embebida = generador_4(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
  lent = len(Ts)
  evolucionar_4(l,celda, celda_embebida,J,H,mu,Tinicial,f)

  for t in Ts:
      energia_, magnetizacion_ = evolucionar_4(l,celda, celda_embebida,J,H,mu,t,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  m_normalizado = np.array(magnetizaciones)/magnetizaciones[0]

  til = "mtferro_q"+str(q)+"_z4.csv"
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
#l8 = int(np.sqrt(2*N))
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
#        futures = {executor.submit(m_vs_T_ferro_4, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")












############### ENERGIA DE RELAJACION ####################################################
def energia_relajacion_4(q,p,l, J,H,mu,T,n):
  """q es la probabilidad de que el nodo sea vacío,
    ,p la probabilidad de que el nodo tenga spin negativo, 
  l**2 es la longitud de un lado de cuadrado, J la energia de interaccion, H la induccion magnetica, mu el momento magnetico, T la temperatura y n el numero de replicas para promediar la energia de relajacion
  T la temperatura y n el numero de pasos Monte Carlo
  """
  celda, celda_embebida = generador_4(l,p,1-p-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno

  energia_relajacion = []
  for j in range(n):
    energia_relajacion.append(evolucionar_4(l,celda,celda_embebida,J,H,mu,T,1)[0])


  til = "Energia de relajacion_q="+str(q)+",T="+str(T)+",z=4,H="+str(H)+".csv"
  with open(til, "w", newline="") as f:
      writer = csv.writer(f)
      writer.writerow(["E"])  # encabezado
      for energia in energia_relajacion:
          writer.writerow([energia])

  plt.show()
  plt.plot(energia_relajacion, 'o', ms=2)
  plt.title(til)
  plt.show()


  return energia_relajacion


N = 1000
l8 = int(np.sqrt(2*N))


#energia_relajacion_2(q,p,l, J,H,mu,T,f)
simulaciones = [
    (0,0,l8, 1,10,0.5,15,15000),
    (0.5,0,l8, 1,10,0.5,15,15000),
    (0.8,0,l8, 1,10,0.5,15,15000)
    
]


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(energia_relajacion_4, *args): args[0] for args in simulaciones}
        for future in concurrent.futures.as_completed(futures):
            q = futures[future]
            try:
                result = future.result()  # Esto fuerza a que la función termine
                print(f"Simulación con q={q} terminada")
            except Exception as e:
                print(f"Simulación con q={q} falló: {e}")

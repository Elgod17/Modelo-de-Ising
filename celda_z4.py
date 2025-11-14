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





################################################ MAGNETIZACION VS TEMPERATURA  Z = 4  #######################################################

################################################ MAGNETIZACION VS TEMPERATURA  Z = 4  #######################################################

def m_vs_T_ferro_4_single(q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f):
    """Una sola réplica de magnetización vs temperatura"""
    energias = []
    magnetizaciones = []
    
    # Generar celda inicial con p=0
    celda, celda_embebida = generador_4(l, 0, 1-q, q)
    
    # Termalizar a temperatura inicial
    evolucionar_4(l, celda, celda_embebida, J, H, mu, Tinicial, f)
    
    # Evolucionar a cada temperatura
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    for t in Ts:
        energia_, magnetizacion_ = evolucionar_4(l, celda, celda_embebida, J, H, mu, t, n)
        energias.append(energia_)
        magnetizaciones.append(magnetizacion_)
    
    return energias, magnetizaciones


# FUNCIÓN AUXILIAR PARA PARALELIZAR RÉPLICAS
def replica_individual_mT_4(args):
    """Ejecuta una réplica individual de m vs T"""
    q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, replica_num = args
    
    energias = []
    magnetizaciones = []
    
    # Generar celda inicial con p=0
    celda, celda_embebida = generador_4(l, 0, 1-q, q)
    
    # Termalizar a temperatura inicial
    evolucionar_4(l, celda, celda_embebida, J, H, mu, Tinicial, f)
    
    # Evolucionar a cada temperatura
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    for t in Ts:
        energia_, magnetizacion_ = evolucionar_4(l, celda, celda_embebida, J, H, mu, t, n)
        energias.append(energia_)
        magnetizaciones.append(magnetizacion_)
    
    return energias, magnetizaciones


def m_vs_T_ferro_4_optimizada(q, N, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, num_replicas=10, max_workers=10):
    """Versión optimizada que paraleliza las réplicas individuales"""
    
    print(f"\nIniciando simulación m vs T: q={q}, N={N}")
    print(f"  Rango de temperaturas: {Tinicial} → {Tfinal} (ΔT={deltaT})")
    print(f"  Ejecutando {num_replicas} réplicas en paralelo con {max_workers} núcleos...")
    
    # Preparar argumentos para cada réplica
    args_replicas = [(q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, i) for i in range(num_replicas)]
    
    # Ejecutar réplicas en paralelo
    todas_energias = []
    todas_magnetizaciones = []
    completadas = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(replica_individual_mT_4, args): i for i, args in enumerate(args_replicas)}
        
        for future in concurrent.futures.as_completed(futures):
            replica_num = futures[future]
            try:
                energias, magnetizaciones = future.result()
                todas_energias.append(energias)
                todas_magnetizaciones.append(magnetizaciones)
                completadas += 1
                if completadas % 5 == 0 or completadas == num_replicas:
                    print(f"    Progreso: {completadas}/{num_replicas} réplicas completadas")
            except Exception as e:
                print(f"    ✗ Réplica {replica_num+1} falló: {e}")
    
    # Convertir a numpy array para calcular estadísticas
    todas_energias = np.array(todas_energias)
    todas_magnetizaciones = np.array(todas_magnetizaciones)
    
    # Calcular promedio y error estándar
    energia_promedio = np.mean(todas_energias, axis=0)
    error_energia = np.std(todas_energias, axis=0) / np.sqrt(len(todas_energias))
    
    mag_promedio = np.mean(todas_magnetizaciones, axis=0)
    error_mag = np.std(todas_magnetizaciones, axis=0) / np.sqrt(len(todas_magnetizaciones))
    
    # Normalizar por el valor inicial
    m_normalizado = mag_promedio / mag_promedio[0]
    error_normalizado = error_mag / mag_promedio[0]
    
    # Temperaturas
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    # Guardar datos individuales en NPZ (comprimido)
    nombre_npz = f"Magnetizacion_vs_T_N={N},q={q},Ti={Tinicial},Tf={Tfinal},z=4,H={H}_individual.npz"
    np.savez_compressed(nombre_npz,
                       temperaturas=Ts,
                       energia_promedio=energia_promedio,
                       error_energia=error_energia,
                       magnetizacion_promedio=mag_promedio,
                       magnetizacion_normalizada=m_normalizado,
                       error_mag=error_mag,
                       error_normalizado=error_normalizado,
                       q=q, N=N, Tinicial=Tinicial, Tfinal=Tfinal, 
                       H=H, J=J, mu=mu,
                       num_replicas=len(todas_magnetizaciones))
    
    print(f"✓ Simulación q={q} completada y guardada en {nombre_npz}\n")
    return q, N, Ts, energia_promedio, mag_promedio, m_normalizado, error_energia, error_mag, error_normalizado


N = 946
# Para z=4 (red cuadrada): N_sitios = l^2/2
l = int(np.ceil(np.sqrt(2*N)))
N_real = l**2 // 2

print(f"N solicitado: {N}")
print(f"l calculado: {l}")
print(f"N real (sitios efectivos): {N_real}")

# Parámetros ajustables
NUM_NUCLEOS = 10  # Número total de núcleos disponibles
NUM_REPLICAS = 150  # Número de réplicas por simulación

# Parámetros de temperatura
TINICIAL = 0.1
TFINAL = 60
DELTA_T = 0.5

# m_vs_T_ferro_4_optimizada(q, N, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, num_replicas, max_workers)
simulaciones = [
    (0,   N_real, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS),
    (0.5, N_real, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS),
    (0.8, N_real, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS)
]


if __name__ == "__main__":
    resultados = []
    
    print(f"\n" + "=" * 70)
    print(f"INICIANDO SIMULACIONES M vs T")
    print(f"=" * 70)
    print(f"Total de configuraciones: {len(simulaciones)}")
    print(f"Réplicas por configuración: {NUM_REPLICAS}")
    print(f"Núcleos disponibles: {NUM_NUCLEOS}")
    print(f"N (sitios efectivos): {N_real}")
    print(f"Rango T: {TINICIAL} → {TFINAL} (ΔT={DELTA_T})")
    print(f"=" * 70)
    
    # ESTRATEGIA: Ejecutar simulaciones secuencialmente, 
    # pero paralelizar las réplicas dentro de cada una
    for args in simulaciones:
        try:
            result = m_vs_T_ferro_4_optimizada(*args)
            resultados.append(result)
        except Exception as e:
            q = args[0]
            print(f"✗ Simulación con q={q} falló: {e}\n")
    
    # Ordenar resultados por q
    resultados.sort(key=lambda x: x[0])
    
    # Crear gráfico con dos subplots (sin energía)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(resultados)))
    
    for idx, (q, N_sim, Ts, energia_promedio, mag_promedio, m_normalizado, 
              error_energia, error_mag, error_normalizado) in enumerate(resultados):
        
        # Gráfico 1: Magnetización absoluta
        ax1.plot(Ts, mag_promedio, '-', color=colores[idx], 
                label=f'q={q}', linewidth=2, alpha=0.8)
        ax1.fill_between(Ts, 
                        mag_promedio - error_mag,
                        mag_promedio + error_mag,
                        color=colores[idx], alpha=0.2)
        
        # Gráfico 2: Magnetización normalizada
        ax2.plot(Ts, m_normalizado, '-', color=colores[idx], 
                label=f'q={q}', linewidth=2, alpha=0.8)
        ax2.fill_between(Ts, 
                        m_normalizado - error_normalizado,
                        m_normalizado + error_normalizado,
                        color=colores[idx], alpha=0.2)
    
    # Configurar gráfico 1
    ax1.set_xlabel('Temperatura (T)', fontsize=12)
    ax1.set_ylabel('Magnetización', fontsize=12)
    ax1.set_title(f'Magnetización vs Temperatura\n(N={N_real}, H={simulaciones[0][8]}, z=4)', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Configurar gráfico 2
    ax2.set_xlabel('Temperatura (T)', fontsize=12)
    ax2.set_ylabel('Magnetización Normalizada (m/m₀)', fontsize=12)
    ax2.set_title(f'Magnetización Normalizada vs Temperatura\n(N={N_real}, H={simulaciones[0][8]}, z=4)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.suptitle(f'{NUM_REPLICAS} réplicas por configuración', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Guardar también la figura
    nombre_figura = f"grafica_m_vs_T_N={N_real}_Ti={TINICIAL}_Tf={TFINAL}_H={simulaciones[0][8]}_z4.png"
    plt.savefig(nombre_figura, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada en: {nombre_figura}")
    
    plt.show()
    
    # Guardar datos consolidados en formato .npz
    datos_guardar = {}
    for q, N_sim, Ts, energia_promedio, mag_promedio, m_normalizado, error_energia, error_mag, error_normalizado in resultados:
        datos_guardar[f'q_{q}_energia'] = energia_promedio
        datos_guardar[f'q_{q}_error_energia'] = error_energia
        datos_guardar[f'q_{q}_mag'] = mag_promedio
        datos_guardar[f'q_{q}_mag_norm'] = m_normalizado
        datos_guardar[f'q_{q}_error_mag'] = error_mag
        datos_guardar[f'q_{q}_error_norm'] = error_normalizado
    
    datos_guardar['temperaturas'] = resultados[0][2]
    datos_guardar['parametros'] = np.array([simulaciones[0][6:9]])  # J, mu, H
    datos_guardar['N'] = N_real
    datos_guardar['num_replicas'] = NUM_REPLICAS
    datos_guardar['z'] = 4
    
    nombre_archivo = f"resultados_consolidados_m_vs_T_N={N_real}_Ti={TINICIAL}_Tf={TFINAL}_H={simulaciones[0][8]}_z4.npz"
    np.savez_compressed(nombre_archivo, **datos_guardar)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Datos consolidados guardados en: {nombre_archivo}")
    print(f"  Claves guardadas: {list(datos_guardar.keys())}")
    print(f"{'=' * 70}")
    print("TODAS LAS SIMULACIONES COMPLETADAS")
    print(f"{'=' * 70}")










#####################################################################################################

#def m_vs_T_ferro_4(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
#  energias = []
#  magnetizaciones = []
#  ## q = 1-probabilidad asignar spin menos uno
#  celda, celda_embebida = generador_4(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero
#
#  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
#  lent = len(Ts)
#  evolucionar_4(l,celda, celda_embebida,J,H,mu,Tinicial,f)
#
#  for t in Ts:
#      energia_, magnetizacion_ = evolucionar_4(l,celda, celda_embebida,J,H,mu,t,n)
#      energias.append(energia_)
#      magnetizaciones.append(magnetizacion_)
#
#  m_normalizado = np.array(magnetizaciones)/magnetizaciones[0]
#
#  til = "mtferro_q"+str(q)+"_z4.csv"
#  with open(til, "w", newline="") as f:
#    writer = csv.writer(f)
#    writer.writerow(["T", "M", "Mn"])  # encabezados opcionales
#    for i in range(len(Ts)):
#        writer.writerow([Ts[i], magnetizaciones[i], m_normalizado[i]])
#  plt.figure()
#  plt.plot(Ts, m_normalizado, 'o', ms=2)
#  plt.title(til)
#  plt.show()
#  return Ts, magnetizaciones

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
################################################ ENERGIA DE RELAJACION  Z = 4  #######################################################

################################################ ENERGIA DE RELAJACION  Z = 4  #######################################################

#def energia_relajacion_2(q, p, N, l, J, H, mu, T, n, num_replicas=10):
#    """q es la probabilidad de que el nodo sea vacío,
#    p la probabilidad de que el nodo tenga spin negativo, 
#    l**2 es tal que da el numero de hexagonos, J la energia de interaccion, 
#    H la induccion magnetica, mu el momento magnetico, T la temperatura 
#    y n el numero de pasos Monte Carlo, num_replicas es el números de simulaciones independientes
#    """
#    
#    # Almacenar todas las réplicas
#    todas_energias = []
#    
#    for replica in range(num_replicas):
#        celda, celda_embebida = generador_4(l, p, 1-p-q, q)
#        energia_relajacion = []
#        
#        for j in range(n):
#            energia_relajacion.append(evolucionar_4(l, celda, celda_embebida, J, H, mu, T, 1)[0])
#        
#        todas_energias.append(energia_relajacion)
#        print(f"  Réplica {replica+1}/{num_replicas} completada para q={q}, p={p}, N={N}")
#    
#    # Convertir a numpy array para calcular estadísticas
#    todas_energias = np.array(todas_energias)
#    
#    # Calcular promedio y error estándar
#    energia_promedio = np.mean(todas_energias, axis=0)
#    error_estandar = np.std(todas_energias, axis=0) / np.sqrt(num_replicas)
#    
#    # Guardar datos individuales en NPZ
#    nombre_npz = f"Energia_relajacion_N={N},q={q},p={p},T={T},z=4,H={H}_individual.npz"
#    np.savez_compressed(nombre_npz,
#                       pasos=np.arange(len(energia_promedio)),
#                       energia_promedio=energia_promedio,
#                       error_estandar=error_estandar,
#                       q=q, p=p, N=N, T=T, H=H)
#    
#    return q, p, N, energia_promedio, error_estandar
#
#
## FUNCIÓN AUXILIAR PARA PARALELIZAR RÉPLICAS
#def replica_individual_4(args):
#    """Ejecuta una réplica individual de la simulación"""
#    q, p, l, J, H, mu, T, n, replica_num = args
#    
#    celda, celda_embebida = generador_4(l, p, 1-p-q, q)
#    energia_relajacion = []
#    
#    for j in range(n):
#        energia_relajacion.append(evolucionar_4(l, celda, celda_embebida, J, H, mu, T, 1)[0])
#    
#    return energia_relajacion
#
#
#def energia_relajacion_2_optimizada(q, p, N, l, J, H, mu, T, n, num_replicas=10, max_workers=10):
#    """Versión optimizada que paraleliza las réplicas individuales"""
#    
#    print(f"\nIniciando simulación: q={q}, p={p}, N={N}")
#    print(f"  Ejecutando {num_replicas} réplicas en paralelo con {max_workers} núcleos...")
#    
#    # Preparar argumentos para cada réplica
#    args_replicas = [(q, p, l, J, H, mu, T, n, i) for i in range(num_replicas)]
#    
#    # Ejecutar réplicas en paralelo
#    todas_energias = []
#    completadas = 0
#    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#        futures = {executor.submit(replica_individual_4, args): i for i, args in enumerate(args_replicas)}
#        
#        for future in concurrent.futures.as_completed(futures):
#            replica_num = futures[future]
#            try:
#                result = future.result()
#                todas_energias.append(result)
#                completadas += 1
#                if completadas % 5 == 0 or completadas == num_replicas:
#                    print(f"    Progreso: {completadas}/{num_replicas} réplicas completadas")
#            except Exception as e:
#                print(f"    ✗ Réplica {replica_num+1} falló: {e}")
#    
#    # Convertir a numpy array para calcular estadísticas
#    todas_energias = np.array(todas_energias)
#    
#    # Calcular promedio y error estándar
#    energia_promedio = np.mean(todas_energias, axis=0)
#    error_estandar = np.std(todas_energias, axis=0) / np.sqrt(len(todas_energias))
#    
#    # Guardar datos individuales en NPZ (comprimido)
#    nombre_npz = f"Energia_relajacion_N={N},q={q},p={p},T={T},z=4,H={H}_individual.npz"
#    np.savez_compressed(nombre_npz,
#                       pasos=np.arange(len(energia_promedio)),
#                       energia_promedio=energia_promedio,
#                       error_estandar=error_estandar,
#                       q=q, p=p, N=N, T=T, H=H, J=J, mu=mu,
#                       num_replicas=len(todas_energias))
#    
#    print(f"✓ Simulación q={q}, p={p} completada y guardada en {nombre_npz}\n")
#    return q, p, N, energia_promedio, error_estandar
#
#
#N = 500
## Para z=4 (red cuadrada): N_sitios = l^2/2, entonces l = sqrt(2*N)
## Redondear hacia arriba para asegurar al menos N sitios
#l8 = int(np.ceil(np.sqrt(2*N)))
#N_real = l8**2 // 2  # Calcular el N real que se usará
#
#print(f"N solicitado: {N}")
#print(f"l calculado: {l8}")
#print(f"N real (sitios efectivos): {N_real}")
#
## Parámetros ajustables
#NUM_NUCLEOS = 10  # Número total de núcleos disponibles
#NUM_REPLICAS = 500  # Número de réplicas por simulación
#
## energia_relajacion_2_optimizada(q, p, N, l, J, H, mu, T, n, num_replicas, max_workers)
#simulaciones = [
#    (0,   0.5, N_real, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS),
#    (0.5, 0.25, N_real, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS),
#    (0.8, 0.1, N_real, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS)
#]
#
#
#if __name__ == "__main__":
#    resultados = []
#    
#    print(f"\n" + "=" * 70)
#    print(f"INICIANDO SIMULACIONES")
#    print(f"=" * 70)
#    print(f"Total de simulaciones: {len(simulaciones)}")
#    print(f"Réplicas por simulación: {NUM_REPLICAS}")
#    print(f"Núcleos disponibles: {NUM_NUCLEOS}")
#    print(f"N (sitios efectivos): {N_real}")
#    print(f"=" * 70)
#    
#    # ESTRATEGIA: Ejecutar simulaciones secuencialmente, 
#    # pero paralelizar las réplicas dentro de cada una
#    for args in simulaciones:
#        try:
#            result = energia_relajacion_2_optimizada(*args)
#            resultados.append(result)
#        except Exception as e:
#            q, p = args[0], args[1]
#            print(f"✗ Simulación con q={q}, p={p} falló: {e}\n")
#    
#    # Ordenar resultados por q
#    resultados.sort(key=lambda x: x[0])
#    
#    # Crear gráfico único con todas las simulaciones
#    plt.figure(figsize=(12, 7))
#    
#    colores = plt.cm.viridis(np.linspace(0, 0.9, len(resultados)))
#    
#    for idx, (q, p, N_sim, energia_promedio, error_estandar) in enumerate(resultados):
#        pasos = np.arange(len(energia_promedio))
#        
#        # Graficar línea promedio
#        plt.plot(pasos, energia_promedio, '-', color=colores[idx], 
#                label=f'q={q}, p={p}', linewidth=1.5, alpha=0.8)
#        
#        # Graficar sombra de error
#        plt.fill_between(pasos, 
#                        energia_promedio - error_estandar,
#                        energia_promedio + error_estandar,
#                        color=colores[idx], alpha=0.2)
#    
#    plt.xlabel('Paso Monte Carlo', fontsize=12)
#    plt.ylabel('Energía', fontsize=12)
#    p_sim = simulaciones[0][1]
#    N_titulo = resultados[0][2]
#    plt.title(f'Energía de relajación promedio (N={N_titulo}, T={simulaciones[0][7]}, H={simulaciones[0][5]}, z=4)\n' + 
#              f'{NUM_REPLICAS} réplicas por configuración', fontsize=13)
#    plt.legend(fontsize=10)
#    plt.grid(True, alpha=0.3)
#    plt.tight_layout()
#    
#    # Guardar también la figura
#    nombre_figura = f"grafica_relajacion_N={N_real}_T={simulaciones[0][7]}_H={simulaciones[0][5]}_z4.png"
#    plt.savefig(nombre_figura, dpi=300, bbox_inches='tight')
#    print(f"\n✓ Gráfica guardada en: {nombre_figura}")
#    
#    plt.show()
#    
#    # Guardar datos consolidados en formato .npz
#    datos_guardar = {}
#    for q, p, N_sim, energia_promedio, error_estandar in resultados:
#        datos_guardar[f'q_{q}_p_{p}_energia'] = energia_promedio
#        datos_guardar[f'q_{q}_p_{p}_error'] = error_estandar
#    
#    datos_guardar['pasos'] = np.arange(len(resultados[0][3]))
#    datos_guardar['parametros'] = np.array([simulaciones[0][4:8]])  # J, H, mu, T
#    datos_guardar['N'] = N_real
#    datos_guardar['num_replicas'] = NUM_REPLICAS
#    datos_guardar['z'] = 4
#    
#    nombre_archivo = f"resultados_consolidados_N={N_real}_T={simulaciones[0][7]}_H={simulaciones[0][5]}_z4.npz"
#    np.savez_compressed(nombre_archivo, **datos_guardar)
#    
#    print(f"\n{'=' * 70}")
#    print(f"✓ Datos consolidados guardados en: {nombre_archivo}")
#    print(f"  Claves guardadas: {list(datos_guardar.keys())}")
#    print(f"{'=' * 70}")
#    print("TODAS LAS SIMULACIONES COMPLETADAS")
#    print(f"{'=' * 70}")
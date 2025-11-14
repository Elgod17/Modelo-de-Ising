import numpy as np
import random
import matplotlib
matplotlib.use('TkAgg')  # o prueba 'Agg' si no vas a mostrar ventanas
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import csv
from numba import njit
import concurrent.futures
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



############################
def m_vs_h_paramagneto_2(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n):
  "q es la dilucion magnetica, l la longitud de la cadena, el campo magnetico H barre valores desde Hinicial hasa Hfinal en pasos de longitud deltaH, J es el parametro de interaccion, mu el moemnto magnetico, T la temperatura y n el numero de pasos de evolucion paara cada valor del campo magnetico H"

  energias = []
  magnetizaciones = []
  ## q = 1-probabilidad asignar spin menos uno
  cadena, cadena_embebida = generador_decadenas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  Hs = np.arange(Hinicial, Hfinal+deltaH, deltaH)
  lenh = len(Hs)
  evolucionar_2(l,cadena, cadena_embebida,J,Hinicial,mu,T,n)

  # calcular energia y magnetizacion para cada campo magnetico
  for h in Hs:
      energia_, magnetizacion_ = evolucionar_2(l,cadena, cadena_embebida,J,h,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)

  m_normalizado = np.array(magnetizaciones)/abs(magnetizaciones[0])
  til = "parama_q"+str(q)+"T"+str(T)+"_z2.csv"
  with open(til, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["H", "M", "Mn"])  # encabezados opcionales
    for i in range(len(Hs)):
        writer.writerow([Hs[i], magnetizaciones[i], m_normalizado[i]])

  plt.figure()
  plt.plot(Hs, magnetizaciones, 'o', ms=2)
  plt.title(til)
  plt.show()
  return Hs, magnetizaciones


#N = 100
#l8 = N
##m_vs_h_paramagneto(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n)
#modH = 120
#simulaciones = [
#    (0, l8, -modH, modH, 0.001, 1, 0.5, 15, 1),
#    (0, l8, -modH, modH, 0.001, 1, 0.5, 9, 1), 
#    (0, l8, -modH, modH, 0.001, 1, 0.5, 4, 1),  # q, l, Binicial, Bfinal, deltab, J, mu, T, n
#    (0.5, l8, -modH, modH, 0.001, 1, 0.5, 15, 1),
#    (0.5, l8, -modH, modH, 0.001, 1, 0.5, 9, 1), 
#    (0.5, l8, -modH, modH, 0.001, 1, 0.5, 4, 1),
#    (0.8, l8, -modH, modH, 0.001, 1, 0.5, 15, 1),
#    (0.8, l8, -modH, modH, 0.001, 1, 0.5, 9, 1), 
#    (0.8, l8, -modH, modH, 0.001, 1, 0.5, 4, 1)
#    
#]
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
#        futures = {executor.submit(m_vs_h_paramagneto_2, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
##########################################################################

def histeresis_2(q,l,Binicial,Bfinal,deltab,J,mu,T,n):

  energias = []
  magnetizaciones = []
  celda, celda_embebida = generador_decadenas(l,1-q,0,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero

  B_aumentando = np.arange(Binicial, Bfinal+deltab, deltab)
  B_disminuyendo = np.arange(Bfinal-deltab,Binicial,-deltab)
  Bs = np.concatenate((B_aumentando,B_disminuyendo))
  lenb = len(Bs)
  evolucionar_2(l,celda, celda_embebida,J,Binicial,mu,T,n)
  for b in Bs:
      energia_, magnetizacion_ = evolucionar_2(l,celda,celda_embebida,J,b,mu,T,n)
      energias.append(energia_)
      magnetizaciones.append(magnetizacion_)
  

  til = "histeresis_q"+str(q)+",z=2"
  nombre_archivo = f"resultados_q{q}.pyz"
  np.savez_compressed(nombre_archivo, Ts=Ts, magnetizaciones=m_normalizado)
  print(f"Datos guardados en {nombre_archivo}")

  plt.figure()
  plt.plot(Bs, magnetizaciones, 'o', ms=2)
  plt.title(til)
  plt.grid()
  plt.show()
  return Bs, magnetizaciones








################################################ MAGNETIZACION VS TEMPERATURA  Z = 2  #######################################################

def m_vs_T_ferro_2_single(q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f):
    """Una sola réplica de magnetización vs temperatura"""
    magnetizaciones = []
    
    # Generar cadena inicial
    cadena, cadena_embebida = generador_decadenas(l, 0, 1-q, q)
    
    # Termalizar a temperatura inicial
    evolucionar_2(l, cadena, cadena_embebida, J, H, mu, Tinicial, f)
    
    # Evolucionar a cada temperatura
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    for t in Ts:
        energia_, magnetizacion_ = evolucionar_2(l, cadena, cadena_embebida, J, H, mu, t, n)
        magnetizaciones.append(magnetizacion_)
    
    return magnetizaciones


# FUNCIÓN AUXILIAR PARA PARALELIZAR RÉPLICAS
def replica_individual_mT_2(args):
    """Ejecuta una réplica individual de m vs T"""
    q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, replica_num = args
    
    magnetizaciones = []
    
    # Generar cadena inicial
    cadena, cadena_embebida = generador_decadenas(l, 0, 1-q, q)
    
    # Termalizar a temperatura inicial
    evolucionar_2(l, cadena, cadena_embebida, J, H, mu, Tinicial, f)
    
    # Evolucionar a cada temperatura
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    for t in Ts:
        energia_, magnetizacion_ = evolucionar_2(l, cadena, cadena_embebida, J, H, mu, t, n)
        magnetizaciones.append(magnetizacion_)
    
    return magnetizaciones


def m_vs_T_ferro_2_optimizada(q, N, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, num_replicas=10, max_workers=10):
    """Versión optimizada que paraleliza las réplicas individuales"""
    
    print(f"\nIniciando simulación m vs T: q={q}, N={N}")
    print(f"  Rango de temperaturas: {Tinicial} → {Tfinal} (ΔT={deltaT})")
    print(f"  Ejecutando {num_replicas} réplicas en paralelo con {max_workers} núcleos...")
    
    # Preparar argumentos para cada réplica
    args_replicas = [(q, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, i) for i in range(num_replicas)]
    
    # Ejecutar réplicas en paralelo
    todas_magnetizaciones = []
    completadas = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(replica_individual_mT_2, args): i for i, args in enumerate(args_replicas)}
        
        for future in concurrent.futures.as_completed(futures):
            replica_num = futures[future]
            try:
                result = future.result()
                todas_magnetizaciones.append(result)
                completadas += 1
                if completadas % 5 == 0 or completadas == num_replicas:
                    print(f"    Progreso: {completadas}/{num_replicas} réplicas completadas")
            except Exception as e:
                print(f"    ✗ Réplica {replica_num+1} falló: {e}")
    
    # Convertir a numpy array para calcular estadísticas
    todas_magnetizaciones = np.array(todas_magnetizaciones)
    
    # Calcular promedio y error estándar
    mag_promedio = np.mean(todas_magnetizaciones, axis=0)
    error_estandar = np.std(todas_magnetizaciones, axis=0) / np.sqrt(len(todas_magnetizaciones))
    
    # Normalizar por el valor inicial
    m_normalizado = mag_promedio / mag_promedio[0]
    error_normalizado = error_estandar / mag_promedio[0]
    
    # Temperaturas
    Ts = np.arange(Tinicial + deltaT, Tfinal + deltaT, deltaT)
    
    # Guardar datos individuales en NPZ (comprimido)
    nombre_npz = f"Magnetizacion_vs_T_N={N},q={q},Ti={Tinicial},Tf={Tfinal},z=2,H={H}_individual.npz"
    np.savez_compressed(nombre_npz,
                       temperaturas=Ts,
                       magnetizacion_promedio=mag_promedio,
                       magnetizacion_normalizada=m_normalizado,
                       error_estandar=error_estandar,
                       error_normalizado=error_normalizado,
                       q=q, N=N, Tinicial=Tinicial, Tfinal=Tfinal, 
                       H=H, J=J, mu=mu,
                       num_replicas=len(todas_magnetizaciones))
    
    print(f"✓ Simulación q={q} completada y guardada en {nombre_npz}\n")
    return q, N, Ts, mag_promedio, m_normalizado, error_estandar, error_normalizado


N = 946
l = N  # Para z=2, N_sitios = l

# Parámetros ajustables
NUM_NUCLEOS = 10  # Número total de núcleos disponibles
NUM_REPLICAS = 90  # Número de réplicas por simulación

# Parámetros de temperatura
TINICIAL = 0.1
TFINAL = 60
DELTA_T = 0.5

# m_vs_T_ferro_2_optimizada(q, N, l, Tinicial, Tfinal, deltaT, J, mu, H, n, f, num_replicas, max_workers)
simulaciones = [
    (0,   N, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS),
    (0.5, N, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS),
    (0.8, N, l, TINICIAL, TFINAL, DELTA_T, 1, 0.5, 10, 1000, 5000, NUM_REPLICAS, NUM_NUCLEOS)
]


if __name__ == "__main__":
    resultados = []
    
    print(f"=" * 70)
    print(f"INICIANDO SIMULACIONES M vs T")
    print(f"=" * 70)
    print(f"Total de configuraciones: {len(simulaciones)}")
    print(f"Réplicas por configuración: {NUM_REPLICAS}")
    print(f"Núcleos disponibles: {NUM_NUCLEOS}")
    print(f"N (sitios): {N}")
    print(f"Rango T: {TINICIAL} → {TFINAL} (ΔT={DELTA_T})")
    print(f"=" * 70)
    
    # ESTRATEGIA: Ejecutar simulaciones secuencialmente, 
    # pero paralelizar las réplicas dentro de cada una
    for args in simulaciones:
        try:
            result = m_vs_T_ferro_2_optimizada(*args)
            resultados.append(result)
        except Exception as e:
            q = args[0]
            print(f"✗ Simulación con q={q} falló: {e}\n")
    
    # Ordenar resultados por q
    resultados.sort(key=lambda x: x[0])
    
    # Crear gráfico único con todas las simulaciones
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    colores = plt.cm.viridis(np.linspace(0, 0.9, len(resultados)))
    
    for idx, (q, N_sim, Ts, mag_promedio, m_normalizado, error_estandar, error_normalizado) in enumerate(resultados):
        
        # Gráfico 1: Magnetización absoluta
        ax1.plot(Ts, mag_promedio, '-', color=colores[idx], 
                label=f'q={q}', linewidth=2, alpha=0.8)
        ax1.fill_between(Ts, 
                        mag_promedio - error_estandar,
                        mag_promedio + error_estandar,
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
    ax1.set_title(f'Magnetización vs Temperatura (N={N}, H={simulaciones[0][8]}, z=2)\n' + 
                  f'{NUM_REPLICAS} réplicas por configuración', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Configurar gráfico 2
    ax2.set_xlabel('Temperatura (T)', fontsize=12)
    ax2.set_ylabel('Magnetización Normalizada (m/m₀)', fontsize=12)
    ax2.set_title(f'Magnetización Normalizada vs Temperatura (N={N}, H={simulaciones[0][8]}, z=2)\n' + 
                  f'{NUM_REPLICAS} réplicas por configuración', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar también la figura
    nombre_figura = f"grafica_m_vs_T_N={N}_Ti={TINICIAL}_Tf={TFINAL}_H={simulaciones[0][8]}_z2.png"
    plt.savefig(nombre_figura, dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfica guardada en: {nombre_figura}")
    
    plt.show()
    
    # Guardar datos consolidados en formato .npz
    datos_guardar = {}
    for q, N_sim, Ts, mag_promedio, m_normalizado, error_estandar, error_normalizado in resultados:
        datos_guardar[f'q_{q}_mag'] = mag_promedio
        datos_guardar[f'q_{q}_mag_norm'] = m_normalizado
        datos_guardar[f'q_{q}_error'] = error_estandar
        datos_guardar[f'q_{q}_error_norm'] = error_normalizado
    
    datos_guardar['temperaturas'] = resultados[0][2]
    datos_guardar['parametros'] = np.array([simulaciones[0][6:9]])  # J, mu, H
    datos_guardar['N'] = N
    datos_guardar['num_replicas'] = NUM_REPLICAS
    datos_guardar['z'] = 2
    
    nombre_archivo = f"resultados_consolidados_m_vs_T_N={N}_Ti={TINICIAL}_Tf={TFINAL}_H={simulaciones[0][8]}_z2.npz"
    np.savez_compressed(nombre_archivo, **datos_guardar)
    
    print(f"\n{'=' * 70}")
    print(f"✓ Datos consolidados guardados en: {nombre_archivo}")
    print(f"  Claves guardadas: {list(datos_guardar.keys())}")
    print(f"{'=' * 70}")
    print("TODAS LAS SIMULACIONES COMPLETADAS")
    print(f"{'=' * 70}")


#def m_vs_T_ferro_2(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f):
#  energias = []
#  magnetizaciones = []
#  ## q = 1-probabilidad asignar spin menos uno
#  cadena, cadena_embebida = generador_decadenas(l,0,1-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno, uno, y cero
#
#
#  Ts = np.arange(Tinicial+deltaT, Tfinal+deltaT, deltaT)
#  evolucionar_2(l,cadena, cadena_embebida,J,H,mu,Tinicial,f)
#
#  for t in Ts:
#      energia_, magnetizacion_ = evolucionar_2(l,cadena,cadena_embebida,J,H,mu,t,n)
#      magnetizaciones.append(magnetizacion_)
#
#  m_normalizado = np.array(magnetizaciones)/magnetizaciones[0]


  
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
#N = 30000
#l8 = N
##m_vs_T_ferro(0,0,l8,0.1,60,0.1,1,0.5,2,400,1)
#
#
#
#simulaciones = [
#    (0,l8,0.1,60,0.1,1,0.5,2,1000,1),
#    (0.5,l8,0.1,60,0.1,1,0.5,2,1000,1),
#    (0.8,l8,0.1,60,1,0.1,0.5,2,1000,1)
#    
#    #(0.5,l8,0.1,60,0.1,1,0.5,2,600,1),
#    #(0.8,l8,0.1,60,0.1,1,0.5,2,600,1)  
#]
#
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
#        futures = {executor.submit(m_vs_T_ferro_2, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
#




#def energia_relajacion_2(q,p,l, J,H,mu,T,n):
#  """q es la probabilidad de que el nodo sea vacío,
#    ,p la probabilidad de que el nodo tenga spin negativo, 
#  l la longitud de la cadena, J la energia de interaccion, H la induccion magnetica, mu el momento magnetico, T la temperatura y n el numero de replicas para promediar la energia de relajacion
#  T la temperatura y n el numero de pasos Monte Carlo
#  """
#  cadena, cadena_embebida = generador_decadenas(l,p,1-p-q,q)  ## numeros de bloques l, probabilidad de asignar spin menos uno
#
#  energia_relajacion = []
#  for j in range(n):
#    energia_relajacion.append(evolucionar_2(l,cadena,cadena_embebida,J,H,mu,T,1)[0])
#
#
#  til = "Energia de relajacion_q="+str(q)+",T="+str(T)+",z=2,H="+str(H)
#  with open(til+".csv", "w", newline="") as f:
#      writer = csv.writer(f)
#      writer.writerow(["E"])  # encabezado
#      for energia in energia_relajacion:
#          writer.writerow([energia])
#
#  plt.figure()
#  plt.plot(energia_relajacion, 'o', ms=2)
#  plt.title(til)
#  plt.show()
#
#  return energia_relajacion
#
#
#N = 1000
#l8 = N
#
##
##energia_relajacion_2(q,p,l, J,H,mu,T,f)
#simulaciones = [
#    (0,0,l8, 1,10,0.5,15,15000),
#    (0.5,0,l8, 1,10,0.5,15,15000),
#    (0.8,0,l8, 1,10,0.5,15,15000),
#     
#]
#
#
#if __name__ == "__main__":
#    with concurrent.futures.ProcessPoolExecutor(max_workers=3) as executor:
#        futures = {executor.submit(energia_relajacion_2, *args): args[0] for args in simulaciones}
#        for future in concurrent.futures.as_completed(futures):
#            q = futures[future]
#            try:
#                result = future.result()  # Esto fuerza a que la función termine
#                print(f"Simulación con q={q} terminada")
#            except Exception as e:
#                print(f"Simulación con q={q} falló: {e}")
#
#



#def f(x,a):
#    return np.tanh(a*x)
#
#a,_ = curve_fit(f, data['H'], data['Mn'])
#
#plt.plot(data['H'], data['Mn'], 'o', ms=2, label='Datos')
#plt.plot(data['H'], f(data['H'], a), '-', label=f'Ajuste: a={a[0]:.4f}')
#plt.xlabel('Campo H')
#plt.grid()
#plt.show()



############### ENERGIA DE RELAJACION ####################################################
#def energia_relajacion_2(q, p, l, J, H, mu, T, n, num_replicas=10):
#    """q es la probabilidad de que el nodo sea vacío,
#    p la probabilidad de que el nodo tenga spin negativo, 
#    l es el numero de nodos, J la energia de interaccion, 
#    H la induccion magnetica, mu el momento magnetico, T la temperatura 
#    y n el numero de pasos Monte Carlo, num_replicas es el número de simulaciones independientes
#    """
#    
#    # Calcular N a partir de l
#    N = l
#    
#    # Almacenar todas las réplicas
#    todas_energias = []
#    
#    for replica in range(num_replicas):
#        celda, celda_embebida = generador_decadenas(l, p, 1-p-q, q)
#        energia_relajacion = []
#        
#        for j in range(n):
#            energia_relajacion.append(evolucionar_2(l, celda, celda_embebida, J, H, mu, T, 1)[0])
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
#    # Guardar datos en CSV
#    til = f"Energia_relajacion_N={N},q={q},p={p},T={T},z=3,H={H}_promedio.csv"
#    with open(til, "w", newline="") as f:
#        writer = csv.writer(f)
#        writer.writerow(["Paso", "E_promedio", "Error_estandar"])
#        for i, (e_prom, err) in enumerate(zip(energia_promedio, error_estandar)):
#            writer.writerow([i, e_prom, err])
#    
#    return q, p, N, energia_promedio, error_estandar
#
#
## FUNCIÓN AUXILIAR PARA PARALELIZAR RÉPLICAS
#def replica_individual(args):
#    """Ejecuta una réplica individual de la simulación"""
#    q, p, l, J, H, mu, T, n, replica_num = args
#    
#    celda, celda_embebida = generador_decadenas(l, p, 1-p-q, q)
#    energia_relajacion = []
#    
#    for j in range(n):
#        energia_relajacion.append(evolucionar_2(l, celda, celda_embebida, J, H, mu, T, 1)[0])
#    
#    return energia_relajacion
#
#
#def energia_relajacion_2_optimizada(q, p, l, J, H, mu, T, n, num_replicas=10, max_workers=10):
#    """Versión optimizada que paraleliza las réplicas individuales"""
#    
#    N = l
#    print(f"\nIniciando simulación: q={q}, p={p}, N={N}")
#    print(f"  Ejecutando {num_replicas} réplicas en paralelo...")
#    
#    # Preparar argumentos para cada réplica
#    args_replicas = [(q, p, l, J, H, mu, T, n, i) for i in range(num_replicas)]
#    
#    # Ejecutar réplicas en paralelo
#    todas_energias = []
#    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
#        futures = {executor.submit(replica_individual, args): i for i, args in enumerate(args_replicas)}
#        
#        for future in concurrent.futures.as_completed(futures):
#            replica_num = futures[future]
#            try:
#                result = future.result()
#                todas_energias.append(result)
#                print(f"    Réplica {replica_num+1}/{num_replicas} completada")
#            except Exception as e:
#                print(f"    Réplica {replica_num+1} falló: {e}")
#    
#    # Convertir a numpy array para calcular estadísticas
#    todas_energias = np.array(todas_energias)
#    
#    # Calcular promedio y error estándar
#    energia_promedio = np.mean(todas_energias, axis=0)
#    error_estandar = np.std(todas_energias, axis=0) / np.sqrt(num_replicas)
#    
#    # Guardar datos en CSV
#    til = f"Energia_relajacion_N={N},q={q},p={p},T={T},z=3,H={H}_promedio.csv"
#    with open(til, "w", newline="") as f:
#        writer = csv.writer(f)
#        writer.writerow(["Paso", "E_promedio", "Error_estandar"])
#        for i, (e_prom, err) in enumerate(zip(energia_promedio, error_estandar)):
#            writer.writerow([i, e_prom, err])
#    
#    print(f"✓ Simulación q={q}, p={p} completada\n")
#    return q, p, N, energia_promedio, error_estandar
#
#
#N = 500
#l8 = N
#
## Parámetros ajustables
#NUM_NUCLEOS = 10  # Número total de núcleos disponibles
#NUM_REPLICAS = 1000  # Número de réplicas por simulación
#
## energia_relajacion_2_optimizada(q, p, l, J, H, mu, T, n, num_replicas, max_workers)
#simulaciones = [
#    (0,   0, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS),
#    (0.5, 0, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS),
#    (0.8, 0, l8, 1, 10, 0.5, 15, 5000, NUM_REPLICAS, NUM_NUCLEOS)
#]
#
#
#if __name__ == "__main__":
#    resultados = []
#    
#    print(f"=" * 70)
#    print(f"INICIANDO SIMULACIONES")
#    print(f"=" * 70)
#    print(f"Total de simulaciones: {len(simulaciones)}")
#    print(f"Réplicas por simulación: {NUM_REPLICAS}")
#    print(f"Núcleos disponibles: {NUM_NUCLEOS}")
#    print(f"N = {N} sitios")
#    print(f"=" * 70)
#    
#    # ESTRATEGIA: Ejecutar simulaciones secuencialmente, 
#    # pero paralelizar las 1000 réplicas dentro de cada una
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
#    plt.title(f'Energía de relajación promedio (N={N_titulo}, T={simulaciones[0][6]}, H={simulaciones[0][4]}, z=3)\n' + 
#              f'{NUM_REPLICAS} réplicas por configuración', fontsize=13)
#    plt.legend(fontsize=10)
#    plt.grid(True, alpha=0.3)
#    plt.tight_layout()
#    plt.show()
#    
#    # Guardar datos en formato .npz
#    datos_guardar = {}
#    for q, p, N_sim, energia_promedio, error_estandar in resultados:
#        datos_guardar[f'q_{q}_p_{p}_energia'] = energia_promedio
#        datos_guardar[f'q_{q}_p_{p}_error'] = error_estandar
#    
#    datos_guardar['pasos'] = np.arange(len(resultados[0][3]))
#    datos_guardar['parametros'] = np.array([simulaciones[0][3:7]])  # J, H, mu, T
#    datos_guardar['N'] = N
#    
#    p_archivo = simulaciones[0][1]
#    nombre_archivo = f"resultados_relajacion_N={N}_T={simulaciones[0][6]}_H={simulaciones[0][4]}.npz"
#    np.savez(nombre_archivo, **datos_guardar)
#    
#    print(f"\n{'=' * 70}")
#    print(f"✓ Datos guardados en: {nombre_archivo}")
#    print(f"  Claves guardadas: {list(datos_guardar.keys())}")
#    print(f"{'=' * 70}")
#    print("TODAS LAS SIMULACIONES COMPLETADAS")
#    print(f"{'=' * 70}")
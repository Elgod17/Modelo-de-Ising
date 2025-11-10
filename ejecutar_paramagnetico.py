import numpy as np
import random
import matplotlib.pyplot as plt
#from numba import njit
from scipy.optimize import curve_fit

import cadena_z2 as z2
import celda_z3 as z3
import celda_z4 as z4
import cubo_z8 as z8

def tanh(t, a,b):
    return a * np.sinh(b * t) / (np.cosh(b * t))

#m_vs_h_paramagneto(q,l,Hinicial,Hfinal,deltaH,J,mu,T,n)
#energia_relajacion(q,p,l, J,H,mu,T,f)
#histeresis(q,l,Binicial,Bfinal,deltab,J,mu,T,n)
#m_vs_T_ferro(q,l,Tinicial,Tfinal,deltaT,J,mu,H,n,f)












H1,M1 = z2.m_vs_h_paramagneto_2(0,300,-100,100,0.01,1,0.5,15,1)

popt_log, pcov_log = curve_fit(tanh,  H1/15, M1/np.max(M1))  # valores iniciales
a,b = popt_log
plt.scatter(H1/15,M1/np.max(M1),s = 10, label = "Simulacion")
plt.plot(H1/15, tanh(H1/15, a,b), label = "Ajuste")
plt.title("M(H)")
plt.xlabel("H")
plt.ylabel("M")
plt.legend()
plt.show()
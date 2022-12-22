#Katharina Rohr, 2424597
#Pilar Mora Bonillo, 2407781

import numpy as np
from scipy.linalg import null_space
from time import perf_counter

L = np.array([[1,1,1,0,0,0,0,0,0,0,0,0],
              [0,1,0,0,1,0,0,0,0,0,0,0],
              [1,1,1,0,0,0,0,0,0,0,0,0],
              [0,0,0,1,1,0,0,0,0,0,0,0],
              [0,0,0,1,1,0,1,0,1,0,0,0],
              [0,0,0,0,1,1,0,0,0,0,0,0],
              [0,0,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,1,0,0,1,0,0,0,0],
              [0,0,0,0,0,1,0,0,1,0,0,0],
              [0,0,0,0,0,0,0,1,0,1,0,0],
              [0,0,0,0,0,0,0,1,0,0,1,0],
              [0,0,0,0,0,0,0,1,0,0,0,1]])

# Matrix A_a erzeugen:

def L_schlange_erstellen(L):
    for i in range(0,12):
        summe = 0
        for k in range(0,12):
            summe = summe + L[i,k]
        if summe == 0:
            for j in range(0,12):
                L[i,j] = 1
    return L

def D_inv_erstellen(L_schlange):
    D = np.diag([1,1,1,1,1,1,1,1,1,1,1,1])
    for n in range(0,12):
        summe = 0
        for k in range(0,12):
            summe = summe + L_schlange[n,k]
        D[n,n] = summe
    D_inv = np.linalg.inv(D)
    return D_inv

def A_a(a):
    L_schlange = L_schlange_erstellen(L)
    L_schlange_T = np.transpose(L_schlange)
    D_inv = D_inv_erstellen(L_schlange)
    E_aN = np.full([12,12], (a/12))

    S_1 = (1-a) * np.dot(L_schlange_T, D_inv)

    A_a = np.add(S_1, E_aN)
    return A_a


# Eigenvektor finden mit Potenzmethode:

def potenzmethode(v0,A_a):
    v = v0
    z = [0]
    y = np.dot(np.dot(np.conj(v),A_a),v) * (1/np.dot(np.conj(v),v))

    while( np.linalg.norm(z[-1]- y*v) > 0.000000001 ): 
        z.append(np.dot(A_a,v))
        zik = np.linalg.norm(z[-1])
        v = z[-1]*(1/zik)
        y = np.dot(np.dot(np.conj(v),A_a),v) * (1/np.dot(np.conj(v),v))
    return(v)

# Eigenvektor berechnen durch Bestimmung des Nullraums:

def nullraum(A_a):
    B = np.subtract(A_a,np.identity(12))
    EV = null_space(B)
    return EV

# Durchfuehrung der beiden Methoden mit Laufzeit:

v0= np.array([1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12,1/12])

t1_start = perf_counter()
print(potenzmethode(v0, A_a(0.1)))
t1_stop = perf_counter()

t2_start = perf_counter()
print(nullraum(A_a(0.1)))
t2_stop = perf_counter()

print("Potenzmethode Laufzeit: ", t1_stop - t1_start)
print("Nullraum Laufzeit: ", t2_stop - t2_start)

# Die Laufzeit der Potenzmethode ist laenger als die Laufzeit der Nullraum-Berechnung.
#Zum Beispiel:
#Potenzmethode Laufzeit:  0.016485299999999925
#Nullraum Laufzeit:  0.0034246

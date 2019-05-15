import matplotlib.pyplot as plt
import random
import numpy as np
from IPython.display import Image
from qutip import *
import example_sat

n = 10
N = 2**n
Theta = np.pi/2
iterations = n
theta = -1*Theta / iterations

# create state vector
Psi = tensor(basis(2),basis(2))
for i in range(2,n):
    Psi = tensor(Psi,basis(2))

# Walsh Hadamard 
for i in range(n):
    Psi = snot(N=n,target=i) * Psi
n = n+1
N = 2**n

# create ancilla
a = basis(2)
Psi = tensor(Psi,a)

# delete(Psi,[[0,1,2],[1,-1,1]]) deletes |010> for qubits 0,1,2
def delete(Psi,clause):

    # define bits and operators
    q0 = abs(clause[0][0])
    q1 = abs(clause[0][1])
    q2 = abs(clause[0][2])
    s0 = np.sign(clause[1][0])
    s1 = np.sign(clause[1][1])
    s2 = np.sign(clause[1][2])
    e0 = (3-s0)/2  # either 1,2
    e1 = (3-s1)/2  # X^1 if sign is 1
    e2 = (3-s2)/2  # X^2 = I if sign is -1
    X0 = 1j * rx(np.pi,N=n,target=q0)
    X1 = 1j * rx(np.pi,N=n,target=q1)
    X2 = 1j * rx(np.pi,N=n,target=q2)
    Ry = ry(theta)
    ancilla = n-1

    # flip the states
    Psi = X0**e0 * X1**e1 * X2**e2 * Psi

    # rotate qubit 2, only if qubits 0,1 are 1
    T = toffoli(N=n,controls=[q0,q1],target=ancilla)
    C = controlled_gate(Ry,N=n,control=ancilla,target=q2,control_value=1)
    Psi = T*C*T*Psi

    # rotate qubit 1, only if qubits 0,2 are 1
    T = toffoli(N=n,controls=[q0,q2],target=ancilla)
    C = controlled_gate(Ry,N=n,control=ancilla,target=q1,control_value=1)
    Psi = T*C*T*Psi

    # rotate qubit 0, only if qubits 1,2 are 1
    T = toffoli(N=n,controls=[q1,q2],target=ancilla)
    C = controlled_gate(Ry,N=n,control=ancilla,target=q0,control_value=1)
    Psi = T*C*T*Psi

    # flip the states back
    Psi = X0**e0 * X1**e1 * X2**e2 * Psi

    return Psi

# computes logic circuit
def logic_circuit(C,x):
    each_clause = []
    for c in range(len(C)):
        clause = C[c]
        bits = clause[0]
        lits = clause[1]
        if lits[0] == 1:
            q0 = x[bits[0]]
        else:
            q0 = not x[bits[0]]
        if lits[1] == 1:
            q1 = x[bits[1]]
        else:
            q1 = not x[bits[1]]
        if lits[2] == 2:
            q2 = x[bits[2]]
        else:
            q2 = not x[bits[2]]

        truth_val = bool(q0 or q1 or q2)
        each_clause.append(truth_val)
        if truth_val == 0:
            print('bits: ', q0,q1,q2)
            print('clause ', c, ' not satisfied')
            print(clause)
        
    return each_clause

# this is for plotting
x = np.zeros(N)
P = Psi.full()
g,p = [],[]
for i in range(N):
    if i % 2 == 0:
        g.append(x[i])
        p.append(P[i])

plt.plot(g,p,'ro')

# load SAT problem
f = open('20.dimacs',"r")
file_list = []
for line in f:
    file_list.append(line)
for i in range(len(file_list)):
    file_list[i] = file_list[i].replace('\n','')
    file_list[i] = file_list[i].split(',')
    for j in range(len(file_list[i])):
        file_list[i][j] = int(file_list[i][j])  

C = []
for i in range(len(file_list)):
    bits = []
    lits = []
    for j in range(len(file_list[i])):
        bits.append(abs(file_list[i][j]) - 1)
        lits.append(np.sign(file_list[i][j]))
    C.append([bits,lits])

# number of clauses
m = len(C)

# iterate through each clause
for it in range(iterations):
    #random.shuffle(C) 
    for clause in range(m):
        Psi = delete(Psi,C[clause])

    P = Psi.full()
    x = x - theta
    g,p = [],[]
    for i in range(N):
        if i % 2 == 0:
            g.append(x[i])
            p.append(P[i])

    plt.title('State Amplitudes vs Iterations')
    plt.ylabel('Pr(iii)')
    plt.xlabel('Iterations')
    plt.plot(g,p,'ro')

# Process output
P = Psi.full()
P = list(P)
for i in range(len(P)):
    P[i] = abs(P[i])
maximum = max(P)
print('max amplitude: ', maximum)
S = bin(P.index(max(P)))
s = S[2:] 
s = s.zfill(n)
s = list(s)
x = []
l = len(s)
for i in range(len(s)):
    x.append(int(s[n-i-1]))
print('bit string of max amplitude: ', x)
satisfied = logic_circuit(C,x)
print('clauses satisfied: ', satisfied)


plot_fock_distribution(Psi)
plt.ylim(0,maximum**2)
plt.show()

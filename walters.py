import matplotlib.pyplot as plt
import random
import numpy as np
from IPython.display import Image
from qutip import *
import example_sat

n = 3
N = 2**n
iterations = n
theta = np.pi/2

# create state vector
Psi = tensor(basis(2),basis(2))
for i in range(2,n):
    Psi = tensor(Psi,basis(2))

# Walsh Hadamard 
for i in range(n):
    Psi = snot(N=n,target=i) * Psi

# create two ancilla
A = 2
a = basis(2)
Psi = tensor(Psi,a)
Psi = tensor(Psi,a)

def print_pretty(Psi):
    P = Psi.full()
    pretty = {}
    counter = 1
    for i in range(len(P)):
        if np.real(P[i]) != 0:
            a = bin(i)
            a = a[2:] 
            a = a.zfill(n+A)
            print(counter,' ',a,' ',np.real(P[i]))
            counter = counter + 1

def max_bit_string(Psi):
    P = Psi.full()
    pretty = {}
    counter = 1
    amplitude = []
    bits = []
    for i in range(len(P)):
        if np.real(P[i]) != 0:
            a = bin(i)
            a = a[2:] 
            a = a.zfill(n+A)
            print(counter,' ',a,' ',np.real(P[i]))
            amplitude.append(np.real(P[i][0]))
            bits.append(a)
            counter = counter + 1
    m = bits[amplitude.index(max(amplitude))][:4]
    m = list(m)
    x = []
    for i in range(len(m)):
        x.append(int(m[n-i-1]))
    print(x)
    return x, max(amplitude)

# shifts amplitude from state |111> to |011>,|101>,|110>
def decimate(Psi,clause):

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
    X0 = 1j * rx(np.pi,N=n+2,target=q0)
    X1 = 1j * rx(np.pi,N=n+2,target=q1)
    X2 = 1j * rx(np.pi,N=n+2,target=q2)
    ancilla0 = n
    ancilla1 = n+1
    Xa = 1j * rx(np.pi,N=n+2,target=ancilla1)
    zero = [[1,0],[0,0]]
    one = [[0,0],[0,1]]
    H = hadamard_transform()

    # flip the states
    Psi = X0**e0 * X1**e1 * X2**e2 * Psi

    # shift a little amplitude into each state differing by one bit. 
    for q in range(3):

        # compute truth value of state into ancilla1
        T0 = toffoli(N=n+2,controls=[q0,q1],target=ancilla0)
        T1 = toffoli(N=n+2,controls=[q2,ancilla0],target=ancilla1)
        Psi = T0*T1*T0*Psi

        # randomize theta
        sigma = 0.087*random.choice([1,-1]) # +/- 5 deg
        Ry = ry(sigma+theta)

        # perform controlled rotation onto the undesired state
        C = controlled_gate(Ry,N=n+2,control=ancilla1,target=q,control_value=1)
        Psi = C*Psi

        # Projection operator for ancilla2
        outcome = random.choice([zero,one])
        projector = Qobj(outcome)
        partial_measure = gate_expand_1toN(projector*H,N=n+2,target=ancilla1)

        # Partial measurement
        Psi = partial_measure * Psi

        # re-normalize the state
        Psi = Psi / np.linalg.norm(np.array(Psi.full())) 

        # reset ancilla 
        if outcome == one:
            Psi = Xa*Psi

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
f = open('4v_15c.dimacs',"r")
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

# tiny SAT problem. Only 000 should remain
C = [[1,1,-1],[1,-1,1],[1,-1,-1],[-1,1,1],[-1,1,-1],[-1,-1,1],[-1,-1,-1]]

# iterate through each clause
############################################################
# So I've tried many different variations of loops, using  #
# the decimation function described in the paper. No       #
# matter what the configuration, the algorithm may still   #
# converge into an undesired state.                        #
############################################################
for loops in range(n):
    for clause in range(len(C)):
        for it in range(iterations):
            Psi = decimate(Psi,[[0,1,2],C[clause]])

            # if using loaded logic circuit
            #Psi = decimate(Psi,C[clause])

            P = Psi.full()
            x = x + theta
            g,p = [],[]
            for i in range(N):
                if i % 2 == 0:
                    g.append(x[i])
                    p.append(P[i])

            plt.title('State Amplitudes vs Iterations')
            plt.ylabel('Pr(iii)')
            plt.xlabel('Iterations')
            plt.plot(g,p,'ro')

# processing solution 
x,maximum = max_bit_string(Psi)

# check truth value of max amplitude if using loaded logic circuit
#satisfied = logic_circuit(C,x)
#print(satisfied)

plot_fock_distribution(Psi)
#plt.ylim(0,maximum**2 + 0.1)
plt.show()

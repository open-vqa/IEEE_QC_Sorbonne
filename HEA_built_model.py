from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ
from qat.core import Observable, Term, Circuit
from qat.lang.AQASM.gates import Gate
import matplotlib as mpl
import numpy as np
from typing import Optional, List
import warnings



def HEA_Linear(
    nqbits: int,
    #theta: List[float],
    n_cycles: int = 1,
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit: #linear entanglement
    """
    This Hardware Efficient Ansatz has the reference from "Nonia Vaquero Sabater et al. Simulating molecules 
    with variational quantum eigensolvers. 2022" -Figure 6 -Link 
    "https://uvadoc.uva.es/bitstream/handle/10324/57885/TFM-G1748.pdf?sequence=1"

    Args:
        nqbits (int): Number of qubits of the circuit.
        n_cycles (int): Number of layers.
        rotation_gates (List[Gate]): Parametrized rotation gates to include around the entangling gate. Defaults to :math:`RY`. Must
            be of arity 1.
        entangling_gate (Gate): The 2-qubit entangler. Must be of arity 2. Defaults to :math:`CNOT`.
    """

    if rotation_gates is None:
        rotation_gates = [RZ]

    n_rotations = len(rotation_gates)

    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    
    theta_curr = n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles)
    
    ind_theta = 0


    
    for i in range(nqbits):

        for rot in rotation_gates:

            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
    
    for k in range(n_cycles):


        for i in range(nqbits - 1):
            prog.apply(CNOT, reg[i], reg[i+1])
            
        for i in range(nqbits):
            for rot in rotation_gates:
                            
                prog.apply(rot(theta[ind_theta]), reg[i])
                ind_theta += 1

    return prog.to_circ(), theta_curr

def HEA_Full(
    nqbits: int,
    n_cycles: int = 1,
    #theta: List[float],
    rotation_gates: List[Gate] = None,
    entangling_gate: Gate = CNOT,
) -> Circuit:
    """
    This Hardware Efficient Ansatz has the reference from "Yunya Liu, Jiakun Liu, Jordan R Raney, and Pai Wang. 
    Quantum computing for solid mechanics and structural engineering–a demonstration with variational quantum eigensolver."
    -Figure 1 - link "https://arxiv.org/pdf/2308.14745.pdf"

    Args:
        nqbits (int): Number of qubits of the circuit.
        n_cycles (int): Number of layers.
        rotation_gates (List[Gate]): Parametrized rotation gates to include around the entangling gate. Defaults to :math:`RY`. Must
            be of arity 1.
        entangling_gate (Gate): The 2-qubit entangler. Must be of arity 2. Defaults to :math:`CNOT`.
    """

    if rotation_gates is None:
        rotation_gates = [RZ]

    n_rotations = len(rotation_gates)

    prog = Program()
    reg = prog.qalloc(nqbits)
    #theta = [np.pi/2 for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    theta = [prog.new_var(float, rf"\theta_{{{i}}}") for i in range(n_rotations * (nqbits + 2 * (nqbits - 1) * n_cycles))]
    
    
    ind_theta = 0


    
    for i in range(nqbits):

        for rot in rotation_gates:

            prog.apply(rot(theta[ind_theta]), reg[i])
            ind_theta += 1
    
    for k in range(n_cycles):


        for i in range(nqbits - 1):
            for j in range(i+1, nqbits):
                prog.apply(CNOT, reg[i], reg[j])
            
        for i in range(nqbits):
            for rot in rotation_gates:
                            
                prog.apply(rot(theta[ind_theta]), reg[i])
                ind_theta += 1

    return prog.to_circ(), theta



def Double_HEA(nqbits, n_layers):
    prog = Program()
    reg = prog.qalloc(nqbits)
    theta = [prog.new_var(float, f"theta_{i}") for i in range(2 * n_layers * nqbits)]
    
    for ind_layer in range(n_layers):
        for qb in range(nqbits):
            # Apply RZ followed by RY to each qubit
            RZ(theta[ind_layer * nqbits * 2 + qb])(reg[qb])
            RY(theta[ind_layer * nqbits * 2 + nqbits + qb])(reg[qb])
        
        # Add entangling CNOT gates in a chain
        for qb in range(nqbits - 1):
            CNOT(reg[qb], reg[qb + 1])
        # Optionally, for circular entanglement, add this:
        CNOT(reg[nqbits - 1], reg[0])
    
    circ = prog.to_circ()
    return circ

def count(gate, mylist):
    """
    count function counts the number of gates in the given list
    params: it takes two parameters. first is which gate you want
    to apply like rx, ry etc. second it take the list of myqlm gates
    instruction.
    returns: it returns number of gates.
    """
    if type(gate) == type(str):
        gate = str(gate)
    if gate == gate.lower():
        gate = gate.upper()
    mylist = [str(i) for i in mylist]
    count = 0
    for i in mylist:
        if i.find("gate='{}'".format(gate)) == -1:
            pass
        else:
            count += 1
    return count







        


        



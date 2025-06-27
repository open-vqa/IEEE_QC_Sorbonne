import numpy as np
import matplotlib.pyplot as plt
from qat.core import Observable, Term
from qat.core.circuit_builder.matrix_util import get_predef_generator
from qat.lang.AQASM import Program, QRoutine, RY, CNOT, RX, Z, H, RZ
from qat.qpus import get_default_qpu
from qat.plugins import ScipyMinimizePlugin

# Define the Heisenberg Hamiltonian for 2 qubits
nqbits = 2
J = 1.0  # coupling constant

# Heisenberg Hamiltonian: H = J(S1^x S2^x + S1^y S2^y + S1^z S2^z)
# In terms of Pauli matrices: H = (J/4)(X1 X2 + Y1 Y2 + Z1 Z2)
pauli_terms = [
    Term(coefficient=J/4, pauli_op="XX", qbits=[0, 1]),  # X1 X2 term
    Term(coefficient=J/4, pauli_op="YY", qbits=[0, 1]),  # Y1 Y2 term  
    Term(coefficient=J/4, pauli_op="ZZ", qbits=[0, 1])   # Z1 Z2 term
]

hamiltonian = Observable(nqbits, pauli_terms=pauli_terms)
print("Heisenberg Hamiltonian:")
print("H =", hamiltonian)

# Function to compute matrix representation
def make_matrix(hamiltonian):
    mat = np.zeros((2**hamiltonian.nbqbits, 2**hamiltonian.nbqbits), np.complex_)
    for term in hamiltonian.terms:
        op_list = ["I"]*hamiltonian.nbqbits
        for op, qb in zip(term.op, term.qbits):
            op_list[qb] = op
        def mat_func(name): return np.identity(2) if name == "I" else get_predef_generator()[name]
        term_mat = mat_func(op_list[0])
        for op in op_list[1:]:
            term_mat = np.kron(term_mat, mat_func(op))
        mat += term.coeff * term_mat
    return mat

# Compute exact ground state energy
H_mat = make_matrix(hamiltonian)
eigvals = np.linalg.eigvalsh(H_mat)
E0 = min(eigvals)
print(f"\nExact ground state energy: E0 = {E0:.6f}")
print(f"All eigenvalues: {eigvals}")

# Create ansatz circuit
def create_ansatz():
    """Function that prepares an ansatz circuit
    
    Returns:
        Circuit: a parameterized circuit (i.e with some variables that are not set)
    """
    
    # For a 2-qubit system, we'll use a simple but effective ansatz
    # Number of parameters: 2 rotations per qubit + 1 additional rotation for entanglement
    nparams = 5  # 2 qubits × 2 rotations + 1 extra rotation
    
    prog = Program()
    reg = prog.qalloc(nqbits)
    
    # define variables using 'new_var' 
    theta = [prog.new_var(float, f'\\theta_{i}')
             for i in range(nparams)]
    
    # Initial rotations on each qubit
    for ind in range(nqbits):
        RY(theta[ind])(reg[ind])
    
    # Add entanglement with CNOT gate
    CNOT(reg[0], reg[1])
    
    # Additional rotation on the second qubit for more expressiveness
    RY(theta[2])(reg[1])
    
    # Another layer of rotations for better expressiveness
    RY(theta[3])(reg[0])
    RY(theta[4])(reg[1])
    
    circ = prog.to_circ()
    return circ

# Create the circuit
circ = create_ansatz()
print(f"\nCircuit created with {len(circ.get_variables())} parameters")

# Create variational job
job = circ.to_job(job_type="OBS",
                  observable=hamiltonian,
                  nbshots=0)  # Use 0 for exact simulation

# Test different optimizers with plotting
methods = ["COBYLA", "Nelder-Mead", "BFGS"]
results = {}
optimization_traces = {}

# Set up plotting
plt.figure(figsize=(12, 8))

for method in methods:
    print(f"\n--- Testing {method} ---")
    
    # Initial parameters
    theta_0 = np.random.random(5) * 2 * np.pi
    
    # Set up QPU and optimizer
    linalg_qpu = get_default_qpu()
    optimizer_scipy = ScipyMinimizePlugin(method=method,
                                          tol=1e-6,
                                          options={"maxiter": 200}, # play with this
                                          x0=theta_0)
    
    # Build variational stack
    qpu = optimizer_scipy | linalg_qpu
    
    # Submit the job
    result = qpu.submit(job)
    results[method] = result.value
    
    # Extract optimization trace
    if 'optimization_trace' in result.meta_data:
        trace = eval(result.meta_data['optimization_trace'])
        optimization_traces[method] = trace
        plt.plot(trace, label=method, linewidth=2, alpha=0.8)
    else:
        print(f"No optimization trace available for {method}")
    
    print(f"VQE energy ({method}) = {result.value:.6f}")
    print(f"Error = {abs(result.value - E0):.6f}")

# Add exact energy line
max_iterations = max([len(trace) for trace in optimization_traces.values()]) if optimization_traces else 200
plt.plot([E0 for _ in range(max_iterations)], '--k', lw=3, label="Exact energy", alpha=0.7)

# Customize the plot
plt.grid(True, alpha=0.3)
plt.legend(loc="best", fontsize=12)
plt.xlabel("Optimization Steps", fontsize=14)
plt.ylabel("Energy", fontsize=14)
plt.title("VQE Convergence Comparison for 2-Qubit Heisenberg Hamiltonian", fontsize=16)
plt.ylim(E0 - 0.1, max([max(trace) for trace in optimization_traces.values()]) + 0.1 if optimization_traces else E0 + 0.5)

# Add text box with final results
textstr = f"Exact E₀ = {E0:.6f}\n"
for method, energy in results.items():
    error = abs(energy - E0)
    textstr += f"{method}: {energy:.6f} (err: {error:.6f})\n"

props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

print(f"\n--- Summary ---")
print(f"Exact ground state energy: {E0:.6f}")
for method, energy in results.items():
    print(f"{method}: {energy:.6f} (error: {abs(energy - E0):.6f})")

# Additional analysis
print(f"\n--- Convergence Analysis ---")
for method, trace in optimization_traces.items():
    if len(trace) > 0:
        initial_energy = trace[0]
        final_energy = trace[-1]
        improvement = initial_energy - final_energy
        convergence_rate = improvement / len(trace) if len(trace) > 1 else 0
        print(f"{method}:")
        print(f"  Initial energy: {initial_energy:.6f}")
        print(f"  Final energy: {final_energy:.6f}")
        print(f"  Total improvement: {improvement:.6f}")
        print(f"  Steps to converge: {len(trace)}")
        print(f"  Average improvement per step: {convergence_rate:.6f}")
        print() 
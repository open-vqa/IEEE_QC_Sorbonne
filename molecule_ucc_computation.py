import numpy as np
from qat.fermion import ElectronicStructureHamiltonian
from qat.fermion.transforms import get_jw_code, recode_integer, transform_to_jw_basis
from qat.fermion.hamiltonians import make_embedded_model
from qat.fermion.circuits import make_shallow_circ, make_ldca_circ
from qat.fermion.chemistry.ucc_deprecated import get_cluster_ops_and_init_guess
from qat.fermion.chemistry.pyscf_tools import perform_pyscf_computation
from qat.fermion.chemistry import MolecularHamiltonian, MoleculeInfo
from qat.fermion.chemistry.ucc import (
    construct_ucc_ansatz,
    guess_init_params,
    get_hf_ket,
    get_cluster_ops,
    convert_to_h_integrals,
)
from qat.qpus import get_default_qpu

from qat.plugins import ScipyMinimizePlugin, MultipleLaunchesAnalyzer

def get_parameters(molecule_symbol):
    if molecule_symbol == "LIH":
        r = 1.45
        geometry = [("Li", (0, 0, 0)), ("H", (0, 0, r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "H2":
        r = 0.75
        geometry = [("H", (0, 0, 0)), ("H", (0, 0, r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "H4":
        # H4
        r = 0.85
        geometry = [
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1 * r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
        ]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "H6":
        r = 4.5
        print("r is:", r)
        geometry = [
            ("H", (0, 0, 0)),
            ("H", (0, 0, 1 * r)),
            ("H", (0, 0, 2 * r)),
            ("H", (0, 0, 3 * r)),
            ("H", (0, 0, 4 * r)),
            ("H", (0, 0, 5 * r)),
        ]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "BeH2":
        r = 1.4
        geometry = [("Be", (0, 0, 0 * r)), ("H", (0, 0, r)), ("H", (0, 0, -r))]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "HO":
        r = 4.0
        geometry = [("H", (0, 0, 0 * r)), ("O", (0, 0, 1 * r))]
        charge = -1
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "H2O":
        r = 1.0285
        theta = 0.538 * np.pi
        geometry = [
            ("O", (0, 0, 0 * r)),
            ("H", (0, 0, r)),
            ("H", (0, r * np.sin(np.pi - theta), r * np.cos(np.pi - theta))),
        ]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "NH3":
        r = 1.0703
        theta = (100.107 / 180) * np.pi
        geometry = [
            ("N", (0, 0, 0 * r)),
            (
                "H",
                (
                    0,
                    2 * (np.sin(theta / 2) / np.sqrt(3)) * r,
                    np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                ),
            ),
            (
                "H",
                (
                    np.sin(theta / 2) * r,
                    -np.sin(theta / 2) / np.sqrt(3) * r,
                    np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                ),
            ),
            (
                "H",
                (
                    -np.sin(theta / 2) * r,
                    -np.sin(theta / 2) / np.sqrt(3) * r,
                    np.sqrt(1 - 4 * np.sin(theta / 2) ** 2 / 3) * r,
                ),
            ),
        ]
        charge = 0
        spin = 0
        basis = "sto-3g"
    elif molecule_symbol == "CO":
        r = 1.0
        geometry = [("C", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.1282))]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "CH4":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (0.6276, 0.6276, 0.6276)),
            ("H", (0.6276, -0.6276, -0.6276)),
            ("H", (-0.6276, 0.6276, -0.6276)),
            ("H", (-0.6276, -0.6276, 0.6276)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "C2H4":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.6695)),
            ("C", (0.0, 0.0, -0.6695)),
            ("H", (0.0, 0.9289, 1.2321)),
            ("H", (0.0, -0.9289, 1.2321)),
            ("H", (0.0, 0.9289, -1.2321)),
            ("H", (0.0, -0.9289, -1.2321)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "CHN":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.0)),
            ("H", (0.0, 0.0, 1.0640)),
            ("N", (0.0, 0.0, -1.1560)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "O2":
        r = 1.0
        geometry = [("O", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.2075))]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "NO":
        r = 1.0
        geometry = [("N", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.1508))]
        basis = "sto-3g"
        spin = 0
        charge = 1
    elif molecule_symbol == "C2H2":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.6063)),
            ("C", (0.0, 0.0, -0.6063)),
            ("H", (0.0, 0.0, 1.6941)),
            ("H", (0.0, 0.0, -1.6941)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    elif molecule_symbol == "CO2":
        r = 1.0
        geometry = [
            ("C", (0.0, 0.0, 0.0)),
            ("O", (0.0, 0.0, -1 * r)),
            ("O", (0.0, 0.0, 1 * r)),
        ]
        basis = "sto-3g"
        spin = 0
        charge = 0
    return r, geometry, charge, spin, basis



def compute_circuit_and_hamiltonian(molecule_name):
    """Compute the quantum circuit and Hamiltonian for a given molecule."""
    r, geometry, charge, spin, basis = get_parameters(molecule_name)

    # Perform quantum chemistry computation
    rdm1, orbital_energies, nuclear_repulsion, nels, one_body_integrals, two_body_integrals, info = perform_pyscf_computation(
        geometry=geometry, basis=basis, spin=spin, charge=charge, run_fci=True
    )

    #print(f"Number of qubits before active space selection: {rdm1.shape[0] * 2}")
    #print("RDM1:", rdm1)
    print("Computation info:", info)

    # Convert integrals to fermionic Hamiltonian
    hpq, hpqrs = convert_to_h_integrals(one_body_integrals, two_body_integrals)
    H = ElectronicStructureHamiltonian(hpq, hpqrs, constant_coeff=nuclear_repulsion)

    # Get cluster operators for UCC ansatz
    noons, basis_change = np.linalg.eigh(rdm1)
    noons = list(reversed(noons))
    noons_full, orb_energies_full = [], []
    for ind in range(len(noons)):
        noons_full.extend([noons[ind], noons[ind]])
        orb_energies_full.extend([orbital_energies[ind], orbital_energies[ind]])

    cluster_ops, theta_0, hf_init = get_cluster_ops_and_init_guess(nels, noons_full, orb_energies_full, H.hpqrs)

    # Transform Hamiltonian and cluster operators to qubit representation
    H_sp = transform_to_jw_basis(H)
    nqbits = H_sp.nbqbits
    print("Number of qubits", nqbits)
    print("Spin representaion of Hamiltonian", H_sp)
    cluster_ops_sp = [transform_to_jw_basis(t_o) for t_o in cluster_ops]
    hf_init_sp = recode_integer(hf_init, get_jw_code(H_sp.nbqbits))

    # Construct UCC ansatz quantum circuit
    qprog = construct_ucc_ansatz(cluster_ops_sp, hf_init_sp)
    circ = qprog.to_circ()
    qpu = get_default_qpu()

    return H_sp, circ, qpu, nqbits, theta_0

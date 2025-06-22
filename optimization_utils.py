import numpy as np

def get_optimization_func(circ, qpu, H_sp, method, nqbits, psi0, energy_list, fid_list):
    # below, I show an example of minimization of the energy where I store the energy and fidelity for each parameter
    # I use a gradient-free procedure.
    def my_func(x):
        """returns energy given parameter x, and stores it + the fidelity of state"""
        circ1 = circ.bind_variables(
            {k: v for k, v in zip(sorted(circ.get_variables()), x)}
        )
        res0 = qpu.submit(circ1.to_job(observable=H_sp))
        energy = res0.value
        energy_list[method].append(energy)

        # additional computation to compute fidelity (just for my own information)
        res = qpu.submit(circ1.to_job())
        psi = np.zeros((2**nqbits,), complex)
        for sample in res:
            psi[sample.state.int] = sample.amplitude
        fid = abs(psi.conj().dot(psi0)) ** 2
        fid_list[method].append(fid)

        return energy

    return my_func


def get_grad_func(circ, qpu, H_sp):
    # here I show a gradient-based minimization strategy
    def my_grad(x):
        grads = circ.to_job(observable=H_sp).gradient()
        grad_list = []
        for var_name in sorted(circ.get_variables()):
            list_jobs = grads[var_name]
            # list_jobs contains jobs to compute E(theta+pi/2) and E(theta-pi/2)
            # the gradient w.r.t theta is then 0.5 (E(theta+pi/2) - E(theta-pi/2))
            grad = 0.0
            for ind in range(len(list_jobs)):
                circ1 = list_jobs[ind].circuit.bind_variables(
                    {k: v for k, v in zip(circ.get_variables(), x)}
                )
                job = circ1.to_job(observable=list_jobs[ind].observable)
                res = qpu.submit(job)
                grad += 0.5 * res.value
            grad_list.append(grad)
        return grad_list

    return my_grad


def compute_eigen_values(H_sp):
    # to compute the eigen values and eigen vectors
    eigvals, eigvec = np.linalg.eigh(H_sp.get_matrix(sparse=False))
    E0 = min(eigvals)
    print("min eigval = ", E0)
    psi0 = eigvec[:, np.argmin(eigvals)]
    return E0, psi0

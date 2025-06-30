
#Auther: HUYBIN TRAN


### 2025 IEEE CIS SUMMER SCHOOL ON QUANTUM COMPUTATIONAL INTELLIGENCE
-----------
This is the presentation + code implementation presented in the Sorbonne Workshop under the title "**Quantum Computing for Quantum Chemistry - Review on spin model**"

**Getting started**

Everything and installation are well defined on the Google Collab but if you want to self-install in your local machine, please follow these steps

- [Tutorial part](https://colab.research.google.com/github/huybinhtr/workshopS/blob/main/Tutorial.ipynb)
- [Excercise part](https://colab.research.google.com/gist/huybinhtr/315974b6c5921d764d47d06a901523aa/excercise.ipynb)

#### Step 1: Install `myQLM-fermion`
```bash
git clone https://github.com/myQLM/myqlm-fermion.git
cd myqlm-fermion
pip install -r requirements.txt
pip install .
```
#### Step 2: Set Up a Conda Environment
```bash
conda create --name myenv
conda activate myenv
pip install myqlm==1.9.4
pip install scipy==1.10.1
```

#### Step 3: Install OpenVQE
```bash
git clone https://github.com/OpenVQE/OpenVQE.git
pip install -e .
```
Last step is to clone this git respo as command below 
```bash
git clone https://github.com/open-vqa/IEEE_QC_Sorbonne
```

Now you can start to hands-on the the `tutorial.ipynb` that composed of three internal .py file:

- `molecule_ucc_computation.py` initialize the choosen molecules library and to construct UCC ansatz quantum circuit & transform Hamiltonian and cluster operators to qubit representation.
- `HEA_built_model.py` is a library for different extension of Hardware Efficient Ansatz with references.
- `optimization_utils.py` is using the Parameter-Shift Rules (PMRS) for the gradient based optimizer; it stores energies & fidelities also to compute eigenvalues of the Hamiltonian.

Afterward you can try yourself on an excerice to see how the VQE method can be used to find the approximate ground state (GS) of a hamiltonian in spin representation (gradient based / non gradient based are taken into consideration)

**Getting in touch**

For any question about this respo (Package installation, research discussion), don't hesitate to get in touch: huybinhfr4120@gmail.com
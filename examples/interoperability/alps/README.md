# QDK/Chemistry → ALPS DMRG

Hands a QDK molecular-orbital Hamiltonian to the [ALPS](https://github.com/ALPSim/ALPS) `dmrg` application and reads back the ground state as a matrix product state (MPS).

```python
from qdk_alps_dmrg import run_dmrg, mps_to_wavefunction

res = run_dmrg(hamiltonian, n_alpha, n_beta, maxstates=256, sweeps=8)
res.energy        # DMRG energy, core energy included
res.mps           # [A_0, ..., A_{L-1}], A_i[left bond, local state, right bond]
res.site_basis    # Nup / Ndown of each local state, per site type
res.dropped       # the (pq|rs) elements that were truncated
wf = mps_to_wavefunction(res, hamiltonian.get_orbitals())   # QDK Wavefunction, small L only
```

## How the Hamiltonian is mapped

Each MO is one ALPS site, in QDK's orbital order, and the lattice is built from the integrals:

- one vertex type per orbital, carrying h<sub>pp</sub> and (pp|pp);
- one edge per orbital pair with a nonzero h<sub>pq</sub> or two-body element over exactly those two orbitals (density, exchange, pair hopping, correlated hopping), each edge its own bond type with numeric coefficients.

Two-body elements over three or four distinct orbitals are **dropped** and listed in `res.dropped`. `truncated_hamiltonian(hamiltonian)` returns the same truncation as a QDK `Hamiltonian`, so QDK solvers can be run on exactly what DMRG solved.

## Requirements

- An ALPS `dmrg` with `SAVE_MPS` (branch `dmrg-save-mps-upstream` of [marcusr2ML/ALPS](https://github.com/marcusr2ML/ALPS)) that also includes [ALPSim/ALPS#221](https://github.com/ALPSim/ALPS/pull/221). Without #221, ALPS keeps 6 significant digits of every coupling.
- `ALPS_BIN` pointing at the directory holding `dmrg` and `parameter2xml`, or both on `PATH`.
- From a build tree rather than an install: `ALPS_XML_PATH=<ALPS source>/src/alps/resources`.
- `h5py`.

## Check

```bash
ALPS_BIN=<ALPS build>/bin ALPS_XML_PATH=<ALPS source>/src/alps/resources python check_dmrg_mps.py
```

For a Hubbard chain and STO-3G H<sub>4</sub>, LiH, H<sub>6</sub> and H<sub>8</sub>, [`check_dmrg_mps.py`](check_dmrg_mps.py) compares the DMRG energy and the saved MPS against dense exact diagonalization and QDK CASCI of the truncated Hamiltonian, and reports the truncation error against CASCI of the full Hamiltonian.

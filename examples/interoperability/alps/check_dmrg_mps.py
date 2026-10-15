# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""QDK MO Hamiltonian -> ALPS dmrg (SAVE_MPS) -> MPS, checked against dense ED.

    ALPS_BIN=<ALPS build>/bin python check_dmrg_mps.py

Per case:
  E_dmrg      dmrg's energy (+ E_core)
  E_trunc     exact ground energy of the truncated Hamiltonian (3/4-orbital
              (pq|rs) zeroed), dense Jordan-Wigner, independent of ALPS
  <H_trunc>   the saved MPS contracted and measured with that same dense
              matrix: checks the tensors and ALPS's fermion ordering, not just
              the energy dmrg printed
  E_casci     QDK macis CASCI on the same truncated Hamiltonian
  |<CI|MPS>|  overlap of the MPS, converted to a QDK Wavefunction, with that
              CASCI wavefunction
  E_full      QDK CASCI on the full Hamiltonian, for the truncation error only
"""

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qdk_alps_dmrg import (
    integrals,
    mps_to_dense,
    mps_to_wavefunction,
    run_dmrg,
    truncated_hamiltonian,
)
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import Element, LatticeGraph, Structure
from qdk_chemistry.utils.model_hamiltonians import (
    create_hubbard_hamiltonian,
)

ANG = 1.8897259886  # Bohr per Angstrom


def molecule(coords_ang, elements, charge=0, mult=1, basis="sto-3g"):
    """SCF -> MO Hamiltonian over all orbitals."""
    s = Structure(np.asarray(coords_ang, float) * ANG, elements)
    with open(os.devnull, "w") as null:  # SCF prints a report
        fd = os.dup(1)
        os.dup2(null.fileno(), 1)
        try:
            _, wf = create("scf_solver").run(s, charge, mult, basis)
            ham = create("hamiltonian_constructor").run(wf.get_orbitals())
        finally:
            os.dup2(fd, 1)
            os.close(fd)
    return ham


def truncate(g):
    """Zero every (pq|rs) over 3 or 4 distinct orbitals."""
    g = g.copy()
    for idx in itertools.product(range(g.shape[0]), repeat=4):
        if len(set(idx)) > 2:
            g[idx] = 0.0
    return g


def dense_hamiltonian(h1, g, na, nb):
    """Sector matrix in modes 2p + spin (0 = up), JW order low mode first."""
    n = h1.shape[0]
    states = [
        s
        for s in range(1 << 2 * n)
        if (s & int("01" * n, 2)).bit_count() == na
        and (s & int("10" * n, 2)).bit_count() == nb
    ]
    index = {s: i for i, s in enumerate(states)}

    def apply(ops, s):
        sign = 1
        for dag, m in reversed(ops):
            if (s >> m) & 1 == dag:
                return None, 0
            if (s & ((1 << m) - 1)).bit_count() % 2:
                sign = -sign
            s ^= 1 << m
        return s, sign

    H = np.zeros((len(states), len(states)))
    for col, s in enumerate(states):
        for p, q in itertools.product(range(n), repeat=2):
            if h1[p, q]:
                for a in (0, 1):
                    t, sg = apply([(1, 2 * p + a), (0, 2 * q + a)], s)
                    if t is not None:
                        H[index[t], col] += h1[p, q] * sg
        for p, q, r, s_ in zip(*np.nonzero(np.abs(g) > 1e-14)):
            v = 0.5 * g[p, q, r, s_]
            for a in (0, 1):
                for b in (0, 1):
                    t, sg = apply(
                        [
                            (1, 2 * p + a),
                            (1, 2 * r + b),
                            (0, 2 * s_ + b),
                            (0, 2 * q + a),
                        ],
                        s,
                    )
                    if t is not None:
                        H[index[t], col] += v * sg
    return H, index


def cases():
    H = Element.H
    yield (
        "Hubbard L=6 U=4",
        create_hubbard_hamiltonian(
            LatticeGraph.chain(6, periodic=False), 0.0, 1.0, 4.0
        ),
        (3, 3),
    )
    yield (
        "H4 chain 0.9 A",
        molecule([[0, 0, 0.9 * i] for i in range(4)], [H] * 4),
        (2, 2),
    )
    yield (
        "H4 chain 1.5 A",
        molecule([[0, 0, 1.5 * i] for i in range(4)], [H] * 4),
        (2, 2),
    )
    yield "LiH 1.6 A", molecule([[0, 0, 0], [0, 0, 1.6]], [Element.Li, H]), (2, 2)
    yield (
        "H6 ring 1.0 A",
        molecule(
            [[np.cos(k * np.pi / 3), np.sin(k * np.pi / 3), 0] for k in range(6)],
            [H] * 6,
        ),
        (3, 3),
    )
    yield (
        "H8 chain 1.0 A",
        molecule([[0, 0, 1.0 * i] for i in range(8)], [H] * 8),
        (4, 4),
    )


def main():
    casci = create("multi_configuration_calculator")
    print(
        f"{'case':18}{'dropped':>8}{'E_dmrg':>16}{'E_trunc':>16}{'|dE|':>9}"
        f"{'|d<H>|':>9}{'E_casci':>16}{'|dE|':>9}{'|<CI|MPS>|':>14}{'E_full':>16}{'trunc err':>11}"
    )
    print("-" * 142)
    bad = 0
    for label, ham, (na, nb) in cases():
        h1, g, core = integrals(ham)
        res = run_dmrg(ham, na, nb, maxstates=256, sweeps=8)
        ham_t = truncated_hamiltonian(ham)
        assert np.array_equal(integrals(ham_t)[1], truncate(g))

        Ht, index = dense_hamiltonian(h1, truncate(g), na, nb)
        e_trunc = np.linalg.eigvalsh(Ht)[0] + core

        psi = np.zeros(len(index))
        for occ, c in mps_to_dense(res).items():
            psi[index[sum(o << k for k, o in enumerate(occ))]] = c
        e_mps = psi @ Ht @ psi / (psi @ psi) + core

        e_ci, wf_ci = casci.run(ham_t, na, nb)
        e_full, _ = casci.run(ham, na, nb)
        wf = mps_to_wavefunction(res, ham.get_orbitals())
        ci = dict(
            zip(
                (d.to_string() for d in wf_ci.get_active_determinants()),
                np.ravel(wf_ci.get_coefficients()),
            )
        )
        ov = abs(
            sum(
                ci.get(d.to_string(), 0.0) * c
                for d, c in zip(
                    wf.get_active_determinants(), np.ravel(wf.get_coefficients())
                )
            )
        )

        ok = (
            abs(res.energy - e_trunc) < 1e-8
            and abs(e_mps - res.energy) < 1e-8
            and abs(res.energy - e_ci) < 1e-7
            and ov > 1 - 1e-6
        )
        bad += not ok
        print(
            f"{label:18}{len(res.dropped):>8}{res.energy:16.10f}{e_trunc:16.10f}"
            f"{abs(res.energy - e_trunc):9.1e}{abs(e_mps - res.energy):9.1e}"
            f"{e_ci:16.10f}{abs(res.energy - e_ci):9.1e}{ov:14.10f}"
            f"{e_full:16.10f}{res.energy - e_full:11.2e}"
            + ("" if ok else "  <-- MISMATCH")
        )
    print("-" * 142)
    print(
        "DMRG MPS = QDK CASCI ground state of the truncated H"
        if not bad
        else f"{bad} MISMATCH(ES)"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

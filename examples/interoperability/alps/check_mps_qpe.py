# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""DMRG MPS (truncated H) -> QDK circuit -> QPE on the full H = CASCI (full H).

    ALPS_BIN=<ALPS build>/bin ALPS_XML_PATH=<ALPS source>/src/alps/resources python check_mps_qpe.py

Needs qiskit for the exact statevector checks. About 22 minutes.

DMRG solves the truncated Hamiltonian; CASCI here is NOT truncated. The MPS
is only the initial state: QPE with the full Jordan-Wigner Hamiltonian
projects it onto a full-H eigenstate, the ground state with probability
p = |<GS_full|MPS>|^2, so the QPE energy must be CASCI's, not DMRG's.

Per case:
  [handoff]  MPS -> Wavefunction -> sparse-isometry circuit; its statevector
             gives <H_trunc> = E_dmrg (state reached the qubits intact)
  [JW]       the full-H JW eigenvector closest to CASCI's wavefunction (also
             prepared as a circuit) has CASCI's energy and overlap 1
  [baseline] <H_full> on the MPS state: variational, above E_casci
  [QPE]      iterative QPE, H_full, MPS initial state, t = pi/||H||: the
             measured bits equal the bin of E_casci (+-1), and that bin is
             not E_dmrg's
"""

import os
import sys
import time

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_dmrg_mps import molecule
from qdk_alps_dmrg import (
    mps_to_wavefunction,
    run_dmrg,
    truncated_hamiltonian,
)
from qdk_chemistry.algorithms import create
from qdk_chemistry.data import AlgorithmRef, Element, MajoranaMapping

NUM_BITS = 12
SEEDS = (11, 12, 13)
# Iterative QPE re-prepares the state for every bit and takes a majority vote;
# 3 shots per bit fails at 81% overlap (stretched H4), 15 holds.
SHOTS_PER_BIT = 15


def statevector(wf):
    qc = create("state_prep", "sparse_isometry_gf2x").run(wf)
    q = qc.get_qiskit_circuit()
    q.remove_final_measurements()
    return qc, Statevector(q).data


def qpe(circuit, qh, t, seed):
    run = create("phase_estimation", "qdk_iterative", shots_per_bit=SHOTS_PER_BIT)
    run.settings().set(
        "qpe_circuit_builder",
        AlgorithmRef(
            "qpe_circuit_builder",
            "qdk_iterative",
            num_bits=NUM_BITS,
            unitary_builder=AlgorithmRef(
                "hamiltonian_unitary_builder", "trotter", time=t
            ),
            controlled_circuit_mapper=AlgorithmRef(
                "controlled_circuit_mapper", "pauli_sequence"
            ),
        ),
    )
    run.settings().set(
        "circuit_executor",
        AlgorithmRef("circuit_executor", "qdk_full_state_simulator", seed=seed),
    )
    return run.run(state_preparation=circuit, qubit_hamiltonian=qh)


def cases():
    H = Element.H
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


def main():
    casci = create("multi_configuration_calculator")
    mapper = create("qubit_mapper", "qdk")
    bad = 0
    for label, ham, (na, nb) in cases():
        n = np.asarray(ham.get_one_body_integrals()[0]).shape[0]
        core = ham.get_core_energy()
        jw = MajoranaMapping.jordan_wigner(2 * n)

        res = run_dmrg(ham, na, nb, maxstates=256, sweeps=8)
        e_ci, wf_ci = casci.run(ham, na, nb)  # full H, not truncated
        circ, psi = statevector(mps_to_wavefunction(res, ham.get_orbitals()))
        _, phi_ci = statevector(wf_ci)

        qh_full = mapper.run(ham, jw)
        h_full = np.asarray(qh_full.to_matrix())
        h_trunc = np.asarray(mapper.run(truncated_hamiltonian(ham), jw).to_matrix())

        e_handoff = (psi.conj() @ h_trunc @ psi).real + core
        e_base = (psi.conj() @ h_full @ psi).real + core
        w, v = np.linalg.eigh(h_full)
        k = int(np.argmax(np.abs(v.conj().T @ phi_ci)))
        e_gs, gs = w[k] + core, v[:, k]
        ov_ci = abs(gs.conj() @ phi_ci)
        p = abs(gs.conj() @ psi) ** 2

        t = np.pi / qh_full.schatten_norm
        bin_width = 2 * np.pi / (t * 2**NUM_BITS)  # |E t| < pi: no phase wrap

        print(
            f"\n== {label}  ({2 * n} qubits, {len(res.dropped)} elements dropped for DMRG)"
        )
        checks = [
            (
                "handoff  <MPS|H_trunc|MPS> = E_dmrg",
                abs(e_handoff - res.energy) < 1e-9,
                f"{e_handoff:.10f} vs {res.energy:.10f}",
            ),
            (
                "JW       full-H eigenvalue = E_casci",
                abs(e_gs - e_ci) < 1e-9,
                f"{e_gs:.10f} vs {e_ci:.10f}",
            ),
            ("JW       |<eigvec|CASCI wf>| = 1", ov_ci > 1 - 1e-9, f"{ov_ci:.12f}"),
            (
                "baseline <MPS|H_full|MPS> >= E_casci",
                e_base >= e_ci - 1e-12,
                f"{e_base:.6f} (+{e_base - e_ci:.2e}),  p = |<GS_full|MPS>|^2 = {p:.4f}",
            ),
            (
                "QPE      resolution separates E_casci, E_dmrg",
                abs(res.energy - e_ci) > 2 * bin_width,
                f"bin {bin_width:.2e} Eh, |E_dmrg - E_casci| = {abs(res.energy - e_ci):.2e}",
            ),
        ]
        hits = 0
        for seed in SEEDS:
            t0 = time.time()
            r = qpe(circ, qh_full, t, seed)
            e_qpe = r.raw_energy + core
            on_gs = abs(e_qpe - e_ci) <= bin_width  # rounding: within one bin
            hits += on_gs
            print(
                f"   QPE seed {seed}: E = {e_qpe:.6f}  (CASCI {e_ci:.6f}, DMRG {res.energy:.6f})  "
                f"dE = {e_qpe - e_ci:+.2e}  {'GS' if on_gs else 'not GS'}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        # Each run lands on the GS with probability ~p; demand a majority.
        checks.append(
            (
                "QPE      majority of runs on the full-H GS",
                hits * 2 > len(SEEDS),
                f"{hits}/{len(SEEDS)} (expected ~{p:.0%})",
            )
        )
        for name, ok, info in checks:
            bad += not ok
            print(f"   {'ok  ' if ok else 'FAIL'} {name:40} {info}")
    print("\nALL PASS" if not bad else f"\n{bad} FAILURE(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

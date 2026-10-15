# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

"""QDK MO Hamiltonian -> ALPS dmrg -> matrix product state.

    from qdk_alps_dmrg import run_dmrg
    res = run_dmrg(ham, n_alpha, n_beta, maxstates=64, sweeps=6)
    res.energy          # DMRG energy, core energy included
    res.mps             # [A_0, ..., A_{L-1}], A_i[left bond, local state, right bond]
    res.site_basis      # {site type: {"Nup": [...], "Ndown": [...]}} per local state
    res.dropped         # the (pq|rs) elements that were truncated

Each MO is one ALPS site, in the order QDK hands them in, so the MPS runs
over the MOs. The lattice is built from the integrals:

* one vertex type per orbital, with SITETERM  h_pp n + (pp|pp) n_up n_down;
* one edge per orbital pair with a nonzero h_pq or two-body element over
  exactly those two orbitals (density, exchange, pair hopping, correlated
  hopping), each its own bond type with a numeric BONDTERM;
* elements over 3 or 4 distinct orbitals are DROPPED and listed in
  ``res.dropped``. ``truncated_hamiltonian`` gives the same truncation as a
  QDK Hamiltonian, so QDK's solvers can be run on what DMRG solved.

Needs an ALPS dmrg with SAVE_MPS (marcusr2ML/ALPS dmrg-save-mps-upstream) and
the full-precision term substitution of ALPSim/ALPS#221; without the latter
every coupling is cut to 6 significant digits. Binaries are looked up in
$ALPS_BIN, then on PATH. Running from a build tree rather than an install
also needs ALPS_XML_PATH=<ALPS source>/src/alps/resources.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

import h5py
import numpy as np

TOL = 1e-10
ALPS_BIN = os.environ.get("ALPS_BIN", "")
MODEL = "qdk dmrg fermion"

# Spinful fermion site: the operators the generated terms use.
FERMION_BASIS = """<SITEBASIS name="fermion">
  <QUANTUMNUMBER name="Nup" min="0" max="1" type="fermionic"/>
  <QUANTUMNUMBER name="Ndown" min="0" max="1" type="fermionic"/>
  <OPERATOR name="c_down" matrixelement="1">
    <CHANGE quantumnumber="Ndown" change="-1"/>
  </OPERATOR>
  <OPERATOR name="cdag_down" matrixelement="1">
    <CHANGE quantumnumber="Ndown" change="1"/>
  </OPERATOR>
  <OPERATOR name="c_up" matrixelement="1">
    <CHANGE quantumnumber="Nup" change="-1"/>
  </OPERATOR>
  <OPERATOR name="cdag_up" matrixelement="1">
    <CHANGE quantumnumber="Nup" change="1"/>
  </OPERATOR>
  <OPERATOR name="n" matrixelement="Nup+Ndown"/>
  <OPERATOR name="n_up" matrixelement="Nup"/>
  <OPERATOR name="n_down" matrixelement="Ndown"/>
</SITEBASIS>
<BASIS name="fermion">
  <SITEBASIS ref="fermion"/>
  <CONSTRAINT quantumnumber="Nup" value="Nup_total"/>
  <CONSTRAINT quantumnumber="Ndown" value="Ndown_total"/>
</BASIS>"""


@dataclass
class DmrgResult:
    energy: float
    mps: list
    site_basis: dict
    site_type: list
    dropped: list = field(default_factory=list)
    workdir: str | None = None


def _tool(name):
    path = os.path.join(ALPS_BIN, name) if ALPS_BIN else ""
    if path and os.path.isfile(path):
        return path
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(f"{name} not found in $ALPS_BIN ({ALPS_BIN!r}) or on PATH")


def integrals(ham):
    """(h1, g, E_core) for a restricted Hamiltonian, g in chemist's order."""
    if ham.has_orbitals() and not ham.get_orbitals().is_restricted():
        raise NotImplementedError(
            "unrestricted Hamiltonian: one set of couplings per spin only"
        )
    h1a, h1b = ham.get_one_body_integrals()
    h1 = np.asarray(h1a, dtype=float)
    if not np.allclose(h1, np.asarray(h1b, dtype=float), atol=TOL):
        raise NotImplementedError("alpha and beta one-body integrals differ")
    n = h1.shape[0]
    g = (
        np.asarray(ham.get_two_body_integrals()[0], dtype=float).reshape(n, n, n, n)
        if ham.has_two_body_integrals()
        else np.zeros((n,) * 4)
    )
    return h1, g, float(ham.get_core_energy())


def orbital_groups(g, tol=TOL):
    """Nonzero two-body elements grouped by the orbitals they touch.

    {sorted tuple of >= 2 distinct orbitals: [(p, q, r, s, (pq|rs)), ...]};
    on-site (pp|pp) is left to the site term.
    """
    groups: dict = {}
    for p, q, r, s in np.array(np.nonzero(np.abs(g) > tol)).T:
        p, q, r, s = int(p), int(q), int(r), int(s)
        key = tuple(sorted({p, q, r, s}))
        if len(key) > 1:
            groups.setdefault(key, []).append((p, q, r, s, float(g[p, q, r, s])))
    return groups


def split_by_arity(g):
    """Two-orbital groups (kept) and the 3/4-orbital elements (dropped)."""
    groups = orbital_groups(g)
    kept = {k: v for k, v in groups.items() if len(k) == 2}
    dropped = [e for k, v in groups.items() if len(k) > 2 for e in v]
    return kept, dropped


def truncated_hamiltonian(ham):
    """The Hamiltonian dmrg actually sees, as a QDK Hamiltonian.

    Same integrals with every (pq|rs) over 3 or 4 distinct orbitals zeroed, so
    QDK's own solvers (CASCI, ...) can be run on exactly what DMRG solved.
    """
    from qdk_chemistry.data import CanonicalFourCenterHamiltonianContainer, Hamiltonian

    h1, g, core = integrals(ham)
    n = h1.shape[0]
    p, q, r, s = np.indices(g.shape)
    distinct = (
        (p != q).astype(int) + (r != p) * (r != q) + (s != p) * (s != q) * (s != r)
    ) + 1
    g = np.where(distinct > 2, 0.0, g)
    fock = (
        np.asarray(ham.get_inactive_fock_matrix()[0], dtype=float)
        if ham.has_inactive_fock_matrix()
        else np.zeros((n, n))
    )
    if fock.shape != (n, n):
        fock = np.zeros((n, n))
    return Hamiltonian(
        CanonicalFourCenterHamiltonianContainer(
            h1, g.reshape(-1), ham.get_orbitals(), core, fock
        )
    )


def _two_body_term(sites, elements):
    """Two-body elements over ``sites`` as an ALPS term on roles r0, r1, ...

    Chemist's notation:
        H2 = 1/2 sum (pq|rs) c+_{p,sigma} c+_{r,tau} c_{s,tau} c_{q,sigma}.
    """
    role = {s: f"r{k}" for k, s in enumerate(sites)}
    out = []
    for p, q, r, s, v in elements:
        c = 0.5 * v
        for a in ("up", "down"):
            for b in ("up", "down"):
                if p == r and a == b:
                    continue  # cdag cdag on one mode: zero
                if q == s and a == b:
                    continue
                out.append(
                    f"({c!r})*cdag_{a}({role[p]})*cdag_{b}({role[r]})"
                    f"*c_{b}({role[s]})*c_{a}({role[q]})"
                )
    return "\n    + ".join(out)


def _bond_term(p, q, h1, elements):
    """Numeric BONDTERM body on roles r0 = p, r1 = q."""
    h1 = np.asarray(h1, dtype=float).tolist()  # plain floats: repr must parse in ALPS
    out = []
    for a in ("up", "down"):
        if abs(h1[p][q]) > TOL:
            out.append(f"({h1[p][q]!r})*cdag_{a}(r0)*c_{a}(r1)")
        if abs(h1[q][p]) > TOL:
            out.append(f"({h1[q][p]!r})*cdag_{a}(r1)*c_{a}(r0)")
    if elements:
        out.append(_two_body_term((p, q), elements))
    return "\n    + ".join(out) if out else "0"


def parm_text(params):
    def fmt(v):
        return f'"{v}"' if isinstance(v, str) else repr(v)

    return "\n".join(f"{k}={fmt(v)}" for k, v in params.items()) + "\n"


def write_inputs(ham, outdir, name="qdk-dmrg"):
    """lattices.xml + models.xml for the truncated Hamiltonian.

    Returns (params, dropped elements).
    """
    h1, g, _ = integrals(ham)
    n = h1.shape[0]
    kept, dropped = split_by_arity(g)

    pairs = sorted(
        set(kept)
        | {
            (p, q)
            for p in range(n)
            for q in range(p + 1, n)
            if abs(h1[p, q]) > TOL or abs(h1[q, p]) > TOL
        }
    )
    lat = [f'<GRAPH name="{name}" vertices="{n}">']
    lat += [f'  <VERTEX id="{p + 1}" type="{p}"/>' for p in range(n)]
    lat += [
        f'  <EDGE type="{k}" source="{p + 1}" target="{q + 1}"/>'
        for k, (p, q) in enumerate(pairs)
    ]
    lat.append("</GRAPH>")

    terms = [
        f'  <SITETERM type="{p}" site="i">\n    '
        f"({float(h1[p, p])!r})*n(i) + ({float(g[p, p, p, p])!r})*n_up(i)*n_down(i)\n  </SITETERM>"
        for p in range(n)
    ]
    terms += [
        f'  <BONDTERM type="{k}" source="r0" target="r1">\n    '
        + _bond_term(p, q, h1, kept.get((p, q), []))
        + "\n  </BONDTERM>"
        for k, (p, q) in enumerate(pairs)
    ]
    model = (
        f'<HAMILTONIAN name="{MODEL}">\n  <BASIS ref="fermion"/>\n'
        + "\n".join(terms)
        + "\n</HAMILTONIAN>"
    )

    os.makedirs(outdir, exist_ok=True)
    lat_path = os.path.join(outdir, "lattices.xml")
    mod_path = os.path.join(outdir, "models.xml")
    with open(lat_path, "w") as fh:
        fh.write("<LATTICES>\n" + "\n".join(lat) + "\n</LATTICES>\n")
    with open(mod_path, "w") as fh:
        fh.write("<MODELS>\n" + FERMION_BASIS + "\n" + model + "\n</MODELS>\n")
    params = {
        "LATTICE_LIBRARY": lat_path,
        "LATTICE": name,
        "MODEL_LIBRARY": mod_path,
        "MODEL": MODEL,
    }
    return params, dropped


def run_dmrg(
    ham,
    n_alpha,
    n_beta,
    maxstates=64,
    sweeps=6,
    workdir=None,
    keep=False,
    lanczos_tol=1e-12,
    extra=None,
):
    """Run ALPS dmrg with SAVE_MPS on the truncated Hamiltonian.

    ``lanczos_tol`` replaces dmtk's default of 1e-7, which leaves ~1e-9 Eh in
    the energy of small systems. ``extra`` adds further ALPS parameters.
    """
    tmp = workdir or tempfile.mkdtemp(prefix="alps_dmrg_")
    try:
        params, dropped = write_inputs(ham, tmp)
        n = int(np.asarray(ham.get_one_body_integrals()[0]).shape[0])
        if n < 4:
            raise ValueError("SAVE_MPS needs at least 4 orbitals")
        os.makedirs(os.path.join(tmp, "tmp"), exist_ok=True)
        params.update(
            {
                "CONSERVED_QUANTUMNUMBERS": "Nup,Ndown",
                "Nup_total": int(n_alpha),
                "Ndown_total": int(n_beta),
                "NUMBER_EIGENVALUES": 1,
                "SAVE_MPS": 1,
                "TEMP_DIRECTORY": os.path.join(tmp, "tmp"),
                "SWEEPS": int(sweeps),
                "MAXSTATES": int(maxstates),
                "LANCZOS_TOLERANCE": float(lanczos_tol),
            }
        )
        params.update(extra or {})
        with open(os.path.join(tmp, "parm"), "w") as fh:
            fh.write(parm_text(params) + "{ }\n")

        p2x = subprocess.run(
            [_tool("parameter2xml"), "parm"],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
        if p2x.returncode:
            raise RuntimeError(
                f"parameter2xml failed: {(p2x.stdout + p2x.stderr).strip()}"
            )
        proc = subprocess.run(
            [_tool("dmrg"), "--write-xml", "parm.in.xml"],
            cwd=tmp,
            capture_output=True,
            text=True,
            check=False,
        )
        with open(os.path.join(tmp, "dmrg.log"), "w") as fh:
            fh.write(proc.stdout + proc.stderr)
        # dmrg exits 0 when it rejects the task files; the missing h5 is the signal
        if proc.returncode or not os.path.exists(
            os.path.join(tmp, "parm.task1.out.h5")
        ):
            raise RuntimeError(f"dmrg failed ({proc.returncode}); see {tmp}/dmrg.log")

        with h5py.File(os.path.join(tmp, "parm.task1.out.h5"), "r") as f:
            if "mps" not in f:
                raise RuntimeError(
                    "no /mps in the result; is this dmrg built with SAVE_MPS?"
                )
            energy = float(np.ravel(f["spectrum/results/Energy/mean/value"][()])[0])
            m = f["mps"]
            mps = [
                np.asarray(m[f"tensors/{i}/values"]).reshape(
                    tuple(m[f"tensors/{i}/shape"])
                )
                for i in range(int(m["length"][()]))
            ]
            site_type = [int(t) for t in m["site_type"]]
            site_basis = {
                int(t): {
                    q: np.asarray(m[f"site_basis/{t}/{q}"])
                    for q in m[f"site_basis/{t}"]
                }
                for t in m["site_basis"]
            }
        return DmrgResult(
            energy + float(ham.get_core_energy()),
            mps,
            site_basis,
            site_type,
            dropped,
            tmp if keep else None,
        )
    finally:
        if not keep and workdir is None:
            shutil.rmtree(tmp, ignore_errors=True)


def mps_to_dense(res):
    """Contract the MPS to psi[occupation] over the 4^L Fock space.

    Returned as a dict {(nup_0, ndn_0, nup_1, ...): amplitude}, nonzero only.
    """
    psi = res.mps[0]
    for A in res.mps[1:]:
        psi = np.tensordot(psi, A, axes=([-1], [0]))
    psi = psi.reshape(psi.shape[1:-1])
    out = {}
    for idx in zip(*np.nonzero(np.abs(psi) > 1e-12)):
        occ = []
        for site, s in enumerate(idx):
            b = res.site_basis[res.site_type[site]]
            occ += [int(b["Nup"][s]), int(b["Ndown"][s])]
        out[tuple(occ)] = float(psi[idx])
    return out


def blocked_sign(occ):
    """Sign relating QDK's blocked mode order to ALPS's site-major one.

    QDK orders all alpha modes then all beta; moving a beta operator past
    every alpha operator on a higher orbital costs one transposition each.
    """
    alpha = [i for i, c in enumerate(occ) if c in "u2"]
    beta = [i for i, c in enumerate(occ) if c in "d2"]
    return (-1) ** sum(sum(1 for a in alpha if a > b) for b in beta)


def mps_to_wavefunction(res, orbitals, cutoff=1e-12):
    """MPS -> QDK Wavefunction (determinant expansion), for small L."""
    from qdk_chemistry.data import Configuration, StateVectorContainer, Wavefunction

    occ_char = {(0, 0): "0", (1, 0): "u", (0, 1): "d", (1, 1): "2"}
    dets, coeffs = [], []
    for occ, c in mps_to_dense(res).items():
        if abs(c) < cutoff:
            continue
        s = "".join(
            occ_char[(occ[2 * p], occ[2 * p + 1])] for p in range(len(occ) // 2)
        )
        dets.append(Configuration.from_spin_half_string(s))
        coeffs.append(blocked_sign(s) * c)
    vec = np.asarray(coeffs)
    vec /= np.linalg.norm(vec)
    if vec[int(np.argmax(np.abs(vec)))] < 0:
        vec = -vec
    return Wavefunction(StateVectorContainer(vec.reshape(-1, 1), dets, orbitals))

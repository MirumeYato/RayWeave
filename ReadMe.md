# RayWeave

RayWeave is a research-oriented Python framework for simulating **radiative transfer** in scattering media.
It implements **Strang splitting** for the Radiative Transfer Equation (RTE), separating streaming and collision operators for accurate and efficient time integration.

The project is based on the methodology described in:

📄 *D.V. Naumov, “Strang Method to solve RTE” (2025)* — see [`Strang.pdf`](./Strang.pdf).

---

## ✨ Features

* **Operator splitting** (Strang splitting, 2nd-order accurate).
* **Collision step** using spherical harmonics expansion and Henyey–Greenstein (HG) scattering phase function.
* **Streaming step** using semi-Lagrangian backtracing and interpolation.
* **Flexible grids** (with future extensions planned: sparse arrays, AMR, octrees, spectral/harmonic grids).
* **NumPy reference implementation** for prototyping.
* **Extendable class design** (Grid, Propagator, Source, Step, BackupData).

---

## 📖 Method Overview

The Radiative Transfer Equation (RTE):

[
\frac{1}{c}\frac{\partial I}{\partial t} + \hat{s}\cdot\nabla I
= -\mu_t I + \mu_s \int_{4\pi} p(\hat{s}\cdot\hat{s}')I(\hat{s}')d\Omega' + \eta
]

is split into two operators:

* **Streaming operator (L):** spatial advection along rays.
* **Collision operator (C):** absorption + scattering in spherical harmonics space.

Strang splitting advances the solution as:

[
I^{n+1} = e^{\tfrac{\Delta t}{2}C} ; e^{\Delta t L} ; e^{\tfrac{\Delta t}{2}C} I^n
]

---

## 📂 Repository Structure

```
RayWeave/
│── src/                # Core Python modules
│   ├── grid.py         # Grid definitions (regular, planned: sparse/AMR)
│   ├── propagator.py   # Strang propagator implementation
│   ├── source.py       # Photon/source initialization
│   ├── step.py         # One propagation step
│   ├── backup.py       # Data saving and checkpointing
│── tests/              # Simple test cases
│── Strang.pdf          # Theory notes and algorithm derivation
│── README.md           # Project documentation
```

---

## 🚀 Getting Started

### Installation

Clone the repository:

```bash
git clone https://github.com/MirumeYato/RayWeave.git
cd RayWeave
```

Install dependencies:

```bash
pip install -r requirements.txt
```

(Dependencies are mostly **NumPy**, **SciPy**, and **Matplotlib** for plotting.)

---

### Example Usage

Run a simple 2D propagation test:

```bash
python tests/simple_test.py
```

This will:

* Initialize a grid with photon sources,
* Propagate intensities over a few Strang steps,
* Save particle trajectories for visualization.

---

## 🔬 Roadmap

* [ ] Add higher-order spherical harmonic truncation schemes.
* [ ] Implement efficient spatial interpolation (monotone cubic / WENO).
* [ ] Extend to adaptive mesh refinement (AMR) and octree grids.
* [ ] Torch/TensorFlow backends for GPU acceleration.
* [ ] Compare with Monte Carlo photon transport methods.

---

## 📚 References

* G. Strang, *On the construction and comparison of difference schemes*, SIAM J. Numer. Anal., 5(3):506–517 (1968).
* D.V. Naumov, *Strang Method to solve RTE* (2025).

---

## 📜 License

MIT License (see [LICENSE](./LICENSE)).
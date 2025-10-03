# RayWeave

RayWeave is a research-oriented Python framework for simulating **radiative transfer** in scattering media.
It implements **Strang splitting** for the Radiative Transfer Equation (RTE), separating streaming and collision operators for accurate and efficient time integration. But also checks some other methods or testcases.

The project is based on the methodology described in:

📄 *D.V. Naumov, “Strang Method to solve RTE” (2025)* — see [`Strang.pdf`](./Strang.pdf).

---

## ✨ Features (planned)

* **Operator splitting** (Strang splitting, 2nd-order accurate).
* **Collision step** using spherical harmonics expansion and Henyey–Greenstein (HG) scattering phase function.
* **Streaming step** using semi-Lagrangian backtracing and interpolation.
* **Flexible grids** (with future extensions planned: sparse arrays, AMR, octrees, spectral/harmonic grids).
* **NumPy reference implementation** for prototyping.
* **Extendable class design** (Grid, Propagator, Source, Step, BackupData).

---

## 📖 Method Overview

The Radiative Transfer Equation (RTE):

$$
\frac{1}{c}\frac{\partial I}{\partial t} + \hat{s}\cdot\nabla I
= -\mu_t I + \mu_s \int_{4\pi} p(\hat{s}\cdot\hat{s}')I(\hat{s}')d\Omega' + \eta
$$

is split into two operators:

* **Streaming operator (L):** spatial advection along rays.
* **Collision operator (C):** absorption + scattering in spherical harmonics space.

Strang splitting advances the solution as:

$$
I^{n+1} = e^{\tfrac{\Delta t}{2}C} ; e^{\Delta t L} ; e^{\tfrac{\Delta t}{2}C} I^n
$$

---

## 📂 Repository Structure

```
RayWeave/
│── lib/                # Core Python modules
│   ├── data/
│      ├── Data.py          # Stores not well designed class for some particle statement
│   ├── physics/
│      ├── Grids.py         # Grid definitions (regular, planned: sparse/AMR)
│      ├── Propagators.py   # Propagator implementation (main calculational part)
│      ├── Sources.py       # Photon/source initialization
│      ├── Steps.py         # One propagation step (it can be streaming, scattering, e.t.c.)
│   ├── results/
│      ├── BackupData.py   # Data saving and checkpointing
│      ├── plot_tools.py   # Some functions for plotting (for now uses not optimized `matplotlib`)
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

You can try to use [Dockerfile](./Dockerfile) for cross-platform usage without having troubles with packages versions.

---

### Example Usage

Run a simple 2D propagation test:

```bash
python tests/simple_test.py
```

This will:

* Initialize a grid with photon sources,
* Propagate intensities over a few streaming steps,
* Save particle trajectories for visualization.

---

## 🔬 Roadmap

* [ ] Physics
    * [ ] Initial conditions
        * [x] Simple source
        * [ ] Gauss like source
        * [ ] Anisotropic source
        * [ ] Realistic Cherenkov source
    * [ ] 5.3 Streaming module
        * [x] Test version
		* [ ] Interpolation
		* [ ] Fourier
    * [ ] 5.2 Scattering module
    * [ ] Nuances
        * [ ] Add higher-order spherical harmonic truncation schemes.
        * [ ] Implement efficient spatial interpolation (monotone cubic / WENO).
        * [ ] Extend to adaptive mesh refinement (AMR) and octree grids.
        * [ ] Torch/TensorFlow backends for GPU acceleration.
* [ ] Results
    * [ ] Visualization (like OPENGL, also will be useful to know more about [pyvista](https://pyvista.org), [pyqtgraph](https://www.pyqtgraph.org))
    * [ ] Compare results
        * [ ] 1D case (we have analytical solution)
        * [ ] Compare with Monte Carlo photon transport methods.
            * [ ] Custom
            * [ ] IceCube
            * [ ] NTSim
        * [ ] RTE (Andrew Sheshukov)

---

## 📚 References

* G. Strang, *On the construction and comparison of difference schemes*, SIAM J. Numer. Anal., 5(3):506–517 (1968).
* D.V. Naumov, *Strang Method to solve RTE* (2025).

---

## 📜 License

MIT License (see [LICENSE](./LICENSE)).
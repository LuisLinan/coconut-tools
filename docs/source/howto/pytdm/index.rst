Adding CMEs with pyTDM
======================

Inject CME flux ropes into **COCONUT** (time-dependent mode) by modifying a
steady-state ``CFmesh`` using **pyTDM**.

Overview
--------
The workflow is:

1. Run a **steady-state COCONUT** simulation → produce ``corona.CFmesh``.
2. Create a **pyTDM configuration** (``.ini``) describing the CME model and parameters.
3. Execute pyTDM → modifies or writes a new CFmesh including the CME.
4. Restart **COCONUT (time-dependent)** from this modified CFmesh.

Models supported:

- **TDM (torus / TDm-type rope)** — *Linan et al., 2023*
- **RBSL (Regularized Biot-Savart Law)** — *Guo et al., 2023*

The details of the physics and model parameters are discussed in those references.

.. toctree::
   :maxdepth: 1

   adding-cmes

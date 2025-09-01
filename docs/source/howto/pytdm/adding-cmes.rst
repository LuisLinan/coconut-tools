Injecting CMEs into COCONUT with pyTDM
======================================

Context & goal
--------------
To simulate CME propagation with **COCONUT (time-dependent)**, you first run a
**steady-state** background to produce a baseline ``corona.CFmesh``. The CME is then
**inserted into that CFmesh** using **pyTDM**, and COCONUT is restarted in **time-dependent**
mode from the modified initial state.

pyTDM was originally developed for PLUTO (F. Regnault) and adapted here to support two
flux-rope models for COCONUT:

- **TDM** (torus/TDm-type rope): see *Linan et al., 2023*.
- **RBSL** (Regularized Biot-Savart Law): see *Guo et al., 2023*.

Module
------
``coconut_tools.pyTDM``

High-level workflow
-------------------
1. **Run steady-state COCONUT** → produce background ``corona.CFmesh``.
2. **Create a pyTDM configuration** (model + parameters).
3. **Execute pyTDM** on that config → writes a **modified CFmesh** with the CME inserted.
4. **Restart COCONUT (time-dependent)** from the modified CFmesh.

Important frame note
--------------------
- The inner boundary in COCONUT comes from a **magnetogram** (HMI/GONG/ADAPT), and in COCONUT
  the **first pixel** of the magnetogram is **Carrington 0°** of the simulation.
- This means the **simulation frame depends on the magnetogram** you used for steady-state.
- Do **not** remap/shift longitudes here. Rotation to **HEEQ** is done later when exporting
  heliospheric boundaries (see :doc:`/howto/boundary/create-dat-rotate`).

Create a CME configuration (`myconfig.createconfig`)
----------------------------------------------------
Generate an ``.ini`` file describing the CME model and parameters. The config is later
consumed by pyTDM to modify the CFmesh.

.. code-block:: python

   from coconut_tools.pyTDM.core_td.myconfig import createconfig

   createconfig(
       path_file='E:/test/',                  # input data context (steady-state run)
       path_tdm='E:/test/',                   # where pyTDM writes intermediates
       path_save='C:/.../pyTDM/config/',      # folder where the .ini is saved
       name='test',                           # case name (used for outputs)

       # Placement/orientation (spherical)
       theta=1.57, phi=3.14, alpha=0,         # colat [rad], lon [rad], tilt [rad]

       # TDm geometry/strength
       D=0.15, A=0.10, R=0.3, delta=0.01,     # distance, radii (minor A / major R), twist
       zeta=5, case_tdm='first',

       # Solver/model
       geometry='spherical', solver='COCONUT',
       flux='Tdm',                            # 'Tdm' or 'RBSL'

       # Auxiliary grid for field construction
       nb_proc=72, nb_r=200, nb_th=200, nb_phi=200, eps=0.01,

       # RBSL-specific parameters (used if flux='RBSL')
       nfr=100, xc=0.5, xh=0.5, hh_fr=120, ll_fr=35, F_flx=20
   )

Parameter glossary
~~~~~~~~~~~~~~~~~~
**Common positioning**

- ``theta``: **colatitude** of the rope center (0 = north pole, π = south pole), in radians.
- ``phi``: longitude of the rope center, in radians.
- ``alpha``: rope **tilt** (rotation around its axis), in radians.

**TDm (``flux='Tdm'``)**

- ``D``: distance from solar surface (apex height parameter).
- ``A``: small (minor) radius of the torus.
- ``R``: large (major) radius of the torus.
- ``delta``: twist parameter.
- ``zeta``: instability/trigger parameter (dimensionless).
- ``case_tdm``: a label for your scenario (e.g. "first").

**RBSL (``flux='RBSL'``)**

- ``nfr``: number of rope field lines used internally.
- ``xc``: center offset (model dependent).
- ``xh``: half-width of the rope cross-section.
- ``hh_fr``: rope height.
- ``ll_fr``: footpoint length.
- ``F_flx``: **total magnetic flux** carried by the rope.

**Auxiliary grid & numerics**

- ``nb_r, nb_th, nb_phi``: spherical grid resolution used by pyTDM to build fields to insert into COCONUT (not the solver mesh).
- ``nb_proc``: number of processes (parallel generation).
- ``eps``: small angular offset used to avoid polar singularities.

Execute pyTDM and write the modified CFmesh
-------------------------------------------
Once the ``.ini`` file(s) exist in a directory, run:

.. code-block:: python

   from coconut_tools.pyTDM.init_TDm import execute_configs

   execute_configs(
       path='C:/Users/.../pyTDM/config/',
       cfmesh=True     # True → write a new corona.CFmesh including the CME
   )

What `execute_configs` does
~~~~~~~~~~~~~~~~~~~~~~~~~~~
- Scans ``path`` for all ``*.ini`` files.
- For each config:

  1) reads parameters via ``myconfig.readconfig_path(...)``,
  2) creates a pyTDM core object,
  3) calls ``TDm.add_TDm_coconut(config, cfmesh)`` to **inject** the CME fields into a CFmesh
     (overwriting or creating a new one),
  4) moves the processed ``.ini`` to a subfolder ``used/``.
- You can queue **multiple configs**: they will be executed sequentially (batch mode).

Restart COCONUT (time-dependent)
--------------------------------
- Point **COCONUT TD** to the **modified CFmesh** as the initial state.
- Configure run time/cadence (and optionally export boundary maps during runtime).
- For heliospheric coupling (EUHFORIA/ICARUS), export boundaries and rotate them to **HEEQ**
  later (see :doc:`/howto/boundary/create-dat-rotate`).

Minimal end-to-end example
--------------------------
.. code-block:: python

   from coconut_tools.pyTDM.core_td.myconfig import createconfig
   from coconut_tools.pyTDM.init_TDm import execute_configs

   # 1) Build a TDM config
   createconfig(
       path_file='/data/coconut/run_ss/',
       path_tdm ='/data/coconut/pytdm_tmp/',
       path_save='/data/coconut/pyTDM/config/',
       name='cme_2020_12_07',
       theta=1.2, phi=2.8, alpha=0.3,
       D=0.12, A=0.08, R=0.28, delta=0.01, zeta=5,
       geometry='spherical', solver='COCONUT', flux='Tdm',
       nb_proc=48, nb_r=200, nb_th=200, nb_phi=200, eps=0.01
   )

   # 2) Inject the CME and write the modified CFmesh
   execute_configs(path='/data/coconut/pyTDM/config/', cfmesh=True)

   # 3) Restart COCONUT (time-dependent) from the new CFmesh
   #    (configure COCONUT TD to pick up that CFmesh as initial condition)

Practical guidance & tips
-------------------------
- **Start conservative**: moderate rope strength (smaller ``A``, moderate ``F_flx``/``zeta``)
  and place away from very intense AR clusters; validate with a quick 3D check
  (:doc:`/howto/vis-analysis/pyvista-threeD`).
- **Angles**: remember that ``theta`` is a **colatitude** (not latitude).
- **TDM vs RBSL**:
  - **TDM**: intuitive radii (``A``, ``R``) and twist (``delta``) for torus-like ropes.
  - **RBSL**: direct control on **total flux** (``F_flx``) and geometric extents (``hh_fr``, ``ll_fr``), convenient for matching target magnetic content.
- **Resolution knobs**: raise ``nb_r/nb_th/nb_phi`` if you need finer rope structure in the initialization; defaults usually suffice for many tests.
- **Batching**: drop multiple ``.ini`` in the same folder to scan various positions/tilts/strengths
  in one go (they’ll be moved under ``used/`` after processing).

References
----------
- Linan, L. *et al.* (2023), *Self-consistent propagation of flux ropes in realistic coronal simulations* (TDM).
- Guo, Y. *et al.* (2023), *COCONUT: Implementation of the regularized Biot-Savart law flux rope model* (RBSL).

See also
--------
- :doc:`/howto/magnetogram/index` — preparing the inner-boundary magnetogram (filtering & frame)
- :doc:`/howto/vis-analysis/pyvista-threeD` — quick 3D inspection of CME-modified snapshots
- :doc:`/howto/boundary/create-dat-rotate` — export & rotate inner boundaries for heliospheric models

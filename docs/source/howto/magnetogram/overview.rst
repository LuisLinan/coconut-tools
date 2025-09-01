Magnetogram filtering for COCONUT (overview)
============================================

Context
-------
In **COCONUT**, the inner boundary is prescribed from the **radial magnetic field (Br)**
of a synoptic magnetogram (HMI, GONG, ADAPT…).  
Directly injecting the raw magnetogram is problematic: the fields are too sharp and
strong, leading to convergence issues. Therefore, **filters** are applied to smooth
the map while preserving the large-scale structure of active regions.

⚠️ Important: in COCONUT the **first pixel** of the magnetogram corresponds to
**Carrington longitude 0°** in the simulation.  
The simulation’s frame thus **depends on the magnetogram** used. Do **not** shift or
reorder the map here — frame rotation is handled later at the heliospheric
boundary stage (see :doc:`/howto/boundary/create-dat-rotate`).

Available filters
-----------------
The package ``coconut_tools.magnetogram`` provides three main filtering approaches:

- **Filtered spherical harmonics (SPH)**: expansion up to degree ``lmax``, with optional
  high-ℓ damping controlled by ``alpha``.
- **Nonlinear diffusion filter (ND)**: Perona–Malik style diffusion that smooths noise
  while preserving edges.
- **Local weighted (Yaroslavsky)**: fast neighborhood-based smoothing with
  spatial and contrast weighting.

Recommendations (based on practice and tests)
---------------------------------------------
- Mesh ~ **2 million cells** → use **SPH** (retains large AR features with fewer cells).
- Mesh ~ **6 million cells** → SPH or ND/Yaroslavsky both acceptable.
- Typical parameter ranges:

  - SPH: start ``lmax=20, alpha=0``. For stronger fields, raise ``lmax`` (30–50) and adjust ``alpha`` between 10⁻⁶ – 10⁻⁵.
  - ND: try ``tau≈5`` with ~6–7 iterations.
  - Yaroslavsky: tune neighborhood radius ``Rn`` and contrast parameter
    ``alpha_factor``; optional Gaussian prefilter.

Performance
-----------
- SPH: minutes for high ``lmax`` (can be parallelized).
- ND: tens of seconds.
- Yaroslavsky: sub-second (depending on grid size).

Further reading
---------------
For an in-depth discussion and benchmark of these techniques, see:

**Murteira, J. et al. (2025)**, *Magnetogram filtering techniques for global coronal modelling*.  


See also
--------
- :doc:`sph-filtering`
- :doc:`nonlinear-diffusion`
- :doc:`local-weighted`

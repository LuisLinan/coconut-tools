Magnetogram Filtering for COCONUT
=================================

Context
-------
COCONUT uses the radial magnetic field, ``Br``, of a synoptic magnetogram as
the inner boundary condition. Raw magnetograms are often too sharp or too
intense to inject directly in a coronal MHD run, so ``coconut_tools`` provides
filters that smooth the map while preserving the large-scale structure of
active regions.

The first longitude pixel written by these tools corresponds to Carrington
longitude 0 degrees in the COCONUT frame. Do not shift or reorder the map in
these filters to prepare a different heliospheric frame; frame rotation is
handled later at the boundary stage, see
:doc:`/howto/boundary/create-dat-rotate`.

Available Filters
-----------------
The package ``coconut_tools.magnetogram`` provides three filtering pipelines:

- Filtered spherical harmonics, implemented in
  ``coconut_tools.magnetogram.sph_filtering``.
- Local weighted filtering, also called Yaroslavsky filtering, implemented in
  ``coconut_tools.magnetogram.Yaroslavsky_filter``.
- Nonlinear diffusion filtering, implemented in
  ``coconut_tools.magnetogram.NLD_implicit_method``.

The three modules expose the same high-level entry point:

.. code-block:: python

   results = process_config(config)

``process_config`` returns one result dictionary per processed target date.

Common Processing Logic
-----------------------
All three filters follow the same pipeline:

1. Build the list of target dates from the configuration.
2. Download the required magnetogram files.
3. Read the magnetogram and, for synchronic GONG, ADAPT, HMI hourly, or
   HMI-FDT, optionally interpolate it in time from four neighboring
   magnetograms.
4. Optionally remove the net magnetic flux.
5. Apply the selected filter.
6. Write the COCONUT ``.dat`` boundary file and, optionally, a diagnostic
   figure.

Latitude Geometry and FITS Headers
----------------------------------
All three pipelines share the same physical FITS latitude decoder. ``Theta``
contains the actual pixel centers, ordered from the north toward the south;
the code does not remap a magnetogram onto another spatial grid before
filtering or spherical-harmonic projection.

The decoder uses ``CRPIX2``, ``CRVAL2``, ``CDELT2`` (or ``CD2_2``),
``CTYPE2``, ``CUNIT2``, and ``LATTYPE``. An explicit ``Sine Latitude`` unit or
``CSLT`` axis is interpreted directly as sine latitude, even when ``CTYPE2``
also contains ``CRLT-CEA``. A standards-compliant CEA axis with an angular
unit is inverted through FITS WCS. Linear latitude axes remain linear in
latitude. Invalid, non-monotonic, non-finite, contradictory, or out-of-domain
axes are rejected.

If the coordinate keywords are incomplete, a warning is emitted before using
the known cell-centered product convention: uniform sine latitude for GONG
and HMI, and uniform latitude for ADAPT and HMI-FDT. These fallback grids do
not put centers at the poles. For example, a 180-row GONG grid starts near
``theta=6.04 degrees`` and a 180-row ADAPT grid starts at
``theta=0.5 degrees``.

With ``resize=True``, the outer physical latitude edges are preserved and new
centers are generated in the same native coordinate. Temporal interpolation
requires all four normalized latitude axes to be identical; incompatible
FITS files raise an error instead of being spatially remapped silently.

Pixel areas are derived from the reconstructed cell edges:

.. math::

   \Delta\Omega_{ij}
   = \left|\cos\theta_{i-1/2}-\cos\theta_{i+1/2}\right|\Delta\phi_j.

The same areas are used for spherical-harmonic projection, flux diagnostics,
surface-mean correction, and polarity scaling. The NLD and Yaroslavsky filter
operators themselves are unchanged.

Single Date and Time Series
---------------------------
The initial date is given with the ``date`` key, using an ISO timestamp:

.. code-block:: python

   "date": "2025-10-09T18:00:00"

To process only one magnetogram, omit ``total_hours``. In that case
``cadence_hours`` is not needed.

To process a time series, set both ``cadence_hours`` and ``total_hours``. The
processed dates are:

.. code-block:: text

   date + k * cadence_hours, while date + k * cadence_hours < date + total_hours

For example, three days with a 3-hour cadence are configured with:

.. code-block:: python

   "cadence_hours": 3,
   "total_hours": 72,

This produces 24 processed magnetograms: the initial date, then every 3 hours,
up to ``date + 69h``.

Temporal Interpolation
----------------------
Temporal interpolation is controlled by a single boolean option:

.. code-block:: python

   "interpolation": True

When enabled for ``GONG_mrzqs``, ``GONG_mrbqs``, ``GONG_mrbqj``, ``ADAPT``,
``HMI_hourly``, or ``HMI_fdt``, the code downloads four magnetograms around
each target date and interpolates ``Br`` at the requested time.
``GONG_mrmqs`` and ``GONG_mrnqs`` are diachronic CAR-frame maps and are
selected as single nearest maps. The interpolation order is selected with:

.. code-block:: python

   "interpolation_order": 2

Use ``1`` for linear interpolation and ``2`` for cubic Hermite interpolation.
The default is cubic Hermite.

Temporal interpolation is implemented for the synchronic GONG variants,
``ADAPT``, ``HMI_hourly``, and ``HMI_fdt``. For ``GONG_mrmqs``,
``GONG_mrnqs``, ``HMI_small``, ``HMI_polfil``, ``HMI_SYNC``, and ``WSO``, the
pipeline uses the single magnetogram selected for the target date or
Carrington rotation.

For ``ADAPT`` and ``HMI_fdt``, ``adapt_map`` selects the realization stored in
the FITS cube. This is a Python zero-based index, so ``adapt_map=6`` selects
the seventh realization.

``HMI_fdt`` downloads HMI-FDTL-driven ADAPT maps from
``https://gong.nso.edu/adapt/maps/hmi-fdtl/``. The server publishes both a
Carrington-fixed product (``adapt40i11``) and a central-meridian-centered
product (``adapt41i11``) at each time. The pipeline deliberately uses only
``adapt40i11`` so all four inputs share the same Carrington longitude grid
before interpolation; Stonyhurst rotation is applied once at the requested
target time.

Net Flux Correction
-------------------
Set ``flux_correct`` to remove the map-averaged net radial magnetic flux before
filtering:

.. code-block:: python

   "flux_correct": True

This subtracts the surface-weighted mean ``Br`` from the input map. It is
applied before the selected filter.

Common Configuration Keys
-------------------------
The following keys are shared by the three filters:

- ``date``: initial ISO timestamp, for example ``"2025-10-09T18:00:00"``.
- ``map_type``: one of ``"GONG_mrzqs"``, ``"GONG_mrbqs"``,
  ``"GONG_mrbqj"``, ``"GONG_mrmqs"``, ``"GONG_mrnqs"``, ``"ADAPT"``,
  ``"HMI_small"``, ``"HMI_polfil"``, ``"HMI_SYNC"``, ``"HMI_hourly"``,
  ``"HMI_fdt"``, or ``"WSO"``.
- ``output_dir``: directory where COCONUT ``.dat`` files are written.
- ``download_dir``: optional directory for downloaded FITS files. If omitted,
  ``output_dir`` is used.
- ``drms_email``: email address used for JSOC DRMS requests when
  ``map_type`` is ``"HMI_SYNC"``. ``jsoc_email`` is also accepted.
- ``cadence_hours`` and ``total_hours``: optional time-series controls.
- ``interpolation``: enable or disable four-magnetogram interpolation for
  synchronic GONG variants, ADAPT, HMI hourly, and HMI-FDT.
- ``interpolation_order``: ``1`` for linear, ``2`` for cubic Hermite.
- ``resize``: resize each normalized input map to ``360 x 720`` before
  temporal interpolation (or resize the selected single map), default
  ``False``.
- ``flux_correct``: enable or disable net-flux correction.
- ``adapt_map``: ADAPT realization index, default ``6``.
- ``lmax``: value included in output filenames. For SPH it is also the
  spherical harmonic truncation degree.
- ``r_st``: radius written in the COCONUT boundary file, default ``1.0``.
- ``write_map``: write the COCONUT ``.dat`` file, default ``True``.
- ``show_map``: write a diagnostic figure, default ``True``.
- ``output_path_fig``: optional diagnostic figure path.
- ``visu_type``: map projection used in the diagnostic figure, default
  ``"sinlat"``. Uniform sine-latitude centers are displayed with ``imshow``
  using their cell edges; nonuniform axes use ``pcolormesh``. Set it to
  ``"lat"`` to display true latitude cell edges with ``pcolormesh``.

Boundary-File Point Count
-------------------------
The COCONUT boundary file keeps the existing ``x y z Br`` columns and writes
the physical coordinates of every retained center. A latitude row is reduced
to one point only when its center is actually ``theta=0`` or ``theta=pi``
within a strict numerical tolerance. All longitudes are written for a
near-polar ring.

Consequently, the declared point count is

.. math::

   N_{\mathrm{points}}
   = N_\theta N_\phi - N_{\mathrm{pole}}(N_\phi-1).

A centered ``180 x 360`` GONG map therefore writes 64,800 points. Historical
grids containing two true pole rows keep the previous count
``(N_theta - 2) * N_phi + 2``.

Output Names
------------
The output filename always includes the filter method and the processed target
date:

.. code-block:: text

   map_gong_lmax20_sph_YYYYMMDDHHMMSS.dat
   map_gong_lmax20_Yaroslavsky_YYYYMMDDHHMMSS.dat
   map_gong_lmax20_NLD_YYYYMMDDHHMMSS.dat

If ``output_path_fig`` is provided for a multi-date run, the timestamp is
inserted before the extension so figures do not overwrite each other. If no
figure path is provided, figures are written by default as:

.. code-block:: text

   output_dir/{map_type_lower}_YYYYMMDDHHMMSS.png

Each diagnostic figure includes the processed date in its title.

Recommendations
---------------
- For meshes around 2 million cells, start with the SPH filter. It keeps the
  dominant active-region structure while reducing small-scale detail.
- For meshes around 6 million cells, SPH, nonlinear diffusion, and Yaroslavsky
  filtering are all viable.
- For SPH, start with ``lmax=20`` and ``alpha=0``. Increase ``lmax`` to keep
  more detail, and increase ``alpha`` to damp high-degree modes.
- For nonlinear diffusion, start with ``tau=5`` and around 6 to 7 iterations.
- For Yaroslavsky filtering, tune ``Rn`` and ``alpha``.

See Also
--------
- :doc:`sph-filtering`
- :doc:`nonlinear-diffusion`
- :doc:`local-weighted`

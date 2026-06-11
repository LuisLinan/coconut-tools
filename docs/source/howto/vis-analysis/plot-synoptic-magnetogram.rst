Plot a synoptic magnetogram aligned to 0° Carrington (HMI / GONG)
==================================================================

Goal
----
Produce a clean **2D plot** of a synoptic magnetogram with the **left edge aligned
to 0° Carrington longitude**, so you can visually check map alignment and compare
with COCONUT inputs / boundary conditions.

Why alignment matters
---------------------
Synoptic products (HMI, GONG) can encode the starting Carrington longitude in
slightly different ways (header keywords, filename suffixes). To make apples-to-apples
comparisons, we first **discover** which longitude represents the left edge of the map,
then **shift/wrap** the columns so plotting runs from **0° → 360°** in Carrington.

Module & entry point
--------------------
The helper lives in ``coconut_tools.magnetogram.play_with_the_frame`` and exposes a single
public entry point:

.. code-block:: python

   fig, ax, info = plot_synoptic_aligned(
       fits_path,
       prefer_filename_for_gong=True,
       vmin=None, vmax=None, cmap="RdBu_r",
       force_degrees=False, force_sine=False,
   )

- Returns a Matplotlib ``(fig, ax)`` plus a diagnostic ``info`` dict describing
  the alignment decision (instrument, step in deg/px, which longitude was used, …). 

What the function does (overview)
---------------------------------
1. **Read** the FITS header/data.  
2. **Build** the native longitude axis from WCS-like keywords (CRPIX/CDELT/CRVAL).  
3. **Infer** the left-edge Carrington longitude from:
   - header only (HMI), or
   - header **and** GONG-style filename suffix (e.g. ``..._268.fits.gz``), then choose which to trust.  
4. **Shift & wrap** longitudes so the left edge is **0° Carrington**, and **reorder columns** accordingly.  
5. **Construct** the latitude axis (degrees or sine-latitude → degrees).  
6. **Render** a diagnostic plot with consistent extent and colorbar. 

Key implementation notes
------------------------
- **Longitude units quirks**: some GONG maps encode **micro-degrees**; the helper auto-detects
  this and converts to **degrees** before alignment. 
- **Latitude encoding**: HMI/GONG may use degrees (``CRLT``) or **sine-latitude** (``CSLT``);
  sine-lat is converted to degrees via ``arcsin``. You can force either mode. 
- **Instrument detection**: based on header (``TELESCOP``, ``INSTRUME``) or filename hints
  (e.g. GONG “MR…”). 
- **Column reordering**: after shifting longitudes, columns are sorted so the x-extent is
  strictly **[0, 360]** deg.

Header math (WCS-like)
----------------------
Longitude world coordinate uses the standard linear relation:

.. code-block:: none

   world = CRVALn + (i - CRPIXn) * CDELTn   (with i starting at 1)

The left-edge longitude (first column, i=1) is:

.. code-block:: none

   lon_start = CRVAL1 + (1 - CRPIX1) * CDELT1

Both step and reference are coerced to **degrees** (with micro-degree detection)
before computing ``lon_start``. 

Choosing which longitude to use
-------------------------------
- **HMI**: use **header** longitude.  
- **GONG**: if a GONG-style filename encodes a suffix (e.g. ``..._268.fits.gz``), you can
  **prefer the filename** value by passing ``prefer_filename_for_gong=True`` (default). 
 
The function reports both values and their difference in the title/``info`` dict. 

Minimal usage
-------------
.. code-block:: python

   from coconut_tools.magnetogram.play_with_the_frame import plot_synoptic_aligned

   # HMI (header-only)
   fig, ax, info = plot_synoptic_aligned("/path/to/hmi.Synoptic_Mr_small.2134.fits",
                                         vmin=-100, vmax=100, force_sine=True)
   print("HMI lon0 used:", info["lon0_used"])

   # GONG (filename may carry the starting Carrington longitude)
   fig, ax, info = plot_synoptic_aligned("mrzqs170404t1814c2189_268.fits.gz",
                                         vmin=-100, vmax=100,
                                         force_sine=True,
                                         prefer_filename_for_gong=True)
   print("GONG lon0 (header/filename/used):",
         info.get("lon0_header"), info.get("lon0_filename"), info.get("lon0_used"))

Returned diagnostics
--------------------
``info`` includes keys like:

- ``instrument`` (``"HMI"`` / ``"GONG"`` / ``"UNKNOWN"``)  
- ``cdelt1_deg`` (deg/pixel), ``lon0_header``, ``lon0_filename`` (if any), ``lon0_used``  
- ``ctype1``/``cunit1`` for the x-axis, and the **latitude mode** chosen (degrees or sine-lat) in the title. 

Customization & tips
--------------------
- **Color scale**: pass ``vmin``/``vmax`` to standardize across maps/runs.  
- **Latitude axis**: ``force_degrees=True`` or ``force_sine=True`` if your product’s header is ambiguous.  
- **Unit on colorbar**: if the header provides ``BUNIT``, it’s shown (defaults to “G”).  
- **Quick alignment check**: the x-axis is always **Carrington [0, 360]** after reordering; if your
  COCONUT boundary map does not line up, revisit the rotation step in :doc:`/howto/boundary/create-dat-rotate`. 

CLI / script demo
-----------------
The module also includes a small ``__main__`` demo illustrating HMI vs GONG usage; adapt the paths
and run as a script if needed. 

See also
--------
- :doc:`/howto/boundary/create-dat-rotate` — rotate COCONUT boundaries into HEEQ before comparison
- :doc:`/howto/comparisons/tomography` — compare density maps with tomography
- :doc:`/api`

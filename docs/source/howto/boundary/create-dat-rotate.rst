Create and rotate inner-boundary ``.dat`` (for EUHFORIA/ICARUS)
================================================================================================================


Goal
----
COCONUT produces ``.CFmesh`` files containing the full 3D state of the corona:
magnetic field and thermodynamic quantities on a numerical grid.

To provide an **inner boundary** to heliospheric models such as **EUHFORIA** or **ICARUS**, we must:

1. **Interpolate** the 3D CFmesh onto a spherical surface (typically 21.5 R⊙).
2. **Save** this surface in ``.dat`` format.
3. **Rotate** the boundary file from the **COCONUT frame** (which depends on the magnetogram) into **HEEQ** coordinates.

---

Step 1 — Create the ``.dat`` from CFmesh
----------------------------------------

The function ``create_boundary_fromcfmesh`` in ``coconut_tools.create_dat`` extracts a spherical surface from a CFmesh file.

.. code-block:: python

   create_boundary_fromcfmesh(
       inputfile: str,
       time: str,
       rad_out: float,
       nb_th: int,
       nb_phi: int,
       eps: float,
       output_dat: str,
       full_output: bool = False
   ) -> None

**Parameters (defaults)**

- ``rad_out`` = 21.5 R⊙ → change if your simulation starts at a different inner boundary (must remain inside the COCONUT domain).
- ``nb_th`` = 180 (colatitude points), ``nb_phi`` = 360 (longitude points) → increase for higher angular resolution.
- ``eps`` = 0.01 rad → a small offset to avoid singularities at the poles.
- ``full_output`` = False → only ``Br, Vr, density, temperature`` (sufficient for solar wind runs).
  Set ``full_output=True`` when following a CME → adds ``Bclt, Blon, Vclt, Vlon``.

**Example**

.. code-block:: python

   from coconut_tools import create_dat
   from datetime import datetime

   cfmesh = "/path/to/corona_flow.CFmesh"
   time = datetime(2020, 12, 7, 15, 0, 0).strftime("%Y-%m-%dT%H:%M:%S")

   create_dat.create_boundary_fromcfmesh(
       inputfile=cfmesh,
       time=time,
       rad_out=21.5,
       nb_th=180,
       nb_phi=360,
       eps=0.01,
       output_dat="boundary_21p5.dat",
       full_output=False
   )

---

Step 2 — Why a rotation is needed
---------------------------------
EUHFORIA and ICARUS expect the inner boundary to be expressed in **HEEQ** coordinates.

However, the **native frame of COCONUT depends on the input magnetogram**:
- if you use GONG, the map has its own Carrington alignment;
- if you use ADAPT, it may be shifted (CM or CAR versions);
- if you use HMI, you need the reference date of the synoptic map.

Therefore, the **longitude grid in the ``.dat`` file must be rotated** to bring it from COCONUT’s Carrington-aligned frame into **Stonyhurst/HEEQ**.
This is done in two steps:
1. Compute the rotation angle with ``compute_rotation_angle``.
2. Apply the rotation to the ``.dat`` file with ``rotation``.

---

Step 3 — Compute rotation angle
-------------------------------

.. code-block:: python

   from coconut_tools.rotation_angle import compute_rotation_angle

   angle_deg = compute_rotation_angle(
       mag_name_path="/path/to/magnetogram.fits",
       date_hmi="2024-07-01T00:00:00"  # required for HMI
   )

This returns the **rotation angle in degrees** needed to convert from Carrington → Stonyhurst/HEEQ.
The function supports **GONG**, **ADAPT (CM/CAR)** and **HMI**.

---

Step 4 — Apply the rotation
---------------------------

Function in ``coconut_tools.create_dat``:

.. code-block:: python

   from coconut_tools.create_dat import rotation

   rotation(
       input_dat="boundary_21p5.dat",
       output_dat="boundary_21p5_heeq.dat",
       angle_degrees=angle_deg,
       full_output=False
   )

This reads the ``.dat`` file, shifts the longitude grid by ``angle_degrees``, and rolls the arrays to preserve continuity.

**Internal logic (simplified)**

.. code-block:: python

   lon_before = [float(x) + np.radians(angle_degrees) for x in lines[lon_start + 1:indices['vr']]]
   lon = [math.fmod(angle, 2 * math.pi) for angle in lon_before]
   min_index = np.argmin(lon)
   lon = np.roll(lon, -min_index)

This ensures that longitude values remain in ``[0, 2π)`` and the data arrays are re-ordered consistently.

---

Batch workflow example
----------------------

In practice you often want to process a **whole sequence of CFmesh snapshots** to build a time series of inner-boundary files.

.. code-block:: python

   import os, glob
   from datetime import datetime, timedelta
   from coconut_tools.create_dat import create_boundary_fromcfmesh, rotation
   from coconut_tools.rotation_angle import compute_rotation_angle

   magnetogram_path = "../hmi.Synoptic_Mr_small.2238.fits"
   date_hmi = "2020-12-07T15:00:00"
   auto_compute_rotation = True
   manual_rotation_angle = 180.0  # fallback

   files = sorted(glob.glob("/run/CFmesh/*.CFmesh"))
   first_time = "2024-04-09T05:04:00"

   rad_out, nb_th, nb_phi, eps = 21.5, 180, 360, 0.01
   full_output = False

   for i, file in enumerate(files):
       date_dt = datetime.strptime(first_time, "%Y-%m-%dT%H:%M:%S")
       date_i = date_dt + timedelta(hours=1 * i * 0.2488 * 0.402)  # adapt cadence
       time_iso = date_i.strftime("%Y-%m-%dT%H:%M:%S")
       timestamp_i = str(int(date_i.timestamp()))

       tmp_dat = f"dat/solar_wind_boundary_{timestamp_i}_tmp.dat"
       out_dat = f"dat/solar_wind_boundary_{timestamp_i}.dat"

       if os.path.exists(out_dat):
           continue

       angle = compute_rotation_angle(magnetogram_path, date_hmi) if auto_compute_rotation else manual_rotation_angle

       create_boundary_fromcfmesh(file, time_iso, rad_out, nb_th, nb_phi, eps,
                                  tmp_dat, full_output=full_output)
       rotation(tmp_dat, out_dat, angle, full_output=full_output)
       os.remove(tmp_dat)

This produces a **time series of inner-boundary files** (one per CFmesh snapshot), all rotated into HEEQ and ready for input into EUHFORIA/ICARUS.

---

Notes & tips
------------
- **Always rotate**: raw COCONUT boundaries are not aligned with HEEQ.
- **Magnetogram matters**: keep track of the file you used, since it sets the rotation angle.
- **CME studies**: set ``full_output=True`` to include B and V components in latitude/longitude.
- **Resolution**: increase ``nb_th``/``nb_phi`` for high-res boundaries, but watch file sizes.
- **Timestamps**: ``time`` must match the CFmesh snapshot in ISO format.

Assess convergence from COCONUT logs (residuals & key fields)
=============================================================

Goal
----
Track whether a **COCONUT** simulation has actually converged by plotting the
evolution of the **residual** and key plasma quantities versus **solver iterations**.
The helper script ``plot_convergence.py`` reads the per-run log file
``convergence.plt-P0.FlowNamespace`` (written by COCONUT) in multiple subfolders
and generates side-by-side comparison plots.

Module
------
``coconut_tools.plot.plot_convergence``

What it reads
-------------
Each case folder must contain a text file named e.g.::

   convergence.plt-P0.FlowNamespace

with tabular content whose first two lines are headers, then data rows like:

.. code-block:: none

   # iter   n      Vx      Vy      Vz      Bx      By      Bz      residual
   ...
   0   1.23e6  -3.1e-3  2.5e-3  ...  1.2e-4  ...

The script expects (after skipping the first two lines) **nine columns** with
the following **headers**:

- ``iter`` (iteration index),
- ``n`` (density),
- ``Vx, Vy, Vz`` (velocity components),
- ``Bx, By, Bz`` (magnetic field components),
- ``residual`` (nonlinear solver residual / norm).

API (core entry)
----------------

.. code-block:: python

   def compare_convergence(
       input_dir: Path,
       output_dir: Path,
       filename: str,
       label_dict: Dict[str, str],
       figsize: Tuple[int, int] = (10, 7)
   ) -> None:
       """Compare convergence curves across several simulation folders."""

Parameters
~~~~~~~~~~
- **input_dir**: directory that contains one subfolder per case (e.g. ``test1/``, ``test2/``).
- **output_dir**: where the figures are written (created if missing).
- **filename**: log filename inside each case folder (usually ``convergence.plt-P0.FlowNamespace``).
- **label_dict**: mapping ``{subfolder_name: "legend label"}`` to control plot legends.
- **figsize**: size of the generated figures.

What it plots
-------------
For each **quantity** in:

- ``n``, ``Vx``, ``Vy``, ``Vz``,
- ``Bx``, ``By``, ``Bz``,
- ``residual``

the function creates a PNG figure ``comparison_<quantity>.png`` saved in ``output_dir``,
with **iterations** on the x-axis and the chosen quantity on the y-axis, one curve per case
(from your ``label_dict``). Grid and legend are included for readability.

Minimal usage
-------------

.. code-block:: python

   from pathlib import Path
   from coconut_tools.plot.plot_convergence import compare_convergence

   input_dir  = Path("/run/media/luis/TOSHIBA EXT/coolfluid/Test_conver/")
   output_dir = input_dir / "plot"
   filename   = "convergence.plt-P0.FlowNamespace"

   label_dict = {
       "test1": "MaxIter 10, AbsNorm -4",
       "test2": "MaxIter 10, AbsNorm -3",
       "test3": "MaxIter 20, AbsNorm -4",
       "test4": "MaxIter 5,  AbsNorm -1",
   }

   compare_convergence(input_dir, output_dir, filename, label_dict, figsize=(10, 10))

How it works (essential bits)
-----------------------------
- For each ``folder_name`` in ``label_dict``, the script reads
  ``input_dir/folder_name/filename``, skips the first two lines, and parses the table:

  .. code-block:: python

     headers = ["iter", "n", "Vx", "Vy", "Vz", "Bx", "By", "Bz", "residual"]
     data = np.loadtxt(lines[2:])
     df = pd.DataFrame(data[:, :9], columns=headers).set_index("iter")

- Then it plots ``df[quantity]`` against ``df.index`` (iterations) and repeats for all quantities of interest.
- One figure per quantity is saved (PNG, DPI=200).

Interpreting the curves
-----------------------
- **Residual** should **decrease** and ideally flatten near machine precision or the chosen tolerance.
  If ``residual`` stalls well above tolerance or increases, the configuration likely **has not converged**.
- **Field components (V, B)** and **density** should stabilize as iterations progress; wildly oscillatory
  behavior often hints at poor solver settings or boundary conditions.
- Compare several **cases** (e.g. different ``MaxIter`` / tolerances) to choose a robust setup.

Tips & variations
-----------------
- **Log scale for residuals**: you can modify the plotting to use ``ax.set_yscale("log")`` for
  ``quantity == "residual"`` if that’s more informative for your runs.
- **Subset of quantities**: clone the function and restrict ``quantities`` to the ones you care about.
- **File robustness**: if some folders lack the file, wrap the reading in ``try/except FileNotFoundError``
  and continue; the rest will still plot.
- **Batch comparisons**: store the ``label_dict`` next to your figures for provenance.
- **Larger fonts/figure size**: increase ``figsize`` and add ``plt.rcParams`` if you target publications.

Output artifacts
----------------
The function writes one PNG per quantity in ``output_dir`` (created automatically), e.g.:

- ``comparison_residual.png``
- ``comparison_Vx.png``, ``comparison_Vy.png``, …

See also
--------
- :doc:`read-output` — load VTU/CFmesh into Python for deeper diagnostics
- :doc:`create-hdf` — precompute time series at fixed points (for in-situ style comparisons)
- :doc:`/api`

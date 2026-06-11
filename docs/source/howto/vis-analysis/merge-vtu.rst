Merge distributed ``.vtu`` files into a single file (ParaView)
==============================================================

Goal
----
When COCONUT (or other MPI-based MHD solvers) writes one ``.vtu`` file **per processor**,
you often want a **single file** to open in ParaView. This utility gathers all pieces
of a snapshot and writes a unified file (``.pvtu`` / merged ``.vtu``) for convenient visualization.

Module
------
``coconut_tools.postprocessing.group_vtu_files``

- Utility to merge ``.vtu`` files from multiple processors into a single file for ParaView visualization.
- For time-dependent runs: filenames like ``corona-flow0-P{j}-iter_<N>.vtu`` (one per processor ``j`` and iteration ``N``).
- For steady-state runs: filenames like ``corona-flow0-P{j}.vtu`` (single iteration).

Function
--------
.. code-block:: python

   def merge_all_snapshots(
       input_dir: str,
       output_dir: str,
       start_time: int,
       timestep: int,
       nbmax: int,
       num_processes: int = 5,
       nb_proc: int = 900
   ) -> None:
       """Launch parallel merging of distributed VTU snapshots."""

Parameters
~~~~~~~~~~
- **input_dir**: directory containing the distributed ``.vtu`` parts.
- **output_dir**: where to write the merged files (create it if needed).
- **start_time**: starting snapshot index (first iteration to process).
- **timestep**: iteration increment between snapshots (e.g. 1 = every iter).
- **nbmax**: number of snapshots to process.
  The **maximum iteration** processed is roughly ``start_time + nbmax * timestep``.
- **num_processes**: parallel workers used to dispatch the merging tasks.
- **nb_proc**: number of **distributed files per snapshot** (i.e. MPI ranks).

Typical usage (time-dependent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from coconut_tools.postprocessing import group_vtu_files

   group_vtu_files.merge_all_snapshots(
       input_dir="/scratch/leuven/342/vsc34280/time-evolving/run/results/",
       output_dir="/scratch/leuven/342/vsc34280/time-evolving/vtu/",
       start_time=0,
       timestep=1,
       nbmax=1,
       num_processes=5,      # tune to your machine
       nb_proc=900           # number of MPI ranks used during the run
   )

This will take, for each iteration ``N``, the files::

   corona-flow0-P0-iter_N.vtu
   corona-flow0-P1-iter_N.vtu
   ...
   corona-flow0-P{nb_proc-1}-iter_N.vtu

and produce a **single merged file** per iteration in ``output_dir`` (ParaView-readable).

Filename patterns
-----------------
Time-dependent (default)
~~~~~~~~~~~~~~~~~~~~~~~~
For the **time-dependent** flavor of COCONUT, the expected pattern is::

   corona-flow0-P{j}-iter_{N}.vtu     (with j = 0..nb_proc-1)

The utility constructs ``N`` as ``start_time * timestep + timestep * index`` while
iterating over ``index in range(nbmax)``.

Steady-state (one iteration)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For **steady-state** runs, each processor writes a single file::

   corona-flow0-P{j}.vtu

You can reuse the same script, **but** you must switch the input pattern inside
``group_vtu_files.py``:

- **Comment** the line that builds the *time-dependent* filenames::

     os.path.join(input_dir, f"corona-flow0-P{j}-iter_{start_time * timestep + timestep * index}.vtu")

- **Uncomment** the line for *steady-state* filenames::

     # os.path.join(input_dir, f"corona-flow0-P{j}.vtu")

(We will likely expose this as a flag later; for now, select the pattern this way.)

Notes & tips
------------
- Set ``nb_proc`` to the **number of MPI ranks** that produced the snapshot, otherwise some parts may be missing.
- ``num_processes`` controls **host-side parallelism** for the merging step; you can increase it until I/O becomes the bottleneck.
- Use a **separate output directory** (e.g. ``.../vtu/``) to keep merged results clean.
- If you are unsure about the parts present, list the files in ``input_dir`` for one iteration to verify the pattern.

See also
--------
- :doc:`/howto/vis-analysis/index`
- :doc:`/api`

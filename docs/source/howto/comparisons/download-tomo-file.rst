Download STEREO tomography files
================================

Goal
----
Fetch STEREO tomography files (``.dat`` and preview ``.jpg``) from the
Aberystwyth archive to compare with COCONUT outputs.

Module
------
``coconut_tools.download_tomo_file``

What it downloads
-----------------
Files of the form:

- ``tomo_sta_cor2a_<YYYYMMDD>_<altitude>.dat``
- ``tomo_sta_cor2a_<YYYYMMDD>_<altitude>.jpg``

From URLs shaped like:

``https://solarphysics.aber.ac.uk/Archives/tomography/cor2a/<year>/distance_<altitude>/``

API
---
.. code-block:: python

   def download_tomography_file(date: str, altitude: str, output_dir: str) -> None:
       """
       Download the .dat and .jpg tomography file for a given date and altitude.

       Args:
           date (str): The date in 'YYYYMMDD' format.
           altitude (str): The altitude string, e.g., '5-0', '4-4'.
           output_dir (str): The directory where to save the files.

       Returns:
           None
       """

Internals (simplified)
----------------------
.. code-block:: python

   year = date[:4]
   url = f"{BASE_URL}/{year}/distance_{altitude}/"
   dat_filename = f"tomo_sta_cor2a_{date}_{altitude}.dat"
   jpg_filename = f"tomo_sta_cor2a_{date}_{altitude}.jpg"

   os.makedirs(output_dir, exist_ok=True)

   for filename in [dat_filename, jpg_filename]:
       file_url = url + filename
       response = requests.get(file_url)
       if response.status_code == 200:
           with open(Path(output_dir) / filename, "wb") as f:
               f.write(response.content)
           logger.info(f"Downloaded: {filename}")
       else:
           raise FileNotFoundError

If the requested altitude is not available for that date, the code logs a list of **available altitudes**:

.. code-block:: python

   def list_available_altitudes(year: str) -> List[str]:
       """Return a list of available altitude strings for a given year."""
       ...

Best practices
--------------
- **Check availability first**: the archive does not provide all altitudes for all dates.
  If a file is missing, use the logged list of available altitudes for that year and choose
  the closest one that matches your COCONUT radius of interest.
- **Keep a local cache**: store downloads in a versioned folder (date/altitude in filename).
- **Compare apples to apples**: use the same date (or closest in time) as your COCONUT snapshot.

See also
--------
- :doc:`/howto/comparisons/tomography` — side-by-side COCONUT vs tomography plots
- :doc:`/howto/vis-analysis/merge-vtu` — merge distributed VTU before comparing
- :doc:`/api`

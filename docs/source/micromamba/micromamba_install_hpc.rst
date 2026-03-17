Micromamba Installation on an HPC User Account
==============================================

This tutorial explains how to install **micromamba** and manage **coconut-tools** with it.

Micromamba is a lightweight package manager compatible with conda environments, easy to manage (and remove if needed).
This will be optimum for a user account on an HPC system and create a Python environment suitable for scientific workflows,
but this can totally be used for your personal laptop as well.

References
----------

- https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html
- https://micro.mamba.pm
- https://conda-forge.org

Prerequisites
-------------

- A Linux HPC system
- Write access to a user data directory
- ``wget``, ``tar`` and ``bash`` available

Installation Steps
------------------

1. Define installation location
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Choose a directory in your HPC storage.

.. code-block:: bash

   export MAMBA_ROOT_PREFIX=/data/leuven/383/$USER/micromamba
   mkdir -p $MAMBA_ROOT_PREFIX
   cd $MAMBA_ROOT_PREFIX


2. Download micromamba
^^^^^^^^^^^^^^^^^^^^^^

Download the latest Linux micromamba package.

.. code-block:: bash

   wget https://micro.mamba.pm/api/micromamba/linux-64/latest -O micromamba.tar.bz2

The downloaded file is a **bzip2 compressed tar archive**, not a plain binary.

It contains the following directory structure::

   bin/
   envs/
   pkgs/

The **micromamba executable is already located in ``bin/``**.

You can confirm the archive type with:

.. code-block:: bash

   file micromamba.tar.bz2


3. Extract micromamba
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   tar -xjf micromamba.tar.bz2

After extraction you should obtain something similar to::

   micromamba/
      bin/
         micromamba
      envs/
      pkgs/

If the archive extracts with a prefix directory, move the contents:

.. code-block:: bash

   mv */* .
   rmdir */

Check the micromamba executable:

.. code-block:: bash

   ./bin/micromamba --version


4. Create the micromamba initialization script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a small script to initialize micromamba:

.. code-block:: bash

   cat > ~/micromamba.sh << EOF
   export MAMBA_ROOT_PREFIX=/data/leuven/383/\$USER/micromamba
   eval "\$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"
   EOF

Load it once in the current shell:

.. code-block:: bash

   source ~/micromamba.sh


5. Automatically load micromamba in new shells
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Add the script to your ``.bashrc``:

.. code-block:: bash

   echo "source ~/micromamba.sh" >> ~/.bashrc

Reload the configuration:

.. code-block:: bash

   source ~/.bashrc


6. Configure package channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``conda-forge`` as the main package source.

.. code-block:: bash

   micromamba config append channels conda-forge
   micromamba config set channel_priority strict


7. Create the Python environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an environment named ``py312``:

.. code-block:: bash

   micromamba create -y -n py312 python=3.12 ipython


8. Activate the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Activate the environment manually:

.. code-block:: bash

   micromamba activate py312

Verify the installation:

.. code-block:: bash

   python --version
   ipython --version


9. Automatically activate the environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To automatically activate the environment in every new terminal:

.. code-block:: bash

   echo "micromamba activate py312" >> ~/.bashrc

Reload the shell configuration:

.. code-block:: bash

   source ~/.bashrc


10. Install scientific Python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Install required scientific libraries:

.. code-block:: bash

   micromamba install -c conda-forge \
   numpy pandas scipy matplotlib \
   astropy sunpy vtk pyvista \
   scikit-learn tqdm requests \
   pyevtk solarmach \
   beautifulsoup4 cmocean colorlog \
   h5py imageio[ffmpeg] natsort


11. Install coconut-tools
^^^^^^^^^^^^^^^^^^^^^^^^^

Clone the repository in your data directory.

.. code-block:: bash

   cd /data/leuven/383/$USER
   git clone https://github.com/LuisLinan/coconut-tools.git

Repository:

https://github.com/LuisLinan/coconut-tools

Install it in editable mode:

.. code-block:: bash

   cd coconut-tools
   python -m pip install -e . --no-deps


12. Test the installation
^^^^^^^^^^^^^^^^^^^^^^^^^

Example Python usage:

.. code-block:: python

   from coconut_tools import group_vtu_files

Example function call:

.. code-block:: python

   group_vtu_files.merge_all_snapshots(
       input_dir="./",
       output_dir="./vtu/",
       start_time=0,
       timestep=1,
       nbmax=1,
       num_processes=1,
       nb_proc=36,
       stat=True,
       remove=False
   )


Quick Installation Summary
--------------------------

For experienced users, the minimal installation workflow:

.. code-block:: bash

   export MAMBA_ROOT_PREFIX=/data/leuven/383/$USER/micromamba
   mkdir -p $MAMBA_ROOT_PREFIX
   cd $MAMBA_ROOT_PREFIX

   wget https://micro.mamba.pm/api/micromamba/linux-64/latest -O micromamba.tar.bz2
   tar -xjf micromamba.tar.bz2

   echo 'export MAMBA_ROOT_PREFIX=/data/leuven/383/$USER/micromamba' > ~/micromamba.sh
   echo 'eval "$($MAMBA_ROOT_PREFIX/bin/micromamba shell hook -s bash)"' >> ~/micromamba.sh

   echo "source ~/micromamba.sh" >> ~/.bashrc
   source ~/.bashrc

   micromamba create -n py312 python=3.12
   micromamba activate py312
   echo "micromamba activate py312" >> ~/.bashrc
   
   
Why use micromamba on HPC systems
---------------------------------

Micromamba is particularly well suited for user environments on HPC systems.

Advantages
^^^^^^^^^^

* **No administrator privileges required**  
  Micromamba installs entirely in user space and does not require root access.

* **Very small footprint**  
  The executable is a single lightweight binary (a few MB) compared to full
  Conda installations.

* **Fast dependency resolution**  
  Micromamba uses the libmamba solver, which is significantly faster than the
  traditional conda solver when creating or updating environments.

* **Clean separation from system software**  
  Environments are isolated inside the user's directory, avoiding conflicts
  with centrally installed HPC modules.

* **Reproducible environments**  
  Environments can be exported and recreated easily, which is essential for
  scientific reproducibility.

Because of these properties, micromamba is commonly recommended for
managing Python environments on HPC infrastructures.


Backing up and reproducing environments
---------------------------------------

To ensure reproducibility or share an environment with collaborators, the
environment specification can be exported to a YAML file.

Export the environment
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   micromamba activate py312
   micromamba env export > py312_environment.yml

This file contains all packages and versions required to recreate the
environment.


Recreate the environment from a backup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To recreate the environment on another system:

.. code-block:: bash

   micromamba create -n py312 -f py312_environment.yml


List existing environments
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   micromamba env list


Remove an environment
^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   micromamba remove -n py312 --all

contact: Q. Noraz

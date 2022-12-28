A quantitative comparison of linear and nonlinear loudspeaker models
====================================================================

This repository provides a Python_ + JAX_ implementation of loudspeaker
differential-equation models and fitting procedures to reproduce the results
from our paper for ICA2022_.

The code was tested on Linux only, apart from ```notebooks/00_data_aquisition.ipynb``, which was run from Windows.

Paper:
   Find it `here <https://github.com/fhchl/quant-comp-ls-mod-ica22/blob/publish-code-orphan/paper.pdf>`_.

Data set:
   Download files from the `release page <https://github.com/fhchl/quant-comp-ls-mod-ica22/releases>`_.

License:
   MIT -- see the file ``LICENSE`` for details.

.. _Python: https://www.python.org/
.. _JAX: https://github.com/google/jax
.. _ICA2022: https://ica2022korea.org


Project overview
----------------

Repo is structured as follows::

   .
   ├── data                           # [data set, get it from release page]
   ├── notebooks                      # [main notebooks and scripts]
   │   ├── 00_data_aquisition.ipynb       # measurement
   │   ├── 01_preprocessing.ipynb         # average and cleanup data
   │   ├── 02_model_training.py           # fit ode models
   │   ├── 03_model_prediction.py         # predict with fitted models
   │   └── 04_analysis.ipynb              # analyze results
   ├── src                            # [models defs, training procedures, etc.]
   ├── README.rst
   ├── environment.yml
   ...


Getting started
---------------

On Linux, create a virtual environment with::

   conda env create -f environment.yml

On Windows, use::

   Conda env create -f environment_windows.yml

Afterwards, activate the environment::

   conda activate mod_comp

Download dataset from release page::

   python data/download_files.py

Run scripts or notebooks in `notebooks` dir.  Enjoy!


Citation
--------

If you found this  codebase useful in your research, please cite::

   @inproceedings{heuchelQuantComp2022,
      author       = "Heuchel, Franz M. and Agerkvist, Finn T.",
      title        = "A quantitative comparison of linear and nonlinear loudspeaker models",
      booktitle    = "Proceedings of the 24th International Congress on Acoustics",
      year         = "2022",
      pages        = "1-8",
   }






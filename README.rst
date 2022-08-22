A quantitative comparison of linear and nonlinear loudspeaker models
====================================================================

This repository provides a Python_ + JAX_ implementation of models and
fitting procedures to reproduce all results from our recent paper for ICA2022_.

The code was tested on Linux only (JAX is not oficially supported on Windows).

Paper:
   Find it `here <https://github.com/fhchl/quant-comp-ls-mod-ica22/blob/publish-code-orphan/paper.pdf>`_

Data set:
   Download files from the `release page <https://github.com/fhchl/quant-comp-ls-mod-ica22/releases>`_.

License:
   MIT -- see the file ``LICENSE`` for details.

.. _Python: https://www.python.org/
.. _JAX: https://github.com/google/jax
.. _ICA2022: https://ica2022korea.org


Quick overview
--------------

Repo is structured as follows::

   .
   ├── data
   │   ├── excitations.hdf5
   │   ├── predictions.dill
   │   ├── processed.hdf5
   │   ├── results.hdf5
   │   └── samples
   ├── notebooks
   │   ├── 00_data_aquisition.ipynb
   │   ├── 01_preprocessing.ipynb
   │   ├── 02_model_training.py
   │   ├── 03_model_prediction.py
   │   └── 04_analysis.ipynb
   ├── src
   │   ├── __init__.py
   │   ├── jaxutil.py
   │   ├── linear_sys_ident.py
   │   ├── measurement.py
   │   ├── model.py
   │   └── util.py
   ├── README.rst
   ├── environment.yml
   ├── setup.cfg
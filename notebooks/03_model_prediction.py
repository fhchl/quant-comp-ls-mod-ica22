import warnings
import os, sys, shutil
import dill
import jax
import h5py
from scipy.signal import welch

# change cwd to `data`
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

sys.path.insert(0, os.path.abspath(os.path.dirname(os.getcwd())))
from src.util import nrmse, raw_decode
from src.jaxutil import clear_caches

warnings.simplefilter("ignore", FutureWarning)  # diffrax tree_util

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

validation_signal = "pink10-10k_2"

db_processed_path = "../data/processed.hdf5"
db_fitted_models_path = "../data/fitted_models.hdf5"
db_predictions_path = "../data/predictions.dill"
columns = ['model', 'driver', 'train level', 'test level', 'train ierror',
           'train verror', 'ierror', 'verror', 'i_psd', 'v_psd', 'ipred_psd',
           'vpred_psd', 'ei_psd', 'ev_psd']

try:
  with open(db_predictions_path, 'rb') as f:
    tab = dill.load(f)
except (FileNotFoundError):
  tab = {}

with (h5py.File(db_fitted_models_path, "r") as db_fitted_models,
      h5py.File(db_processed_path, "r") as db_processed):
  for ls in db_processed:
    rec = db_processed[ls][validation_signal]['rec']
    levels = db_processed[ls][validation_signal].attrs['levels']
    sr = db_processed[ls][validation_signal].attrs['sr']
    for modelname in db_fitted_models:
      for trainl, train_lvl in enumerate(levels):
        try:
          model = raw_decode(
            db_fitted_models[modelname][ls][str(train_lvl)].attrs['model_fitted_blob'])
          print(modelname, ls, train_lvl)
          ierror_train, verror_train = (db_fitted_models[modelname][ls][str(train_lvl)]
                                        .attrs['training_errors'])
        except Exception as e:
          print(modelname, ls, train_lvl, e)
          continue

        for testl, test_lvl in enumerate(levels):
          if (modelname, ls, train_lvl, test_lvl) in tab:
            continue
          print(test_lvl)
          u, i, v = rec[..., testl]
          ipred, vpred = model.predict(u, sr).T
          ierror = nrmse(i, ipred)
          verror = nrmse(v, vpred)

          nperseg = 2**14
          f, i_psd = welch(i, fs=sr, nperseg=nperseg)
          f, v_psd = welch(v, fs=sr, nperseg=nperseg)
          f, ei_psd = welch(ipred-i, fs=sr, nperseg=nperseg)
          f, ev_psd = welch(vpred-v, fs=sr, nperseg=nperseg)
          f, ipred_psd = welch(ipred, fs=sr, nperseg=nperseg)
          f, vpred_psd = welch(vpred, fs=sr, nperseg=nperseg)
          print(ierror, verror)
          tab[(modelname, ls, train_lvl, test_lvl)] = (ierror_train, verror_train,
                                             ierror, verror, i_psd, v_psd, ipred_psd,
                                             vpred_psd, ei_psd, ev_psd)
          # save to file
          with open(db_predictions_path, 'wb') as f:
            dill.dump(tab, f)
          shutil.copyfile(db_predictions_path, db_predictions_path+".bak")
        clear_caches()

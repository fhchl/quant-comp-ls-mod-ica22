import warnings
warnings.simplefilter("ignore", FutureWarning)  # diffrax tree_util
import shutil
import os, sys
from datetime import datetime
sys.path.insert(0, os.path.abspath(os.path.dirname(os.getcwd())))
import gc, psutil
import time
from src.util import nrmse

import diffrax as dfx
import jax
import h5py
from tqdm import tqdm

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)

db_processed_path = "../data/processed.hdf5"
db_fitted_models_path = "../data/fitted_models.hdf5"


#### Helper functions for training

def clear_caches():
  """Clear all kind of jax and diffrax caches."""
  # see https://github.com/patrick-kidger/diffrax/issues/142
  process = psutil.Process()
  if process.memory_info().rss > 4 * 2**30: # >500MB memory usage
    for module_name, module in sys.modules.items():
      if module_name.startswith("jax"):
        for obj_name in dir(module):
          obj = getattr(module, obj_name)
          if hasattr(obj, "cache_clear"):
            obj.cache_clear()
  dfx.diffeqsolve._cached.clear_cache()
  gc.collect()
  print("Cache cleared")


### Define models

from src.model import *

key = jax.random.PRNGKey(42)
model_dict = {
  "Simplest": lambda: FittableModel(system=SimplestDyn()),
  "LinearL2R2": lambda: FittableModel(system=LinearL2R2Dyn()),
  "LinearSLS": lambda: FittableModel(system=LinearSLSDyn()),
  "LinearL2R2SLS": lambda: FittableModel(system=LinearL2R2SLSDyn()),
  "PolyNonLinL2R2Cun4": lambda: FittableModel(system=PolyNonLinL2R2CunDyn()),
  "PolyNonLinSLSL2R2GenCunLi4": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn()),
  "PolyNonLinSLSL2R2GenCunLi4_Euler": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Euler()),
  "PolyNonLinSLSL2R2GenCunLi4_Heun": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Heun()),
  "PolyNonLinSLSL2R2GenCunLi4_Midpoint": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Midpoint()),
  "PolyNonLinSLSL2R2GenCunLi4_Ralston": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Ralston()),
  "PolyNonLinSLSL2R2GenCunLi4_Bosh3": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Bosh3()),
  "PolyNonLinSLSL2R2GenCunLi4_Tsit5": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Tsit5()),
  "PolyNonLinSLSL2R2GenCunLi4_Dopri5": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(), solver=dfx.Dopri5()),
  "PolyNonLinSLSL2R2CunLi4": lambda: FittableModel(system=PolyNonLinSLSL2R2CunLiDyn()),
  "PolyNonLinSLSL2R2Li4": lambda: FittableModel(system=PolyNonLinSLSL2R2LiDyn()),
  "PolyNonLinSLSL2R2GenCun4": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunDyn()),
  "PolyNonLinL2R2GenCunLi4": lambda: FittableModel(system=PolyNonLinL2R2GenCunLiDyn()),
  "PolyNonLinSLSGenCunLi4": lambda: FittableModel(system=PolyNonLinSLSGenCunLiDyn()),
  "PolyNonLinSLSL2R2GenCunLi7": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=7, nK=7, nL=7)),
  "PolyNonLinSLSL2R2GenCunLi6": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=6, nK=6, nL=6)),
  "PolyNonLinSLSL2R2GenCunLi5": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=5, nK=5, nL=5)),
  "PolyNonLinSLSL2R2GenCunLi4": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=4, nK=4, nL=4)),
  "PolyNonLinSLSL2R2GenCunLi3": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=3, nK=3, nL=3)),
  "PolyNonLinSLSL2R2GenCunLi2": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=2, nK=2, nL=2)),
  "PolyNonLinSLSL2R2GenCunLi1": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=1, nK=1, nL=1)),
  "PolyNonLinSLSL2R2GenCunLi0": lambda: FittableModel(system=PolyNonLinSLSL2R2GenCunLiDyn(nBl=0, nK=0, nL=0)),
}

# get overview of data
with h5py.File(db_processed_path, "r") as db_data:
  all_ls = list(db_data.keys())
  print(all_ls)
  all_sig = list(db_data[all_ls[0]].keys())
  print(all_sig)


### training

from src.util import create_or_return_group, raw_decode, raw_encode

training_signal = "pink10-10k_1"

# for each model
with (h5py.File(db_fitted_models_path, "a") as db_results,
      h5py.File(db_processed_path, "r") as db_processed):
  try:
    for modelname in tqdm(model_dict.keys(), desc="model", leave=False):
      res_mod_grp, _ = create_or_return_group(db_results, modelname)

      for ls in tqdm(all_ls, desc="ls", leave=False):
        res_mod_ls_grp, isnew = create_or_return_group(res_mod_grp, ls)

        # load training data
        train_sig_grp = db_processed[ls][training_signal]
        sr = train_sig_grp.attrs['sr']
        levels = train_sig_grp.attrs['levels']
        rec = train_sig_grp['rec']

        # create a fresh model instance
        model = model_dict[modelname]()

        for lvl, level in tqdm(list(enumerate(levels)), desc="level"):
          res_mod_ls_lvl_grp, isnew = create_or_return_group(res_mod_ls_grp, str(level))

          u, i, v = rec[..., lvl]
          num_train = len(u)
          num_pred = len(u)
          try:
            # check if model is already trained on this data
            if (blob := res_mod_ls_lvl_grp.attrs.get('model_blob')) and raw_decode(blob) == model:
              print(f"{modelname} on {ls} with level {level} exists, skipping.")
              continue

            # fit model
            print("\n\n")
            print(f"Fitting {modelname} on {ls} with level {level} ...")
            print(f"====================================================")
            start = time.process_time()
            model_fitted = model.fit(u[:num_train], i[:num_train], v[:num_train], sr, reg=1e-4, max_nfev=100, ftol=1e-4)
            end = time.process_time()
            fittime = end - start
            print(f"took {fittime:.2f} sec.")
            print("Fitted model:", model_fitted)

            # predict on training data
            ipred, vpred = model_fitted.predict(u[:num_pred], sr).T
            ierror = nrmse(i[:num_pred], ipred)
            verror = nrmse(v[:num_pred], vpred)
            print(f"Training error (i, v): {ierror, verror}.")

            # save model
            res_mod_ls_lvl_grp.attrs['model_blob'] = raw_encode(model)
            res_mod_ls_lvl_grp.attrs['model_fitted_blob'] = raw_encode(model_fitted)
            res_mod_ls_lvl_grp.attrs['training_time_sec'] = fittime
            res_mod_ls_lvl_grp.attrs['training_errors'] = (ierror, verror)
            res_mod_ls_lvl_grp.attrs['level'] = level

            # force write to file
            db_results.flush()
            # make a backup in case something happens
            shutil.copyfile(db_fitted_models_path, db_fitted_models_path+".bak")

            print("Saved", res_mod_ls_lvl_grp)

          except ValueError as e:
            print(f"Fitting failed: {e}")

        clear_caches()

  except KeyboardInterrupt:
    print("KeyboardInterrupt")



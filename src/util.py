import dill
import h5py
import numpy as np
from scipy.signal import resample_poly

# from https://stackoverflow.com/a/53340677/2629879
def descend_obj(obj, sep='\t'):
  """
  Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
  """
  if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
    for key in obj.keys():
      print(sep, '-', key, ':', obj[key])
      descend_obj(obj[key], sep=sep+'\t')
  elif type(obj) == h5py._hl.dataset.Dataset:
    for key in obj.attrs.keys():
      print(sep+'\t', '-', key, ':', obj.attrs[key])

def h5dump(path,group='/'):
  """
  print HDF5 file metadata

  group: you can give a specific group, defaults to the root group
  """
  with h5py.File(path, 'r') as f:
     descend_obj(f[group])


def create_or_return_group(parent, name):
  if name in parent:
    grp = parent[name]
    isnew = False
  else:
    grp = parent.create_group(name)
    isnew = True
  return grp, isnew

def create_or_overwrite_dataset(parent, name, **kwargs):
  try:
    dset = parent.create_dataset(name, **kwargs)
    isnew = False
  except ValueError:
    del parent[name]
    dset = parent.create_dataset(name, **kwargs)
    isnew = True
  return dset, isnew

def nmse(target, prediction):
  """Normalized mean squared error."""
  return np.mean(np.abs(target-prediction)**2, axis=0)/np.mean(np.abs(target)**2, axis=0)

def nrmse(target, prediction):
  """Normalized root mean squared error."""
  return np.sqrt(nmse(target, prediction))

def nmae(target, prediction):
  """Normalized mean absolute error."""
  return np.mean(np.abs(target-prediction), axis=0)/np.std(target, axis=0)

# https://docs.h5py.org/en/stable/strings.html#how-to-store-raw-binary-data
def raw_encode(obj): return np.void(dill.dumps(obj))
def raw_decode(blob): return dill.loads(blob.tobytes())

def find_nearest(array, value):
    """Find nearest value and its index in array."""
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx
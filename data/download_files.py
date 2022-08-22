import urllib.request
import os

# change cwd to `data`
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

urls = [
  'https://github.com/fhchl/quant-comp-ls-mod-ica22/releases/download/reproduces-results/excitations.hdf5',
  'https://github.com/fhchl/quant-comp-ls-mod-ica22/releases/download/reproduces-results/fitted_models.hdf5',
  'https://github.com/fhchl/quant-comp-ls-mod-ica22/releases/download/reproduces-results/predictions.dill',
  'https://github.com/fhchl/quant-comp-ls-mod-ica22/releases/download/reproduces-results/processed.hdf5',
]

for url in urls:
  filename = url.split('/')[-1]
  print("downloading", filename, "...")
  urllib.request.urlretrieve(url, filename)
print("Done.")
import numpy as np
import scipy.signal as sig
from scipy.optimize import least_squares
from scipy.optimize._optimize import MemoizeJac
import jax
import jax.numpy as jnp

from .jaxutil import value_and_jacfwd

def tf_matching(model, x, y, sr, x0, nperseg=2**14, reg=1e-6, **kwargs):
  """Estimate linear parameters by matching of cross spectral densities."""
  f, S_xx = sig.welch(x[:, None], fs=sr, nperseg=nperseg, axis=0)
  f, S_yx = sig.csd(x[:, None], y, fs=sr, nperseg=nperseg, axis=0)
  weight = np.std(S_yx) * np.sqrt(len(f))
  x0 = np.asarray(x0)

  def residuals(params):
    hatG_yx = model(f, params)
    hatS_yx = hatG_yx * S_xx
    res = (S_yx - hatS_yx) / weight
    regterm = params / x0 * reg
    return jnp.concatenate((
      jnp.real(res).reshape(-1),
      jnp.imag(res).reshape(-1),
      regterm # high param values may lead to stiff ODEs
    ))

  fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
  jac = fun.derivative
  res = least_squares(fun, x0, jac=jac, x_scale='jac', bounds=(0, np.inf), **kwargs)
  return res.x.tolist()
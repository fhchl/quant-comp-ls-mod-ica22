import functools
import abc

import jax
import jax.tree_util as jtu
import equinox as eqx
import jax.numpy as jnp
import diffrax as dfx
import numpy as np

from scipy.optimize import least_squares
from scipy.optimize._optimize import MemoizeJac
from .linear_sys_ident import tf_matching
from .jaxutil import value_and_jacfwd

class DynamicalSystem(eqx.Module):
  n_states: int = eqx.static_field()
  n_params: int = eqx.static_field()

  @abc.abstractmethod
  def vector_field(self, x, u=None, t=None): pass

  @abc.abstractmethod
  def output(self, x, t=None): pass


class ControlAffine(DynamicalSystem):
  @abc.abstractmethod
  def f(self, x, t=None): pass

  @abc.abstractmethod
  def g(self, x, t=None): pass

  @abc.abstractmethod
  def h(self, x, t=None): pass

  def vector_field(self, x, u, t=None):
    return self.f(x, t) + self.g(x, t)*u

  def output(self, x, t=None):
    return self.h(x, t)


class ForwardModel(eqx.Module):
  system: DynamicalSystem
  solver: dfx.AbstractAdaptiveSolver = eqx.static_field()
  step: dfx.AbstractStepSizeController = eqx.static_field()
  bounds: tuple = eqx.static_field()

  def __init__(self, system, solver=dfx.Dopri5(),
               step=dfx.ConstantStepSize()):
    self.system = system
    self.solver = solver
    self.step = step
    self.bounds = _positive_linear_bounds(system)

  def __call__(self, t, x0, ufun):
    vector_field = lambda t, x, self: self.system.vector_field(x, ufun(t), t)
    term = dfx.ODETerm(vector_field)
    saveat = dfx.SaveAt(ts=t)
    sol = dfx.diffeqsolve(term, self.solver, t0=t[0], t1=t[-1], dt0=t[1],
                          y0=x0, saveat=saveat, max_steps=1000*len(t),
                          stepsize_controller=self.step,
                          args=self)
    y = jax.vmap(self.system.output)(sol.ys)
    return y


def _positive_linear_bounds(sys: DynamicalSystem):
  # upper limit is +inf for all params
  ub = jtu.tree_map(lambda _: np.inf, sys)
  ub, _ = jtu.tree_flatten(ub)  # to list
  # for LinearParams, lower limit is zero, else -inf
  lb = jtu.tree_map(lambda x: 0. if isinstance(x, LinearParameter) else -np.inf, sys)
  lb, _ = jtu.tree_flatten(lb)  # to list
  return lb, ub

def _fit_ml(model: ForwardModel, t, u, y, x0, constrained=True, **kwargs):
  """Fit forward model via maximum likelihood."""
  t = jnp.asarray(t)
  u = jnp.asarray(u)
  y = jnp.asarray(y)
  coeffs = dfx.backward_hermite_coefficients(t, u)
  cubic = dfx.CubicInterpolation(t, coeffs)
  ufun = lambda t: cubic.evaluate(t)
  init_params, treedef = jtu.tree_flatten(model)
  std_y = np.std(y, axis=0)
  bounds = model.bounds if constrained else (-np.inf, np.inf)

  # scale parameters and bounds
  def residuals(params):
    model = treedef.unflatten(params)
    pred_y = model(t, x0, ufun)
    res = ((y - pred_y)/std_y).reshape(-1)
    return res / np.sqrt(len(res))

  # compute primal and sensitivties in one forward pass
  fun = MemoizeJac(jax.jit(lambda x: value_and_jacfwd(residuals, x)))
  jac = fun.derivative
  try:
    # use https://lmfit.github.io/lmfit-py/index.html instead?
    res = least_squares(fun, init_params, jac=jac, x_scale='jac', bounds=bounds,
                        **kwargs)
  except:
    print(model)
    raise
  params = res.x
  return treedef.unflatten(params)


class FittableModel(ForwardModel):
  """A forward model that can be fitted."""

  @property
  def n_params(self): return self.system.n_params
  @property
  def n_states(self): return self.system.n_states

  def fit_linear_params(self, t, u, y, sr, **kwargs):
    out = jax.vmap(self.system.h)
    tf_iv = lambda f, p: out(self.system.tf(f, p))
    linear_params = tf_matching(tf_iv, u, y, sr, x0=self.system.init_linear_params, **kwargs)
    # replace linear parameters by newly found parameters
    filter_spec = lambda x: isinstance(x, LinearParameter)
    lin_tree, nonlin_tree = eqx.partition(self, filter_spec, replace=None)
    _, lintreedef = jtu.tree_flatten(lin_tree)
    lin_tree = lintreedef.unflatten(linear_params)
    model = eqx.combine(lin_tree, nonlin_tree)
    return model

  def fit(self, u, i, v, sr, reg=0, **kwargs):
    """Fit to equisampled data."""
    t = jnp.arange(len(u))/sr
    u = jnp.asarray(u)
    y = jnp.stack((i, v), axis=-1)
    x0 = jnp.zeros(self.system.n_states)
    # fit linear parameters
    fwd_model = self.fit_linear_params(t, u, y, sr, reg=reg, **kwargs)
    # fit all parameters
    fwd_model = _fit_ml(fwd_model, t, u, y, x0, verbose=2, **kwargs)
    return fwd_model

  @functools.partial(jax.jit, static_argnames=['self', 'sr'])
  def predict(self, u, sr):
    """Predict from equisampled input."""
    t = jnp.arange(len(u))/sr
    u = jnp.asarray(u)
    coeffs = dfx.backward_hermite_coefficients(t, u)
    cubic = dfx.CubicInterpolation(t, coeffs)
    ufun = lambda t: cubic.evaluate(t)
    x0 = jnp.zeros(self.system.n_states)
    return self(t, x0, ufun)


def _simple_tf(f, params):
  """Compute frequency response for simplest linear model."""
  Bl, Re, Rm, K, L, M = params[:6]
  s = 1j*2*jnp.pi*f
  sZm = s**2*M + Rm*s + K # derivative of mech. impedance
  Ze = Re + s*L          # electrical impedance
  D = Bl / (sZm*Ze + Bl**2*s)
  V = s*D
  I = (1 - Bl*s*D) / Ze
  return jnp.stack((I, D, V), axis=-1)


def _l2r2_tf(f, params):
  """Compute frequency response with L2R2 model."""
  Bl, Re, Rm, K, L, M, L2, R2 = params[:8]
  s = 1j*2*jnp.pi*f
  sZm = s**2*M + Rm*s + K  # mechanical impedance
  Ze = Re + s*L + R2*L2*s/(R2+L2*s)  # electrical impedance
  D = Bl / (sZm*Ze + Bl**2*s)
  V = s*D
  I = (1 - Bl*s*D) / Ze
  return jnp.stack((I, D, V), axis=-1)


def _sls_tf(f, params):
  """Frequency response with Standard Linear Solid model in Maxwell form."""
  Bl, Re, Rm, K, L, M, K2, Rm2 = params[:8]
  s = 1j*2*jnp.pi*f
  sZm = s**2*M + Rm*s + K + s*Rm2*K2/(K2 + s*Rm2)  # derivative of mech. impedance
  Ze = Re + s*L  # electrical impedance
  D = Bl / (sZm*Ze + Bl**2*s)
  V = s*D
  I = (1 - Bl*s*D) / Ze
  return jnp.stack((I, D, V), axis=-1)


def _l2r2_sls_tf(f, params):
  """Frequency response with SLS and L2R2."""
  Bl, Re, Rm, K, L, M, L2, R2, K2, Rm2 = params[:10]
  s = 1j*2*jnp.pi*f
  sZm = s**2*M + Rm*s + K + s*Rm2*K2/(K2 + s*Rm2)  # derivative of mech. impedance
  Ze = Re + s*L + R2*L2*s/(R2+L2*s)  # electrical impedance
  D = Bl / (sZm*Ze + Bl**2*s)
  V = s*D
  I = (1 - Bl*s*D) / Ze
  return jnp.stack((I, D, V), axis=-1)


class LinearParameter(float): pass


class PolyNonLinSLSL2R2GenCunLiDyn(ControlAffine):
  """The full model."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L20: LinearParameter
  R20: LinearParameter
  K2: LinearParameter
  K2divR2: LinearParameter
  Bln: list
  Kn: list
  Ln: list
  Li: list
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1., 1e3, 1e3]

  def __init__(self, linear_params=None, nBl=4, nK=4, nL=4, nLi=2):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L20 = linear_params[6]
    self.R20 = linear_params[7]
    self.K2 = linear_params[8]
    self.K2divR2 = linear_params[9]
    self.Bln = list(np.zeros(nBl))
    self.Kn = list(np.zeros(nK))
    self.Ln = list(np.zeros(nL))
    self.Li = list(np.zeros(nLi))
    self.n_states = 5
    self.n_params = nBl + nK + nL + nLi + 10

  def tf(self, f, params):
    return _l2r2_sls_tf(f, params)

  def f(self, x, t=None):
    i, d, v, i2, d2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d - K2*d2) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    dd2 = v - K2divR2*d2   # \dot x2 = \dot x - K2/R2 \dot x2
    return jnp.array([di, dd, dv, di2, dd2])

  def g(self, x, t=None):
    i, d, _, _, _ = x
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    return jnp.array([1/(L + i*L_i), 0., 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class PolyNonLinL2R2GenCunLiDyn(ControlAffine):
  """Full model without SLS."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L20: LinearParameter
  R20: LinearParameter
  Bln: list
  Kn: list
  Ln: list
  Li: list
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1.]

  def __init__(self, linear_params=None, nBl=4, nK=4, nL=4, nLi=2):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L20 = linear_params[6]
    self.R20 = linear_params[7]
    self.Bln = list(np.zeros(nBl))
    self.Kn = list(np.zeros(nK))
    self.Ln = list(np.zeros(nL))
    self.Li = list(np.zeros(nLi))
    self.n_states = 4
    self.n_params = nBl + nK + nL + nLi + 8

  def tf(self, f, params):
    return _l2r2_tf(f, params)

  def f(self, x, t=None):
    i, d, v, i2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    return jnp.array([di, dd, dv, di2])

  def g(self, x, t=None):
    i, d, _, _ = x
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    return jnp.array([1/(L + i*L_i), 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class PolyNonLinSLSGenCunLiDyn(ControlAffine):
  """Full without L2R2."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  K2: LinearParameter
  K2divR2: LinearParameter
  Bln: list
  Kn: list
  Ln: list
  Li: list
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e3, 1e3]

  def __init__(self, linear_params=None, nBl=4, nK=4, nL=4, nLi=2):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.K2 = linear_params[6]
    self.K2divR2 = linear_params[7]
    self.Bln = list(np.zeros(nBl))
    self.Kn = list(np.zeros(nK))
    self.Ln = list(np.zeros(nL))
    self.Li = list(np.zeros(nLi))
    self.n_states = 4
    self.n_params = nBl + nK + nL + nLi + 8

  def tf(self, f, params):
    return _sls_tf(f, params)

  def f(self, x, t=None):
    i, d, v, d2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2
    # state evolution
    di = (-Re*i - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = (Bl*i + 0.5*L_d*i**2 - Rm*v - K*d - K2*d2) / M
    dd2 = v - K2divR2*d2   # \dot x2 = \dot x - K2/R2 \dot x2
    return jnp.array([di, dd, dv, dd2])

  def g(self, x, t=None):
    i, d, _, _ = x
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    return jnp.array([1/(L + i*L_i), 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class PolyNonLinSLSL2R2GenCunDyn(ControlAffine):
  """Full model without Li."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L20: LinearParameter
  R20: LinearParameter
  K2: LinearParameter
  K2divR2: LinearParameter
  Bln: list
  Kn: list
  Ln: list
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1., 1e3, 1e3]

  def __init__(self, linear_params=None, nBl=4, nK=4, nL=4):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L20 = linear_params[6]
    self.R20 = linear_params[7]
    self.K2 = linear_params[8]
    self.K2divR2 = linear_params[9]
    self.Bln = list(np.zeros(nBl))
    self.Kn = list(np.zeros(nK))
    self.Ln = list(np.zeros(nL))
    self.n_states = 5
    self.n_params = nBl + nK + nL + 10

  def tf(self, f, params):
    return _l2r2_sls_tf(f, params)

  def f(self, x, t=None):
    i, d, v, i2, d2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Lfun = lambda d: jnp.polyval(Ld_coefs, d)
    L = Lfun(d)
    L_d = jax.grad(Lfun, argnums=0)(d) # dL/dd
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / L
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d - K2*d2) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / L2
    dd2 = v - K2divR2*d2   # \dot x2 = \dot x - K2/R2 \dot x2
    return jnp.array([di, dd, dv, di2, dd2])

  def g(self, x, t=None):
    _, d, _, _, _ = x
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    L = jnp.polyval(Ld_coefs, d)
    return jnp.array([1/L, 0., 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class PolyNonLinSLSL2R2CunLiDyn(PolyNonLinSLSL2R2GenCunLiDyn):
  """Full model with old Cunningham."""

  def f(self, x, t=None):
    i, d, v, i2, d2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = (Bl*i + 0.5*(L_d*i**2 + L2_d*i2**2) - Rm*v - K*d - K2*d2) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    dd2 = v - K2divR2*d2   # \dot x2 = \dot x - K2/R2 \dot x2
    return jnp.array([di, dd, dv, di2, dd2])


class PolyNonLinSLSL2R2LiDyn(PolyNonLinSLSL2R2GenCunLiDyn):
  """The SOTA model w/o Cunningham."""

  def f(self, x, t=None):
    i, d, v, i2, d2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Li_coefs = jnp.append(jnp.atleast_1d(self.Li), 1.) # L(i) = poly + 1
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d) * jnp.polyval(Li_coefs, i) # L = L(d)L(i)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    L_i = jax.grad(Lfun, argnums=1)(d, i) # dL/di
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    L2_i = L20 * L_i/L0
    # standard linear solid
    # NOTE: how to scale these parameters? Like L2R2?
    K2 = self.K2
    K2divR2 = self.K2divR2
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / (L + i*L_i)
    dd = v
    dv = (Bl*i - Rm*v - K*d - K2*d2) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / (L2 + i*L2_i)
    dd2 = v - K2divR2*d2   # \dot x2 = \dot x - K2/R2 \dot x2
    return jnp.array([di, dd, dv, di2, dd2])


class PolyNonLinL2R2GenCunDyn(ControlAffine):
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L20: LinearParameter
  R20: LinearParameter
  Bln: list
  Kn: list
  Ln: list
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1.]

  def __init__(self, linear_params=None, nBl=4, nK=4, nL=4):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L20 = linear_params[6]
    self.R20 = linear_params[7]
    self.Bln = list(np.zeros(nBl))
    self.Kn = list(np.zeros(nK))
    self.Ln = list(np.zeros(nL))
    self.n_states = 4
    self.n_params = nBl + nK + nL + 8

  def tf(self, f, params):
    return _l2r2_tf(f, params)

  def f(self, x, t=None):
    i, d, v, i2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / L
    dd = v
    dv = ((Bl + 0.5*(L_d*i + L2_d*i2))*i - Rm*v - K*d) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / L2
    return jnp.array([di, dd, dv, di2])

  def g(self, x, t=None):
    i, d, _, _ = x
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d)
    L = Lfun(d, i)
    return jnp.array([1/L, 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class PolyNonLinL2R2CunDyn(PolyNonLinL2R2GenCunDyn):
  """Old cunningham."""

  def f(self, x, t=None):
    i, d, v, i2 = jnp.moveaxis(x, -1, 0)
    Bl = jnp.polyval(jnp.append(jnp.asarray(self.Bln), self.Bl), d)
    Re = self.Re
    Rm = self.Rm
    K = jnp.polyval(jnp.append(jnp.asarray(self.Kn), self.K), d)
    M = self.M
    # position and current dependent inductance
    Ld_coefs = jnp.append(jnp.asarray(self.Ln), self.L)
    Lfun = lambda d, i: jnp.polyval(Ld_coefs, d)
    L = Lfun(d, i)
    L_d = jax.grad(Lfun, argnums=0)(d, i) # dL/dd
    # lossy inductance with co-variant elemts
    L20 = self.L20
    R20 = self.R20
    L0 = Ld_coefs[-1] # == L(d=0, i=0)
    L2 = L20 * L/L0
    R2 = R20 * L/L0
    L2_d = L20 * L_d/L0
    # state evolution
    di = (-(Re + R2)*i + R2*i2 - (Bl + L_d*i)*v) / L
    dd = v
    dv = ((Bl*i + 0.5*(L_d*i**2 + L2_d*i2**2)) - Rm*v - K*d) / M
    di2 = (R2 * (i - i2) - L2_d*i2*v) / L2
    return jnp.array([di, dd, dv, di2])


class SimplestDyn(ControlAffine):
  """Simplest linear model."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3]

  def tf(self, f, p):
    return _simple_tf(f, p)

  def __init__(self, linear_params=None):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.n_states = 3
    self.n_params = 6

  def f(self, x, t=None):
    i, d, v = x
    Bl = self.Bl
    Re = self.Re
    Rm = self.Rm
    K = self.K
    M = self.M
    L = self.L
    di = (-Re*i - Bl*v) / L
    dd = v
    dv = (Bl*i - Rm*v - K*d) / M
    return jnp.array([di, dd, dv])

  def g(self, x, t=None):
    return jnp.array([1/self.L, 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class LinearL2R2Dyn(ControlAffine):
  """Linear with L2R2."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L2: LinearParameter
  R2: LinearParameter
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1]

  def tf(self, f, p):
    return _l2r2_tf(f, p)

  def __init__(self, linear_params=None):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L2 = linear_params[6]
    self.R2 = linear_params[7]
    self.n_states = 4
    self.n_params = 8

  def f(self, x, t=None):
    i, d, v, i2 = x
    Bl = self.Bl
    Re = self.Re
    Rm = self.Rm
    K = self.K
    M = self.M
    L = self.L
    L2 = self.L2
    R2 = self.R2
    di = (-(Re + R2)*i + R2*i2 - Bl*v) / L
    dd = v
    dv = (Bl*i - Rm*v - K*d) / M
    di2 = R2 * (i - i2) / L2
    return jnp.array([di, dd, dv, di2])

  def g(self, x, t=None):
    return jnp.array([1/self.L, 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class LinearSLSDyn(ControlAffine):
  """Linear with SLS."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  K2: LinearParameter
  K2divR2: LinearParameter
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e3, 1e3]

  def tf(self, f, p):
    return _sls_tf(f, p)

  def __init__(self, linear_params=None):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.K2 = linear_params[6]
    self.K2divR2 = linear_params[7]
    self.n_states = 4
    self.n_params = 8

  def f(self, x, t=None):
    i, d, v, d2 = x
    Bl = self.Bl
    Re = self.Re
    Rm = self.Rm
    K = self.K
    M = self.M
    L = self.L
    K2 = self.K2
    K2divR2 = self.K2divR2
    di = (-Re*i - Bl*v) / L
    dd = v
    dv = (Bl*i - Rm*v - K*d - K2*d2) / M
    dd2 = v - K2divR2*d2
    return jnp.array([di, dd, dv, dd2])

  def g(self, x, t=None):
    return jnp.array([1/self.L, 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]


class LinearL2R2SLSDyn(ControlAffine):
  """Linear with SLS+L2R2."""
  Bl: LinearParameter
  Re: LinearParameter
  Rm: LinearParameter
  K: LinearParameter
  L: LinearParameter
  M: LinearParameter
  L2: LinearParameter
  R2: LinearParameter
  K2: LinearParameter
  K2divR2: LinearParameter
  init_linear_params = [1., 1., 1., 1e3, 1e-3, 1e-3, 1e-3, 1., 1e3, 1e3]

  def tf(self, f, p):
    return _l2r2_sls_tf(f, p)

  def __init__(self, linear_params=None):
    if linear_params is None:
      linear_params = self.init_linear_params
    linear_params = [LinearParameter(p) for p in linear_params]
    self.Bl = linear_params[0]
    self.Re = linear_params[1]
    self.Rm = linear_params[2]
    self.K = linear_params[3]
    self.L = linear_params[4]
    self.M = linear_params[5]
    self.L2 = linear_params[6]
    self.R2 = linear_params[7]
    self.K2 = linear_params[8]
    self.K2divR2 = linear_params[9]
    self.n_states = 5
    self.n_params = 10

  def f(self, x, t=None):
    i, d, v, i2, d2 = x
    Bl = self.Bl
    Re = self.Re
    Rm = self.Rm
    K = self.K
    M = self.M
    L = self.L
    L2 = self.L2
    R2 = self.R2
    K2 = self.K2
    K2divR2 = self.K2divR2
    di = (-(Re + R2)*i + R2*i2 - Bl*v) / L
    dd = v
    dv = (Bl*i - Rm*v - K*d - K2*d2) / M
    di2 = R2 * (i - i2) / L2
    dd2 = v - K2divR2*d2
    return jnp.array([di, dd, dv, di2, dd2])

  def g(self, x, t=None):
    return jnp.array([1/self.L, 0., 0., 0., 0.])

  def h(self, x, t=None):
    return x[np.array([0, 2])]
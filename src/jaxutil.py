import jax
import jax.numpy as jnp
import functools
import psutil
import diffrax as dfx
import sys, gc

def value_and_jacfwd(f, x):
  """Create a function that evaluates both fun and its foward-mode jacobian.

  Only works on ndarrays, not pytrees.
  Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
  """
  pushfwd = functools.partial(jax.jvp, f, (x,))
  basis = jnp.eye(x.size, dtype=x.dtype)
  y, jac = jax.vmap(pushfwd, out_axes=(None, 1))((basis,))
  return y, jac


def value_and_jacrev(f, x):
  """Create a function that evaluates both fun and its reverse-mode jacobian.

  Only works on ndarrays, not pytrees.
  Source: https://github.com/google/jax/pull/762#issuecomment-1002267121
  """
  y, pullback = jax.vjp(f, x)
  basis = jnp.eye(y.size, dtype=y.dtype)
  jac = jax.vmap(pullback)(basis)
  return y, jac


def clear_caches():
  """Clear all kind of jax and diffrax caches."""
  # see https://github.com/patrick-kidger/diffrax/issues/142
  process = psutil.Process()
  if process.memory_info().rss > 4 * 2**30: # >4GB memory usage
    for module_name, module in sys.modules.items():
      if module_name.startswith("jax"):
        for obj_name in dir(module):
          obj = getattr(module, obj_name)
          if hasattr(obj, "cache_clear"):
            obj.cache_clear()
    dfx.diffeqsolve._cached.clear_cache()
    gc.collect()
    print("Cache cleared")

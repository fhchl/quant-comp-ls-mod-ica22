from .util import find_nearest
import numpy as np
import scipy.signal as sig
from sympy import primerange
import warnings
import nidaqmx as ni
from nidaqmx.constants import WAIT_INFINITELY

def query_devices():
  local = ni.system.System.local()
  driver = local.driver_version

  print(f'DAQmx {0}.{1}.{2}'.format(
    driver.major_version, driver.minor_version, driver.update_version))

  for device in local.devices:
    print('Device Name: {0}, Product Category: {1}, Product Type: {2}'.format(
      device.name, device.product_category, device.product_type))

from warnings import warn

def playrec(
  outdata, sr=96000, input_mapping=['Dev2/ai0', 'Dev2/ai1', 'Dev2/ai2'],
  output_mapping=['Dev2/ao0']
):
  """Simultaneous playback and recording though NI device.

  Parameters:
  -----------
  outdata: ndarray shape(nsamples, noutchan)
    Send to device output

  Returns
  -------
  indata: ndarray shape(nsamples, ninchan)
    Recorded data

  """
  max_out_range = 3.5 # output range of USB-4431
  max_outdata = np.max(np.abs(outdata))
  if max_outdata > max_out_range:
    fac = max_out_range / max_outdata
    outdata *= fac
    raise ValueError(
      f"outdata amplitude ({max_outdata:.2f}) larger than allowed range (+-{max_out_range}). "
      +"Reducing amplitude by factor {fac:.2f} to prevent clipping.")

  outdata = np.asarray(outdata)
  if len(output_mapping) > 1:
    assert outdata.ndim == 2
    outdata = outdata.T
    assert len(output_mapping) == outdata.shape[0]
    nsamples = outdata.shape[1]
  else:
    nsamples = outdata.shape[0]

  with ni.Task() as read_task, ni.Task() as write_task:
    for o in output_mapping:
      aochan = write_task.ao_channels.add_ao_voltage_chan(o)
      aochan.ao_max = max_out_range
      aochan.ao_min = -max_out_range
    for i in input_mapping:
      aichan = read_task.ai_channels.add_ai_voltage_chan(i)
      aichan.ai_min = -10
      aichan.ai_max = 10

    for task in (read_task, write_task):
      task.timing.cfg_samp_clk_timing(rate=sr, source='OnboardClock', samps_per_chan=nsamples)

    # trigger write_task as soon as read_task starts
    write_task.triggers.start_trigger.cfg_dig_edge_start_trig(read_task.triggers.start_trigger.term)
    write_task.write(outdata, auto_start=False)

    from nidaqmx.constants import TaskMode
    # commiting task reduces lag between read and write
    read_task.control(TaskMode.TASK_COMMIT)
    write_task.control(TaskMode.TASK_COMMIT)

    write_task.start()  # write_task doesn't start at read_task's start_trigger without this
    indata = read_task.read(nsamples, timeout=WAIT_INFINITELY)

  return np.asarray(indata).T


def _post_process_signal(x, sr, fade=0, maxamp=None, post_silence=0,
                         bandpass=None, rms=None, clip=None):
  if bandpass is not None:
    bandpass = np.asarray(bandpass)
    sos = sig.butter(8, bandpass / (sr / 2), "bandpass", output='sos')
    x = sig.sosfilt(sos, x)
  if fade > 0:
    n_fade = round(fade * sr)
    fading_window = sig.hann(2 * n_fade)
    x[:n_fade] = x[:n_fade] * fading_window[:n_fade]
    x[-n_fade:] = x[-n_fade:] * fading_window[-n_fade:]
  if maxamp is not None and rms is None:
    x *= maxamp/np.max(np.abs(x))
  elif rms is not None and maxamp is None:
    x *= rms/np.std(x)
  else:
    raise ValueError("specify either rms or maxamp")
  if post_silence > 0:
    silence = np.zeros(int(round(post_silence * sr)))
    x = np.concatenate((x, silence))
  if clip is not None:
    np.clip(x, -clip, clip, out=x)
  return x


def powerlaw_psd_gaussian(exponent, size, fmin=0):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper. It is not actually
        zero, but 1/samples.

    Returns
    -------
    out : array
        The samples.


    Examples:
    ---------

    # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)

    Adapted from `colorednoise` package.
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = np.fft.rfftfreq(samples)

    # Build scaling factors for all frequencies
    s_scale = f
    fmin = max(fmin, 1./samples) # Low frequency cutoff
    ix   = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w      = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2. # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale     = s_scale[(np.newaxis,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    sr = np.random.normal(scale=s_scale, size=size)
    si = np.random.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2): si[...,-1] = 0

    # Regardless of signal length, the DC component must be real
    si[...,0] = 0

    # Combine power + corrected phase to Fourier components
    s  = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = np.fft.irfft(s, n=samples, axis=-1) / sigma

    return y


def pink_noise(T, sr, fknee=0, **kwargs):
  """Generate pink noise."""
  x = powerlaw_psd_gaussian(1, int(sr*T), fmin=fknee/sr)
  kwargs.setdefault("rms", 1)
  return _post_process_signal(x, sr, **kwargs)





def _frac_octave_spread(sr, nperseg, octave_fraction, f_start, f_end):
  prime_periods = np.array(list(primerange(1, nperseg+1)))
  wavlen = nperseg / prime_periods
  freqs = sr/wavlen

  filtered_freqs = set()
  f = f_start
  while True:
    filtered_freqs.add(find_nearest(freqs, f)[0])
    f = f*2**(octave_fraction)
    if f > f_end:
      break

  return sorted(list(filtered_freqs))


def line_tone_generator(T, sr, octave_fraction=1/2, nperseg=2**14, f_start=16, f_end=20e3):
  filtered_freqs = _frac_octave_spread(
    sr, nperseg, octave_fraction, f_start, f_end)

  # build tones
  N = int(np.round(T*sr/nperseg))*nperseg
  t = np.arange(N)/sr
  np.random.seed(1)
  for f in filtered_freqs:
    randphase = np.random.random() * 2*np.pi
    yield f, np.sin(2*np.pi*f*t + randphase)


def multitone(T, sr, octave_fraction=1/2, nperseg=2**14, f_start=16, f_end=20e3, **kwargs):
  """Generate multitone signal."""
  tone_gen = line_tone_generator(T, sr, octave_fraction, nperseg, f_start, f_end)
  x = 0
  fs = []
  for f, tone in tone_gen:
    fs.append(f)
    x += tone

  kwargs.setdefault("rms", 1)
  return fs, _post_process_signal(x, sr, **kwargs)


def analyze_multitone(f, x, sr, nperseg=2**14, dbrange=100):
  fall, s = sig.welch(x, sr, window='boxcar', nperseg=nperseg)
  idx_tones = np.isin(np.round(fall, 3), np.round(f, 3))
  eps = 1e-16
  s_tone = np.where(idx_tones, s, eps)
  s_else = np.where(idx_tones, eps, s)
  plt.plot(fall, 20*np.log10(s_tone))
  plt.plot(fall, 20*np.log10(s_else))
  maxval = np.max(20*np.log10(s_tone))
  plt.ylim((maxval-dbrange, maxval+3))


def _default_sweep_params(T, sr, f_start, f_end):
  nsamples = int(np.round(T * sr))
  if f_start is None:
      f_start = sr / nsamples
  if f_end is None:
      f_end = sr / 2
  assert f_start < f_end
  assert f_end <= sr / 2
  return nsamples, f_start, f_end


def exponential_sweep(T, sr, f_start=None, f_end=None, **kwargs):
  """Generate exponential sweep."""
  n_tap, f_start, f_end = _default_sweep_params(T, sr, f_start, f_end)
  t = np.linspace(0, T, n_tap, endpoint=False)
  sweep = sig.chirp(t, f_start, T, f_end, method='logarithmic', phi=-90)
  return _post_process_signal(sweep, sr, **kwargs)


def synchronized_swept_sine(Tapprox, sr, f_start=None, f_end=None, **kwargs):
    """Generate synchronized swept sine.

    Sweep constructed in time domain as described by `Novak`_.

    Parameters
    ----------
    T : float
        length of sweep
    sr : int
        sampling frequency
    f_start, f_end: int, optional

    Returns
    -------
    ndarray, floats
        The sweep. It will have a length approximately to Tapprox. If end frequency is
        integer multiple of start frequency the sweep will start and end with a 0.

    .. _Novak:
       A. Novak, P. Lotton, and L. Simon, “Transfer-Function Measurement with Sweeps *,”
       Journal of the Audio Engineering Society, vol. 63, no. 10, pp. 786–798, 2015.

    """
    _, f_start, f_end = _default_sweep_params(Tapprox, sr, f_start, f_end)

    print(Tapprox * f_start / np.log(f_end / f_start))
    k = round(Tapprox * f_start / np.log(f_end / f_start))
    if k == 0:
        raise ValueError("Choose arguments s. t. f1 / log(f_2 / f_1) * T >= 0.5")
    L = k / f_start
    T = L * np.log(f_end / f_start)
    n = int(np.ceil(sr * T))
    t = np.linspace(0, T, n, endpoint=False)
    x = np.sin(2 * np.pi * f_start * L * np.exp(t / L))

    return _post_process_signal(x, sr, **kwargs)


def delay_higher_order_ir(T, sr, order, f_start=None, f_end=None, post_silence=0):
    """Delay of harmonic impulse response.

    From `Farina`_.

    Parameters
    ----------
    T : float
        Iength of impulse response.
    order : int > 0
        Order of harmonic. First harmonic is fundamental.
    f_start, f_end : float or None, optional
        Start and stop frequencies of exponential sweep.

    Returns
    -------
    float
        dt

    .. _Farina:
       A. Farina, “Simultaneous
       of impulse response and distortion
       with a swept-sine techniqueMinnaar, Pauli,” in Proc. AES 108th conv,
       Paris, France, 2000, pp. 1–15.

    """
    _, f_start, f_end = _default_sweep_params(T, sr, f_start, f_end)
    return T * np.log(order) / np.log(f_end / f_start) - post_silence


def split_sweep_ir_into_higher_order_irs(h, sr, order=10, f_start=None, f_end=None, post_silence=0):
    """Energy of non-linear components of sweept impulse response.

    Parameters
    ----------
    h : ndarray
        Impulse response from sine sweept measurement.
    sr : int
        sample rate
    order : int, optional
        Number of harmonics.

    Returns
    -------
    list, of length order
        Higher order impulse responses

    """
    T = len(h)/sr
    _, f_start, f_end = _default_sweep_params(T, sr, f_start, f_end)

    # find max and circshift it to tap 0
    h = np.roll(h, -np.argmax(np.abs(h) ** 2))

    # delays of non-linear components
    orders = np.arange(1, order + 2)
    dts = delay_higher_order_ir(T, sr, orders, f_start, f_end)
    dns = np.round((T - dts) * sr).astype(int)

    hoir = []

    # fundamental
    n_start = int(round((dns[1] + dns[0]) / 2))
    n_end = int(round((0 + dns[-1] / 2)))
    hoir.append(np.concatenate((h[n_start:], h[:n_end])))

    # higher orders
    for i in orders[:-2]:
        n_start = int(round((dns[i + 1] + dns[i]) / 2))
        n_end = int(round((dns[i] + dns[i - 1]) / 2))
        hoir.append(h[n_start:n_end])

    return hoir


def transfer_function(
    ref,
    meas,
    ret_time=True,
    axis=-1,
    reg=0,
    reg_lim_dB=None,
):
    """Compute transfer-function between time domain signals.

    Parameters
    ----------
    ref : ndarray, float
        Reference signal.
    meas : ndarray, float
        Measured signal.
    ret_time : bool, optional
        If True, return in time domain. Otherwise return in frequency domain.
    axis : integer, optional
        Time axis
    Ywindow : Tuple or None, optional
        Apply a frequency domain window to `meas`. (fs, startwindow, stopwindow) before
        the FFT to avoid numerical problems close to Nyquist frequency due to division
        by small numbers.
    fftwindow : None, optional
        Apply a Tukey time window to meas before doing the fft removing clicks at the
        end and beginning of the recording.
    reg : float
        Regularization in deconvolution
    reg_lim_dB: float
        Regularize such that reference has at least reg_lim_dB below of maximum energy
        in each bin.

    Returns
    -------
    h : ndarray, float
        Transfer-function between ref and meas.

    """
    R = np.fft.rfft(ref, axis=axis)  # no need for normalization because
    Y = np.fft.rfft(meas, axis=axis)  # of division

    # FIXME: next two paragraphs are not very elegant
    R[R == 0] = np.finfo(complex).eps  # avoid devision by zero

    # Avoid large TF gains that lead to Fourier Transform numerical errors
    TOO_LARGE_GAIN = 1e9
    too_large = np.abs(Y / R) > TOO_LARGE_GAIN
    if np.any(too_large):
        warnings.warn(
            f"TF gains larger than {20*np.log10(TOO_LARGE_GAIN):.0f} dB. Setting to 0"
        )
        Y[too_large] = 0

    if reg_lim_dB is not None:
        # maximum of reference
        maxRdB = np.max(20 * np.log10(np.abs(R)), axis=axis)

        # power in reference should be at least
        minRdB = maxRdB - reg_lim_dB

        # 10 * log10(reg + |R|**2) = minRdB
        reg = 10 ** (minRdB / 10) - np.abs(R) ** 2
        reg[reg < 0] = 0

    H = Y * R.conj() / (np.abs(R) ** 2 + reg)

    if ret_time:
        h = np.fft.irfft(H, axis=axis, n=ref.shape[axis])
        return h
    else:
        return H
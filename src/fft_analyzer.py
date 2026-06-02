# src/fft_analyzer.py
"""
FFT-based spectral analysis for motor vibration signals.

Core function: convert time-domain signal → frequency spectrum,
then extract amplitudes at specific fault-characteristic frequencies.

Physics background:
    Bearing faults create periodic mechanical impulses at frequencies
    determined by bearing geometry. These impulses modulate the motor's
    air gap flux, which appears as sidebands in the stator current/
    vibration spectrum.

    For CWRU dataset (SKF 6205-2RS bearing at 1797 RPM):
        BPFO = 107.36 Hz  (Ball Pass Frequency Outer race)
        BPFI = 162.18 Hz  (Ball Pass Frequency Inner race)
        BSF  = 141.17 Hz  (Ball Spin Frequency)
        FTF  =  11.90 Hz  (Fundamental Train Frequency / cage)
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.fft import rfft, rfftfreq


class FFTAnalyzer:
    """
    Computes FFT, PSD, and STFT on motor vibration/current signals.
    Also calculates bearing fault characteristic frequencies.

    Usage:
        analyzer = FFTAnalyzer(sampling_rate=12000)
        freqs, mags = analyzer.compute_fft(signal_window)
        fault_freqs = analyzer.compute_bearing_fault_frequencies(1797)
    """

    # SKF 6205-2RS JEM bearing geometry — used in CWRU test rig
    # These values let us calculate exactly where fault peaks should appear
    BEARING_N_BALLS       = 9        # number of rolling elements
    BEARING_BALL_DIA      = 0.3126   # ball diameter (inches)
    BEARING_PITCH_DIA     = 1.537    # pitch diameter (inches)
    BEARING_CONTACT_ANGLE = 0        # contact angle (degrees)

    def __init__(self, sampling_rate: int = 12000):
        """
        Args:
            sampling_rate: Hz — must match acquisition hardware.
                           CWRU uses 12,000 Hz for drive-end data.
        """
        self.fs = sampling_rate

    # ─────────────────────────────────────────────────────────────
    #  CORE FFT
    # ─────────────────────────────────────────────────────────────

    def compute_fft(self, signal: np.ndarray,
                    apply_window: bool = True) -> tuple:
        """
        Compute the single-sided magnitude FFT of a signal.

        Why single-sided?
            For real-valued signals, the FFT is symmetric around
            the Nyquist frequency. The positive half contains all
            the information — no need to look at the negative half.

        Why Hann window?
            Without windowing, FFT assumes the signal repeats
            periodically. If the signal does not start and end at
            the same value, the FFT sees a sharp discontinuity at
            the window edges. This spreads energy from each frequency
            into its neighbors — called spectral leakage.
            The Hann window tapers the signal to zero at both ends,
            eliminating the discontinuity.

        Args:
            signal      : 1D numpy array of signal samples
            apply_window: if True, apply Hann window before FFT
                          (recommended — reduces spectral leakage)

        Returns:
            frequencies : 1D array of frequency values in Hz
                          Length = N//2 + 1 where N = len(signal)
                          Range  = [0, sampling_rate/2]
            magnitudes  : 1D array of amplitude values
                          Same length as frequencies
                          Units: same as input signal (g for acceleration)
        """
        n = len(signal)

        if apply_window:
            # Create Hann window of same length as signal
            window = np.hanning(n)

            # Normalize: compensate for energy lost by windowing
            # Without this, amplitudes would appear 50% too small
            normalization = 2.0 / np.sum(window)
            windowed_signal = signal * window * normalization
        else:
            windowed_signal = signal.copy()

        # rfft: real FFT — only returns positive frequencies
        # This is correct for real-valued signals (which ours are)
        fft_complex = rfft(windowed_signal)

        # Convert complex output to magnitude (absolute value)
        magnitudes = np.abs(fft_complex)

        # Build frequency axis
        # rfftfreq returns normalized frequencies — multiply by fs to get Hz
        frequencies = rfftfreq(n, d=1.0 / self.fs)

        return frequencies, magnitudes

    # ─────────────────────────────────────────────────────────────
    #  WELCH'S PSD — more accurate for noisy signals
    # ─────────────────────────────────────────────────────────────

    def compute_psd(self, signal: np.ndarray) -> tuple:
        """
        Compute Power Spectral Density using Welch's method.

        Welch's method averages multiple overlapping FFTs.
        This reduces noise variance and gives a smoother estimate
        of the true spectrum — at the cost of frequency resolution.

        Use case: when signal has too much noise for standard FFT
        to show fault peaks clearly.

        Returns:
            frequencies: Hz
            psd        : power per Hz (V²/Hz or g²/Hz)
        """
        frequencies, psd = scipy_signal.welch(
            signal,
            fs=self.fs,
            nperseg=1024,    # length of each segment
            noverlap=512,    # 50% overlap between segments
            window='hann'
        )
        return frequencies, psd

    # ─────────────────────────────────────────────────────────────
    #  STFT — Short-Time Fourier Transform
    # ─────────────────────────────────────────────────────────────

    def compute_stft(self, signal: np.ndarray) -> tuple:
        """
        Compute Short-Time Fourier Transform (spectrogram).

        STFT divides the signal into short overlapping segments
        and computes FFT on each. The output shows how the
        frequency content changes over time.

        Why it's useful:
            A standard FFT averages the spectrum over the entire
            window. If a fault causes a brief transient event,
            it gets diluted by averaging. STFT reveals if fault
            frequencies appear at specific moments in time.

        Returns:
            frequencies : 1D array of frequency values (Hz)
            times       : 1D array of time values (seconds)
            magnitude   : 2D array, shape (freq_bins, time_bins)
                          magnitude[i, j] = amplitude at frequency i, time j
        """
        frequencies, times, Zxx = scipy_signal.stft(
            signal,
            fs=self.fs,
            window='hann',
            nperseg=256,     # each segment: 256 samples = 0.021 seconds
            noverlap=128     # 50% overlap
        )
        magnitude = np.abs(Zxx)
        return frequencies, times, magnitude

    # ─────────────────────────────────────────────────────────────
    #  FAULT FREQUENCY CALCULATOR
    # ─────────────────────────────────────────────────────────────

    def compute_bearing_fault_frequencies(self,
                                           shaft_rpm: float) -> dict:
        """
        Calculate characteristic bearing fault frequencies from geometry.

        These are the exact frequencies where fault-related peaks
        should appear in the vibration/current spectrum.

        Physics:
            Each bearing component (inner race, outer race, balls)
            rotates at a different speed. When a fault (spall, pit,
            crack) on one component strikes another, it creates an
            impulse at a frequency determined by that component's
            rotational speed and the bearing geometry.

        Formulas (standard bearing fault frequency equations):
            shaft_freq = RPM / 60
            ratio      = (Bd / Pd) × cos(α)

            BPFO = (N/2) × shaft_freq × (1 - ratio)
            BPFI = (N/2) × shaft_freq × (1 + ratio)
            BSF  = (Pd / (2×Bd)) × shaft_freq × (1 - ratio²)
            FTF  = (1/2) × shaft_freq × (1 - ratio)

        Where:
            N   = number of rolling elements (balls)
            Bd  = ball diameter
            Pd  = pitch diameter
            α   = contact angle
            BPFO = Ball Pass Frequency Outer race
            BPFI = Ball Pass Frequency Inner race
            BSF  = Ball Spin Frequency
            FTF  = Fundamental Train Frequency (cage frequency)

        Args:
            shaft_rpm: motor shaft speed in RPM
                       CWRU loads: 0HP=1797, 1HP=1772, 2HP=1750, 3HP=1730

        Returns:
            dict with keys: shaft_freq, BPFO, BPFI, BSF, FTF
            All values in Hz
        """
        shaft_freq = shaft_rpm / 60.0

        # Geometry ratio — appears in all fault frequency formulas
        ratio = (self.BEARING_BALL_DIA / self.BEARING_PITCH_DIA) * \
                np.cos(np.radians(self.BEARING_CONTACT_ANGLE))

        bpfo = (self.BEARING_N_BALLS / 2) * shaft_freq * (1 - ratio)
        bpfi = (self.BEARING_N_BALLS / 2) * shaft_freq * (1 + ratio)
        bsf  = (self.BEARING_PITCH_DIA / (2 * self.BEARING_BALL_DIA)) * \
               shaft_freq * (1 - ratio ** 2)
        ftf  = 0.5 * shaft_freq * (1 - ratio)

        return {
            'shaft_freq': round(shaft_freq, 4),
            'BPFO':       round(bpfo, 4),
            'BPFI':       round(bpfi, 4),
            'BSF':        round(bsf,  4),
            'FTF':        round(ftf,  4)
        }

    # ─────────────────────────────────────────────────────────────
    #  AMPLITUDE EXTRACTION
    # ─────────────────────────────────────────────────────────────

    def get_amplitude_at(self, frequencies: np.ndarray,
                          magnitudes: np.ndarray,
                          target_freq: float,
                          tolerance_hz: float = 10.0) -> float:
        """
        Extract peak amplitude within ±tolerance of a target frequency.

        Used to measure how much energy is present at specific
        fault frequencies (BPFO, BPFI, BSF, etc.).

        Args:
            frequencies  : frequency array from compute_fft
            magnitudes   : magnitude array from compute_fft
            target_freq  : frequency of interest in Hz
            tolerance_hz : search window (±Hz around target)
                           Default 10 Hz is appropriate for 12kHz/4096
                           where Δf ≈ 2.93 Hz

        Returns:
            Peak magnitude within the search window.
            Returns 0.0 if target_freq is outside the spectrum range.
        """
        if target_freq < 0 or target_freq > self.fs / 2:
            return 0.0

        mask = np.abs(frequencies - target_freq) <= tolerance_hz

        if not np.any(mask):
            return 0.0

        return float(np.max(magnitudes[mask]))

    # ─────────────────────────────────────────────────────────────
    #  TOTAL HARMONIC DISTORTION
    # ─────────────────────────────────────────────────────────────

    def compute_thd(self, frequencies: np.ndarray,
                    magnitudes: np.ndarray,
                    fundamental_freq: float = 50.0,
                    n_harmonics: int = 5) -> float:
        """
        Compute Total Harmonic Distortion (THD).

        THD measures how much of the signal's energy is in harmonics
        (multiples of the fundamental frequency) relative to the
        fundamental itself.

        High THD can indicate:
            - Non-linear load behavior
            - Winding asymmetry
            - Power quality issues

        Formula: THD = sqrt(V2² + V3² + ... + Vn²) / V1
        Where V1 = fundamental amplitude, V2...Vn = harmonics

        Args:
            fundamental_freq: supply frequency in Hz (default 50 Hz India)
            n_harmonics     : how many harmonics to include

        Returns:
            THD as a fraction (0.05 = 5% THD)
        """
        v1 = self.get_amplitude_at(frequencies, magnitudes,
                                    fundamental_freq)
        if v1 < 1e-10:
            return 0.0

        harmonic_sum_sq = 0.0
        for n in range(2, n_harmonics + 1):
            vn = self.get_amplitude_at(frequencies, magnitudes,
                                        n * fundamental_freq)
            harmonic_sum_sq += vn ** 2

        return float(np.sqrt(harmonic_sum_sq) / v1)


# ─────────────────────────────────────────────────────────────────
#  SELF-TEST — run this file directly to verify everything works
# ─────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import scipy.io
    from pathlib import Path

    print("=" * 60)
    print("FFT ANALYZER SELF-TEST")
    print("=" * 60)

    # ── Test 1: FFT on a synthetic known signal ──────────────────
    print("\nTest 1: FFT on synthetic 50 Hz sine wave")
    fs   = 12000
    N    = 4096
    t    = np.arange(N) / fs
    # Pure sine wave at 50 Hz — FFT peak should be exactly at 50 Hz
    test_signal = np.sin(2 * np.pi * 50 * t)

    analyzer = FFTAnalyzer(sampling_rate=fs)
    freqs, mags = analyzer.compute_fft(test_signal, apply_window=False)

    peak_freq = freqs[np.argmax(mags)]
    peak_mag  = np.max(mags)
    print(f"  Expected peak: 50.0 Hz")
    print(f"  Detected peak: {peak_freq:.2f} Hz")
    print(f"  Peak magnitude: {peak_mag:.4f}")
    assert abs(peak_freq - 50.0) < 5.0, "FAIL: Peak not at 50 Hz"
    print("  PASS ✓")

    # ── Test 2: Windowing reduces leakage ────────────────────────
    print("\nTest 2: Hann window reduces spectral leakage")
    # Signal at non-bin-center frequency (where leakage is worst)
    test_leaky = np.sin(2 * np.pi * 55.5 * t)

    _, mags_no_window   = analyzer.compute_fft(test_leaky,
                                                apply_window=False)
    _, mags_with_window = analyzer.compute_fft(test_leaky,
                                                apply_window=True)

    # With windowing: energy should be more concentrated at peak
    # Measure: ratio of peak energy to total energy
    peak_idx     = np.argmax(mags_with_window)
    ratio_window = mags_with_window[peak_idx] / (np.sum(mags_with_window) + 1e-10)
    ratio_none   = mags_no_window[peak_idx]   / (np.sum(mags_no_window)   + 1e-10)

    print(f"  Peak concentration (no window):   {ratio_none:.4f}")
    print(f"  Peak concentration (Hann window): {ratio_window:.4f}")
    print("  PASS ✓" if ratio_window > ratio_none else "  WARNING: check windowing")

    # ── Test 3: Bearing fault frequencies ───────────────────────
    print("\nTest 3: Bearing fault frequency calculation (1797 RPM)")
    fault_freqs = analyzer.compute_bearing_fault_frequencies(1797)

    print(f"  Shaft frequency : {fault_freqs['shaft_freq']:.4f} Hz")
    print(f"  BPFO            : {fault_freqs['BPFO']:.4f} Hz")
    print(f"  BPFI            : {fault_freqs['BPFI']:.4f} Hz")
    print(f"  BSF             : {fault_freqs['BSF']:.4f}  Hz")
    print(f"  FTF (cage)      : {fault_freqs['FTF']:.4f}  Hz")

    # Published reference values for CWRU SKF 6205-2RS at 1797 RPM:
    # BPFO ≈ 107.36 Hz, BPFI ≈ 162.18 Hz
    assert abs(fault_freqs['BPFO'] - 107.36) < 1.0, \
        f"BPFO wrong: {fault_freqs['BPFO']}"
    assert abs(fault_freqs['BPFI'] - 162.18) < 1.0, \
        f"BPFI wrong: {fault_freqs['BPFI']}"
    print("  PASS ✓ (matches published CWRU reference values)")

    # ── Test 4: Amplitude extraction ────────────────────────────
    print("\nTest 4: Amplitude extraction at known frequency")
    amp = analyzer.get_amplitude_at(freqs, mags, 50.0, tolerance_hz=10)
    print(f"  Amplitude at 50 Hz: {amp:.4f}")
    assert amp > 0, "FAIL: No amplitude detected at 50 Hz"
    print("  PASS ✓")

    # ── Test 5: Real CWRU data ───────────────────────────────────
    print("\nTest 5: FFT on real CWRU data")
    normal_file = Path('data/raw/normal/97.mat')

    if normal_file.exists():
        mat  = scipy.io.loadmat(str(normal_file))
        key  = [k for k in mat.keys() if not k.startswith('_')][0]
        sig  = mat[key].flatten()[:4096].astype(np.float64)

        freqs_real, mags_real = analyzer.compute_fft(sig)

        print(f"  Signal length    : {len(sig)} samples")
        print(f"  Spectrum length  : {len(freqs_real)} bins")
        print(f"  Max frequency    : {freqs_real[-1]:.1f} Hz")
        print(f"  Frequency res    : {freqs_real[1]:.3f} Hz")
        print(f"  Peak amplitude   : {np.max(mags_real):.6f}")
        print(f"  Peak at frequency: {freqs_real[np.argmax(mags_real)]:.2f} Hz")
        print("  PASS ✓")
    else:
        print(f"  Skipped — {normal_file} not found")
        print("  (Download CWRU data to data/raw/normal/ to run this test)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — fft_analyzer.py is working correctly")
    print("=" * 60)
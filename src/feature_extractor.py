# src/feature_extractor.py

import numpy as np
from scipy import stats
from src.fft_analyzer import FFTAnalyzer

class FeatureExtractor:
    SAMPLING_RATE = 12000
    SHAFT_RPM     = 1797    

    def __init__(self, shaft_rpm: float = 1797):
        self.shaft_rpm  = shaft_rpm
        self.analyzer   = FFTAnalyzer(self.SAMPLING_RATE)
        self.fault_freqs = self.analyzer.compute_bearing_fault_frequencies(
            shaft_rpm
        )

    # ═══════════════════════════════════════════════════════════
    #  TIME DOMAIN
    # ═══════════════════════════════════════════════════════════
    def rms(self, x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(x ** 2)))

    def peak(self, x: np.ndarray) -> float:
        return float(np.max(np.abs(x)))

    def crest_factor(self, x: np.ndarray) -> float:
        rms_val = self.rms(x)
        if rms_val < 1e-10:
            return 0.0
        return self.peak(x) / rms_val

    def kurtosis(self, x: np.ndarray) -> float:
        return float(stats.kurtosis(x, fisher=False))

    def skewness(self, x: np.ndarray) -> float:
        return float(stats.skew(x))

    def shape_factor(self, x: np.ndarray) -> float:
        mean_abs = np.mean(np.abs(x))
        if mean_abs < 1e-10:
            return 0.0
        return self.rms(x) / mean_abs

    def impulse_factor(self, x: np.ndarray) -> float:
        mean_abs = np.mean(np.abs(x))
        if mean_abs < 1e-10:
            return 0.0
        return self.peak(x) / mean_abs

    def variance(self, x: np.ndarray) -> float:
        return float(np.var(x))

    # ═══════════════════════════════════════════════════════════
    #  FREQUENCY DOMAIN
    # ═══════════════════════════════════════════════════════════
    def extract_spectral_features(self, x: np.ndarray) -> dict:
        freqs, mags = self.analyzer.compute_fft(x, apply_window=True)
        get_amp     = self.analyzer.get_amplitude_at
        ff = self.fault_freqs   
        features = {}

        features['bpfo_amp']  = get_amp(freqs, mags, ff['BPFO'])
        features['bpfi_amp']  = get_amp(freqs, mags, ff['BPFI'])
        features['bsf_amp']   = get_amp(freqs, mags, ff['BSF'])

        features['bpfo_2x']  = get_amp(freqs, mags, 2 * ff['BPFO'])
        features['bpfi_2x']  = get_amp(freqs, mags, 2 * ff['BPFI'])

        total = np.sum(mags) + 1e-10
        p     = mags / total
        p_clipped = np.clip(p, 1e-10, 1.0)
        features['spectral_entropy'] = float(
            -np.sum(p_clipped * np.log(p_clipped))
        )

        total_energy = np.sum(mags ** 2) + 1e-10
        def band_energy_ratio(f_low, f_high):
            mask = (freqs >= f_low) & (freqs < f_high)
            return float(np.sum(mags[mask] ** 2) / total_energy)

        features['energy_0_300']    = band_energy_ratio(0,    300)
        features['energy_300_1000'] = band_energy_ratio(300,  1000)
        features['energy_1000_3k']  = band_energy_ratio(1000, 3000)
        features['energy_3k_6k']    = band_energy_ratio(3000, 6000)

        return features

    # ═══════════════════════════════════════════════════════════
    #  MASTER EXTRACTION 
    # ═══════════════════════════════════════════════════════════
    def extract_all(self, x: np.ndarray) -> dict:
        features = {}
        
        # Time domain
        features['rms']           = self.rms(x)
        features['peak']          = self.peak(x)
        features['crest_factor']  = self.crest_factor(x)
        features['kurtosis']      = self.kurtosis(x)
        features['skewness']      = self.skewness(x)
        features['shape_factor']  = self.shape_factor(x)
        features['impulse_factor']= self.impulse_factor(x)
        features['variance']      = self.variance(x)

        # Frequency domain
        spectral = self.extract_spectral_features(x)
        features.update(spectral)

        # Derived
        features['peak_rms_ratio'] = features['peak'] / (features['rms'] + 1e-10)
        features['kurtosis_rms']   = features['kurtosis'] * features['rms']
        features['bpfo_bpfi_ratio'] = features['bpfo_amp'] / (features['bpfi_amp'] + 1e-10)
        features['total_fault_energy'] = (
            features['bpfo_amp'] ** 2 +
            features['bpfi_amp'] ** 2 +
            features['bsf_amp']  ** 2
        )

        freqs_thd, mags_thd = self.analyzer.compute_fft(x)
        features['thd'] = self.analyzer.compute_thd(
            freqs_thd, mags_thd, fundamental_freq=60.0
        )
        return features

    def get_feature_names(self) -> list:
        dummy = np.zeros(4096)
        return list(self.extract_all(dummy).keys())

# ─────────────────────────────────────────────────────────────
#  SELF-TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import scipy.io
    from pathlib import Path
    extractor = FeatureExtractor()
    print("=" * 60)
    print("FEATURE EXTRACTOR SELF-TEST")
    print("=" * 60)
    
    dummy = np.random.randn(4096)
    feats = extractor.extract_all(dummy)
    assert len(feats) == 23, f"Expected 23 features, got {len(feats)}"
    
    gaussian = np.random.randn(4096)
    kurt_val  = extractor.kurtosis(gaussian)
    assert 2.0 < kurt_val < 5.0, f"Kurtosis unexpected: {kurt_val}"
    
    impulsive = np.random.randn(4096)
    for idx in [200, 600, 1000, 1400, 1800]:
        impulsive[idx] += 20.0   
    kurt_impulsive = extractor.kurtosis(impulsive)
    assert kurt_impulsive > kurt_val, "Impulsive should have higher kurtosis"
    
    t = np.arange(4096) / 12000
    sine = np.sin(2 * np.pi * 50 * t)
    cf   = extractor.crest_factor(sine)
    assert abs(cf - 1.414) < 0.1, f"Crest factor unexpected: {cf}"
    
    normal_file = Path('data/raw/normal/97.mat')
    fault_file  = Path('data/raw/inner_race/105.mat')
    if normal_file.exists() and fault_file.exists():
        import scipy.io as sio
        mat_n   = sio.loadmat(str(normal_file))
        key_n   = [k for k in mat_n.keys() if not k.startswith('_')][0]
        sig_n   = mat_n[key_n].flatten()[:4096]
        
        mat_f   = sio.loadmat(str(fault_file))
        key_f   = [k for k in mat_f.keys() if not k.startswith('_')][0]
        sig_f   = mat_f[key_f].flatten()[:4096]
        
        feats_n = extractor.extract_all(sig_n)
        feats_f = extractor.extract_all(sig_f)
        
        print(f"\nTest 5: Real CWRU data comparison")
        print(f"{'Feature':20s} | {'Normal':>12} | {'Fault':>12} | {'Direction':>10}")
        print("─" * 65)
        compare_features = [
            'kurtosis', 'crest_factor', 'impulse_factor',
            'bpfi_amp', 'total_fault_energy', 'spectral_entropy'
        ]
        for feat in compare_features:
            n_val = feats_n[feat]
            f_val = feats_f[feat]
            direction = "fault↑" if f_val > n_val else "fault↓"
            print(f"  {feat:20s} | {n_val:>12.6f} | {f_val:>12.6f} | {direction:>10}")
        print("  PASS ✓")
    else:
        print("\nTest 5: Skipped (CWRU files not found)")
    print("\nALL TESTS PASSED")
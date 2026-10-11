"""Rotor-noise-whitened localizer vs SRP-PHAT on a simulated quadcopter mic array."""
import json, os, sys
import numpy as np
from scipy.signal import stft
from sklearn.model_selection import train_test_split, KFold

FS, N, C = 16000, 8000, 343.0
NMIC, NROT, NCLIP, NCAL = 8, 4, 400, 40
SNRS = np.array([-20, -15, -10, -5, 0])
CAL_SPEEDS = [90, 120, 150, 180]          # calibration rotor speeds (Hz); test speeds are drawn continuously
EPS_GRID = [1e-3, 1e-2, 1e-1, 1.0]        # diagonal loading of noise covariance (tuned by CV on train)
SEEDS = [42, 43, 44]
GRID = np.deg2rad(np.arange(360.0))       # 1-degree azimuth scan

# Drone geometry: 8 mics on a ring (r=0.10 m), 4 rotors at r=0.21 m (near field)
ma = np.deg2rad(22.5 + 45 * np.arange(NMIC))
MICS = np.c_[0.10 * np.cos(ma), 0.10 * np.sin(ma), np.full(NMIC, 0.05)]
ra = np.deg2rad(45 + 90 * np.arange(NROT))
ROTORS = np.c_[0.21 * np.cos(ra), 0.21 * np.sin(ra), np.zeros(NROT)]
DIST = np.linalg.norm(ROTORS[:, None] - MICS[None], axis=-1)       # (R, M)
FREQ_FULL = np.fft.rfftfreq(N, 1 / FS)
STFT_F = np.linspace(0, FS / 2, 257)
BAND = (STFT_F >= 200) & (STFT_F <= 5000)
FB = STFT_F[BAND]
# far-field steering vectors (F, M, A), shared by both methods
TAU = -(MICS[:, :2] @ np.stack([np.cos(GRID), np.sin(GRID)]))[:, :] / C  # (M, A)
A = np.exp(-2j * np.pi * FB[:, None, None] * TAU[None])


def simulate(rng, f0, snr_db=None, az=None):
    """One clip (M, N): near-field rotor noise (+ far-field broadband source if snr_db given)."""
    t = np.arange(N) / FS
    fr = f0 * (1 + 0.02 * rng.standard_normal(NROT))
    k = np.arange(1, 21)[None, :, None]
    ph = rng.uniform(0, 2 * np.pi, (NROT, 20, 1))
    amp = (1 + 0.2 * rng.standard_normal((NROT, 1, 1))) / k
    s = (amp * np.sin(2 * np.pi * k * fr[:, None, None] * t + ph)).sum(1)
    s += 0.7 * rng.standard_normal((NROT, N))                       # broadband rotor noise
    H = np.exp(-2j * np.pi * FREQ_FULL[None, None] * DIST[:, :, None] / C) / DIST[:, :, None]
    noise = np.fft.irfft(np.einsum('rf,rmf->mf', np.fft.rfft(s, axis=-1), H), N, axis=-1)
    noise /= np.sqrt((noise ** 2).mean())                           # unit noise power
    x = noise + 0.03 * rng.standard_normal((NMIC, N))               # sensor noise
    if snr_db is not None:
        u = np.array([np.cos(az), np.sin(az)])
        tau = -(MICS[:, :2] @ u) / C
        src = rng.standard_normal(N)
        src *= 10 ** (snr_db / 20) / src.std()
        x += np.fft.irfft(np.fft.rfft(src)[None] * np.exp(-2j * np.pi * FREQ_FULL[None] * tau[:, None]), N, axis=-1)
    return x


def feats(x):
    """Shared STFT features: sample covariance R and frame-wise PHAT-normalised covariance, per bin."""
    X = stft(x, FS, nperseg=512, axis=-1)[2][:, BAND, :]            # (M, F, T)
    o = np.einsum('mft,nft->ftmn', X, X.conj())
    return o.mean(1), (o / (np.abs(o) + 1e-12)).mean(1)


def srp_phat(R, Rp, _):
    """Baseline: steered response power on PHAT-normalised cross-spectra, summed over bins."""
    return np.einsum('fma,fmn,fna->a', A.conj(), Rp, A).real


def make_whitened(Sigma, eps):
    """Improved: scan after whitening each bin by the calibrated rotor-noise covariance."""
    Sl = Sigma + eps * np.trace(Sigma, axis1=1, axis2=2).real[:, None, None] / NMIC * np.eye(NMIC)
    Si = np.linalg.inv(Sl)
    SA = Si @ A                                                      # Sigma^-1 a
    den = (A.conj() * SA).sum(1).real                                # ||W a||^2
    def scan(R, Rp, _):
        num = (SA.conj() * (R @ SA)).sum(1).real                     # a^H S^-1 R S^-1 a
        tr = np.einsum('fmn,fnm->f', Si, R).real                     # tr(whitened R)
        return (num / (den * tr[:, None])).sum(0)                    # per-bin normalised power
    return scan


def predict(scan, R, Rp):
    return np.rad2deg(GRID[[np.argmax(scan(R[i], Rp[i], None)) for i in range(len(R))]])


def abs_err(pred, true_deg):
    d = np.abs(pred - true_deg) % 360
    return np.minimum(d, 360 - d)


def run_seed(seed):
    rng = np.random.default_rng(seed)
    az = rng.uniform(0, 360, NCLIP)
    snr = SNRS[np.arange(NCLIP) % len(SNRS)]
    f0 = rng.uniform(80, 200, NCLIP)
    F = [feats(simulate(rng, f0[i], snr[i], np.deg2rad(az[i]))) for i in range(NCLIP)]
    R, Rp = np.stack([f[0] for f in F]), np.stack([f[1] for f in F])
    cal = [feats(simulate(rng, CAL_SPEEDS[i % 4]))[0] for i in range(NCAL)]
    Sigma = np.mean(cal, axis=0)                                     # noise-only covariance
    tr, te = train_test_split(np.arange(NCLIP), test_size=0.25, random_state=seed, stratify=snr)
    # select diagonal loading with 3-fold CV on the training clips only
    cv = {}
    for eps in EPS_GRID:
        sc = make_whitened(Sigma, eps)
        errs = abs_err(predict(sc, R[tr], Rp[tr]), az[tr])
        cv[eps] = np.mean([errs[v].mean() for _, v in KFold(3, shuffle=True, random_state=seed).split(tr)])
    best = min(cv, key=cv.get)
    print(f"seed {seed}: CV MAE by eps {cv} -> eps={best}", flush=True)
    out = {"best_eps": best}
    for name, sc in [("baseline", srp_phat), ("improved", make_whitened(Sigma, best))]:
        e = abs_err(predict(sc, R[te], Rp[te]), az[te])              # test set used once
        out[name] = {"mae": e.mean(), "acc10": (e <= 10).mean(),
                     **{f"mae_snr{s}": e[snr[te] == s].mean() for s in SNRS}}
        print(f"seed {seed} {name}: MAE={e.mean():.2f} acc10={out[name]['acc10']:.3f}", flush=True)
    return out, len(tr), len(te)


res = []
for s in SEEDS:
    r, ntr, nte = run_seed(s)
    res.append(r)
mean = lambda m, k: float(np.mean([r[m][k] for r in res]))
std = lambda m, k: float(np.std([r[m][k] for r in res]))
metrics = {}
for m in ["baseline", "improved"]:
    metrics[f"{m}_score"], metrics[f"{m}_score_std"] = mean(m, "mae"), std(m, "mae")
    metrics[f"{m}_acc10"], metrics[f"{m}_acc10_std"] = mean(m, "acc10"), std(m, "acc10")
    for s in SNRS:
        metrics[f"{m}_mae_snr{s}"] = mean(m, f"mae_snr{s}")
metrics["improvement"] = metrics["baseline_score"] - metrics["improved_score"]
metrics["improvement_acc10"] = metrics["improved_acc10"] - metrics["baseline_acc10"]
if not all(np.isfinite(v) for v in metrics.values()):
    sys.exit("non-finite metrics")

os.makedirs("figures", exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
w = 0.38
plt.figure(figsize=(6, 4))
plt.bar(np.arange(5) - w / 2, [metrics[f"baseline_mae_snr{s}"] for s in SNRS], w, label="SRP-PHAT")
plt.bar(np.arange(5) + w / 2, [metrics[f"improved_mae_snr{s}"] for s in SNRS], w, label="Noise-whitened")
plt.xticks(range(5), [f"{s} dB" for s in SNRS]); plt.ylabel("Mean abs azimuth error (deg)")
plt.legend(); plt.tight_layout(); plt.savefig("figures/mae_by_snr.png", dpi=120)

json.dump({
    "question": "Can a rotor-noise-whitened frequency-domain localizer beat a standard SRP-PHAT localizer at finding the direction of a sound source from a simulated quadcopter microphone array across low-SNR and varying rotor-speed conditions?",
    "method": "Numpy simulation of an 8-mic drone array, far-field broadband source and near-field rotor noise (4 rotors, harmonics + broadband), SNR -20..0 dB, rotor speeds 80-200 Hz. Same STFT features and 1-degree azimuth grid for both methods; noise covariance from separate noise-only calibration clips. Mean over 3 seeds (each regenerates the data).",
    "dataset": {"name": "Simulated quadcopter microphone-array clips (stand-in)", "source_url": "simulated in numpy/scipy (no download)",
                "n_rows": NCLIP, "n_train": ntr, "n_test": nte, "same_as_paper": False, "baseline_is_papers_method": False},
    "metric_name": "Mean absolute azimuth error (degrees)", "higher_is_better": False,
    "baseline": "Standard SRP-PHAT azimuth scan (frame-wise PHAT-normalised cross-spectra, summed over 200-5000 Hz bins).",
    "improved": "Per-bin whitening with a rotor-noise spatial covariance from noise-only calibration clips (diagonal loading chosen by 3-fold CV on train), then a whitened, per-bin-normalised beamformer-power (MVDR/MUSIC-style) scan.",
    "metrics": metrics,
    "notes": f"Simulated stand-in, not the paper's dataset; results depend on the simulator assumptions (free-field propagation, no reverberation, fixed rotor geometry shared by calibration and test). Calibration uses 4 discrete rotor speeds; test speeds are continuous. Selected loading per seed: {[r['best_eps'] for r in res]}. Models are deterministic; std is across 3 data seeds. SNR is source power vs total rotor-noise power per microphone. The loading CV is on labelled training clips; the noise covariance is not fit on them.",
}, open("results.json", "w", encoding="utf-8"), indent=2)
print("done", metrics["baseline_score"], metrics["improved_score"])

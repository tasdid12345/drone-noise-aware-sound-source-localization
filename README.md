# Rotor-noise-whitened source localization on a simulated quadcopter array

## Motivation
The paper "Audio-visual sensing from a quadcopter: dataset and baselines for source localization and sound enhancement" (2019) provides baselines for localizing sound sources from a drone, where strong rotor noise masks the target. This experiment asks whether exploiting the known spatial structure of rotor noise improves a standard localizer.

## Question
Can a rotor-noise-whitened frequency-domain localizer beat a standard SRP-PHAT localizer at finding the direction of a sound source from a simulated quadcopter microphone array across low-SNR and varying rotor-speed conditions?

## Dataset
The paper's dataset has no public download link, so this is **not** the paper's data. A numpy simulation generates an 8-microphone drone array, a far-field broadband source at a random azimuth, and near-field rotor noise (4 rotors, harmonic tones plus broadband) at SNRs from -20 to 0 dB and varying rotor speeds. Noise-only calibration clips are kept separate from the labelled clips, which are split once into train and held-out test sets (random angles, SNRs, rotor speeds). The whole pipeline is repeated for seeds 42, 43, 44.

## Models
- **Baseline:** standard SRP-PHAT azimuth scan.
- **Improved:** each frequency bin is whitened with a rotor-noise spatial covariance estimated from the calibration clips (diagonal loading chosen by cross-validation on the training clips), followed by a whitened MVDR/MUSIC-style scan.

Both use the same STFT, steering grid and test clips. Main metric: mean absolute azimuth error in degrees (lower is better); secondary: accuracy within 10 degrees.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`).

## Reference

This project is motivated by the following paper:

> L Wang, R Sanchez-Matilla (2019). *Audio-visual sensing from a quadcopter: dataset and baselines for source localization and sound enhancement*. 2019 IEEE/RSJ. https://ieeexplore.ieee.org/abstract/document/8968183/

Paper author: Prof. Andrea Cavallaro, Queen Mary University of London (a.cavallaro@qmul.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._

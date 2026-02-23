# CTC & Seq2Seq for Polyphonic Automatic Music Transcription



**Phillip Olshausen, Livia Kastrati**

------------------------------------------------------------------------

## Overview

This repository presents a rigorous comparative study of two
fundamentally different paradigms for **polyphonic Automatic Music
Transcription (AMT)** on a constrained 50-track MusicNet subset:

-   **Connectionist Temporal Classification (CTC)** — frame-synchronous
    pitch prediction  
-   **Seq2Seq with Additive Attention** — autoregressive event-token
    generation

Rather than a leaderboard-style benchmark, this project provides a
**mechanistic, statistically grounded, and diagnostically rich
comparison** under unified preprocessing and encoder architecture.

The full seminar report is available in the `Report/` folder. Both approaches take raw audio as input and aim to recover symbolic musical structure (pitch, onset, offset, duration).  
All experiments, models, preprocessing, decoding, and visual analysis are implemented in a **two respective Jupyter notebooks**, of which the Seq2Seq notebook contains a **fully documented, block-by-block explanation of the entire codebase** . You can also find a PPT folder that presents the full project and results.


------------------------------------------------------------------------


## Project overview

**Input**
- Audio waveform (`.wav`, MusicNet)

**Ground truth**
- Note-level annotations from MusicNet (`.csv`)
- Sample-accurate `start_time` / `end_time` and MIDI pitch

**Features**
- Log-mel spectrogram segments with fixed hop length

**Outputs**
- **CTC:** frame-wise pitch probabilities + blank, collapsed via CTC decoding
- **Seq2Seq:** autoregressive **event-token sequences** (time-shift, note-on, note-off)

**Evaluation**
- Token-level metrics (TER, token F1)
- Musical event metrics (onset / offset F1 with tolerance)

---

## 🔎 Research Question

> How do frame-synchronous CTC decoding and attention-based Seq2Seq
> event generation behave when transcribing polyphonic classical
> excerpts under constrained data and compute?

We investigate stability, data efficiency, alignment behavior, blank
dynamics, pitch confusion, entropy, and decoding robustness.

------------------------------------------------------------------------

## 📁 Repository Structure

    CTC/           → Frame-synchronous CTC implementation (CQT / HCQT)
    Seq2Seq/       → Event-based Seq2Seq model with additive attention
    Subset/        → MusicNet 50-track subset construction pipeline
    Report/        → Final seminar report (LaTeX PDF)
    Presentation/  → Seminar presentation slides

------------------------------------------------------------------------

## 🎵 Dataset

**MusicNet (Thickstun et al., 2017)**

-   50-track subset
-   32 train / 8 validation / 10 test (work-disjoint split)
-   76% solo piano dominance
-   Median ≈ 886 note events per track
-   Dense polyphony (p95 ≈ 4 simultaneous notes)

Subset creation is fully reproducible via the `Subset/` folder.

------------------------------------------------------------------------

## 🧠 Encoder Architecture

Both models use a:

**CNN → BiLSTM Encoder**

-   3 Conv-BN-ReLU blocks
-   Asymmetric pooling (1×2 frequency-only pooling)
-   No temporal compression
-   BiLSTM (hidden size = 256 per direction)
-   Output dimension: 512 per frame

This preserves time resolution for: - CTC alignment feasibility -
Precise attention localization

------------------------------------------------------------------------

# 🔷 CTC Model

## Formulation

-   Vocabulary: 88 MIDI pitches + BLANK
-   Frame-wise posterior distributions
-   Greedy decoding with blank-penalty calibration
-   Forward–backward marginalization

## Acoustic Features

-   Constant-Q Transform (CQT)
-   Harmonic CQT (HCQT, 5 harmonic channels)

HCQT exposes harmonic structure explicitly and improves robustness in
dense polyphony.

## Evaluation

-   Token F1 (micro-averaged)
-   Token Error Rate (Levenshtein-based)
-   Length ratio
-   Per-pitch F1
-   Blank dominance diagnostics
-   Entropy analysis
-   Cross-seed robustness

## Best Configuration (HCQT, no augmentation)

Mean across 5 seeds:

Test Token F1 = 0.7145 ± 0.0176  
Test TER = 0.6858 ± 0.0688  
95% CI F1 = \[0.6927, 0.7363\]

### Observations

-   Stable convergence
-   Controlled blank dynamics
-   Strong CNN gradient contribution
-   Harmonic preprocessing beneficial
-   Data augmentation harmful without label consistency

Under constrained data, CTC proved robust and data-efficient.

------------------------------------------------------------------------

# 🔶 Seq2Seq Model (Event-Based Transcription)

## Event Vocabulary

-   TIME_SHIFT
-   NOTE_ON(p)
-   NOTE_OFF(p)
-   EOS

Polyphony is encoded explicitly via consecutive NOTE_ON tokens and
duration modeling through NOTE_OFF.

## Architecture

Encoder → Additive Attention → LSTM Decoder

Additive attention:

e\_{t,i} = vᵀ tanh(W_h s_t + W_s h_i)

Provides dynamic alignment between acoustic frames and symbolic events.

## Training

-   Weighted cross-entropy
-   Scheduled sampling (linear increase)
-   AdamW optimizer
-   Gradient clipping
-   Constrained greedy decoding

### Decoding Constraints

-   Valid NOTE_ON / NOTE_OFF transitions enforced
-   Polyphony caps
-   Max events per frame
-   EOS scheduling
-   Termination control

## Diagnostics

-   Attention heatmaps
-   Attention peak trajectories
-   Attention entropy evolution
-   Pitch confusion matrices
-   Onset precision / recall

## Behavioral Findings

-   Learned symbolic event grammar
-   Structurally valid sequences
-   Weaker onset localization under small data
-   More data-intensive than CTC

Despite only 32 training tracks, Seq2Seq achieved meaningful structure
modeling — suggesting strong scaling potential under pretraining (e.g.,
MAESTRO-scale corpora).

------------------------------------------------------------------------

# ⚖️ Comparative Summary

| Aspect               | CTC                         | Seq2Seq                 |
|----------------------|-----------------------------|-------------------------|
| Alignment            | Monotonic                   | Learned (attention)     |
| Duration modeling    | No                          | Yes                     |
| Polyphony modeling   | Linearized                  | Explicit grammar        |
| Stability (low data) | High                        | Moderate                |
| Data efficiency      | Strong                      | More data-hungry        |
| Interpretability     | Blank & entropy diagnostics | Attention visualization |

Under constrained data, **CTC outperformed in robustness and
stability**.

Seq2Seq, however, demonstrated strong symbolic modeling potential and
may surpass CTC at scale.

------------------------------------------------------------------------

# 🚧 Limitations

-   Small 50-track subset
-   Piano-heavy bias
-   CTC Token F1 is order-invariant
-   Greedy decoding only
-   No language-model prior
-   CTC lacks duration modeling

------------------------------------------------------------------------

# 🔮 Future Work

-   Label-consistent pitch-shift augmentation
-   Multi-label CTC (chord tokens)
-   Hybrid CTC + attention decoding
-   Transformer decoder replacement
-   Beam search with pitch-language prior
-   Large-scale Seq2Seq pretraining

------------------------------------------------------------------------


## 📄 Full Report

See `Report/CTC_Seq2Seq_Report.pdf` for:

-   Mathematical derivations
-   Hyperparameter tables
-   Full experimental results
-   Statistical analysis
-   Extended diagnostics
-   Reproducibility appendix

------------------------------------------------------------------------

© 2026 – HTW Berlin – Deep Learning Seminar

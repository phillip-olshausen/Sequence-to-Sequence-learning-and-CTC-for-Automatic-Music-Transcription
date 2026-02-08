# Polyphonic Music Transcription on MusicNet — CTC vs Seq2Seq (Attention)

**Authors:** Phillip Olshausen, Livia Kastrati

This repository studies **polyphonic music transcription** (multiple simultaneous notes) on the **MusicNet** dataset by comparing two fundamentally different modeling paradigms:

1. **CTC-based transcription** — frame-wise pitch prediction with blanks and monotonic alignment  
2. **Seq2Seq transcription with additive attention** — event-token sequence modeling

Both approaches take raw audio as input and aim to recover symbolic musical structure (pitch, onset, offset, duration).  
All experiments, models, preprocessing, decoding, and visual analysis are implemented in a **single Jupyter notebook**, which contains a **fully documented, block-by-block explanation of the entire codebase**.

---

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

## Repository structure

This repository is intentionally notebook-centric.  
All logic lives in one reproducible notebook, supported by cached artifacts:

```
.
├─ notebook.ipynb
├─ mel_event_npz/
├─ checkpoints/
├─ figures/
├─ manifest.csv
└─ vocab_info.json
```

---

## Dataset setup (MusicNet)

### Expected layout
```
musicnet_small/
├─ train_data/
├─ train_labels/
├─ test_data/
├─ test_labels/
```

### Label semantics
- `start_time`, `end_time`: sample indices  
- `note`: MIDI pitch  
- `instrument`: instrument ID  

Beat-based timing is ignored; all alignment uses sample-accurate timing.

---

## Shared acoustic encoder

Both models share a **CNN–BiLSTM encoder**:
- CNN front-end for time–frequency feature extraction  
- Bidirectional LSTM for temporal context  
- Output: sequence of encoder states

---

# Part A — CTC-based Transcription (Baseline)

### What CTC predicts
CTC predicts **frame-wise pitch classes plus a blank symbol**.  
There are no explicit onset, offset, or duration tokens; note structure is inferred implicitly through alignment.

### Strengths / limitations

**Pros**
- stable training  
- strong monotonic alignment bias  
- fast decoding  

**Cons**
- limited sequence expressiveness  
- weak global musical structure modeling  

---

# Part B — Seq2Seq with Additive Attention

### Core idea
Seq2Seq models transcription as **event-level sequence generation**:

p(y | X) = ∏ p(y_t | y_<t, X)

The model emits symbolic **music events** rather than frame-wise pitches.

---

## Event-token vocabulary

- `TIME_SHIFT(k)` — advance time by k frames  
- `NOTE_ON(p)` — start pitch p  
- `NOTE_OFF(p)` — end pitch p  

Special tokens: `SOS`, `EOS`, `PAD`

This representation explicitly encodes timing, duration, and polyphony.

---

## Architecture

- Shared CNN–BiLSTM encoder  
- LSTM decoder with **additive (Bahdanau) attention**  
- Autoregressive token prediction  

---

## Training

- Teacher forcing with cross-entropy loss  
- Padding masked  
- Class-weighted loss to address token imbalance  
- Scheduled sampling to reduce exposure bias  

---

## Constrained decoding

Decoding includes musical constraints:
- no invalid note-on/off events  
- polyphony limits  
- EOS biasing and minimum-length enforcement  

A validation decode sweep selects optimal constraints without retraining.

---

## Evaluation metrics

- **TER** (token error rate)  
- **Token F1**  
- **Onset / offset F1** with tolerance  

---

## Visual diagnostics

The notebook generates:
- attention heatmaps  
- alignment overlays  
- token-type confusion matrices  
- pitch-level NOTE_ON confusion  

---

## Reproducible execution order

1. Configuration & vocabulary  
2. Dataset indexing & manifest  
3. Feature preprocessing  
4. Data loaders  
5. Encoder definition  
6. CTC training & evaluation  
7. Seq2Seq training  
8. Full decoding & metrics  
9. Attention & confusion visualizations  

---



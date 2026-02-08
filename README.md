[README (5).md](https://github.com/user-attachments/files/25169582/README.5.md)
# Polyphonic Music Transcription on MusicNet — CTC vs Seq2Seq (Attention)
Authors: Phillip olshausen, Livia kastrati

This repository compares two complementary approaches to **polyphonic music transcription** (multiple notes at once) using the **MusicNet** dataset:

1. **CTC-based transcription** (frame-wise / monotonic alignment)
2. **Seq2Seq transcription with additive attention** (event-token language modeling)

Both models share the same overall goal: given an audio waveform, predict the underlying musical note events (pitch + time), and evaluate the output with token-level and onset/offset-based metrics.

---

## What this project does (high level)

**Input:** audio waveform (MusicNet `.wav`)  
**Labels:** per-track note events from MusicNet `.csv` (sample indices for start/end + pitch + instrument metadata)  
**Features:** log-mel spectrogram segments extracted with a fixed hop length  
**Outputs:**
- **CTC:** per-frame token predictions collapsed by CTC rules
- **Seq2Seq:** autoregressive **event token** sequence: time-shifts + note-on + note-off

---

## Repository structure 



---

## Dataset setup (MusicNet)

### Expected dataset layout (example)
```
musicnet_small/
├─ train_data/
│  ├─ 2156.wav
│  └─ ...
├─ train_labels/
│  ├─ 2156.csv
│  └─ ...
├─ test_data/
├─ test_labels/
└─ ...
```

### What the label CSV contains
The CSV typically includes:
- `start_time`, `end_time`: **sample indices** in the waveform (not milliseconds)
- `note`: MIDI pitch number
- `instrument`: instrument ID (optional for this project)
- `start_beat`, `end_beat`, `note_value`: beat-based metadata (not required for timing)

**Timing detail (important):**  
Even if `start_beat` / `end_beat` exists, the pipeline uses the **sample-accurate** `start_time` / `end_time` for alignment. You convert samples → frames using:

- `frame_index = floor(sample_index / hop_length)`

So note length is learned because the event encoding contains **time shifts** between note-on and note-off.

---

## Preprocessing pipeline

### 1) Segmenting audio
Audio is split into fixed-length segments (e.g., a few seconds). For each segment you store:

- `X`: log-mel spectrogram, shape `(T, M)`  
- `y`: tokenized event sequence, shape `(L,)`  
- `meta`: segment timing metadata (track id, segment start sample/frame, etc.)

These are cached as `.npz` files for fast training:
```
mel_event_npz/
├─ train_id2156_seg00001.npz
├─ val_id2620_seg00012.npz
└─ ...
```

### 2) Manifest
A `manifest.csv` lists `split`, `npz_path`, and metadata. This becomes the single source of truth for train/val/test loaders.

---

## Shared audio encoder (used by both models)

Both CTC and Seq2Seq use the same **CNN–BiLSTM encoder**:

1. **Log-mel spectrogram input**: `X ∈ R^{T×M}`
2. **CNN blocks**: local time–frequency feature extraction, optional time downsampling
3. **BiLSTM**: contextualize features in both directions over time  
4. **Encoder outputs**: `H = (h_1, …, h_{T'})`, `h_t ∈ R^{d_enc}`

The encoder yields a time-indexed representation; downstream heads interpret it differently.

---

# Part A — CTC Model (currently being updated)

## Why CTC?
CTC (Connectionist Temporal Classification) is useful when:
- alignment between audio frames and labels is **monotonic** but unknown
- you want frame-wise emissions and alignment marginalization

## Model
- Encoder outputs `H`
- Linear projection + log-softmax to get per-frame token probabilities
- **CTC loss** sums over all valid alignments between frames and the target sequence

## Decoding
- Greedy (collapse repeats + remove blank)
- Optional beam search if implemented

## Strengths / Weaknesses
**Pros**
- strong monotonic alignment bias
- stable training even with limited data
- relatively fast decoding

**Cons**
- less expressive sequence modeling than autoregressive event generation
- global musical constraints require post-processing or constrained decoding

---

# Part B — Seq2Seq with Additive Attention (Detailed)

## What Seq2Seq is doing here
Seq2Seq treats transcription as **event language modeling** conditioned on audio:

\[
p(y \mid X) = \prod_{t=1}^{L} p(y_t \mid y_{<t}, X)
\]

Yes—the model **predicts tokens** one by one.

---

## Event-token vocabulary (what each token means)

Your token stream represents the performance as discrete events:

### 1) `TIME_SHIFT(k)`
Advances time forward by `k` frames (each frame = `hop_length / sr` seconds).  
Used to express gaps and note durations.

### 2) `NOTE_ON(p)`
Starts a note at pitch `p`.

### 3) `NOTE_OFF(p)`
Ends a note at pitch `p`.

### Special tokens
- `SOS`: start of sequence
- `EOS`: end of sequence
- `PAD`: padding for batching

---

## How timing works (beats vs milliseconds)

Even if the label CSV includes beat information, the effective timing for learning/decoding comes from **sample indices**:

- `start_time`, `end_time` are in samples
- features have a hop of `hop_length` samples

So:
- `start_frame = floor(start_time / hop_length)`
- `end_frame   = floor(end_time / hop_length)`
- note duration in frames is `end_frame - start_frame`

This becomes time-shifts in the token encoding.

**Concept example**
- `NOTE_ON(60)`
- `TIME_SHIFT(12)`
- `NOTE_OFF(60)`

This means the note lasts `12` frames, i.e. `12 * hop_length / sr` seconds.

---

## Seq2Seq architecture

### Encoder
- CNN → BiLSTM → encoder states `H`

### Decoder (Attention-RNN)
At each step `t`:
1. Embed previous token: `e_t = Emb(y_{t-1})`
2. Compute additive attention weights:
\[
lpha_{t,i} = \mathrm{softmax}\left(v^	op 	anh(W_h s_{t-1} + W_s h_i)
ight)
\]
3. Context vector:
\[
c_t = \sum_i lpha_{t,i} h_i
\]
4. LSTM update on `[e_t ; c_t]`
5. Output logits over vocabulary and pick next token

This matches your implementation: linear projections, tanh, vector `v`, masking, softmax, weighted sum.

---

## Training objective (teacher forcing)

During training you use **teacher forcing** (feed ground-truth previous token).  
Loss:
\[
\mathcal{L}(	heta)=\sum_{t=1}^{L} -\log p_	heta(y_t \mid y_{<t}, X)
\]

Implementation details:
- CrossEntropyLoss with `ignore_index=PAD`
- **class weights** to reduce imbalance (time shifts, EOS)

---

## Scheduled sampling

Teacher forcing creates exposure bias (test-time uses its own predictions).  
Scheduled sampling gradually replaces ground-truth inputs with model outputs.

Example schedule:
- `ss_prob = min(0.2, 0.02*(epoch-1))`

---

## Constrained decoding (why it matters)

On small datasets, naive decoding can:
- loop on time-shifts
- spam note-ons
- stop too early or too late

Your constrained greedy decoder adds:
- penalties for long time-shift runs
- polyphony limits (`max_notes_per_frame`, `max_active_notes`)
- validity rules (no NOTE_OFF for inactive notes, no NOTE_ON for active notes)
- EOS ramp after `min_len`

This is why decode “retuning” can change metrics substantially without retraining.

---

## Metrics

### Token-level
- **TER**: edit distance / target length
- **Token-F1**: micro precision/recall/F1 from token counts

### Musical onset/offset metrics
- Convert tokens → onset/offset times
- F1 with tolerance window (e.g., ±2 or ±3 frames)

---

## Reproducible run order (single notebook)

1. Config + vocab + manifest
2. Precompute segments → NPZ
3. Dataloaders
4. Encoder definition
5. **CTC** training/eval
6. **Seq2Seq** training (fast epoch prints)
7. Full decode sweep + final eval + visuals
8. Inference: audio → tokens → MIDI + piano-roll

---

## References (short)
- MusicNet dataset (Thickstun et al.)
- CTC: Graves et al.
- Additive attention: Bahdanau et al.
- Scheduled sampling: Bengio et al.

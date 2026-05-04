# Music Similarity Project

Project question:

Do audio embeddings capture genre, rhythm, and mood-like acoustic similarity equally well, or are they biased toward genre?

## Data Layout

Place the FMA files here:

```text
data/raw/
  fma_metadata/
    tracks.csv
    genres.csv
    ...
  fma_small/
    000/
    001/
    ...
```

## Code Layout

```text
src/shared/
  Dataset manifests, generic embedding evaluation, plotting

src/handcrafted/
  Handcrafted feature extraction and handcrafted-only retrieval baselines

src/pretrained/
  VGGish pretrained embedding extraction

src/trained/
  Mel-spectrogram caching and CNN/ResNet training from scratch

src/trained/legacy/
  Earlier CNN scripts kept for reference; not the final pipeline
```

If the downloaded files are still zipped, move them into `data/raw/` and unzip them there:

```bash
mv fma_metadata.zip fma_small.zip data/raw/
cd data/raw
unzip fma_metadata.zip
unzip fma_small.zip
```

## First Pipeline Step

After `fma_metadata` is unzipped, run:

```bash
python3 src/shared/build_manifest.py \
  --tracks-csv data/raw/fma_metadata/tracks.csv \
  --genres-csv data/raw/fma_metadata/genres.csv \
  --audio-root data/raw/fma_small \
  --output data/processed/fma_small_manifest.csv
```

This creates one clean CSV with:

- `track_id`
- `audio_path`
- `genre_top`
- `genre_id`
- `genre_title`
- `exists`

If `fma_small` has not finished downloading yet, the script can still run, but `exists` will be `false`.

For FMA medium, download and unzip `fma_medium.zip`, then build a manifest using both small and medium audio roots under one folder. One layout is:

```text
data/raw/fma_medium_audio/
  000/
  001/
  ...
```

Build the medium manifest:

```bash
python3 src/shared/build_manifest.py \
  --tracks-csv data/raw/fma_metadata/tracks.csv \
  --genres-csv data/raw/fma_metadata/genres.csv \
  --audio-root data/raw/fma_medium_audio \
  --subset medium \
  --output data/processed/fma_medium_manifest.csv
```

## Handcrafted Features

After the manifest shows that audio files exist, run a small test first:

```bash
python3 src/handcrafted/extract_features.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/handcrafted_features_debug.csv \
  --limit 25
```

Then run the full feature extraction:

```bash
python3 src/handcrafted/extract_features.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/handcrafted_features.csv
```

This produces tempo, energy, spectral features, and MFCC summary statistics.

## Retrieval Evaluation

Evaluate handcrafted feature groups with nearest-neighbor retrieval:

```bash
python src/handcrafted/evaluate_retrieval.py \
  --manifest data/processed/fma_small_manifest.csv \
  --features data/processed/handcrafted_features.csv \
  --output outputs/reports/handcrafted_retrieval_metrics.csv
```

The script reports:

- genre Precision@K
- rhythm Precision@K using tempo within 10 BPM
- mood-proxy Precision@K, where relevant tracks are the closest 5% in energy/spectral proxy space

It evaluates four report-facing handcrafted representations:

- `tempo_only`: rhythm reference based only on BPM
- `mood_proxy`: energy/brightness reference based on RMS and spectral features
- `mfcc_only`: timbre-focused handcrafted baseline
- `handcrafted_full`: all extracted handcrafted features

Plot the metrics:

```bash
python src/shared/plot_retrieval_metrics.py \
  --metrics outputs/reports/handcrafted_retrieval_metrics.csv \
  --output outputs/reports/handcrafted_retrieval_metrics.png
```

## CNN Genre Baseline

Cache 15-second mel spectrograms. Start with a balanced debug run:

```bash
python src/trained/cache_mel_spectrograms.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output-dir data/processed/mels_15s_debug \
  --index-output data/processed/mel_spectrograms_15s_debug.csv \
  --limit-per-genre 8
```

Train the debug CNN:

```bash
python src/trained/train_cnn_genre.py \
  --mel-index data/processed/mel_spectrograms_15s_debug.csv \
  --output-dir outputs/models/cnn_genre_debug \
  --embeddings-output data/processed/cnn_embeddings_debug.csv \
  --epochs 2 \
  --batch-size 16
```

Then cache the full mel set:

```bash
python src/trained/cache_mel_spectrograms.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output-dir data/processed/mels_15s \
  --index-output data/processed/mel_spectrograms_15s.csv
```

Train the full CNN genre baseline:

```bash
python src/trained/train_cnn_genre.py \
  --mel-index data/processed/mel_spectrograms_15s.csv \
  --output-dir outputs/models/cnn_genre \
  --embeddings-output data/processed/cnn_embeddings.csv \
  --epochs 10 \
  --batch-size 64
```

The CNN is a lightweight genre-supervised baseline. Its goal is not state-of-the-art genre classification; it gives us penultimate-layer embeddings trained specifically with genre labels. The training script uses a stratified train/validation/test split, saves the best model by validation accuracy, reports test accuracy, and exports embeddings for all cached tracks.

If the small CNN trains quickly, run a stronger medium CNN experiment:

```bash
python src/trained/train_cnn_genre.py \
  --mel-index data/processed/mel_spectrograms_15s.csv \
  --output-dir outputs/models/cnn_genre_medium \
  --embeddings-output data/processed/cnn_embeddings_medium.csv \
  --epochs 20 \
  --batch-size 128 \
  --model-size medium \
  --embedding-dim 128 \
  --dropout 0.4 \
  --label-smoothing 0.05 \
  --patience 5
```

Use the medium model only if validation/test accuracy and retrieval metrics improve; otherwise keep the simpler CNN as the cleaner baseline.

For the final FMA-medium CNN experiment, prefer the compact ResNet-style model:

```bash
python src/trained/train_cnn_genre.py \
  --mel-index data/processed/mel_spectrograms_15s_medium.csv \
  --output-dir outputs/models/cnn_genre_resnet_medium \
  --embeddings-output data/processed/cnn_embeddings_resnet_medium.csv \
  --epochs 30 \
  --batch-size 128 \
  --model-size resnet \
  --embedding-dim 128 \
  --dropout 0.4 \
  --label-smoothing 0.05 \
  --patience 6
```

Evaluate CNN embeddings on test queries:

```bash
python src/shared/evaluate_model_embeddings.py \
  --manifest data/processed/fma_small_manifest.csv \
  --embeddings data/processed/cnn_embeddings.csv \
  --target-features data/processed/handcrafted_features.csv \
  --output outputs/reports/cnn_retrieval_metrics_test_queries.csv \
  --representation-name cnn_genre \
  --query-track-ids outputs/models/cnn_genre/cnn_genre_split.csv \
  --query-split test
```

## Pretrained VGGish Embeddings

Extract VGGish embeddings from a manifest:

```bash
python src/pretrained/extract_vggish_embeddings.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/pretrained_embeddings.csv
```

Debug run:

```bash
python src/pretrained/extract_vggish_embeddings.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/pretrained_embeddings.csv \
  --debug
```

The output format matches the shared evaluator:

```text
track_id, emb_000, emb_001, ...
```

## Planned Pipeline

1. Build clean manifest from FMA metadata.
2. Extract handcrafted audio features with `librosa`.
3. Define genre, rhythm, and mood-proxy similarity labels.
4. Train a small CNN for genre classification.
5. Extract CNN embeddings from the penultimate layer.
6. Evaluate nearest-neighbor retrieval with Precision@K and correlations.
7. Create UMAP/t-SNE plots and final report tables.

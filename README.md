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
python3 src/build_manifest.py \
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

## Handcrafted Features

After the manifest shows that audio files exist, run a small test first:

```bash
python3 src/extract_handcrafted_features.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/handcrafted_features_debug.csv \
  --limit 25
```

Then run the full feature extraction:

```bash
python3 src/extract_handcrafted_features.py \
  --manifest data/processed/fma_small_manifest.csv \
  --output data/processed/handcrafted_features.csv
```

This produces tempo, energy, spectral features, and MFCC summary statistics.

## Planned Pipeline

1. Build clean manifest from FMA metadata.
2. Extract handcrafted audio features with `librosa`.
3. Define genre, rhythm, and mood-proxy similarity labels.
4. Train a small CNN for genre classification.
5. Extract CNN embeddings from the penultimate layer.
6. Evaluate nearest-neighbor retrieval with Precision@K and correlations.
7. Create UMAP/t-SNE plots and final report tables.

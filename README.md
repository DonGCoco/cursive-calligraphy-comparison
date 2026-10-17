# Chinese–Japanese Cursive Calligraphy Comparison

An exploratory computer-vision project comparing visual characteristics of Chinese and Japanese cursive calligraphy.

## Purpose

Measure character shape, spatial organization and ink distribution to support comparative study and digital exploration of calligraphic heritage. Computational descriptions complement expert interpretation; they do not establish artistic or cultural value.

## Status

This implementation was newly written from the project description in October 2026. It is not an archive of the original 2024 code or a reproduction of historical results. Original datasets, trained project models and research results are not bundled. The sample manifest illustrates the input format only.

## Run

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
cp data/manifest.example.csv data/manifest.csv
# Edit the CSV and place actual character images under data/raw/.
python calligraphy.py data/manifest.csv --output outputs/run01
```

On Windows activate with `.venv\Scripts\activate`. Image paths are relative to the manifest CSV. Tradition must be `Chinese` or `Japanese`; artist and work IDs must be filled consistently. Character text may be empty when unknown. Use manually checked single-character crops, dark ink on light paper. Page segmentation is not automatic.

## Method

1. Tight foreground crop using a configurable threshold.
2. Uniform resize and white padding to 224 × 224 pixels, preserving aspect ratio.
3. Eight interpretable image descriptors.
4. Descriptive summaries by tradition and means by artist.
5. Boxplots and standardized PCA exploration.
6. Optional pretrained ResNet18 embeddings.

No default denoising or geometric augmentation is applied, because it may change fine strokes. Inspect every normalized crop; stains, borders and seals may be detected as ink. Configure `--size 224 --threshold 180` as needed.

| Output | Purpose |
| --- | --- |
| `normalized/` | Reviewable normalized images |
| `features.csv` | Metadata and per-image descriptors |
| `summary.csv` | Count, mean, standard deviation and median |
| `artist_means.csv` | Mean descriptors within each artist/tradition |
| `descriptors.png` | Descriptive boxplots |
| `pca.png` | PCA plot when data vary sufficiently |
| `audit.json` | Sample balance, artist diversity, duplicates and parameters |

A nonempty output directory is rejected to avoid overwriting earlier runs. Original images are ignored by Git by default. Track source URLs/catalogue IDs, crop coordinates, period, scan resolution and redistribution rights separately before publishing datasets.

## Optional deep representations

```bash
pip install -r requirements-deep.txt
python deep_features.py outputs/run01/features.csv --output outputs/run01/resnet18.npz
```

Uses official torchvision ImageNet ResNet18 weights; the first run downloads them. This is a frozen feature extractor, not a project-trained model. It exports 512-dimensional vectors in manifest order. The optional path has not been run with downloaded weights or actual project data in this release.

## Interpretation

The baseline measures aspect ratio, ink fraction, connected components, centroid x/y, left/right and top/bottom mirror mismatch, and boundary-to-ink ratio. The two symmetry fields store mismatch: lower values mean greater symmetry. Aspect ratio is measured before normalization; other features use the normalized threshold mask. Components are not literal strokes or pen lifts. Static images cannot establish writing speed or physical brush pressure.

Artist, source work, character identity, period and scanning conditions can confound the comparison. Three Chinese artists and a single Japanese work would not support broad tradition-level claims. PCA separation is exploratory, not evidence of cultural causation or model generalization. No significance tests, accuracy scores or findings are supplied.

For future predictive evaluation, split by artist or source work, check duplicate leakage, and fit scaling/PCA only on training data. For inference, define the sampling unit, collect independent artists in both groups, assess matched-character subsets, report effect sizes and uncertainty, and account for multiple comparisons. Do not treat crops from one work as independent artists.

## Validation

Tests use synthetic strokes to check aspect preservation, blank rejection, CSV validation, complete output generation and overwrite protection. They provide software checks, not calligraphy research evidence.

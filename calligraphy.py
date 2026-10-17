"""Exploratory visual comparison; no historical research results bundled."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES = ['aspect_ratio', 'ink_fraction', 'components', 'centroid_x', 'centroid_y', 'horizontal_symmetry', 'vertical_symmetry', 'edge_fraction']


def normalize(path, size=224, margin=16, threshold=180):
    """Dark ink on light paper. Tight crop + isotropic resize + white padding."""
    if size <= 2 * margin or margin < 0 or not 0 < threshold < 255:
        raise ValueError('Invalid size, margin or threshold')
    with Image.open(path) as src:
        rgba = ImageOps.exif_transpose(src).convert('RGBA')
        paper = Image.new('RGBA', rgba.size, 'white')
        paper.alpha_composite(rgba)
        gray = paper.convert('L')
    ink = np.asarray(gray) < threshold
    ys, xs = np.where(ink)
    if len(xs) == 0:
        raise ValueError(f'No dark foreground: {path}')
    box = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    crop = gray.crop(box)
    ratio = crop.width / crop.height
    scale = (size - 2 * margin) / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new('L', (size, size), 255)
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    return canvas, ratio


def features(image, ratio, threshold=180):
    mask = np.asarray(image) < threshold
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError('Foreground vanished after resize')
    _, count = ndimage.label(mask, structure=np.ones((3, 3)))
    boundary = mask & ~ndimage.binary_erosion(mask)
    return dict(zip(FEATURES, [ratio, mask.mean(), count,
        xs.mean() / (mask.shape[1] - 1), ys.mean() / (mask.shape[0] - 1),
        np.logical_xor(mask, np.fliplr(mask)).mean(),
        np.logical_xor(mask, np.flipud(mask)).mean(), boundary.sum() / mask.sum()]))


def load_manifest(path):
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {'image_path', 'tradition', 'artist_id', 'work_id', 'character'}
    if not required.issubset(df.columns) or df.empty:
        raise ValueError(f'Manifest needs nonempty rows and columns: {sorted(required)}')
    if not set(df.tradition).issubset({'Chinese', 'Japanese'}):
        raise ValueError('tradition must be Chinese or Japanese')
    for col in ['image_path', 'artist_id', 'work_id']:
        if df[col].str.strip().eq('').any():
            raise ValueError(f'Missing {col}')
    paths = [(Path(path).resolve().parent / p).resolve() for p in df.image_path]
    if len(set(paths)) != len(paths):
        raise ValueError('Duplicate image paths')
    if any(not p.is_file() for p in paths):
        raise ValueError('One or more image paths do not exist')
    return df, paths


def run(manifest, output, size=224, threshold=180):
    df, paths = load_manifest(manifest)
    out = Path(output)
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output directory must be new or empty')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'normalized').mkdir()
    records, hashes = [], []
    for i, (row, path) in enumerate(zip(df.to_dict('records'), paths)):
        image, ratio = normalize(path, size=size, threshold=threshold)
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        hashes.append(digest)
        name = f'normalized/{i:06d}.png'
        image.save(out / name)
        records.append({**row, 'normalized_path': name, 'sha256_normalized': digest, **features(image, ratio, threshold)})
    table = pd.DataFrame(records)
    table.to_csv(out / 'features.csv', index=False)
    table.groupby('tradition')[FEATURES].agg(['count', 'mean', 'std', 'median']).to_csv(out / 'summary.csv')
    # Artists are separate descriptive units; pooled character counts are not independent subjects.
    table.groupby(['tradition', 'artist_id'])[FEATURES].mean().to_csv(out / 'artist_means.csv')
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    groups = sorted(table.tradition.unique())
    for ax, feature in zip(axes.flat, FEATURES):
        ax.boxplot([table.loc[table.tradition == g, feature] for g in groups], tick_labels=groups)
        ax.set_title(feature.replace('_', ' '))
    fig.suptitle('Exploratory image descriptors — samples are not independent artists')
    fig.tight_layout()
    fig.savefig(out / 'descriptors.png', dpi=160)
    plt.close(fig)
    if len(table) >= 2:
        x = StandardScaler().fit_transform(table[FEATURES])
        if np.any(np.var(x, axis=0) > 0):
            pca = PCA(n_components=2)
            coords = pca.fit_transform(x)
            fig, ax = plt.subplots(figsize=(7, 6))
            for group in groups:
                keep = table.tradition == group
                ax.scatter(coords[keep, 0], coords[keep, 1], label=group, alpha=.7)
            ax.set(xlabel=f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', ylabel=f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', title='Exploratory PCA of visual descriptors')
            ax.legend()
            fig.tight_layout()
            fig.savefig(out / 'pca.png', dpi=160)
            plt.close(fig)
    audit = {'samples': len(table), 'counts': table.tradition.value_counts().to_dict(),
             'artists_per_tradition': table.groupby('tradition').artist_id.nunique().to_dict(),
             'duplicate_normalized_images': len(hashes) - len(set(hashes)),
             'size': size, 'threshold': threshold,
             'manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
             'interpretation': 'Descriptive only. Control artist, source work, character identity and scan conditions before making tradition-level claims.'}
    (out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf8')
    return table


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--threshold', type=int, default=180)
    args = parser.parse_args()
    run(args.manifest, args.output, args.size, args.threshold)

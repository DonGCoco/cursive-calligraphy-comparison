"""Optional frozen ImageNet ResNet18 features; downloads official weights."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torchvision.models import resnet18, ResNet18_Weights


def extract(csv, output, batch_size=32):
    source, out = Path(csv), Path(output)
    if batch_size < 1 or out.exists():
        raise ValueError('Batch size must be positive and output must not exist')
    table = pd.read_csv(source, keep_default_na=False)
    if table.empty:
        raise ValueError('No samples')
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    transform = weights.transforms()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(table), batch_size):
            batch = []
            for name in table.normalized_path.iloc[start:start + batch_size]:
                with Image.open(source.parent / name) as image:
                    batch.append(transform(image.convert('RGB')))
            vectors.append(model(torch.stack(batch)).cpu().numpy())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('xb') as handle:
        np.savez_compressed(handle, embeddings=np.concatenate(vectors), image_paths=table.image_path.to_numpy(dtype=str), traditions=table.tradition.to_numpy(dtype=str), model='resnet18', weights=str(weights))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('features_csv')
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    a = p.parse_args()
    extract(a.features_csv, a.output, a.batch_size)

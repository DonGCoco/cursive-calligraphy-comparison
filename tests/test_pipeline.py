import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from calligraphy import normalize, run, load_manifest


class PipelineTests(unittest.TestCase):
    def test_aspect_and_padding(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'ink.png'
            image = Image.new('L', (100, 180), 255)
            ImageDraw.Draw(image).rectangle((40, 20, 59, 159), fill=0)
            image.save(path)
            result, ratio = normalize(path)
            self.assertEqual(result.size, (224, 224))
            self.assertAlmostEqual(ratio, 20 / 140)
            mask = np.asarray(result) < 180
            y, x = np.where(mask)
            self.assertLess(abs((x.max()-x.min()+1)/(y.max()-y.min()+1)-ratio), .01)
            self.assertFalse(mask[:16].any())

    def test_blank_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'blank.png'
            Image.new('L', (20, 20), 255).save(p)
            with self.assertRaises(ValueError):
                normalize(p)

    def test_end_to_end(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            rows = []
            for i in range(4):
                image = Image.new('L', (100, 100), 255)
                ImageDraw.Draw(image).line([(20, 20), (30+i*10, 80), (80, 40)], fill=0, width=3+i)
                image.save(root / f'{i}.png')
                rows.append(dict(image_path=f'{i}.png', tradition='Chinese' if i<2 else 'Japanese', artist_id=f'a{i}', work_id=f'w{i}', character='synthetic'))
            manifest = root / 'manifest.csv'
            pd.DataFrame(rows).to_csv(manifest, index=False)
            self.assertEqual(len(run(manifest, root / 'out')), 4)
            for name in ['features.csv', 'summary.csv', 'artist_means.csv', 'descriptors.png', 'pca.png', 'audit.json']:
                self.assertTrue((root / 'out' / name).is_file(), name)
            with self.assertRaises(ValueError):
                run(manifest, root / 'out')
            rows.append(rows[0])
            pd.DataFrame(rows).to_csv(manifest, index=False)
            with self.assertRaises(ValueError):
                load_manifest(manifest)


if __name__ == '__main__':
    unittest.main()

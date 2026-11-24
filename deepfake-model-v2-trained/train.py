"""Train an ELA feature classifier and evaluate two checkpoints on one split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from ela import prepare

SEED = 2027


def features(ela):
    # Preserve regional and JPEG-block structure at a fixed feature dimension.
    x = ela.astype(np.float32) / 255
    tiles = x.reshape(4, 64, 4, 64, 3)
    phases = x.reshape(32, 8, 32, 8, 3)
    return np.concatenate([tiles.mean((1, 3)).ravel(), tiles.std((1, 3)).ravel(),
                           phases.mean((0, 2)).ravel(),
                           np.quantile(x, [.1, .5, .9, .99], axis=(0, 1)).ravel()]).astype('float32')


def write_csv(path, columns, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)


def metrics(labels, scores):
    pred = scores >= .5
    tp = int(np.sum(pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    fp, fn = int(sum(labels == 0)) - tn, int(sum(labels == 1)) - tp
    pos, neg = scores[labels == 1, None], scores[labels == 0]
    return dict(accuracy=float((tp+tn)/len(scores)), fake_f1=2*tp/(2*tp+fp+fn),
                roc_auc=float(((pos > neg) + .5*(pos == neg)).mean()),
                tp=tp, tn=tn, fp=fp, fn=fn)


def run(dataset, original, out):
    out.mkdir(exist_ok=True, parents=True)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
    rng = np.random.default_rng(SEED)
    records, vectors, seen = [], [], {}
    with ZipFile(dataset) as archive:
        for name in sorted(n for n in archive.namelist() if n.lower().endswith('.jpg')):
            data = archive.read(name)
            label = {'Real': 0, 'Fake': 1}[Path(name).parent.name]
            with Image.open(BytesIO(data)) as image:
                key = hashlib.sha256(image.convert('RGB').tobytes()).hexdigest()
            if key in seen and seen[key] != label:
                raise ValueError(f'Conflicting labels for {name}')
            seen[key] = label
            records.append((name, label, key))
            vectors.append(features(prepare(data)))
        partitions = {}
        for label in [0, 1]:
            keys = rng.permutation(sorted(k for k, v in seen.items() if v == label))
            a, b = int(.7*len(keys)), int(.85*len(keys))
            for split, group in zip(['train','validation','test'], np.split(keys,[a,b])):
                partitions.update(dict.fromkeys(group, split))
        subsets = np.array([partitions[r[2]] for r in records])
        write_csv(out/'split.csv', ['image','fake','pixel_sha256','split'],
                  [(*r,s) for r,s in zip(records,subsets)])
        x = np.stack(vectors)
        y = np.array([r[1] for r in records])
        train, val, test = [subsets == s for s in ['train','validation','test']]
        norm = tf.keras.layers.Normalization()
        norm.adapt(x[train])
        model = tf.keras.Sequential([tf.keras.Input((x.shape[1],)), norm,
                    tf.keras.layers.Dense(64, activation='relu'), tf.keras.layers.Dropout(.25),
                    tf.keras.layers.Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        history = model.fit(x[train], y[train], validation_data=(x[val],y[val]),
                  epochs=60, batch_size=64, verbose=0, callbacks=[
                  tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
        model.save(out/'v2.keras')
        scores = tf.keras.models.load_model(out/'v2.keras')(x[test], training=False).numpy().ravel()
        baseline = tf.keras.models.load_model(original, compile=False)
        names = [records[i][0] for i in np.flatnonzero(test)]
        old = []
        for i in range(0,len(names),8):
            batch = np.stack([prepare(archive.read(n)) for n in names[i:i+8]])
            old.extend(baseline(batch, training=False).numpy().ravel())
    write_csv(out/'predictions.csv',['image','fake','original_score','v2_score'],zip(names,y[test],old,scores))
    result = dict(counts={s:int(sum(subsets==s)) for s in sorted(set(subsets))},
                  original_sha256=hashlib.file_digest(open(original,'rb'),'sha256').hexdigest(),
                  best_epoch=int(np.argmin(history.history['val_loss']))+1,
                  original=metrics(y[test], np.array(old)), v2=metrics(y[test],scores))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',type=Path)
    parser.add_argument('original',type=Path)
    parser.add_argument('--out',type=Path,default=Path('run'))
    args = parser.parse_args()
    run(args.dataset,args.original,args.out)

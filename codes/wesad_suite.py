"""Resumable corrected WESAD suite. Run from the project root.

Full runs: 15 LOSO folds x 5 architectures, five HABC grouped splits,
and four additional fixed-split ablations (HABC reuses grouped split 1).
One prespecified initialization seed; repeated-seed uncertainty is not claimed.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
import numpy as np
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from wesad_corrected import ROOT, CHANNELS, CLASS_NAMES, load_subject, build_habc, TemporalAttention
import tensorflow as tf
from tensorflow import keras

OUT = ROOT / 'results' / 'wesad_complete'
CACHE = OUT / 'cache'
VARIANTS = ['HABC', 'CNN', 'BiLSTM', 'BiLSTM_Attention', 'CNN_BiLSTM']
SEED = 42
CONFIG = {'version': 1, 'seed': SEED, 'epochs': 100, 'batch_size': 64,
          'early_stop_patience': 20, 'lr_patience': 8, 'learning_rate': .001,
          'weight_decay': .004, 'validation_subjects': 3, 'window_seconds': 10,
          'overlap': .5, 'class_purity': .95, 'target_hz': 70,
          'variants': VARIANTS, 'corruption_seeds': [42, 43, 44, 45, 46],
          'scope': 'WESAD only; one training seed per fit; no row-level standard split'}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def status(stage, **kwargs):
    write_json(OUT / 'status.json', {'stage': stage, 'updated_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **kwargs})
    print(stage, kwargs, flush=True)


def fingerprint():
    sources = {name: hashlib.sha256((ROOT / 'codes' / name).read_bytes()).hexdigest()
               for name in ['wesad_corrected.py', 'wesad_suite.py']}
    files = {p.name: {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns}
             for p in sorted((ROOT / 'datasets' / 'WESAD').glob('S*.pkl'))}
    return {'config': CONFIG, 'source_sha256': sources, 'data_inventory': files}


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    manifest = fingerprint()
    manifest_path = OUT / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise RuntimeError('Code/config/dataset differs from existing run. Use a new output directory.')
    else:
        write_json(manifest_path, manifest)
    paths = sorted((ROOT / 'datasets' / 'WESAD').glob('S*.pkl'), key=lambda p: int(p.stem[1:]))
    expected = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17}
    assert {int(p.stem[1:]) for p in paths} == expected
    for p in paths:
        sid = int(p.stem[1:])
        dest = CACHE / f'S{sid}.npz'
        info_path = CACHE / f'S{sid}.json'
        if dest.exists() and info_path.exists():
            continue
        status('preprocessing', subject=sid)
        X, y, info = load_subject(p)  # no pilot subsampling
        assert len(X) == len(y) and X.shape[1:] == (700, 8)
        np.savez(dest, X=X, y=y, starts=np.array(info['selected_start_samples']))
        write_json(info_path, info)
        del X, y
        gc.collect()
    data = []
    labels = []
    subjects = []
    starts = []
    for p in paths:
        sid = int(p.stem[1:])
        with np.load(CACHE / f'S{sid}.npz') as z:
            data.append(z['X'])
            labels.append(z['y'])
            starts.append(z['starts'])
            subjects.append(np.full(len(z['y']), sid))
    X, y, groups, starts = map(np.concatenate, [data, labels, subjects, starts])
    assert np.isfinite(X).all() and set(y) == {0, 1, 2}
    audit = {'subjects': sorted(expected), 'total_windows': len(y), 'shape': list(X.shape),
             'class_counts': np.bincount(y, minlength=3).tolist(),
             'per_subject': {str(s): np.bincount(y[groups == s], minlength=3).tolist() for s in sorted(expected)},
             'channels': CHANNELS, 'class_names': CLASS_NAMES}
    write_json(OUT / 'data_audit.json', audit)
    return X, y, groups, starts


def make_plan(groups):
    subjects = np.unique(groups)
    jobs = []
    # Fixed validation subjects chosen only from the outer training people.
    def add(name, train_pool, test, variant):
        rng = np.random.default_rng(SEED)
        val = sorted(rng.choice(sorted(train_pool), CONFIG['validation_subjects'], replace=False).tolist())
        train = sorted(set(train_pool) - set(val))
        jobs.append({'id': name, 'variant': variant, 'train': train, 'validation': val,
                     'test': sorted(map(int, test)), 'seed': SEED})
    for s in subjects:
        pool = subjects[subjects != s].tolist()
        for variant in VARIANTS:
            add(f'loso_S{s}_{variant}', pool, [s], variant)
    splitter = GroupShuffleSplit(n_splits=5, test_size=.3, random_state=SEED)
    for i, (tr, te) in enumerate(splitter.split(groups, groups=groups), 1):
        add(f'cs_{i}_HABC', np.unique(groups[tr]).tolist(), np.unique(groups[te]).tolist(), 'HABC')
    first = next(j for j in jobs if j['id'] == 'cs_1_HABC')
    for variant in VARIANTS[1:]:
        add(f'ablation_{variant}', first['train'] + first['validation'], first['test'], variant)
    for j in jobs:
        t, v, e = map(set, [j['train'], j['validation'], j['test']])
        assert not(t & v or t & e or v & e)
        assert t | v | e == set(subjects)
    write_json(OUT / 'split_plan.json', jobs)
    return jobs


def build_variant(variant):
    if variant == 'HABC':
        return build_habc()
    L = keras.layers
    inputs = {}
    embeddings = []
    if variant in ['CNN', 'CNN_BiLSTM']:
        inp = keras.Input((700, 8, 1), name='cnn_input')
        inputs['cnn_input'] = inp
        x = inp
        for filters, kernel in [(32, (5, 1)), (64, (3, 1))]:
            x = L.Conv2D(filters, kernel, padding='same')(x)
            x = L.Activation('relu')(L.BatchNormalization()(x))
            x = L.MaxPooling2D((2, 1))(x)
        embeddings.append(L.Dropout(.3)(L.Flatten()(x)))
    if variant != 'CNN':
        inp = keras.Input((700, 8), name='lstm_input')
        inputs['lstm_input'] = inp
        h = L.Bidirectional(L.LSTM(128, return_sequences=True, dropout=.3, recurrent_dropout=.3))(inp)
        attention = variant == 'BiLSTM_Attention'
        h = L.Bidirectional(L.LSTM(64, return_sequences=attention, dropout=.3, recurrent_dropout=.3))(h)
        if attention:
            h = TemporalAttention()(h)
        embeddings.append(L.Dropout(.3)(h))
    z = embeddings[0] if len(embeddings) == 1 else L.Concatenate()(embeddings)
    for units, rate in [(256, .5), (128, .4)]:
        z = L.Dense(units)(z)
        z = L.Dropout(rate)(L.Activation('relu')(L.BatchNormalization()(z)))
    model = keras.Model(inputs, L.Dense(3, activation='softmax')(z), name=variant)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=.001, weight_decay=.004),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def inputs(X, variant):
    result = {}
    if variant in ['HABC', 'CNN', 'CNN_BiLSTM']:
        result['cnn_input'] = X[..., None]
    if variant != 'CNN':
        result['lstm_input'] = X
    return result


def metrics(y, probabilities):
    assert probabilities.shape == (len(y), 3)
    assert np.isfinite(probabilities).all() and np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    pred = probabilities.argmax(axis=1)
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(y, pred, labels=[0, 1, 2], average='weighted', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
            'report': classification_report(y, pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                            output_dict=True, zero_division=0)}


def scaled_split(job, X, y, groups):
    idx = {r: np.flatnonzero(np.isin(groups, job[r])) for r in ['train', 'validation', 'test']}
    scaler = StandardScaler().fit(X[idx['train']].reshape(-1, 8))
    arrays = {r: scaler.transform(X[i].reshape(-1, 8)).reshape(-1, 700, 8).astype('float32') for r, i in idx.items()}
    assert int(scaler.n_samples_seen_) == len(idx['train']) * 700
    assert np.allclose(arrays['train'].mean(axis=(0, 1), dtype=np.float64), 0, atol=1e-4)
    return idx, scaler, arrays


class Progress(keras.callbacks.Callback):
    def __init__(self, job, number, total):
        super().__init__()
        self.job, self.number, self.total = job, number, total
        self.started = time.time()

    def on_epoch_end(self, epoch, logs=None):
        status('training', job=self.job, job_number=self.number, total_jobs=self.total,
               epoch=epoch + 1, elapsed_seconds=round(time.time() - self.started, 1),
               metrics={k: float(v) for k, v in (logs or {}).items()})


def fit_job(job, X, y, groups, starts, number, total):
    dest = OUT / 'runs' / job['id']
    if (dest / 'result.json').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    status('starting_fit', job=job['id'], job_number=number, total_jobs=total)
    idx, scaler, arrays = scaled_split(job, X, y, groups)
    write_json(dest / 'split.json', job)
    np.savez(dest / 'scaler.npz', mean=scaler.mean_, scale=scaler.scale_, n_samples_seen=scaler.n_samples_seen_)
    keras.backend.clear_session()
    keras.utils.set_random_seed(job['seed'])
    model = build_variant(job['variant'])
    write_json(dest / 'model.json', json.loads(model.to_json()))
    weights = compute_class_weight('balanced', classes=np.arange(3), y=y[idx['train']])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8),
                 keras.callbacks.ModelCheckpoint(str(dest / 'best.weights.h5'), monitor='val_loss',
                                                 save_best_only=True, save_weights_only=True),
                 keras.callbacks.CSVLogger(str(dest / 'epochs.csv')),
                 Progress(job['id'], number, total)]
    before = time.time()
    history = model.fit(inputs(arrays['train'], job['variant']), y[idx['train']],
                        validation_data=(inputs(arrays['validation'], job['variant']), y[idx['validation']]),
                        epochs=100, batch_size=64, class_weight=dict(enumerate(weights.tolist())),
                        callbacks=callbacks, verbose=2)
    model.load_weights(dest / 'best.weights.h5')
    p = model.predict(inputs(arrays['test'], job['variant']), batch_size=64, verbose=0)
    result = metrics(y[idx['test']], p)
    np.savez(dest / 'predictions.npz', probabilities=p, y_true=y[idx['test']],
             y_pred=p.argmax(axis=1), indices=idx['test'], subjects=groups[idx['test']], starts=starts[idx['test']])
    result.update({'job': job, 'epochs': len(history.history['loss']), 'history': history.history,
                   'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
                   'seconds': time.time() - before, 'parameter_count': model.count_params(),
                   'class_weights': weights.tolist()})
    write_json(dest / 'result.json', result)
    del model, arrays
    keras.backend.clear_session()
    gc.collect()


def robustness(job, X, y, groups, starts):
    dest = OUT / 'robustness'
    if (dest / 'result.json').exists():
        return
    dest.mkdir(exist_ok=True)
    idx, _, arrays = scaled_split(job, X, y, groups)
    test = arrays['test']
    truth = y[idx['test']]
    keras.backend.clear_session()
    model = build_variant('HABC')
    model.load_weights(OUT / 'runs' / job['id'] / 'best.weights.h5')
    conditions = [('clean', None)] + [(f'snr_{v}', v) for v in [30, 20, 10, 5, 0]] + [(f'missing_{v}', v / 100) for v in [5, 10, 15, 20, 30]]
    results = {}
    for name, value in conditions:
        results[name] = []
        seeds = [SEED] if name == 'clean' else CONFIG['corruption_seeds']
        for seed in seeds:
            status('robustness', condition=name, seed=seed)
            rng = np.random.default_rng(seed)
            perturbed = test.copy()
            if name.startswith('snr'):
                power = np.mean(test.astype(np.float64) ** 2)
                perturbed += rng.normal(0, np.sqrt(power / (10 ** (value / 10))), test.shape).astype('float32')
            elif name.startswith('missing'):
                perturbed[rng.random(test.shape) < value] = 0
            p = model.predict(inputs(perturbed, 'HABC'), batch_size=64, verbose=0)
            np.savez(dest / f'{name}_seed{seed}.npz', y_true=truth, probabilities=p, indices=idx['test'])
            results[name].append({'seed': seed, **metrics(truth, p)})
    write_json(dest / 'result.json', {'source_job': job, 'conditions': results,
               'interpretation': 'Gaussian standardized-value noise / independent scalar zero masking; not specific motion artifacts or packet loss'})


def paired_tests(plan):
    folds = {}
    for variant in VARIANTS:
        folds[variant] = [json.loads((OUT / 'runs' / j['id'] / 'result.json').read_text())['accuracy']
                          for j in plan if j['id'].startswith('loso_') and j['variant'] == variant]
    comparisons = []
    for variant in VARIANTS[1:]:
        d = np.array(folds['HABC']) - np.array(folds[variant])
        if np.all(d == 0):
            t, p, w, wp = 0., 1., 0., 1.
        else:
            t, p = stats.ttest_rel(folds['HABC'], folds[variant])
            w, wp = stats.wilcoxon(folds['HABC'], folds[variant])
        comparisons.append({'baseline': variant, 't': float(t), 'p_t': float(p),
                            'w': float(w), 'p_w': float(wp), 'mean_difference': float(d.mean())})
    # Holm correction of the prespecified primary Wilcoxon family (four comparisons).
    ordered = sorted(range(len(comparisons)), key=lambda k: comparisons[k]['p_w'])
    previous = 0.
    for rank, k in enumerate(ordered):
        previous = max(previous, min(1., (len(ordered) - rank) * comparisons[k]['p_w']))
        comparisons[k]['p_w_holm'] = previous
        comparisons[k]['significant_primary_005'] = previous < .05
    write_json(OUT / 'significance.json', {'fold_accuracy': folds, 'comparisons': comparisons,
               'primary_test': 'two-sided Wilcoxon, Holm across four baseline comparisons',
               'caveat': 'LOSO training folds overlap; one initialization per fit; non-significance is not equivalence'})


def validate(X, y, groups, starts, plan, require_complete=False):
    checks = []
    for j in plan:
        folder = OUT / 'runs' / j['id']
        if not (folder / 'result.json').exists():
            if require_complete:
                raise AssertionError(f'Missing job {j["id"]}')
            continue
        result = json.loads((folder / 'result.json').read_text())
        with np.load(folder / 'predictions.npz') as z:
            indices = np.flatnonzero(np.isin(groups, j['test']))
            assert np.array_equal(z['indices'], indices)
            assert np.array_equal(z['y_true'], y[indices])
            assert np.array_equal(z['subjects'], groups[indices])
            assert np.array_equal(z['starts'], starts[indices])
            assert np.array_equal(z['y_pred'], z['probabilities'].argmax(axis=1))
            measured = metrics(z['y_true'], z['probabilities'])
            for key in ['accuracy', 'macro_f1', 'weighted_f1']:
                assert np.isclose(measured[key], result[key], atol=1e-12)
            assert measured['confusion_matrix'] == result['confusion_matrix']
        # Independently refit only from recorded training people to check saved scaler.
        train = X[np.isin(groups, j['train'])].reshape(-1, 8)
        scaler = StandardScaler().fit(train)
        with np.load(folder / 'scaler.npz') as z:
            assert np.allclose(z['mean'], scaler.mean_, atol=1e-12)
            assert np.allclose(z['scale'], scaler.scale_, atol=1e-12)
            assert int(z['n_samples_seen']) == len(train)
        assert result['best_epoch'] == int(np.argmin(result['history']['val_loss']) + 1)
        checks.append(j['id'])
    robustness_path = OUT / 'robustness' / 'result.json'
    robustness_checked = 0
    if robustness_path.exists():
        robust = json.loads(robustness_path.read_text())
        job = robust['source_job']
        test_indices = np.flatnonzero(np.isin(groups, job['test']))
        for condition, runs in robust['conditions'].items():
            for run in runs:
                with np.load(OUT / 'robustness' / f'{condition}_seed{run["seed"]}.npz') as z:
                    assert np.array_equal(z['indices'], test_indices)
                    assert np.array_equal(z['y_true'], y[test_indices])
                    measured = metrics(z['y_true'], z['probabilities'])
                    assert np.isclose(measured['accuracy'], run['accuracy'], atol=1e-12)
                    assert measured['confusion_matrix'] == run['confusion_matrix']
                robustness_checked += 1
    if require_complete:
        assert robustness_checked == 51
    audit = {'validated_jobs': checks, 'completed': len(checks), 'expected': len(plan),
             'complete': len(checks) == len(plan) and robustness_checked == 51,
             'robustness_predictions_checked': robustness_checked,
             'checks': ['test identities/labels/start positions',
             'probability normalization', 'metrics recomputation', 'train-only scaler refit', 'validation best epoch']}
    write_json(OUT / 'validation.json', audit)
    return audit


def summarize(plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = {j['id']: json.loads((OUT / 'runs' / j['id'] / 'result.json').read_text()) for j in plan}
    aggregate = {}
    for variant in VARIANTS:
        selected = [r for k, r in rows.items() if k.startswith('loso_') and r['job']['variant'] == variant]
        aggregate[variant] = {key: {'mean': float(np.mean([r[key] for r in selected])),
                                   'std_population': float(np.std([r[key] for r in selected]))}
                              for key in ['accuracy', 'macro_f1', 'weighted_f1']}
    cs = [r for k, r in rows.items() if k.startswith('cs_')]
    cs_summary = {key: {'mean': float(np.mean([r[key] for r in cs])),
                        'std_population': float(np.std([r[key] for r in cs]))}
                  for key in ['accuracy', 'macro_f1', 'weighted_f1']}
    ablation = {'HABC': rows['cs_1_HABC']}
    ablation.update({v: rows[f'ablation_{v}'] for v in VARIANTS[1:]})
    write_json(OUT / 'summary.json', {'LOSO': aggregate, 'grouped_splits': cs_summary,
               'fixed_split_ablation': {v: {k: r[k] for k in ['accuracy', 'macro_f1', 'weighted_f1']} for v, r in ablation.items()},
               'fixed_split_class_report': rows['cs_1_HABC']['report'],
               'fixed_split_source': 'cs_1_HABC', 'initialization_seeds_per_fit': 1})
    lines = ['# Corrected WESAD results', '',
             'Subject-disjoint validation, train-only scaling, anti-alias downsampling, 95% window purity.',
             'One initialization seed per fit. These replace neither old scores nor claimed protocols silently.', '',
             '| Model | LOSO accuracy (%) | Macro F1 (%) |', '|---|---:|---:|']
    for v, r in aggregate.items():
        lines.append(f'| {v} | {100*r["accuracy"]["mean"]:.2f} ± {100*r["accuracy"]["std_population"]:.2f} | {100*r["macro_f1"]["mean"]:.2f} |')
    lines += ['', '## Fixed-split ablation', '', '| Model | Accuracy (%) | Macro F1 (%) |', '|---|---:|---:|']
    for v, r in ablation.items():
        lines.append(f'| {v} | {100*r["accuracy"]:.2f} | {100*r["macro_f1"]:.2f} |')
    lines += ['', 'Statistical tests are in significance.json (primary: Holm-adjusted Wilcoxon).',
              'No superiority, equivalence, or real sensor-artifact claims follow automatically from these scores.']
    (OUT / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    cm = np.asarray(rows['cs_1_HABC']['confusion_matrix'])
    for normalized in [False, True]:
        shown = cm / cm.sum(axis=1, keepdims=True) if normalized else cm
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(shown, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f'{shown[i,j]:.3f}' if normalized else str(cm[i,j]), ha='center', va='center')
        ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
               xlabel='Predicted', ylabel='True', title='WESAD fixed subject split')
        fig.tight_layout()
        fig.savefig(OUT / ('confusion_normalized.png' if normalized else 'confusion_counts.png'), dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    positions = np.arange(len(VARIANTS))
    ax.bar(positions - .18, [100*ablation[v]['accuracy'] for v in VARIANTS], width=.36, label='Accuracy')
    ax.bar(positions + .18, [100*ablation[v]['macro_f1'] for v in VARIANTS], width=.36, label='Macro F1')
    ax.set(xticks=positions, xticklabels=VARIANTS, ylabel='Score (%)', ylim=(0, 100))
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / 'ablation.png', dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.experimental.enable_op_determinism()
    try:
        X, y, groups, starts = prepare()
        plan = make_plan(groups)
        write_json(OUT / 'runtime.json', {'python': sys.version, 'tensorflow': tf.__version__,
                   'platform': platform.platform(), 'gpu': [str(x) for x in tf.config.list_physical_devices('GPU')],
                   'free_disk_bytes': shutil.disk_usage(OUT).free, 'jobs': len(plan)})
        if args.prepare_only:
            status('prepared', jobs=len(plan), windows=len(y))
            return
        if args.validate_only:
            print(validate(X, y, groups, starts, plan), flush=True)
            return
        # Model fits run serially to avoid competing for RAM/CPU.
        for i, job in enumerate(plan, 1):
            fit_job(job, X, y, groups, starts, i, len(plan))
        robustness(next(j for j in plan if j['id'] == 'cs_1_HABC'), X, y, groups, starts)
        paired_tests(plan)
        audit = validate(X, y, groups, starts, plan, require_complete=True)
        summarize(plan)
        status('complete', **audit)
    except BaseException as exc:
        status('failed', error=repr(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()

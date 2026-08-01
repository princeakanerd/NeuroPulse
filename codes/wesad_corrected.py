"""Subject-disjoint WESAD pilot. Scores are smoke checks, not paper results."""
from pathlib import Path
import gc
import json
import pickle
import random

import numpy as np
from scipy.signal import resample_poly
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ['ACC_x', 'ACC_y', 'ACC_z', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
CLASS_NAMES = ['Baseline', 'Stress', 'Amusement']


def load_subject(path, *, max_per_class=None, seed=42):
    """10-second windows, 50% overlap, >=95% single valid class.

    Anti-alias resampling operates on each continuous subject recording before
    window extraction. No normalization is learned here. Input pickle files
    must be the trusted WESAD files supplied for this project.
    """
    with Path(path).open('rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    labels = np.asarray(data['label']).reshape(-1)
    chest = data['signal']['chest']
    parts = []
    shapes = {}
    for key in ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']:
        signal = np.asarray(chest[key])
        if signal.ndim == 1:
            signal = signal[:, None]
        expected = 3 if key == 'ACC' else 1
        if signal.shape != (len(labels), expected):
            raise ValueError(f'{path}: invalid {key} shape {signal.shape}')
        shapes[key] = list(signal.shape)
        parts.append(signal.astype(np.float32))
    continuous = np.concatenate(parts, axis=1)
    if not np.isfinite(continuous).all():
        raise ValueError(f'{path}: nonfinite sensor values')
    # 700 Hz -> 70 Hz with a polyphase low-pass filter (not stride slicing).
    filtered = resample_poly(continuous, up=1, down=10, axis=0).astype(np.float32)
    label_counts = {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))}
    del data, chest, parts, continuous, signal
    gc.collect()
    indices = {0: [], 1: [], 2: []}
    for start in range(0, len(labels) - 7000 + 1, 3500):
        counts = np.bincount(labels[start:start + 7000].astype(int), minlength=4)
        winner = int(np.argmax(counts))
        if winner in (1, 2, 3) and counts[winner] / 7000 >= 0.95:
            indices[winner - 1].append(start)
    available = {str(k): len(v) for k, v in indices.items()}
    rng = np.random.default_rng(seed)
    selected = []
    for cls, starts in indices.items():
        if not starts:
            raise ValueError(f'{path}: class {cls} absent after purity filtering')
        if max_per_class is not None and len(starts) > max_per_class:
            starts = rng.choice(starts, max_per_class, replace=False).tolist()
        selected.extend((start, cls) for start in starts)
    selected.sort()
    X = np.stack([filtered[start // 10:start // 10 + 700] for start, _ in selected])
    y = np.array([cls for _, cls in selected], dtype=np.int64)
    if X.shape[1:] != (700, 8):
        raise ValueError(f'Unexpected window shape: {X.shape}')
    info = {'file': str(path), 'raw_samples': len(labels), 'raw_signal_shapes': shapes,
            'raw_label_counts': label_counts, 'eligible_windows_per_class': available,
            'selected_windows': len(y), 'selected_start_samples': [s for s, _ in selected]}
    return X, y, info


@keras.utils.register_keras_serializable(package='HABC')
class TemporalAttention(keras.layers.Layer):
    def build(self, input_shape):
        self.score = keras.layers.Dense(1, activation='tanh')
        self.score.build(input_shape)
        super().build(input_shape)

    def call(self, inputs):
        scores = self.score(inputs)
        weights = tf.nn.softmax(scores, axis=1)
        return tf.reduce_sum(inputs * weights, axis=1)


def build_habc(time_steps=700, feature_dim=8, num_classes=3):
    """Paper 1 topology, explicitly channels-last; shared scalar score bias."""
    L = keras.layers
    cnn_input = keras.Input((time_steps, feature_dim, 1), name='cnn_input')
    x = cnn_input
    for filters, kernel in [(32, (5, 1)), (64, (3, 1))]:
        x = L.Conv2D(filters, kernel, padding='same', data_format='channels_last')(x)
        x = L.BatchNormalization()(x)
        x = L.Activation('relu')(x)
        x = L.MaxPooling2D((2, 1), data_format='channels_last')(x)
    x = L.Dropout(0.3)(L.Flatten()(x))
    recurrent_input = keras.Input((time_steps, feature_dim), name='lstm_input')
    h = recurrent_input
    for units in (128, 64):
        h = L.Bidirectional(L.LSTM(units, return_sequences=True,
                                  dropout=0.3, recurrent_dropout=0.3))(h)
    h = L.Dropout(0.3)(TemporalAttention(name='temporal_attention')(h))
    z = L.Concatenate()([x, h])
    for units, dropout in [(256, 0.5), (128, 0.4)]:
        z = L.Dense(units)(z)
        z = L.BatchNormalization()(z)
        z = L.Activation('relu')(z)
        z = L.Dropout(dropout)(z)
    output = L.Dense(num_classes, activation='softmax')(z)
    model = keras.Model([cnn_input, recurrent_input], output, name='HABC_Paper1_corrected')
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.004),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def model_inputs(X):
    return {'cnn_input': X[..., None], 'lstm_input': X}


def run_pilot():
    """One epoch, 12 train / 6 validation / 6 test windows; no final claims."""
    seed = 42
    keras.backend.clear_session()
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    out = ROOT / 'results' / 'wesad_pilot'
    out.mkdir(parents=True, exist_ok=True)
    split = {'train': [2], 'validation': [3], 'test': [4]}
    assert not (set(split['train']) & set(split['validation']))
    assert not (set(split['train']) & set(split['test']))
    assert not (set(split['validation']) & set(split['test']))
    data = {}
    inspections = {}
    for role, subjects in split.items():
        sid = subjects[0]
        print(f'Inspecting/loading S{sid}: {role}', flush=True)
        X, y, info = load_subject(ROOT / 'datasets' / 'WESAD' / f'S{sid}.pkl',
                                  max_per_class=4 if role == 'train' else 2, seed=seed)
        data[role] = (X, y)
        inspections[role] = info
        print(f'{role}: {X.shape}, class counts={np.bincount(y).tolist()}', flush=True)
    scaler = StandardScaler()
    scaler.fit(data['train'][0].reshape(-1, 8))
    expected = len(data['train'][0]) * 700
    assert int(scaler.n_samples_seen_) == expected
    for role, (X, y) in data.items():
        data[role] = (scaler.transform(X.reshape(-1, 8)).reshape(X.shape).astype(np.float32), y)
        assert np.isfinite(data[role][0]).all()
    assert np.allclose(data['train'][0].mean(axis=(0, 1)), 0, atol=1e-5)
    np.savez(out / 'training_scaler.npz', mean=scaler.mean_, scale=scaler.scale_)
    model = build_habc()
    print('Parameters:', model.count_params(), flush=True)
    (out / 'model_architecture.json').write_text(model.to_json(), encoding='utf-8')
    model.summary(print_fn=lambda line: print(line, flush=True))
    classes = np.arange(3)
    weights = compute_class_weight('balanced', classes=classes, y=data['train'][1])
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=20,
                                             restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=8),
                 keras.callbacks.ModelCheckpoint(str(out / 'best_model.keras'),
                                                 monitor='val_loss', save_best_only=True)]
    history = model.fit(model_inputs(data['train'][0]), data['train'][1],
                        validation_data=(model_inputs(data['validation'][0]), data['validation'][1]),
                        epochs=1, batch_size=4, class_weight=dict(zip(classes.tolist(), weights.tolist())),
                        callbacks=callbacks, verbose=2)
    # Use the validation-selected checkpoint, not any test-based selection.
    model = keras.models.load_model(out / 'best_model.keras')
    probabilities = model.predict(model_inputs(data['test'][0]), batch_size=2, verbose=0)
    assert probabilities.shape == (6, 3)
    assert np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    truth = data['test'][1]
    predicted = probabilities.argmax(axis=1)
    np.savez(out / 'test_predictions.npz', y_true=truth, y_pred=predicted,
             probabilities=probabilities,
             subject_ids=np.full(len(truth), 4),
             start_samples=inspections['test']['selected_start_samples'])
    result = {'status': 'pilot_passed', 'purpose': 'one-epoch pipeline smoke check; NOT paper performance',
              'seed': seed, 'split_subjects': split, 'tensorflow_version': tf.__version__,
              'gpu_devices': [str(d) for d in tf.config.list_physical_devices('GPU')],
              'channels': CHANNELS, 'class_names': CLASS_NAMES,
              'preprocessing': {'raw_hz': 700, 'target_hz': 70, 'resampling': 'scipy.signal.resample_poly',
                                'window_seconds': 10, 'overlap': 0.5, 'single_class_purity': 0.95,
                                'scaler_fit_partition': 'train', 'scaler_fit_samples': expected},
              'model_parameters': model.count_params(), 'optimizer': 'AdamW', 'weight_decay': 0.004,
              'attention': 'tanh(Dense(1)) with shared scalar bias; softmax over real time samples',
              'inspections': inspections, 'history': history.history,
              'confusion_matrix': confusion_matrix(truth, predicted, labels=classes).tolist(),
              'classification_report': classification_report(truth, predicted, labels=classes,
                                        target_names=CLASS_NAMES, output_dict=True, zero_division=0)}
    (out / 'pilot_report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('PILOT PASSED. These tiny-sample metrics are not research results.', flush=True)
    return result


if __name__ == '__main__':
    run_pilot()

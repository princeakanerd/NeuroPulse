import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# --- Suppress TensorFlow INFO and WARNING messages ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# --- Configuration for MUSE ---
# IMPORTANT: This path should point directly to your 'muse_v3.csv' file.
MUSE_FILE_PATH = r"F:\BTP_M\muse_v3.csv"

# Number of emotion classes.
# Note: The mapping below creates 3 classes (low, medium, high valence).
NUM_CLASSES = 3

BATCH_SIZE = 64
EPOCHS = 100

# --- Data Loading and Preprocessing for MUSE ---
def load_and_preprocess_muse_from_csv(file_path):
    """
    Loads and preprocesses the MUSE dataset from a single CSV file.
    It handles feature selection and mapping a numerical label to classes.
    """
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please check the path.")
        return np.array([]), np.array([])

    print(f"Loading and preprocessing data from {file_path}...")

    try:
        df = pd.read_csv(file_path)

        # The error messages confirm the correct label column is 'valence_tags'
        label_column = 'valence_tags'

        if label_column not in df.columns:
            print(f"Error: '{label_column}' column not found in the CSV file.")
            return np.array([]), np.array([])

        # Drop columns that are not useful for a numeric model (like text-based track, artist, etc.)
        # and other non-numeric columns. You can adjust this list.
        # In the load_and_preprocess_muse_from_csv function
        columns_to_drop = [
        'lastfm_url', 'track', 'artist', 'seeds', 'spotify_id', 'genre', 'mbid'
        ]

        # Identify columns to keep as features
        features = df.drop(columns=columns_to_drop, errors='ignore')

        # Let's map the numerical 'valence_tags' values into three categories for classification.
        # This mapping is based on common practice for sentiment analysis on a 1-5 scale,
        # but you might need to adjust it based on the exact range of your data.
        def map_valence(val):
            if val <= 2.0:
                return 0 # Low Valence
            elif val <= 3.5:
                return 1 # Medium Valence
            else:
                return 2 # High Valence

        # Create the features and labels
        X = features.drop(label_column, axis=1)
        y = features[label_column].apply(map_valence)

        # Drop any rows with missing data
        df_final = pd.concat([X, y], axis=1).dropna()

        X_data = df_final.drop(label_column, axis=1).values
        y_data = df_final[label_column].values

    except Exception as e:
        print(f"Error loading or processing {file_path}: {e}.")
        return np.array([]), np.array([])

    print(f"Finished loading data. Features shape: {X_data.shape}, Labels shape: {y_data.shape}")
    return X_data, y_data

# --- Load and Prepare Data ---
X_data, y_data = load_and_preprocess_muse_from_csv(MUSE_FILE_PATH)

if X_data.size == 0:
    print("\nError: No data loaded from the CSV file. Please check MUSE_FILE_PATH and column names.")
    exit()

# Split the data into training, validation, and testing sets
X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42, stratify=y_data
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_raw, y_train_raw, test_size=0.2, random_state=42, stratify=y_train_raw
)

# --- Data Normalization (StandardScaler) ---
print("\nFitting StandardScaler on training data...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_raw)
print("StandardScaler fit complete.")

# One-Hot Encode Labels
train_label = tf.keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
val_label = tf.keras.utils.to_categorical(y_val, num_classes=NUM_CLASSES)
test_label = tf.keras.utils.to_categorical(y_test_raw, num_classes=NUM_CLASSES)

# --- Model Definition (Multi-Layer Perceptron) ---
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_data.shape[1],)),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# --- Callbacks for Robust Training ---
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
model_checkpoint = ModelCheckpoint(filepath='best_model_muse_mlp.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
callbacks_list = [early_stop, model_checkpoint]

# --- Model Training ---
print("\n--- Starting Model Training ---")
history = model.fit(
    X_train_scaled,
    train_label,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=1,
    validation_data=(X_val_scaled, val_label),
    callbacks=callbacks_list
)
print("--- Model Training Finished ---")

# --- Evaluate and Visualize Results ---
print("\n--- Evaluating Model on Test Data ---")
try:
    from keras.models import load_model
    model_best = load_model('best_model_muse_mlp.keras')
    score = model_best.evaluate(X_test_scaled, test_label, verbose=0)
    print(f"\nTest Accuracy (Best Model): {score[1]:.4f}")
except Exception as e:
    print(f"\nCould not load best model ('best_model_muse_mlp.keras'). Error: {e}")
    print("Evaluating the model from the last training epoch instead.")
    score = model.evaluate(X_test_scaled, test_label, verbose=0)
    print(f"\nTest Accuracy (Last Epoch Model): {score[1]:.4f}")
    model_best = model

y_pred = model_best.predict(X_test_scaled, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test_raw

# Define class names based on your mapping
class_names = ['Low Valence', 'Medium Valence', 'High Valence']
cm = confusion_matrix(y_true, y_pred_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Confusion Matrix for MUSE Classification")
plt.show()

print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred_classes, target_names=class_names, zero_division=0))

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
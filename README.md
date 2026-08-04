# Stress, Affect, and Activity Recognition

A deep learning research project exploring stress detection, affective descriptors, cognitive states, and human activity using physiological and sensor data.

The project implements a Hybrid Attention-Based CNN-BiLSTM Classifier (HABC), combining parallel convolutional and recurrent branches with attention and feature fusion.

## Model Overview

The architecture consists of:

- **CNN branch:** learns patterns from the input representation.
- **BiLSTM branch:** models relationships across encoded input positions in both directions.
- **Attention mechanism:** assigns weights to recurrent hidden states.
- **Feature fusion and classification:** combines branch embeddings and predicts class probabilities through dense layers and softmax.

Input representations and output classes depend on the dataset and experiment.

## Datasets

| Dataset | Focus | Input |
| --- | --- | --- |
| WESAD | Stress and affect classification | Wearable physiological signals |
| SWELL-KW | Cognitive-state analysis | HRV and behavioral features |
| MuSe | Affective-descriptor classification | Valence, arousal, dominance, and tag counts |
| UTD-MHAD | Human activity recognition | Skeleton joint recordings |

Dataset files are organized under `datasets/`. WESAD recordings must be obtained separately and placed in `datasets/WESAD/`. Refer to each notebook for its required input files and paths.

## Repository Structure

```text
codes/                         Python scripts and supporting notebooks
datasets/                     Dataset files and loading information
MuSe_Corrected_Colab.ipynb      MuSe training and evaluation
UTD1_Experiment.ipynb          UTD-MHAD experiments
Wesad1_Experiment.ipynb         WESAD experiments
Wesad_Colab_Run.ipynb           WESAD Colab launcher
Wesad_Corrected_Pilot.ipynb     WESAD pilot experiment
Muse1_Experiment.ipynb          MuSe experiments
Swell_KW_1_Experiment.ipynb     SWELL-KW experiments
Copy_of_BTP_UTD (1).ipynb       UTD-MHAD implementation
requirements.txt               Python dependencies
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/Jasleenxddd/Stress_detection_model.git
cd Stress_detection_model
```

Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m jupyter notebook
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m jupyter notebook
```

Open the notebook for the desired dataset, set the dataset paths, and run the cells in order. Colab notebooks can also be uploaded to Google Colab.

To launch the WESAD evaluation suite from the project root:

```powershell
.\.venv\Scripts\python.exe -u codes/wesad_suite.py
```

## Experiments

The code includes dataset-specific preprocessing, model training, baseline comparisons, component ablations, and evaluation using accuracy, precision, recall, F1-score, and confusion matrices. Selected experiments also examine subject-based generalization, additive noise, and missing input values.

MuSe classes are defined from valence-arousal regions and are evaluated as a descriptor classification task. Interpret each experiment according to its input representation and evaluation protocol.

## Technology Stack

Python, TensorFlow/Keras, NumPy, Pandas, SciPy, scikit-learn, Matplotlib, Seaborn, Jupyter Notebook, and Google Colab.

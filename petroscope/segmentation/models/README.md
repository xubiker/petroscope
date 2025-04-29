# Segmentation Models

This directory contains segmentation models and a unified training pipeline for the Petroscope project.

## Architecture

The segmentation models are organized with a common base class and model-specific implementations:

```
models/
├── abstract.py            # Abstract GeoSegmModel base class
├── base.py                # PatchSegmentationModel implementation
├── train.py               # Unified training script for all models
├── config.yaml            # Common configuration file for all models
├── resunet/               # ResUNet model
│   ├── model.py           # ResUNet model class
│   └── nn.py              # ResUNet neural network architecture
├── pspnet/                # PSPNet model
│   ├── model.py           # PSPNet model class
│   └── nn.py              # PSPNet neural network architecture
└── upernet/               # UPerNet model
    ├── model.py           # UPerNet model class
    └── nn.py              # UPerNet neural network architecture
```

## Patch Segmentation Model

The `PatchSegmentationModel` class in `base.py` provides common functionality for all patch-based segmentation models:

- Training pipeline with data balancing
- Model evaluation
- Checkpoint saving and loading
- Prediction methods

## Model-Specific Implementations

Each model extends the base class and implements:

- Model architecture initialization
- Model-specific checkpoint data
- Pre-trained model loading

## Unified Training Pipeline

The unified training script (`train.py`) can be used to train any of the supported models:

```bash
# Train ResUNet model
python -m petroscope.segmentation.models.train model_type=resunet

# Train PSPNet model
python -m petroscope.segmentation.models.train model_type=pspnet

# Train UPerNet model
python -m petroscope.segmentation.models.train model_type=upernet
```

## Configuration

All models share a single configuration file (`config.yaml`) that contains settings for all model types. The configuration includes a `model_type` parameter that specifies which model to use. You can override any configuration parameter using Hydra's command-line syntax:

```bash
python -m petroscope.segmentation.models.train model_type=pspnet train.batch_size=64 train.epochs=100
```

## Adding a New Model

To add a new segmentation model:

1. Create a new directory for your model (e.g., `models/newmodel/`)
2. Create the model class that extends `PatchSegmentationModel`
3. Implement the neural network architecture
4. Update the `create_model` function in `train.py` to support your model
5. Add model-specific parameters to the shared `config.yaml` file

## Data Balancing

All models use the `SelfBalancingDataset` for training, which ensures balanced class representation in the training data. The balancing parameters can be configured in the configuration file.

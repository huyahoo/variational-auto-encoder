# AutoEncoder Modes

This script allows you to train, evaluate, sampling, and interpolate an AutoEncoder model in PyTorch on the MNIST dataset.

## Arguments
- --mode: **Required.** Mode to run the AutoEncoder. Options: "train", "evaluate", "sampling", "interpolate".
- --alpha: Alpha value for interpolation. Only used in "interpolate" mode. Default is 0.5.
- --noise_level: Noise level for adding noise to the images. Default is 0.5.
- --latent_dim: Latent dimension for the AutoEncoder model. Default is 10.
- --lr: Learning rate for the AutoEncoder model. Default is 0.001.
- --epochs: Number of epochs for training the AutoEncoder model. Default is 10.
- --batch_size: Batch size for training the AutoEncoder model. Default is 64.

## Modes
- train: Train the AutoEncoder model.
- evaluate: Evaluate the AutoEncoder model.
- sampling: Sample from the latent space of the AutoEncoder model.
- interpolate: Interpolate between two points in the latent space of the AutoEncoder model using the specified alpha value.

## Usage
``` $bash
python3 autoencoder.py --mode <mode> [--alpha <alpha>] [--noise_level <noise_level>] [--latent_dim <latent_dim>] [--lr <lr>] [--epochs <epochs>] [--batch_size <batch_size>]
```

## Example

``` $bash
python3 autoencoder.py --mode train

python3 autoencoder.py --mode evaluate

python3 autoencoder.py --mode sampling

python3 autoencoder.py --mode interpolate
```
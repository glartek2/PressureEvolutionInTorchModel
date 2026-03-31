# Models Documentation

This directory contains all core deep learning components used in the project:

* **Autoencoder** – learns latent representation of snake images
* **Classifier** – predicts whether a snake is venomous

These two models interact during the evolutionary experiment:

* Autoencoder defines the **search space**
* Classifier defines the **optimization objective - discriminator**

---

# Structure
| model
|| evolution
||| evolution_engine.py           # engine running evolution experyment
||| fitness.py                    # calculating fitness using classifier
||| mutate.py                     # doing mutation in latent space
||| population.py                 # generate/load from dataset images to run through evolution
|
|| generative
||| autoencoder.py                # definition of autoencoder and its components (Block, Encoder, Decoder)
||| latent_utils.py               # transform functions to augment data
||| train_auto.py                 # autoencoder training function
|
|| architecture.py                # load model that will be base for classifier (ResNet-50)
|| dataset.py                     # load dataset and return train and test sets
|| utils.py                       # transform functions to augment data - (normilzed to ResNet-50)
|| weights                        # saved .pt weights of trained models, becuase of file size I can't commit last v.
|| classifier
||| train.py                      # training - fine tuning ResNet to become snake classifier

---

# Autoencoder

## Purpose

The Autoencoder learns a compressed latent representation of snake images and reconstructs them back.

It is used to:

* map images → latent space (encoder)
* map latent vectors → images (decoder)
* enable evolution directly in latent space

---

## Architecture

The model is a **convolutional encoder-decoder with residual connections**.

### Key components:

#### Encoder

* Progressive downsampling via strided convolutions
* Residual blocks (`ResBlock`)
* Group Normalization + SiLU activation
* Final projection to latent space via `1x1 Conv`

#### Decoder

* Upsampling via nearest-neighbor interpolation
* Residual blocks for refinement
* Final `tanh` output

#### Latent Space

* Dimensionality: `256`
* Spatial latent (not flattened due to previous attempts with lack of information)

---

### Residual Block

```python
x = Conv → GN → SiLU → Conv → GN → skip connection
```

Helps with:

* stable training
* deeper architecture

---

### (Optional) Attention

The architecture includes an `AttentionBlock`, but it is currently disabled (I commented it because even without attention training was long enough).

---

## Training

### Loss Function

* **L1 Loss (primary)**

```python
loss = L1(reconstruction, input)
```

Chosen because:

* produces sharper images than MSE
* better preserves textures

Tried mix loss (0.8 l1 and 0.2 mse, but again calculating was a little too much for my hardware)

---

### Optimization

* Optimizer: `AdamW`
* Learning rate: `2e-4`
* Weight decay: `1e-4`

---

### Training Details

* Mixed precision training (`torch.amp`)
* Gradient clipping (`max_norm=10`) - prevent gradient overflow but adds a lot of overhead.
* Channels-last memory format (performance optimization)

---

### Data Pipeline

* Custom dataset loader (`get_datasets`)
* Augmentations via `get_transforms` (generative/latent_utils):
  * transforms.Resize
  * RandomResizedCrop
  * RandomHorizontalFlip
  * RandomRotation
  * ColorJitter
  * RandomGrayscale
  * ToTensor
  * Normalize

---

## Results

The Autoencoder successfully learns:

* global structure of snakes
* dominant colors and textures

However:
* model captures background along with the snake

Limitations:
* No disentangled latent space
* Background leakage into representation
* Limited training epochs (compute constraints)

---

# Classifier

## Purpose

Binary classifier that predicts:

* `0 → non-venomous`
* `1 → venomous`

It is later used as a **fitness signal** in the evolutionary process.

---

## Architecture

* Backbone defined in `get_model()`
  (pretrained CNN ResNet-50)

* Final layer:
  * Fully connected (`fc`) → 2 classes

---

## Training Strategy

Training is done in **two phases**:

---

### Phase 1 – Train classifier head

* Freeze entire model
* Train only final FC layer

```python
freeze(model)
train(model.fc)
```

* Optimizer: `Adam`
* Learning rate: `1e-3`
* Epochs: `3`

---

### Phase 2 – Fine-tuning

* Unfreeze `layer4` (top convolutional block)

```python
unfreeze_layer4(model)
```

* Train:

  * `layer4` (low LR)
  * `fc` (higher LR)

```python
lr(layer4) = 1e-5
lr(fc)     = 1e-4
```

* Epochs: `15`

---

## Optimization

* Loss: `CrossEntropyLoss`
* Scheduler: `CosineAnnealingLR`
* Optimizer:

  * Phase 1: Adam
  * Phase 2: Adam (layer-wise LR)

---

## Evaluation

Metrics:

* Accuracy
* Confusion matrix
* Classification report (precision / recall / F1)

```python
print(classification_report(...))
print(confusion_matrix(...))
```

---

## Observations

* Model achieves reasonable classification performance
* However, it tends to rely on:

  * background features
  * lighting conditions
  * color distribution

This bias significantly affects downstream evolution.

---

# Integration in Evolution

Pipeline:

```python
image → encoder → latent vector
latent vector → decoder → image
image → classifier → probability
```

* Evolution operates entirely in latent space
* Classifier output defines component of fitness

---

# Known Issues

* Latent space is not semantically structured
* Classifier is biased toward non-biological features
* Reconstruction artifacts propagate into evolution

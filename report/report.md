# Homework 3: PyTorch Training Pipeline Report

**Student Name:** Muha Ioana

**Course:** Advanced Topics in Neural Networks

**Due Date:** Week 9

---

## 1. Setup & How to Run

### Installation
To replicate the environment, please use the provided `requirements.txt`.

```bash
# 1. Install miniconda and create a clean Conda environment (Recommended Python 3.10)
conda create -n hw3 python=3.10
conda activate hw3

# 2. Install dependencies inside the environment and use that environment as project interpreter
pip install -r requirements.txt
```

### Running the Pipeline
The pipeline is fully configurable via command line arguments or YAML configuration files.

**Basic Training Run:**
```bash
# Run with default configuration (CIFAR-10, ResNet18)
python main.py --config config_cifar10.yaml

# Run with overrides (e.g., specific model and optimizer)
python main.py --dataset cifar100 --model_name resnet50 --optimizer adamw --num_epochs 5
```

---

## 2. Pipeline Architecture

I have implemented a modular, device-agnostic training pipeline using PyTorch. The project structure separates concerns into `data`, `models`, `train`, and `utils` modules.

### Key Features Implemented:
* **Device Agnostic:** The code automatically detects and utilizes CUDA if available, falling back to CPU otherwise.
* **Configurable Datasets:** Support is implemented for `MNIST`, `CIFAR-10`, `CIFAR-100`, and `OxfordIIITPet`.
    * *Implementation:* See `data/datasets.py`. Logic handles grayscale-to-RGB conversion for MNIST.
* **Model Support:** The pipeline integrates `timm` to support `resnet18`, `resnet50`, `resnest14d`, `resnest26d`. A custom `SimpleMLP` class was implemented to satisfy the MLP requirement.
* **Advanced Optimization:**
    * **Optimizers:** Factory supports `SGD`, `Adam`, `AdamW`, `SAM`, and `Muon`.
    * **Schedulers:** Integrated `StepLR` and `ReduceLROnPlateau`.
* **Metrics Reporting:** Integrated with **WandB** (and Tensorboard) to log training/validation loss and accuracy in real-time.
* **Early Stopping:** A custom `EarlyStopping` class monitors validation loss and checkpoints the best model, halting training if no improvement is seen after a defined patience.

---

## 3. Efficiency Analysis

To satisfy the efficiency requirements, I implemented several optimizations to maximize throughput on limited hardware.

### Measurements & Motivation

1.  **Data Loading Efficiency:**
    * Used `num_workers=4` and `pin_memory=True` in DataLoaders to ensure the GPU is not starved of data by the CPU.
    * *Impact:* Reduces data loading bottlenecks significantly.

2.  **Batch Size Scheduler:**
    * Implemented a mechanism to dynamically increase batch size at specific epochs. This allows the training to start with smaller, noisier batches for exploration and shift to larger batches for stability, maximizing GPU memory usage in later stages.

3.  **Automatic Mixed Precision (AMP):**
    * Implemented `torch.amp.GradScaler` and `autocast`.
    * *Note on Stability:* While the code fully supports AMP, I observed numerical instability (NaN loss) during specific high-LR experiments with the SAM optimizer on my hardware. For those specific runs, I utilized a flag `use_amp: False` to prioritize stability over speed.

---

## 5. Experimental Results: Pretraining vs. Scratch

I conducted a targeted experiment to compare the performance of training from scratch versus transfer learning on CIFAR-100.

### A. No Pretraining (From Scratch)
* **Strategy:** Used `ResNeSt26d` with the `SAM` optimizer to maximize generalization. Trained for 100 epochs.
* **Best Accuracy:** **66.02%**

### B. With Pretraining (Transfer Learning)
* **Strategy:** Used `ResNet50` pretrained on ImageNet. Images were resized to 224x224 to match the pretrained input resolution.
* **Best Accuracy:** **79.76%**

### Comparison Table

| Strategy | Model | Image Size | Epochs | Best Accuracy |
| :--- | :--- | :--- |:-------|:--------------|
| **No Pretraining** | ResNeSt26d | 32x32 | 100    | **66.02%**    |
| **Pretraining** | ResNet50 | 224x224 | 10     | **79.76%**    |


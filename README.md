# Anti-Spoofing with LCNN

This project detects whether an audio recording contains real human speech or speech generated or modified by computer systems.

The model was trained and evaluated on the **Logical Access part of the ASVspoof 2019 dataset**. The main evaluation metric is **Equal Error Rate (EER)**. A lower EER means that the model separates real and fake recordings more accurately.

## Model

The project uses a **Light Convolutional Neural Network (LCNN)**.

Before an audio recording is passed to the model, it is converted into **LFCC features**. These features describe important frequency information in the audio signal.

The model includes:

- Max-Feature-Map convolutional layers;
- max-pooling layers;
- batch normalization;
- dropout;
- a final classification layer.

All recordings are processed at a sample rate of 16 kHz. Each recording is converted to a fixed length of **112,000 samples**, which is equal to 7 seconds.

Main implementation files:

```text
src/model/lcnn.py
src/transforms/lfcc.py
src/datasets/asvspoof_dataset.py
src/loss/weighted_cross_entropy.py
src/metrics/eer.py
```

## Dataset

The project uses the **ASVspoof 2019 Logical Access dataset**.

The dataset contains three parts:

- `train` — used to train the model;
- `dev` — used to check model quality during training and select the best checkpoint;
- `eval` — used for the final evaluation.

The dataset can be added to a Kaggle Notebook from:

[ASVspoof 2019 Dataset on Kaggle](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset)

## Project structure

```text
.
├── train.py
├── inference.py
├── requirements.txt
├── saved/
│   └── lcnn_lfcc_final/
│       └── model_best.pth
└── src/
    ├── configs/       # Training and inference settings
    ├── datasets/      # Dataset loading
    ├── logger/        # Experiment logging
    ├── loss/          # Loss functions
    ├── metrics/       # Accuracy and EER
    ├── model/         # LCNN model
    ├── trainer/       # Training and inference logic
    ├── transforms/    # LFCC feature extraction
    └── utils/         # Supporting functions
```

## Running the project in Kaggle

Create a new Kaggle Notebook, enable a GPU accelerator and add the ASVspoof 2019 dataset to the notebook inputs.

### 1. Clone the repository

```python
!git clone https://github.com/kirillstrx/dl-voice-project.git
%cd dl-voice-project
```

### 2. Install the required packages

```python
!pip install -q hydra-core torchmetrics wandb comet_ml soundfile
```

### 3. Set the dataset path

```python
DATA_ROOT = "/kaggle/input/datasets/awsaf49/asvpoof-2019-dataset/LA/LA"
```

### 4. Configure Weights & Biases

Training metrics can be saved to Weights & Biases.


```python
!wandb login your_login
```

## Training

Run the following command to train the model:

```python
!python -u train.py \
    -cn=asvspoof_lfcc \
    trainer.device=cuda \
    trainer.n_epochs=30 \
    datasets.train.root={DATA_ROOT} \
    datasets.dev.root={DATA_ROOT} \
    datasets.train.num_samples=112000 \
    datasets.dev.num_samples=112000 \
    datasets.train.limit=null \
    datasets.dev.limit=null \
    dataloader.batch_size=64 \
    writer.mode=online \
    writer.run_name=lcnn_lfcc_final
```

During training, the project calculates loss, accuracy and EER on the development dataset. The checkpoint with the lowest development EER is saved as the best model:

```text
saved/lcnn_lfcc_final/model_best.pth
```

Run the following command to evaluate the best model:

```python
!TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python -u inference.py \
    -cn=asvspoof_lfcc_inference \
    datasets.eval.root={DATA_ROOT} \
    datasets.eval.limit=null \
    dataloader.batch_size=64 \
    inferencer.device=cuda \
    inferencer.save_path=asvspoof_lfcc_7s_eval \
    inferencer.csv_name=predictions.csv
```

This command loads the best checkpoint, runs the model on the full evaluation dataset and saves the prediction scores to:

```text
data/saved/asvspoof_lfcc_7s_eval/predictions.csv
```

## Results

The best model was evaluated on the full ASVspoof 2019 LA evaluation dataset.

| Metric | Result |
|---|---:|
| Accuracy | 91.97% |
| EER | 3.85% |


## References

The project is based on the following PyTorch project template:

[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)

## Author

Kirill Strashnov

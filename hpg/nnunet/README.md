# nnU-Net Baseline (HiPerGator)

This repository provides setup instructions, dataset preparation guidelines, and Slurm batch scripts for running **nnU-Net v2** on the **HiperGator** HPC cluster 

The repository contains code only. Rat data and model results live on HPG group storage. Follow the shared [data contract](../../docs/DATA_CONTRACT.md) for class IDs and prediction format.

## 1. Repository Setup on HiPerGator

Log into HiPerGator via SSH or Open OnDemand terminal, navigate to your team's storage directory, clone the repository, and switch to the `feature/nnunet` branch:

```bash
# Navigate to your personal folder under the dream_team group allocation
cd /blue/dream_team/<username>

# Clone the TOAD repository
git clone https://github.com/DreamTeamSE/TOAD.git
cd TOAD

# Checkout the nnU-Net feature branch
git checkout feature/nnunet

```
## Environment Setup

Initial One-Time Setup
From the repository root (TOAD), run the environment setup script once to create the nnunet Conda environment from environment_nnunet.yml and verify required dependencies (PyTorch, CUDA, nnU-Net v2):

```bash 
hpg/nnunet/setup_env.sh
```

Routine Activation (Every Session Thereafter)
When you open a new terminal session or start an interactive job on HiPerGator, simply load Conda and activate the environment:

```bash
  module load conda
  conda activate nnunet
```

nnU-Net v2 relies on three core environment variables to locate raw data, preprocessed data, and trained model weights. 
While these were exported while creating the Conda encironment, they do not carry over to later shells or SLURM jobs. 

You can set them again in each job or interactive session:

```bash
export DATA_ROOT="/blue/dream_team/$USER/seko_storage"
export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
export nnUNet_results="$DATA_ROOT/nnUNet_results"

# Create directories if they do not exist
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```
Alternatively, add them to your ~/.bashrc file so HiPerGator automatically loads them every time you log in:

```bash
  echo 'export DATA_ROOT="/blue/<your-group>/$USER/seko_storage"' >> ~/.bashrc
  echo 'export nnUNet_raw="$DATA_ROOT/nnUNet_raw"' >> ~/.bashrc
  echo 'export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"' >> ~/.bashrc
  echo 'export nnUNet_results="$DATA_ROOT/nnUNet_results"' >> ~/.bashrc
  source ~/.bashrc
```

## Data Conversion

Dataset Structuring (Dataset101_KneeHistology)
nnU-Net requires a strict folder layout and file naming convention:
$nnUNet_raw/
└── Dataset101_KneeHistology/
    ├── dataset.json
    ├── imagesTr/
    │   ├── slide_001_0000.png
    │   ├── slide_002_0000.png
    │   └── ...
    ├── labelsTr/
    │   ├── slide_001.png
    │   ├── slide_002.png
    │   └── ...
    └── imagesTs/
        ├── slide_078_0000.png
        └── ...
Image Files (_0000.png / .tif): Input RGB histology channels must end with a 4-digit channel identifier (e.g., _0000.png).
Label Masks (.png): Pixel values must directly correspond to integer class IDs:
0: Background
1: Bone
2: Cartilage
3: Growth Plate
4: Marrow
5: Osteophyte


## Slurm Batch Job Scripts for HiPerGator

Submit computational tasks (preprocessing, training, and inference) using sbatch.

# Step 1: Preprocessing (submit_preprocess.sh)
```bash
  #!/bin/bash
  #SBATCH --job-name=nnunet_preprocess
  #SBATCH --output=logs/preprocess_%j.log
  #SBATCH --error=logs/preprocess_%j.err
  #SBATCH --account=dream_team
  #SBATCH --qos=dream_team
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=8
  #SBATCH --mem=32gb
  #SBATCH --time=04:00:00

  module load conda
  conda activate nnunet

  export DATA_ROOT="/blue/dream_team/$USER/seko_storage"
  export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
  export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
  export nnUNet_results="$DATA_ROOT/nnUNet_results"
```

# Plan and preprocess dataset ID 101

``` bash
  nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity
  Submit job:
  sbatch submit_preprocess.sh
```

# Step 2: Training (submit_train.sh)

```bash 
  #!/bin/bash
  #SBATCH --job-name=nnunet_train_2d
  #SBATCH --output=logs/train_%j.log
  #SBATCH --error=logs/train_%j.err
  #SBATCH --account=dream_team
  #SBATCH --qos=dream_team
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=8
  #SBATCH --gpus=1
  #SBATCH --mem=32gb
  #SBATCH --time=30:00:00

  module load conda
  module load cuda/12.1
  conda activate nnunet

  export DATA_ROOT="/blue/dream_team/$USER/seko_storage"
  export nnUNet_raw="$DATA_ROOT/nnUNet_raw"
  export nnUNet_preprocessed="$DATA_ROOT/nnUNet_preprocessed"
  export nnUNet_results="$DATA_ROOT/nnUNet_results"

  # Train 2D configuration on Fold 0
  nnUNetv2_train 101 2d 0

Submit job:
  sbatch submit_train.sh
```
# Step 3: Inference on Test Set (submit_predict.sh)
```bash
  #!/bin/bash
  #SBATCH --job-name=nnunet_predict
  #SBATCH --output=logs/predict_%j.log
  #SBATCH --error=logs/predict_%j.err
  #SBATCH --account=dream_team
  #SBATCH --qos=dream_team
  #SBATCH --nodes=1
  #SBATCH --ntasks=1
  #SBATCH --cpus-per-task=4
  #SBATCH --gpus=1
  #SBATCH --mem=16gb
  #SBATCH --time=02:00:00

  module load conda
  module load cuda/12.1
  conda activate nnunet

  export DATA_ROOT="/blue/dream_team/\$USER/seko_storage"
  export nnUNet_raw="\$DATA_ROOT/nnUNet_raw"
  export nnUNet_preprocessed="\$DATA_ROOT/nnUNet_preprocessed"
  export nnUNet_results="\$DATA_ROOT/nnUNet_results"

  INPUT_DIR="\$nnUNet_raw/Dataset101_KneeHistology/imagesTs"
  OUTPUT_DIR="outputs/nnunet_predictions"

  # Run inference
  nnUNetv2_predict -i \\(INPUT_DIR -o \\)OUTPUT_DIR -d 101 -c 2d -f 0
```
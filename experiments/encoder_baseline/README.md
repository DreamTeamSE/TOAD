# ResNet-34 U-Net Baseline

Purpose: This experiment establishes an initial PyTorch based image segmentation inference pipeline for TOAD, separate from the Keras/SEKO pipeline we inherited from the lab.

Architecture:
    Framework: PyTorch
    Model: U-Net from SMP
    Encoder: ResNet-34
    Decoder: Standard decoder for U-Net from SMP
    Encoder Weights: Random
    Input: 3-channel image (192x256)
    Output: Five class scores per pixel

Classes:
    0: Background
    1: Bone
    2: Cartilage
    3: Growth Plate
    4: Bone Marrow
(Osteophyte labels mapped to cartilage)

# Running Test 

- Activate toad-pytorch Conda environment and run from repository root: python experiments/encoder_baseline/run_inference.py --input-dir PATH_TO_IMAGES --output-dir outputs/encoder_baseline --limit 2
    Replace PATH_TO_IMAGES with the path to a directory containing files

Script saves one single channel PNG prediction mask per processed image

# Status/Limitations 
- Forward pass/output dimensions tested succesfully
- Image loading and mask-saving tested using synthetic images
- UNTRAINED & used random weights -> Predictions are not yet meaningful tissue segmentations
- No real data yet (waiting on access to HiperGator)

# Next Steps
- Run inference test on actual TOAD images when dataset/HiperGator access becomes available
- Confirm preprocessing and mask formatting with actual histology images
- Validate saves masks (using actual histology images) against DATA_CONTRACT.md
- Extend experiment to accept configurable encoders and test each against SEKO baseline & possible nnUnet baseline
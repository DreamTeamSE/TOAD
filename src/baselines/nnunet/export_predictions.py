"""
export_predictions.py — nnU-Net prediction output -> docs/DATA_CONTRACT.md format.

Planned responsibilities:
  - Take nnU-Net's native nnUNetv2_predict output (run on imagesTs) and convert
    it to the exact docs/DATA_CONTRACT.md spec: one PNG per input image, same
    filename stem as the input, single-channel, 8-bit, pixel values = class
    IDs 0-4.
  - Since dataset.json's labels already map 1:1 to the contract, this should
    mostly be a format/naming pass-through — explicitly verify:
      * bit depth is 8-bit (not nnU-Net's sometimes-16-bit output)
      * single-channel (not an RGB visualization)
      * filenames don't carry extra nnU-Net-added suffixes that would break
        stem-matching against the input images
"""

raise NotImplementedError("export_predictions.py scaffold stub")

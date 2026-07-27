# PLNet

Official training and evaluation code for **PLNet: An Efficient Parameter Aggregation Network
for Multimodal Whole Heart Segmentation**.

PLNet combines a Pyramid Pooling Transformer encoder with a Large Spatial
Calibration encoder and uses Learnable Residual Enhancement and Edge-Aware
Blocks for lightweight decoding.

## Data

The dataset is not included in this repository. Place MM-WHS2017 CT/MR data
under `data/`. The 40/5/15 train/validation/test splits are defined in:

- `configs/data/manifests/mmwhs_ct.json`
- `configs/data/manifests/mmwhs_mr.json`

## Training

```bash
bash scripts/train_mmwhs_ct.sh
bash scripts/train_mmwhs_mr.sh
```

The default setup uses an `80×160×160` crop, AdamW, 40,000 iterations, batch
size 1, initial learning rate `1e-4`, and weight decay `5e-4`.

## Evaluation

```bash
bash scripts/test_mmwhs_ct.sh --checkpoint /path/to/ct_checkpoint.pth
bash scripts/test_mmwhs_mr.sh --checkpoint /path/to/mr_checkpoint.pth
```

Evaluation uses the original sliding-window overlap and `0.5` sigmoid
threshold. Predictions are saved as class indices `0-7`; add `--raw_labels`
to save the original MM-WHS label values.

Labels are ordered as background, myocardium, left atrium, left ventricle,
right atrium, right ventricle, ascending aorta, and pulmonary artery. The main
output uses Dice and BCE losses; three decoder outputs use class-balanced BCE
for deep supervision with weights `0.5`, `0.25`, and `0.25`.

## License

Original PLNet contributions are released under the
[Apache License 2.0](LICENSE). Third-party-derived portions remain subject to
their upstream terms; see [NOTICE](NOTICE) for attribution and details.

# Food-101 Fine-Grained Classification (PyTorch)

End-to-end notebook workflow comparing three CNN approaches on the Food-101 dataset:
- Baseline CNN (scratch)
- Custom CNN (BatchNorm + GlobalAvgPool)
- EfficientNet-B0 (two-phase transfer learning)

## Results (latest)
- Baseline CNN: 0.99% top-1, 4.95% top-5, loss 4.6152
- Custom CNN: 57.19% top-1, 81.82% top-5, loss 1.7600
- EfficientNet-B0: 79.00% top-1, 93.43% top-5, loss 0.8580
- Best val accuracy: 0.7958 (EfficientNet-B0), 0.5742 (Custom), 0.0099 (Baseline)
- Split: stratified 80/10/10 (train/val/test)

## Repository Layout
- `food101_data_pipeline.ipynb` — main notebook with data loading, augmentation, training, evaluation, and plots
- `Baseline_CNN_best.pt`, `Custom_CNN_best.pt`, `EfficientNet-B0_Phase_1_(Frozen_Backbone)_best.pt` — saved checkpoints

## Requirements
- Python 3.10+
- PyTorch with CUDA (recommended)
- torchvision, datasets (HuggingFace), scikit-learn, pandas, matplotlib, seaborn, tqdm

## Quickstart
1) Create and activate a virtual environment
```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```
2) Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install datasets scikit-learn pandas matplotlib seaborn tqdm
```
3) Open and run the notebook
- Launch VS Code / Jupyter and run `food101_data_pipeline.ipynb` top-to-bottom.

## Training Notes
- Data: pulls Food-101 via `datasets.load_dataset('food101')` and re-splits stratified 80/10/10.
- Augmentation: resize/crop, flip, small rotation, mild contrast jitter; normalization with ImageNet stats.
- Optimizer: Adam; optional ReduceLROnPlateau scheduler; early stopping with patience.
- EfficientNet-B0: Phase 1 trains head only; Phase 2 unfreezes top layers and fine-tunes at 0.1× LR.

## Reproducing Results
- Ensure CUDA is available; set `CONFIG['DEVICE']` auto-selects CUDA if present.
- NUM_WORKERS=0 is used in the current config; increase if your environment supports it for faster epochs.
- Run the full notebook to regenerate metrics and figures (confusion matrix, per-class accuracy, learning curves, comparison table).

## Export / Inference
- Saved checkpoints can be loaded with `torch.load` and standard PyTorch `state_dict` restore.
- For production, consider exporting EfficientNet-B0 to ONNX and adding test-time augmentation.

## License
MIT (add your preferred license if different).

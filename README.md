# Blurred-Image-restore-project

CNN-based blurred image restoration with three model options:
- Simple CNN
- Residual CNN
- U-Net

## Setup

```bash
pip install -r requirements.txt
```

## Run inference

```bash
python infer.py --input path/to/blurred.png --output path/to/restored.png --model unet
```

Supported `--model` values: `simple`, `residual`, `unet`.

If you have trained weights, provide them with `--weights`:

```bash
python infer.py --input path/to/blurred.png --output path/to/restored.png --model residual --weights path/to/model.pt
```

## Tests

```bash
python -m unittest tests/test_models.py
```

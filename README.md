# LayoutTransformer-JDC
Joint Generation of Discrete and Continuous Values using Autoregressive Models

## Baseline

### Training
```bash
cd layout_transformer_baseline
python main.py \
    --train_json path/to/train/json \
    --val_json path/to/val/json \
    --exp publaynet \
    --epochs 30
```

### Sample
```bash
# under layout_transformer_baseline
python sample.py \
    --ckpt path/to/pth/file \
    --json_path path/to/test/json
```

### Evaluation
```bash
# under layout_transformer_baseline
python eval.py \
    --ckpt path/to/ckpt/file \
    # ... other options for model
```

## With Joint Discrete-Continuous Generation
### Training
```bash 
cd layout_transformer_jdc
python main.py \
    --train_json path/to/train/json \
    --val_json path/to/val/json \
    --exp publaynet \
    --epochs 30 
```

### Sample 
```bash
# under layout_transformer_jdc 
python sample.py \
    --ckpt path/to/pth/file \
    --json_path path/to/test/json
```

### Evaluation
```bash 
# under layout_transformer_jdc 
python eval.py \
    --ckpt path/to/pth/file \
    # ... other options for model
```
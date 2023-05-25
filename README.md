# GAN

## Train

Use `train.py` with the following parameters:

```bash
 python train.py \
  --data {DATASET_PATH} \
  --outdir {OUT_DIR} \
  --epochs {NUM_EPOCHS} \
  --snap {SNAP} \
  --batch {BATCH_SIZE} \
  --size {IMAGE_SIZE} \
  --noise_dim {NOISE_DIM} \
  --lr {LEARNING_RATE} \
  --beta {BETA} \
  --seed {SEED} \
  --gen {GEN} \  # to continue training
  --disc {DISC} \  # to continue training
  --cuda  # or --mps on Apple Silicon      
```

## Generate

Use `generate.py` with the following parameters:

```bash
python generate.py \
  --parameters {PARAMS_PATH} \
  --gen {MODEL_PATH} \
  --seeds {SEEDS} \
  --outdir {OUT_DIR} \
  --cuda  # or --mps on Apple Silicon
```
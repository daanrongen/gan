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
  --gen {GEN_PATH} \  # to continue training
  --disc {DISC_PATH} \  # to continue training
  --cuda  # or --mps on Apple Silicon      
```

## Generate

Use `generate.py` with the following parameters:

```bash
python generate.py \
  --parameters {PARAMS_PATH} \
  --gen {GEN_PATH} \
  --seeds {SEEDS} \
  --outdir {OUT_DIR} \
  --cuda  # or --mps on Apple Silicon
```

To create a video of an animated interpolation inbetween seeds change the parameters to:

```bash
python generate.py \
  --parameters {PARAMS_PATH} \
  --gen {GEN_PATH} \
  --seeds {SEEDS} \
  --outdir {OUT_DIR} \
  --interpolate \
  --fps {FPS} \
  --frames {FRAMES} \
  --cuda  # or --mps on Apple Silicon
```
# GAN

## Train

Use `train.py` with the following parameters:

```bash
 python train.py \
  --data ./path/to/dataset \
  --out ./runs/ \
  --cuda # or --mps on Apple Silicon      
```

## Generate

Use `generate.py` with the following parameters:

```bash
python generate.py \
  --parameters ./path/to/parameters.json \
  --outdir ./out/ \
  --gen .path/to/gen_epoch_XX.pth \
  --seeds 1-10 \
  --cuda # or --mps on Apple Silicon
```


Profiler used: cProfile

Target script:  evaluate.py

The evaluation script was profiled using Python’s built-in cProfile, sorted by cumulative runtime:

```bash

rm -f evaluate.prof

uv run python -m cProfile -s cumulative -o evaluate.prof \
  src/anomaly_detection/evaluate.py \
  --data_root ./data/raw \
  --class_name carpet \
  --weights_path ./models/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --memory_bank_path ./models/memory_bank.pt \
  --output_dir ./reports/figures/eval \
  --img_size 224 \
  --batch_size 8 \
  --k 10

```

To inspect the most time-consuming functions:

```bash
uv run python -c "import pstats; p=pstats.Stats('evaluate.prof'); p.sort_stats('cumulative').print_stats(20)"
```


Visualization
The profiling results were visualized using SnakeViz:

```bash
uv add snakeviz
uv run snakeviz evaluate.prof
```



Training and evaluation scripts use Loguru for lightweight logging.

- Logs are printed to the terminal
- Logs are also written to `logs/`
- Log files are timestamped per run

This helps with debugging and reproducibility without cluttering the codebase.


Evaluation logs key metrics such as image-level and pixel-level ROC AUC.
Each run produces a timestamped log file in `logs/` for traceability.
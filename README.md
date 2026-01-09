# anomaly_detection

ML project for the 02476 MLOPS course at DTU. Using DINOv3 to find anomalies in object from images.

# Project Overview: Zero-Shot Anomaly Detection with Vision Foundation Models

This project explores whether cutting-edge vision foundation models can be used to detect anomalies in industrial settings-without ever seeing defective examples during training.

In real-world manufacturing, defective samples are often rare, expensive to collect, or simply unavailable when building a detection system. Our goal is to design and test a pipeline that relies only on normal (defect-free) images and requires no task-specific fine-tuning. By leveraging the rich visual features learned by these models, we aim to detect and pinpoint anomalies both at the image and pixel level, in a way that is both data-efficient and adaptable across different products.

We’ll start by using the MVTec Anomaly Detection (MVTec AD) dataset, a standard benchmark for unsupervised and zero-shot anomaly detection. It includes high-resolution images from various industrial categories, such as textures (carpet, leather) and objects (screws, bottles). The training set contains only normal images, while the test set includes both normal and defective images, complete with precise pixel-level annotations. This setup is ideal for evaluating our approach using metrics like image-level and pixel-level AUROC. While MVTec AD is our primary dataset, the method is designed to be flexible and could be extended to other industrial or real-world datasets in the future.

At the core of our approach is DINOv3, a self-supervised Vision Transformer trained without labeled data. We’ll use DINOv3 as a fixed feature extractor to obtain patch-level embeddings from input images. From these, we’ll build a “memory bank” of normal patch features using the training data. During inference, we’ll compute anomaly scores by comparing test image patches to this memory bank, using distance-based methods such as k-nearest neighbors (kNN). We might also explore other evaluation methods.

Ultimately, this project aims to show that powerful pretrained vision models can enable accurate, robust anomaly detection without the need for explicit supervision-opening the door to scalable industrial applications.



## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).
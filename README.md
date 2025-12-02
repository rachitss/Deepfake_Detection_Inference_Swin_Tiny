## About

The project uses a model based on swin tiny transformer, trained on a small DFDC sample dataset to predict if a video is fake or real.

### API service branch

Work on the `api-service` branch focuses on exposing the inference pipeline as a FastAPI service so external clients can submit video paths/URIs instead of relying on the desktop GUI. Keep changes isolated to this branch before merging to `main`.

### How to run inference

Run only the infer.py script to run an inference

### Python version

The codebase targets **Python 3.13.2**. Use this interpreter locally (pyenv/virtualenv/uv).

### Large Files (Git LFS)

This repository uses **Git LFS** to store model weights.

To clone the repository with all large files included, install Git LFS first.


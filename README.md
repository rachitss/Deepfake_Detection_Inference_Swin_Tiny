## About

The project uses a model based on swin tiny transformer, trained on a small DFDC sample dataset to predict if a video is fake or real.

### How to run inference

**CLI**

```
python -m app.infer --video path/to/video.mp4 --model-path swin_tiny.pth
```

**FastAPI service**

1. `pip install -r requirements.txt`
2. `python run_api.py` or simply run the script run_api.py.
3. Open `http://localhost:8000/docs`, choose `POST /inference`, upload a video, and execute the request. The service stores the upload in a temporary file, extracts faces, runs the transformer model, and returns a JSON response. All temporary assets are deleted after each request.

Responses contain `prediction` (0 real / 1 fake) and `probability`, which averages all sigmoid confidences collected across batches. The legacy Tkinter drag-and-drop flow has been commented out so uploads always pass through the CLI or FastAPI surface.

### Python version

The scripts rely on **Python 3.13.2**. Other python versions might cause compatibility errors.

### Large Files (Git LFS)

This repository uses **Git LFS** to store model weights.

To clone the repository with all large files included, install Git LFS first.




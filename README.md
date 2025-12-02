## About

The project uses a model based on swin tiny transformer, trained on a small DFDC sample dataset to predict if a video is fake or real.

### API service branch

Work on the `api-service` branch focuses on exposing the inference pipeline as a FastAPI service so external clients can submit video paths/URIs instead of relying on the desktop GUI.

### How to run inference

**CLI**

```
python -m app.infer --video path/to/video.mp4 --model-path swin_tiny.pth
```

**FastAPI service**

1. `pip install -r requirements.txt`
2. `uvicorn app.api_service:app --host 0.0.0.0 --port 8000`
3. Open `http://localhost:8000/docs`, choose `POST /inference`, upload a video (MP4/MOV/AVI/MKV), and execute the request. The service stores the upload in a temporary file, extracts faces, runs the transformer model, and returns a JSON response. All temporary assets are deleted after each request.

The legacy Tkinter drag-and-drop flow has been commented out so uploads always pass through the CLI or FastAPI surface.

### Python version

The scripts rely on **Python 3.13.2**. Other python versions might cause compatibility errors.

### Large Files (Git LFS)

This repository uses **Git LFS** to store model weights.

To clone the repository with all large files included, install Git LFS first.




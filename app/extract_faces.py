import os
import torch
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
 


from upload_video import get_video



class FaceExtractor:
    def __init__(self, detector, n_frames=None, resize=None):
        """
        Parameters:
            n_frames {int} -- Total number of frames to load. These will be evenly spaced
                throughout the video. If not specified (i.e., None), all frames will be loaded.
                (default: {None})
            resize {float} -- Fraction by which to resize frames from original prior to face
                detection. A value less than 1 results in downsampling and a value greater than
                1 result in upsampling. (default: {None})
        """

        self.detector = detector
        self.n_frames = n_frames
        self.resize = resize
    
    def __call__(self, filename, save_dir):
        """Load frames from an MP4 video, detect faces and save the results.

        Parameters:
            filename {str} -- Path to video.
            save_dir {str} -- The directory where results are saved.
        """

        # Create video reader and find length
        v_cap = cv2.VideoCapture(filename)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Pick 'n_frames' evenly spaced frames to sample
        if self.n_frames is None:
            sample = np.arange(0, v_len)
        else:
            sample = np.linspace(0, v_len - 1, self.n_frames).astype(int)

        # Loop through frames
        for j in tqdm(range(v_len), desc="Extracting faces"):
            success = v_cap.grab()
            if j in sample:
                # Load frame
                success, frame = v_cap.retrieve()
                if not success:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                
                # Resize frame to desired size
                if self.resize is not None:
                    frame = frame.resize([int(d * self.resize) for d in frame.size])

                save_path = os.path.join(save_dir, f'{j}.png')

                self.detector([frame], save_path=save_path)

        v_cap.release()



def get_frames():
    VID_PATH = get_video()
    TMP_DIR = 'MTCNN EXTRACTED/'

    # os.makedirs(VID_PATH, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)


    SCALE = 0.25
    N_FRAMES = None

    # Load face detector
    face_detector = MTCNN(margin=14, keep_all=True, factor=0.5, device=device).eval()

    # Define face extractor
    face_extractor = FaceExtractor(detector=face_detector, n_frames=N_FRAMES, resize=SCALE)

    with torch.no_grad():
        file_name = VID_PATH.split('/')[-1]

        save_dir = os.path.join(TMP_DIR, file_name.split(".")[0])

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Detect all faces appear in the video and save them.
        face_extractor(VID_PATH, save_dir)
        
        return save_dir
    


if __name__ == "__main__":
    print(get_frames())


# import os
# import glob
# import json
# import torch
# import cv2
# from PIL import Image
# import numpy as np
# import pandas as pd
# from tqdm.notebook import tqdm
# from facenet_pytorch import MTCNN

# from torch.utils.data import Dataset, DataLoader
# from torch import nn, optim
# from torch.nn import functional as F
# from torchvision.models import resnet18
# from albumentations import Normalize, Compose
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# import multiprocessing as mp
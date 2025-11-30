from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
from tqdm.auto import tqdm
import torch
import os
from transformers import logging
logging.set_verbosity_error()



from extract_faces import get_frames



def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

    base = "D:/W/VS/VS Folder/DFD/DFD-T"
    model_path = f'{base}/swin_tiny.pth'



    # Select a specific row for inference
    path = get_frames()
    print("\n--------------Selected video------------------")
    print(os.path.basename(path))



    # Load the model and processor
    processor = AutoImageProcessor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model = AutoModelForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224", num_labels=1, ignore_mismatched_sizes=True)



    # Load weights
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)



    # Inference
    batch_size = 36
    pred_classes = []
    total_batches = 0
    probabs = []
    total_batches = 0

    image_files = sorted([ os.path.join(path, f)
    for f in os.listdir(path)
    if f.lower().endswith('.png')
    ])

    print("\n--------------Running Inference------------------")
    model.eval()
    for i in tqdm(range(0, len(image_files), batch_size), desc='Image Batches'):
        total_batches += 1
        batch_paths = image_files[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        inputs = processor(images=images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            batch_logits = outputs.logits

            prob=torch.sigmoid(batch_logits).detach().cpu().view(-1).tolist()
            probabs.extend(prob)

            pred = (torch.sigmoid(batch_logits) > 0.5).int()

            pred_classes.extend(pred.cpu().numpy().flatten().tolist())
            


    pred = int(sum(pred_classes) > len(pred_classes) / 2)
    print("\n--------------Result------------------")
    print(f"Predicted class: {pred}")

    # Cleanup
    import shutil
    shutil.rmtree(path)



if __name__ == "__main__":
    main()

import numpy as np
from cellpose import models
from PIL import Image
import torch

def load_cellpose_model(model_type='cyto', gpu=True):
    model = models.CellposeModel(pretrained_model=model_type, gpu=True)
    return model

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)
    return img_np

def extract_cell_features(image_path, cellpose_model,  device):
    img = preprocess_image(image_path)
    
    # 4. 使用Cellpose进行细胞分割
    masks, flows, styles = cellpose_model.eval(img, diameter=None, channels=[0, 0])
    return masks

if __name__ == "__main__":
    image_path = "/home/zhaoyh/MUF-Clust/outputs/segmentation/11_Scan1/tiles/nuclei_edge_overlay_13312_23552.png"
    cellpose_model = load_cellpose_model()
    # ctranspath_model = load_ctranspath_model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = extract_cell_features(image_path, cellpose_model, device)
    Image.fromarray(features.astype(np.uint8)).save("/home/zhaoyh/MUF-Clust/pytest/nuclei_edge_overlay_13312_23552_masks.png")
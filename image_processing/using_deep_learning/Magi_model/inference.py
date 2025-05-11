from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import os

# 1. Load your image as a full‑color PIL image & numpy array
PATH = "/content/001_Nagraj_0010.jpg"
orig = Image.open(PATH).convert("RGB")
img = np.array(orig)

# 2. Load the Magi model
model = AutoModel.from_pretrained(
    "ragavsachdeva/magi",
    trust_remote_code=True
).cuda().eval()

# 3. Run detection with a lower panel threshold to split adjacent panels
with torch.no_grad():
    results = model.predict_detections_and_associations(
        [img],
        panel_detection_threshold=0.10,        # ← lower from 0.20 to pick up smaller panels
        character_detection_threshold=0.30,    # optional: adjust if you use character boxes
        text_detection_threshold=0.25          # optional: adjust if you use text boxes
    )

pred = results[0]
panel_boxes = pred["panels"]  # list of [x1, y1, x2, y2]

# 4. Crop each panel from the original colour image and save
out_dir = "panels_color"
os.makedirs(out_dir, exist_ok=True)

for i, (x1, y1, x2, y2) in enumerate(panel_boxes):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    crop = orig.crop((x1, y1, x2, y2))
    fname = os.path.join(out_dir, f"001_Nagraj_0010_panel_{i}.png")
    crop.save(fname)
    print(f"Saved {fname}")

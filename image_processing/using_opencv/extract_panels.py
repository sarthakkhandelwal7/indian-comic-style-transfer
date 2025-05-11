import os
from PIL import Image
import imageio
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.morphology import dilation
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
import numpy as np
import sys

def extract_panels_from_image(image_path, panels_subdir, min_panel_area=3000, num_dilations=3, canny_low=0.1, canny_high=0.3):
    im = imageio.imread(image_path)
    if im.ndim == 3 and im.shape[-1] == 4:
        im = im[:, :, :3]
    if im.ndim == 3:
        grayscale = rgb2gray(im)
    elif im.ndim == 2:
        grayscale = im.astype(float) / 255.0
    edges = canny(grayscale, low_threshold=canny_low, high_threshold=canny_high)
    thick_edges = edges
    for _ in range(num_dilations):
        thick_edges = dilation(thick_edges)
    segmentation = ndi.binary_fill_holes(thick_edges)
    labels = label(segmentation)
    properties = regionprops(labels)
    panel_count = 0
    for i, prop in enumerate(properties):
        if prop.area > min_panel_area:
            min_row, min_col, max_row, max_col = prop.bbox
            panel_image_array = im[min_row:max_row, min_col:max_col]
            panel_image_pil = Image.fromarray(panel_image_array)
            panel_count += 1
            filename = f"panel_{panel_count:03d}.png"
            save_path = os.path.join(panels_subdir, filename)
            panel_image_pil.save(save_path)
            print(f"Saved {save_path} (Original Label {i+1}) with bounding box {prop.bbox} and area {prop.area}")
        else:
            print(f"Skipping small region (Original Label {i+1}) with area {prop.area}")
    print(f"Panel separation complete for {os.path.basename(image_path)} (filtered by area).\n")
    
    

def process_folder(folder_path):
    images_dir = os.path.join(folder_path, 'images')
    panels_dir = os.path.join(folder_path, 'panels')
    os.makedirs(panels_dir, exist_ok=True)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        image_name = os.path.splitext(image_file)[0]
        panels_subdir = os.path.join(panels_dir, image_name)
        os.makedirs(panels_subdir, exist_ok=True)
        extract_panels_from_image(image_path, panels_subdir)

def main():
    if len(sys.argv) < 2:
        folder_name = input("Enter the folder name (e.g., 001_Nagraj_jp2): ")
    else:
        folder_name = sys.argv[1]
    folder_path = os.path.join(os.getcwd(), 'Data', folder_name)
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return
    process_folder(folder_path)

if __name__ == "__main__":
    main()
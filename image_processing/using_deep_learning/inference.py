import os.path as osp
import hashlib
from sklearn.cluster import MeanShift
import cv2
from PIL import Image
import pickle
import pytorch_lightning as pl
import torch
from torch import nn
from segment_anything import SamPredictor, sam_model_registry, apply_transform_to_pil_without_sam_model
from segment_anything.modeling.mask_decoder import MLP
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F
import numpy as np
import requests
from io import BytesIO
import matplotlib.pyplot as plt

# --- Helper Functions (copied from original model.py or adapted) ---

class DictWrapper: # Copied from args.py, needed for loading pickled args
    def __init__(self, d):
        self.__d = d
    def __getattr__(self, name):
        return self.__d.get(name)
    def __setattr__(self, name, value):
        if name == '_DictWrapper__d':
            super().__setattr__(name, value)
        else:
            self.__d[name] = value
    def __getitem__(self, name):
        return self.__d.get(name)
    def __setitem__(self, name, value):
        self.__d[name] = value
    def __contains__(self, name) :
        return name in self.__d
    def to_dict(self) :
        return self.__d
    def __str__(self):
        return str(self.__d)
    def __repr__(self):
        return repr(self.__d)

def topk (arr, k) :
    return np.argsort(arr)[-k:]

def avg (lst) :
    lst = list(lst)
    if len(lst) == 0 :
        return 0
    return sum(lst) / len(lst)

def filter_predicted_polygons (polygon_preds, confidence_scores, top_k=40, cluster_size_threshold=0.05) :
    if len(confidence_scores) == 0:
        return []
    if len(confidence_scores) < top_k:
        top_k = len(confidence_scores)

    top_idx = topk(confidence_scores, top_k)
    polygon_preds_np = np.array([polygon_preds[i].cpu().numpy() for i in top_idx])
    confidence_scores_np = np.array([confidence_scores[i].cpu().numpy() for i in top_idx])

    if polygon_preds_np.ndim == 1:
        polygon_preds_np = polygon_preds_np.reshape(top_k, -1, 2)
    elif polygon_preds_np.shape[1] != 2 :
        polygon_preds_np = polygon_preds_np.reshape(top_k, -1, 2)

    X = polygon_preds_np.reshape(top_k, -1)

    # Handle cases where all points are identical, causing MeanShift to fail or hang
    if X.shape[0] > 0 and np.all(X == X[0, :], axis=0).all():
         return [polygon_preds_np[0].astype(int)]

    clustering = MeanShift(cluster_all=False, bin_seeding=True).fit(X) # bin_seeding can help with speed
    labels = clustering.labels_

    # Handle cases where no clusters are formed (-1 label)
    if np.all(labels == -1):
        # Fallback: return the polygon with the highest original confidence score
        if len(polygon_preds_np) > 0:
            return [polygon_preds_np[0].astype(int)]
        else:
            return []

    unique_labels = np.unique(labels[labels != -1])
    if not len(unique_labels): return []

    bins = [[] for _ in range(max(unique_labels) + 1)]
    conf_bins = [[] for _ in range(max(unique_labels) + 1)]

    for i in range(top_k):
        if labels[i] != -1:
            bins[labels[i]].append(polygon_preds_np[i])
            conf_bins[labels[i]].append(confidence_scores_np[i])

    final_polygons = []
    for cluster_idx in unique_labels:
        if len(bins[cluster_idx]) >= cluster_size_threshold * top_k :
            avg_poly = np.mean(np.array(bins[cluster_idx]), axis=0).astype(int)
            final_polygons.append(avg_poly)
            
    return final_polygons


# --- ComicFramePredictorModule (adapted for inference) ---
class ComicFramePredictorModule(pl.LightningModule):
    def __init__(self, args):
        super(ComicFramePredictorModule, self).__init__()

        sam_checkpoint_path = args.sam_ckpt_path if hasattr(args, 'sam_ckpt_path') and args.sam_ckpt_path else None
        
        if sam_checkpoint_path and not osp.exists(sam_checkpoint_path):
            print(f"SAM checkpoint {sam_checkpoint_path} not found. Please ensure it's available or will be downloaded.")


        self.sam_model = sam_model_registry[args.sam_model_type](checkpoint=sam_checkpoint_path)

        self.projector_x = MLP(2 * 256, 256, 4, 3).float()
        self.projector_y = MLP(2 * 256, 256, 4, 3).float()
        self.point_confidence_score_predictor = MLP(2 * 256, 256, 1, 3).float()
        self.args = args
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def forward(self, batch): 
        current_device = self.device
        with torch.no_grad():
            if 'features' in batch:
                features = batch['features'].to(current_device)
            else:
                img_tensor = batch['img'].to(current_device)
                features = self.sam_model.image_encoder(img_tensor)

            point_coords = batch['point_coords'].to(current_device)
            point_labels = batch['point_labels'].to(current_device)
            
            N = point_coords.shape[0] if point_coords.nelement() > 0 else 0

            if N == 0: 
                 return {'pred': torch.empty(0, 4, 2, device=current_device), 
                        'iou_predictions': torch.empty(0,1, device=current_device),
                        'point_confidence': torch.empty(0,1, device=current_device),
                        'low_res_masks': torch.empty(0,1,256,256, device=current_device)}


            points_tuple = (point_coords.unsqueeze(0), point_labels.unsqueeze(0)) 

            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=points_tuple,
                boxes=None,
                masks=None,
            )
            sparse_embeddings = sparse_embeddings.to(current_device)
            dense_embeddings = dense_embeddings.to(current_device)

            low_res_masks, iou_predictions, prompt_tokens = self.sam_model.mask_decoder(
                image_embeddings=features, 
                image_pe=self.sam_model.prompt_encoder.get_dense_pe().to(current_device),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=True,
                interleave=False 
            )
            

            prompt_tokens_reshaped = prompt_tokens.squeeze(0) 
            
            out_x = self.projector_x(prompt_tokens_reshaped) 
            out_y = self.projector_y(prompt_tokens_reshaped)
            point_confidence_score_pred = self.point_confidence_score_predictor(prompt_tokens_reshaped)
            
            pred_polygons = torch.cat((out_x[..., None], out_y[..., None]), 2)

        return {
            'pred': pred_polygons,
            'iou_predictions': iou_predictions.squeeze(0).to(current_device),
            'point_confidence': point_confidence_score_pred.to(current_device),
            'low_res_masks': low_res_masks.squeeze(0).to(current_device)
        }

    @torch.no_grad()
    def encode_image_pil(self, pil_img):
        current_device = self.device
        
        np_img_original_shape = np.array(pil_img) 

        input_image_resized_np = self.transform.apply_image(np.array(pil_img))

        input_tensor_uint8 = torch.as_tensor(input_image_resized_np, device=current_device)
        input_tensor_uint8_nchw = input_tensor_uint8.permute(2, 0, 1).contiguous()[None, :, :, :]

        input_tensor_float_nchw = input_tensor_uint8_nchw.to(torch.float32)
        
        x_preprocessed = self.sam_model.preprocess(input_tensor_float_nchw)
        
        features = self.sam_model.image_encoder(x_preprocessed)
        
        return features, np_img_original_shape.shape[:2]

    @torch.no_grad()
    def run_inference_full(self, pil_img, num_sample_points=20, top_k_polygons=10, cluster_size_thresh=0.1):
        current_device = self.device
        self.eval()
        img_features, original_size = self.encode_image_pil(pil_img)
        
        h, w = original_size
        if num_sample_points == 0:
            return [], np.array(pil_img)

        sampled_points = torch.rand(num_sample_points, 2, device=current_device) 
        sampled_points[:, 0] *= w
        sampled_points[:, 1] *= h
        sampled_points = sampled_points.int()
        
        point_labels = torch.ones(num_sample_points, device=current_device).int()

        batch_for_forward = {
            'features': img_features,
            'point_coords': sampled_points,
            'point_labels': point_labels,
            'original_size': original_size,
            'input_size': tuple(self.transform.apply_image(np.array(pil_img)).shape[:2])
        }

        outputs = self(batch_for_forward)
        
        predicted_polygons_tensor = outputs['pred']
        confidence_scores_tensor = outputs['point_confidence'].squeeze()

        polygons_for_filtering = [p.cpu().numpy() for p in predicted_polygons_tensor]
        confidences_for_filtering = [c.cpu().numpy() for c in confidence_scores_tensor]

        if not polygons_for_filtering: 
            return [], np.array(pil_img)

        final_polygons = filter_predicted_polygons(
            polygons_for_filtering,
            confidences_for_filtering, 
            top_k=min(top_k_polygons, len(polygons_for_filtering)),
            cluster_size_threshold=cluster_size_thresh
        )
        
        vis_image = np.array(pil_img).copy()
        for poly in final_polygons:
            cv2.polylines(vis_image, [poly.reshape(-1, 1, 2).astype(np.int32)], isClosed=True, color=(0, 255, 0), thickness=2)
            
        return final_polygons, vis_image

def load_image_from_path_or_url(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        response = requests.get(image_path_or_url)
        response.raise_for_status() # Raise an exception for bad status codes
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    return image

def load_model_for_colab(checkpoint_path, args_path, sam_checkpoint_path="sam_vit_h_4b8939.pth"):
    effective_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Attempting to load model onto: {effective_device}")

    if not osp.exists(sam_checkpoint_path):
        print(f"Downloading SAM checkpoint to {sam_checkpoint_path}...")
        torch.hub.download_url_to_file("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth", sam_checkpoint_path)
        print("SAM checkpoint downloaded.")

    with open(args_path, 'rb') as f:
        args_dict = pickle.load(f)
    
    args_dict['sam_ckpt_path'] = sam_checkpoint_path
    args_dict['sam_model_type'] = args_dict.get('sam_model_type', 'vit_h')
    args_dict['device'] = effective_device 

    args_wrapper = DictWrapper(args_dict)

    model = ComicFramePredictorModule.load_from_checkpoint(
        checkpoint_path, 
        args=args_wrapper, 
        strict=False, 
        map_location=effective_device 
    )
    model.eval()
    model.to(effective_device)
    
    model.sam_model.to(effective_device)
    model.projector_x.to(effective_device)
    model.projector_y.to(effective_device)
    model.point_confidence_score_predictor.to(effective_device)

    print(f"Model loaded. Main module device: {next(model.parameters()).device}")
    try:
        print(f"SAM encoder device: {next(model.sam_model.image_encoder.parameters()).device}")
        print(f"Projector_x device: {next(model.projector_x.parameters()).device}")
    except StopIteration: 
        print("Could not query device for one or more submodules (possibly no parameters).")
    return model

# --- Example Usage (for Colab) ---
if __name__ == '__main__':


    YOUR_CHECKPOINT_PATH = "epoch=40-step=55432.ckpt" 
    YOUR_ARGS_PKL_PATH = "args.pkl" 
    SAM_CHECKPOINT_FILENAME = "sam_vit_h_4b8939.pth" 


    EXAMPLE_IMAGE_PATH = "/content/001_Nagraj_0010.jpg"


    # --- Check if checkpoint and args exist before loading ---
    if not osp.exists(YOUR_CHECKPOINT_PATH):
        print(f"ERROR: Checkpoint file not found at '{YOUR_CHECKPOINT_PATH}'. Please upload it and update the path.")
    elif not osp.exists(YOUR_ARGS_PKL_PATH):
        print(f"ERROR: Args file not found at '{YOUR_ARGS_PKL_PATH}'. Please upload it and update the path.")
    else:
        print("Loading model...")
        predictor_model = load_model_for_colab(YOUR_CHECKPOINT_PATH, YOUR_ARGS_PKL_PATH, SAM_CHECKPOINT_FILENAME)
        print("Model loaded.")

        print(f"Running inference on: {EXAMPLE_IMAGE_PATH}")
        try:
            input_pil_image = load_image_from_path_or_url(EXAMPLE_IMAGE_PATH)

            detected_polygons, visualization_image = predictor_model.run_inference_full(
                input_pil_image,
                num_sample_points=30, 
                top_k_polygons=10,    
                cluster_size_thresh=0.05 
            )
            
            print(f"Detected {len(detected_polygons)} panels.")
            for i, poly in enumerate(detected_polygons):
                print(f"Panel {i+1} points: {poly.tolist()}")

            # Display using matplotlib
            plt.figure(figsize=(10, 10))
            plt.imshow(visualization_image)
            plt.title(f"Detected Panels ({len(detected_polygons)})")
            plt.axis('off')
            plt.show()

        except FileNotFoundError:
            print(f"ERROR: Example image not found at '{EXAMPLE_IMAGE_PATH}'. Please provide a valid image path or URL.")
        except Exception as e:
            print(f"An error occurred during inference: {e}")
            import traceback
            traceback.print_exc() 
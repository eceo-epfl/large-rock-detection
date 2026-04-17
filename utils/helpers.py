from pathlib import Path
import json
import yaml
import random
import utils.paths as paths
import os
from torch.utils.data import Subset
from tifffile import imwrite
import cv2
import torch
from ultralytics.utils.ops import non_max_suppression, xywhn2xyxy
from ultralytics.utils.metrics import ConfusionMatrix
from utils.constants import TILE_SIZE_PX


# --------------- DATA PREPROCESSING ----------------

def convert_annotation_to_yolo(annotation, patch_size=TILE_SIZE_PX, bbox_width=16, bbox_height=16):
    """
    Convert a single annotation to YOLO format.
    """
    class_id = 0 # We only have one class (rocks)
    center_x, center_y = annotation["relative_within_patch_location"]
    rel_width = bbox_width / patch_size
    rel_height = bbox_height / patch_size
    return f"{class_id} {center_x:.6f} {center_y:.6f} {rel_width:.6f} {rel_height:.6f}"


def create_yolo_annotation_files(dataset, output_dir, patch_size=TILE_SIZE_PX, bbox_width=32, bbox_height=32):
    """
    Generate YOLO-format annotation files (the .txt files) for each image patch in the given dataset.
    """
    # Ensure output_dir is a Path object and exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for sample in dataset:
        # Extract the base filename (without extension) to create the corresponding .txt file.
        base_filename = sample["file_name"].removesuffix(".tif")
        txt_file_path = output_dir / f"{base_filename}.txt"
        
        # Open the text file for writing
        with open(txt_file_path, 'w') as f:
            # For each rock annotation, convert it to YOLO format and write it on a new line.
            for annotation in sample.get("rocks_annotations", []):
                yolo_line = convert_annotation_to_yolo(annotation, patch_size, bbox_width, bbox_height)
                f.write(yolo_line + "\n")


def prepare_yolo_training_files(dataset, root_dest_dir, split, patch_size=TILE_SIZE_PX, bbox_width=32, bbox_height=32):

    # destination directories for each split
    dest_images_dir = root_dest_dir  / "images" / split
    dest_labels_dir = root_dest_dir / "labels" / split

    dest_images_dir.mkdir(parents=True, exist_ok=True)
    dest_labels_dir.mkdir(parents=True, exist_ok=True)

    # copy each image (from the sample)
    for sample in dataset:
        src_img = sample["image"]
        dst_file = dest_images_dir / sample["file_name"]
        if src_img is not None:
            imwrite(dst_file, src_img)
        else:
            print(f"Source file not found, error when retrieving the image")

    # Generate YOLO annotation files for the given samples
    create_yolo_annotation_files(
        dataset, 
        output_dir=dest_labels_dir, 
        patch_size=patch_size, 
        bbox_width=bbox_width, 
        bbox_height=bbox_height
    )
    
    print("YOLO data preparation complete for split.")

def prepare_yolo_training_files_all_splits(train_samples, val_samples, test_samples, patch_size=TILE_SIZE_PX, bbox_width=32, bbox_height=32):
    root_dest_dir = paths.PROCESSED_DATA_DIR
    root_dest_dir.mkdir(parents=True, exist_ok=True)
    prepare_yolo_training_files(train_samples, root_dest_dir, "train", patch_size, bbox_width, bbox_height)
    prepare_yolo_training_files(val_samples, root_dest_dir, "val", patch_size, bbox_width, bbox_height)
    prepare_yolo_training_files(test_samples, root_dest_dir, "test", patch_size, bbox_width, bbox_height)
    print("----------- YOLO data preparation complete for all splits. ------------")

# --------------- DATA SPLITTING ----------------

def do_overlap(coord1, coord2):
    '''
    checks overlap between 2 patches, where coord = [xmin, ymin, xmax, ymax]
    '''
    x_min1, y_min1, x_max1, y_max1 = coord1
    x_min2, y_min2, x_max2, y_max2 = coord2

    if x_max1 <= x_min2 or x_max2 <= x_min1:
        return False

    if y_max1 <= y_min2 or y_max2 <= y_min1:
        return False

    return True

def find_overlapping_patches(dataset):
    """
    Groups patches based on overlap.
    """
    patches = [sample["coord"] for sample in dataset]
    n = len(patches)
    
    # Build an undirected graph where each patch is a node.
    # Two nodes are connected if their patches overlap.
    graph = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if do_overlap(patches[i], patches[j]):
                graph[i].add(j)
                graph[j].add(i)
    
    # Find connected components using a simple DFS.
    visited = set()
    overlap_groups = []
    not_overlaping = set()
    
    for i in range(n):
        if i not in visited:
            stack = [i]
            component = set()
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.add(node)
                for neighbor in graph[node]:
                    if neighbor not in visited:
                        stack.append(neighbor)
            # If the component size is 1, the patch does not overlap with any other.
            if len(component) == 1:
                not_overlaping.update(component)
            else:
                overlap_groups.append(component)
    
    return {'overlap_groups': overlap_groups, 'not_overlaping': not_overlaping}

def split_without_overlap(dataset, split=(0.8, 0.1, 0.1), seed=None):
    """
    Split dataset into train, val, and test subsets without overlapping patches.
    """
    overlap = find_overlapping_patches(dataset)
    subgroups = overlap['overlap_groups'] # there is only overlapping groups with each 16 images

    # shuffle the groups
    if seed is not None:
        random.seed(seed)
    random.shuffle(subgroups)

    # split the groups
    n = len(subgroups)

    nb_val = int(n * split[1])
    nb_test = int(n * split[2])

    # flatten the groups for each split
    train_indices = flatten_groups(subgroups[nb_val+nb_test:])
    val_indices = flatten_groups(subgroups[:nb_val])
    test_indices = flatten_groups(subgroups[nb_val:nb_val+nb_test])
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices), Subset(dataset, test_indices)

def flatten_groups(groups):
    """Flatten a list of lists into a single list."""
    return [idx for group in groups for idx in group]


def save_metrics(results, output_dir, file_name):
    """
    Save the training metrics to a csv file.
    Precision. Recall, F1, F2, mAP50, mAP50/95
    """
    # ectract metrics
    map5095 = results.box.map
    map50 = results.box.map50
    f1 = results.box.f1[0]
    precision = results.box.p[0]
    recall = results.box.r[0]
    f2 = (5 * precision * recall) / (4 * precision + recall)
    
    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, "metrics_results_" + file_name + ".csv")
    # save to file
    with open(output_file_path, "w") as f:
        f.write("precision, recall, f1, f2, map50, map5095\n")
        f.write(f"{precision:.4f}, {recall:.4f}, {f1:.4f}, {f2:.4f}, {map50:.4f}, {map5095:.4f}\n")


# ------------- Active teacher helper function ---------------------

def validate_model(
    model,
    dataloader,
    device=None,
    conf_thres: float = 0.25,
    iou_thres: float = 0.6,
    verbose: bool = True
    ) -> dict:
    """
    Validate a YOLO model by building a ConfusionMatrix and using its process_batch logic.

    Args:
        model: a loaded ultralytics YOLO model instance
        dataloader: DataLoader yielding dicts with keys ['img','batch_idx','cls','bboxes']
        device: torch device (defaults to model's)
        conf_thres: confidence threshold for filtering detections
        iou_thres: IoU threshold for matching
        verbose: print metrics

    Returns:
        dict with 'precision','recall','f1'
    """
    device = device or next(model.parameters()).device
    # build confusion matrix for detection
    cm = ConfusionMatrix(
        nc=model.nc,
        conf=conf_thres,
        iou_thres=iou_thres,
        task='detect'
    )

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            imgs      = batch['img'].float().div(255.0).to(device)
            batch_idx = batch['batch_idx'].to(device)
            true_cls  = batch['cls'].to(device)
            true_bb   = batch['bboxes'].to(device)

            # inference + NMS
            preds = non_max_suppression(
                model(imgs), conf_thres, iou_thres
            )

            # per-image update
            for si, dets in enumerate(preds):
                gt_mask = batch_idx == si
                gt_cls  = true_cls[gt_mask]
                # convert gt to xyxy for matching
                gt_boxes_xyxy = xywhn2xyxy(true_bb[gt_mask], imgs.shape[3], imgs.shape[2])

                # ensure detections tensor
                if dets is None:
                    dets_t = torch.zeros((0, 6), device=device)
                else:
                    dets_t = dets.to(device)

                # update confusion matrix using YOLO's process_batch
                cm.process_batch(
                    dets_t,
                    gt_boxes_xyxy,
                    gt_cls
                )

    # compute metrics from confusion matrix
    tp, fp = cm.tp_fp()
    # false negatives per class = sum over column minus TP
    fn = cm.matrix.sum(0)[:-1] - tp

    # overall precision and recall
    total_tp = tp.sum().item()
    total_fp = fp.sum().item()
    total_fn = fn.sum().item()
    precision = total_tp / (total_tp + total_fp + 1e-16)
    recall    = total_tp / (total_tp + total_fn + 1e-16)
    f1        = 2 * precision * recall / (precision + recall + 1e-16)
    f2 = (1 + 2.0**2) * precision * recall / (2.0**2 * precision + recall + 1e-16)

    if verbose:
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, F2: {f2:.3f}")

    return {
        'precision': precision,
        'recall':    recall,
        'f1':        f1,
        'f2':        f2
    }
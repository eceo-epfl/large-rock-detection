import argparse

from utils.constants import TILE_SIZE_PX


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO Training Script Argument Parser"
    )

    parser.add_argument(
        '--model',
        type=str,
        default="yolov8x.pt",
        help="Path to the model file for training. Accepts a .pt pretrained model or a .yaml config file."
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help="Total number of training epochs (default: 200)."
    )
    parser.add_argument(
        '--time',
        type=float,
        default=None,
        help="Maximum training time in hours (default: None)."
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=100,
        help="Number of epochs to wait for improvement before early stopping (default: 100)."
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help="Batch size (default: 16)."
    )
    parser.add_argument(
        '--bbox_size',
        type=int,
        default=16,
        help="Bounding box size (default: 16)."
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help="Number of worker threads for data loading (default: 8)."
    )
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help="Name of the training run (default: None)."
    )
    parser.add_argument(
        '--project',
        type=str,
        default=None,
        help="Name of the project directory where training outputs are saved"
    )
    parser.add_argument(
        '--cos_lr',
        action='store_true',
        default=False,
        help="Enable cosine learning rate scheduler (default: False)."
    )
    parser.add_argument(
        '--close_mosaic',
        type=int,
        default=10,
        help="Epochs before disabling mosaic augmentation (default: 10)."
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help="Initial learning rate (default: 0.01)."
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help="Final learning rate as a fraction of lr0 (default: 0.01)."
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help="Momentum factor (default: 0.937)."
    )
    parser.add_argument(
        '--box',
        type=float,
        default=7.5,
        help="Weight of the box loss component (default: 7.5)."
    )
    parser.add_argument(
        '--cls',
        type=float,
        default=0.5,
        help="Weight of the classification loss (default: 0.5)."
    )
    parser.add_argument(
        '--dfl',
        type=float,
        default=1.5,
        help="Weight of the distribution focal loss (default: 1.5)."
    )
    parser.add_argument(
        '--pose',
        type=float,
        default=12.0,
        help="Weight of the pose loss; set to 0 for rock detection (default: 0.0)."
    )
    parser.add_argument(
        '--kobj',
        type=float,
        default=2.0,
        help="Weight of the keypoint objectness loss; set to 0 for rock detection (default: 0.0)."
    )
    parser.add_argument(
        '--plots',
        action='store_true',
        default=True,
        help="Generate and save plots during training (default: False)."
    )
    parser.add_argument(
        '--single_cls',
        action='store_true',
        default=False,
        help="Enable single class training (default: False)."
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help="Translation augmentation factor (default: 0.1)."
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.0,
        help="Scale augmentation factor; set to 0 for rock detection (default: 0.0)."
    )
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.5,
        help="Probability of vertical flip; recommended to set to 0.5 (default: 0.5)."
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help="Probability of horizontal flip (default: 0.5)."
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Computational device(s) for training (default: None)."
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        help="Choice of optimizer (default: auto)."
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help="L2 regularization term (default: 0.0005)."
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help="Resume training from the last saved checkpoint."
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=TILE_SIZE_PX,
        help="Image size for training (default: 640)."
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=1.0,
        help="Weight of the RGB image. (1 - alpha) is used for hillshade."
    )
    parser.add_argument(
        '--rgb_channel',
        type=int,
        default=-1,
        help="Channel to be replaced by HS value"
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=255,
        help="threshold at which RGB is replaced by black pixels"
    )
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help="Mosaic augmentation factor (default: 1.0)."
    )
    parser.add_argument(
        '--hsv_h',
        type=float,
        default=0.015,
        help="Hue augmentation factor (default: 0.015)."
    )
    parser.add_argument(
        '--hsv_s',
        type=float,
        default=0.7,
        help="Saturation augmentation factor (default: 0.7)."
    )
    parser.add_argument(
        '--hsv_v',
        type=float,
        default=0.4,
        help="Value (brightness) augmentation factor (default: 0.4)."
    )
    parser.add_argument(
        '--erasing',
        type=float,
        default=0.4,
        help="Erasing augmentation factor (default: 0.4)."
    )
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help="Rotation augmentation in degrees (default: 0.0)."
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)."
    )

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    print(args)
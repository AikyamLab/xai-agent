"""
Debug script to visualize which pixels are selected for suggested_bounding_box.
Usage:
    python debug_bbox.py --npy path/to/heatmap.npy [--image path/to/image.png] [--pct 0.25]
    python debug_bbox.py --dir path/to/xai_output_dir [--stl10 path/to/stl10_binary] [--row_no 0]
    python debug_bbox.py --dir path/to/xai_output_dir --image path/to/image.png
    python debug_bbox.py --dir path/to/xai_output_dir --cub /path/to/CUB_200_2011 --row_no 0 [--split test]
    python debug_bbox.py --npy path/to/heatmap.npy --cub /path/to/CUB_200_2011 --row_no 0
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def load_cub_image(dataset_root, row_no, split='test'):
    """Load a single CUB-200-2011 image by positional index within the split (raw PIL, no transform).

    Also returns the ground-truth bounding box (from bounding_boxes.txt) in
    [x_min, y_min, x_max, y_max] pixel coordinates of the *original* image
    (before any resize), or None if unavailable.
    """
    import pandas as pd

    df_img = pd.read_csv(
        os.path.join(dataset_root, 'images.txt'),
        sep=' ', header=None, names=['ID', 'Image'], index_col=0
    )
    df_label = pd.read_csv(
        os.path.join(dataset_root, 'image_class_labels.txt'),
        sep=' ', header=None, names=['ID', 'Label'], index_col=0
    )
    df_split = pd.read_csv(
        os.path.join(dataset_root, 'train_test_split.txt'),
        sep=' ', header=None, names=['ID', 'Train'], index_col=0
    )
    df_bbox = pd.read_csv(
        os.path.join(dataset_root, 'bounding_boxes.txt'),
        sep=' ', header=None, names=['ID', 'x', 'y', 'width', 'height'], index_col=0
    )

    df = pd.concat([df_img, df_label, df_split, df_bbox], axis=1)
    df['Label'] = df['Label'] - 1  # 1-based → 0-based

    train_flag = 1 if split == 'train' else 0
    df_filtered = df[df['Train'] == train_flag].reset_index()

    if row_no >= len(df_filtered):
        raise IndexError(f"row_no={row_no} out of range for CUB split='{split}' (size={len(df_filtered)})")

    row = df_filtered.iloc[row_no]
    img_path = os.path.join(dataset_root, 'images', row['Image'])
    from PIL import Image as PILImage
    img = PILImage.open(img_path).convert('RGB')

    # bounding box in original image coords: [x_min, y_min, x_max, y_max]
    gt_bbox = [
        int(row['x']),
        int(row['y']),
        int(row['x'] + row['width']),
        int(row['y'] + row['height']),
    ]
    label = int(row['Label'])

    # Load class name
    classes_path = os.path.join(dataset_root, 'classes.txt')
    label_name = None
    with open(classes_path) as f:
        for line in f:
            parts = line.strip().split()
            if int(parts[0]) - 1 == label:
                label_name = parts[1]
                break

    print(f"CUB image: {row['Image']}")
    print(f"Label: {label} ({label_name})")
    print(f"GT bbox (original image): {gt_bbox}  (x,y,x2,y2)")

    return img, gt_bbox


def load_stl10_image(dataset_root, row_no, split='test'):
    """Load a single STL-10 image by index (raw PIL, no transform)."""
    from torchvision.datasets import STL10
    dataset = STL10(root=dataset_root, split=split, download=False, transform=None)
    img, label = dataset[row_no]
    return img


def compute_top_pct_bbox(heatmap, pct=0.25):
    nonzero = heatmap[heatmap > 0]
    if len(nonzero) == 0:
        return None
    threshold = float(np.percentile(nonzero, (1.0 - pct) * 100))
    mask = heatmap >= threshold
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())], mask, threshold


def visualize_bbox_selection(heatmap_path, image=None, pct=0.25, gt_bbox=None):
    """
    image: numpy array (H,W,3) uint8, or None.
    heatmap_path: path to *_heatmap.npy (the abs-value normalized heatmap saved by each tool).
    """
    heatmap = np.load(heatmap_path)
    name = Path(heatmap_path).stem
    print(f"\n{'='*60}")
    print(f"File: {name}")
    print(f"Heatmap shape: {heatmap.shape}, dtype: {heatmap.dtype}")
    print(f"Value range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")

    total_pixels = heatmap.size
    nonzero_pixels = int((heatmap > 0).sum())
    print(f"Total pixels: {total_pixels}")
    print(f"Nonzero pixels: {nonzero_pixels} ({nonzero_pixels/total_pixels*100:.1f}%)")

    if nonzero_pixels == 0:
        print("No nonzero pixels — cannot compute bbox.")
        return

    nonzero_vals = heatmap[heatmap > 0]
    threshold = float(np.percentile(nonzero_vals, (1.0 - pct) * 100))
    mask = heatmap >= threshold
    selected = int(mask.sum())
    print(f"Threshold (top {pct*100:.0f}% of nonzero): {threshold:.4f}")
    print(f"Selected pixels: {selected} ({selected/total_pixels*100:.1f}% of image)")

    ys, xs = np.where(mask)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    print(f"Bbox: {bbox}  ({bw}x{bh} = {bw*bh}px, {bw*bh/total_pixels*100:.1f}% of image)")

    # ── Plot ──────────────────────────────────────────────────────────
    ncols = 4 if image is not None else 3
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(name, fontsize=10)

    col = 0
    if image is not None:
        axes[col].imshow(image)
        title = 'Original image'
        if gt_bbox is not None:
            gx, gy, gx2, gy2 = gt_bbox
            axes[col].add_patch(patches.Rectangle(
                (gx, gy), gx2 - gx, gy2 - gy,
                linewidth=2, edgecolor='lime', facecolor='none', label='GT bbox'
            ))
            title += '\n(green = GT bbox)'
        axes[col].set_title(title)
        axes[col].axis('off')
        col += 1

    # Raw heatmap
    axes[col].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[col].set_title(f'Heatmap\n(nonzero: {nonzero_pixels/total_pixels*100:.1f}%)')
    axes[col].axis('off')
    col += 1

    # Selected pixels mask + bbox
    axes[col].imshow(mask, cmap='Reds', vmin=0, vmax=1)
    axes[col].set_title(f'Selected pixels\n(thresh={threshold:.3f}, n={selected})')
    axes[col].axis('off')
    axes[col].add_patch(patches.Rectangle(
        (bbox[0], bbox[1]), bw, bh,
        linewidth=2, edgecolor='blue', facecolor='none'
    ))
    col += 1

    # Heatmap + bbox overlay
    axes[col].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[col].add_patch(patches.Rectangle(
        (bbox[0], bbox[1]), bw, bh,
        linewidth=2, edgecolor='cyan', facecolor='none'
    ))
    axes[col].set_title(f'Heatmap + bbox\n{bbox} ({bw}x{bh})')
    axes[col].axis('off')

    plt.tight_layout()
    out_path = str(heatmap_path).replace('_heatmap.npy', '_bbox_debug.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npy', type=str, help='Path to a single *_heatmap.npy file')
    parser.add_argument('--dir', type=str, help='Directory to search for all *_heatmap.npy files')
    parser.add_argument('--image', type=str, default=None, help='Path to original image file (PNG/JPG)')
    parser.add_argument('--stl10', type=str, default=None,
                        help='Path to STL-10 dataset root (parent of stl10_binary/). '
                             'Used with --row_no to load the original image.')
    parser.add_argument('--cub', type=str, default=None,
                        help='Path to CUB-200-2011 dataset root (contains images.txt, bounding_boxes.txt, …). '
                             'Used with --row_no to load the original image and show ground-truth bbox.')
    parser.add_argument('--row_no', type=int, default=None, help='Dataset test-set positional index')
    parser.add_argument('--split', type=str, default='test', help='Dataset split (default: test)')
    parser.add_argument('--pct', type=float, default=0.25, help='Top percentage (default 0.25)')
    args = parser.parse_args()

    # Resolve image once
    image_np = None
    gt_bbox_overlay = None  # optional GT bbox to draw on plots
    if args.cub and args.row_no is not None:
        print(f"Loading CUB image: split={args.split}, row_no={args.row_no}")
        pil_img, gt_bbox_overlay = load_cub_image(args.cub, args.row_no, split=args.split)
        image_np = np.array(pil_img)
        print(f"Image shape: {image_np.shape}")
    elif args.stl10 and args.row_no is not None:
        print(f"Loading STL-10 image: split={args.split}, row_no={args.row_no}")
        pil_img = load_stl10_image(args.stl10, args.row_no, split=args.split)
        image_np = np.array(pil_img)
        print(f"Image shape: {image_np.shape}")
    elif args.image:
        from PIL import Image as PILImage
        image_np = np.array(PILImage.open(args.image).convert('RGB'))

    if args.npy:
        visualize_bbox_selection(args.npy, image=image_np, pct=args.pct, gt_bbox=gt_bbox_overlay)
    elif args.dir:
        npy_files = sorted(Path(args.dir).rglob('*_heatmap.npy'))
        if not npy_files:
            print(f"No *_heatmap.npy files found in {args.dir}")
            return
        print(f"Found {len(npy_files)} heatmap files")
        for f in npy_files:
            visualize_bbox_selection(str(f), image=image_np, pct=args.pct, gt_bbox=gt_bbox_overlay)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

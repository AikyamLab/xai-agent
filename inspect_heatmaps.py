import numpy as np
import os

def inspect_npy(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    data = np.load(path)
    print(f"Path: {path}")
    print(f"Shape: {data.shape}")
    print(f"Min: {data.min():.6f}")
    print(f"Max: {data.max():.6f}")
    print(f"Mean: {data.mean():.6f}")
    print(f"Std: {data.std():.6f}")
    print(f"Non-zero elements: {(data > 0).sum()} / {data.size}")

test_ig_path = "test_outputs/xai_outputs/vision/stl10_resnet/q1/7/ig_stl10_resnet_q1_7_integrated_gradients_class1_heatmap.npy"
baseline_ig_path = "baseline_outputs/Qwen30B_VL_top25/xai_outputs/vision/stl10_resnet/q1/7/ig_stl10_resnet_q1_7_integrated_gradients_class1_heatmap.npy"
test_gb_path = "test_outputs/xai_outputs/vision/stl10_resnet/q1/7/guided_bp_stl10_resnet_q1_7_guided_backprop_class1_heatmap.npy"
baseline_gb_path = "baseline_outputs/Qwen30B_VL_top25/xai_outputs/vision/stl10_resnet/q1/7/guided_bp_stl10_resnet_q1_7_guided_backprop_class1_heatmap.npy"

test_sg_path = "test_outputs/xai_outputs/vision/stl10_resnet/q1/7/smoothgrad_stl10_resnet_q1_7_smoothgrad_class1_heatmap.npy"
baseline_sg_path = "baseline_outputs/Qwen30B_VL_top25/xai_outputs/vision/stl10_resnet/q1/7/smoothgrad_stl10_resnet_q1_7_smoothgrad_class1_heatmap.npy"

print("IG Inspection:")
inspect_npy(test_ig_path)
print("-" * 20)
inspect_npy(baseline_ig_path)
print("\nGuided Backprop Inspection:")
inspect_npy(test_gb_path)
print("-" * 20)
inspect_npy(baseline_gb_path)
print("\nSmoothGrad Inspection:")
inspect_npy(test_sg_path)
print("-" * 20)
inspect_npy(baseline_sg_path)

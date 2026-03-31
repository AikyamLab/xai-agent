
def execute_integrated_gradients(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    n_steps: int = 200,
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute Integrated Gradients analysis with robust post-processing.
    """
    try:
        original_size = image.size # (W, H)
        img_np = np.array(image)
        model.eval()

        # 1. Preprocess ensuring spatial alignment (NO CenterCrop)
        if input_tensor is None:
            input_tensor = _preprocess_image(image, model_type, processor, device)
        else:
            input_tensor = input_tensor.to(device).float()
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

        # baseline: black image
        baseline = torch.zeros_like(input_tensor)

        inplace_states = {}
        for name, module in model.named_modules():
            if hasattr(module, 'inplace') and module.inplace:
                inplace_states[name] = True
                module.inplace = False

        try:
            ig = IntegratedGradients(model)
            attribution = ig.attribute(
                input_tensor, 
                baselines=baseline, 
                target=target_class, 
                n_steps=n_steps,
                internal_batch_size=16 
            )
        finally:
            for name, module in model.named_modules():
                if name in inplace_states:
                    module.inplace = True

        # 2. Extract and Aggregate: [C, H, W]
        attr_np = attribution.squeeze(0).cpu().detach().numpy()
        
        # Aggregate across channels: Sum signed gradients first! 
        # This allows noise to cancel out across R,G,B channels.
        if len(attr_np.shape) == 3:
            attr_np = attr_np.sum(axis=0) # [H, W]
        
        # 3. Robust Normalization: Percentile Clipping
        # CLIP Top 0.5% outliers to prevent them from washing out the heatmap.
        vmax = np.percentile(np.abs(attr_np), 99.5)
        attr_np = np.clip(attr_np, -vmax, vmax)
        
        # Convert to absolute importance
        heatmap = np.abs(attr_np)

        # 4. Spatial Alignment: Resize back to original
        heatmap = cv2.resize(heatmap, (original_size[0], original_size[1]))
        
        # Gentle smoothing
        sigma = max(0.5, original_size[0] * 0.005)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        # Save visualization
        output_dir = get_output_dir()
        viz_path = str(output_dir / f"ig_{image_id}_class{target_class}.png")
        _save_heatmap_visualization(img_np, heatmap, viz_path, "Integrated Gradients")
        heatmap_path = _save_raw_heatmap(heatmap, viz_path)

        p90 = float(np.percentile(heatmap, 90))
        mean_val = float(heatmap.mean())
        flat_idx = np.argsort(heatmap.flatten())[-10:]
        top_coords = [{"y": int(np.unravel_index(idx, heatmap.shape)[0]), "x": int(np.unravel_index(idx, heatmap.shape)[1]), "value": float(heatmap.flatten()[idx])} for idx in flat_idx]

        # Compute suggested bounding box from top 1% of heatmap pixels
        threshold = np.percentile(heatmap, 99)
        ys, xs = np.where(heatmap >= threshold)
        suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

        return {
            "success": True, "method": "IntegratedGradients", "target_class": target_class,
            "original_image_size": {"width": original_size[0], "height": original_size[1]},
            "visualization_path": viz_path, "heatmap_path": heatmap_path,
            "suggested_bounding_box": suggested_bbox,
            "statistics": {"high_importance_threshold": round(p90, 4), "mean_importance": round(mean_val, 4), "top_importance_coords": top_coords},
            "description": f"IG shows pixels most responsible for class {target_class}. Percentile clipping applied."
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(), "method": "IntegratedGradients"}

def execute_guided_backprop(
    image: Image.Image,
    model: Any,
    model_type: str,
    processor: Any,
    target_class: int,
    device: torch.device,
    image_id: str = "temp",
    input_tensor: Optional[torch.Tensor] = None
) -> Dict[str, Any]:
    """
    Execute Guided Backpropagation with spatial alignment and noise reduction.
    """
    try:
        original_size = image.size
        img_np = np.array(image)
        model.eval()

        if input_tensor is None:
            input_tensor = _preprocess_image(image, model_type, processor, device)
        else:
            input_tensor = input_tensor.to(device).float()
            if input_tensor.dim() == 3:
                input_tensor = input_tensor.unsqueeze(0)

        inplace_states = {}
        for name, module in model.named_modules():
            if hasattr(module, 'inplace') and module.inplace:
                inplace_states[name] = True
                module.inplace = False

        try:
            gbp = GuidedBackprop(model)
            attribution = gbp.attribute(input_tensor, target=target_class)
        finally:
            for name, module in model.named_modules():
                if name in inplace_states:
                    module.inplace = True

        # Process: [C, H, W]
        attr_np = attribution.squeeze(0).cpu().detach().numpy()
        
        # 1. Aggregate: Sum signed across channels to cancel noise
        if len(attr_np.shape) == 3:
            attr_np = attr_np.sum(axis=0)
        
        # 2. Outlier removal: clip top 0.5%
        vmax = np.percentile(np.abs(attr_np), 99.5)
        attr_np = np.clip(attr_np, -vmax, vmax)
        
        # 3. Absolute importance
        heatmap = np.abs(attr_np)

        # 4. Resize and Normalize
        heatmap = cv2.resize(heatmap, (original_size[0], original_size[1]))
        sigma = max(0.5, original_size[0] * 0.005)
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=sigma, sigmaY=sigma)

        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        else:
            heatmap = np.zeros_like(heatmap)

        output_dir = get_output_dir()
        viz_path = str(output_dir / f"guided_bp_{image_id}_class{target_class}.png")
        _save_heatmap_visualization(img_np, heatmap, viz_path, "Guided Backprop")
        heatmap_path = _save_raw_heatmap(heatmap, viz_path)

        mean_val = float(heatmap.mean())
        flat_idx = np.argsort(heatmap.flatten())[-10:]
        top_coords = [{"y": int(np.unravel_index(idx, heatmap.shape)[0]), "x": int(np.unravel_index(idx, heatmap.shape)[1]), "value": float(heatmap.flatten()[idx])} for idx in flat_idx]

        # Compute suggested bounding box from top 1% of heatmap pixels
        threshold = np.percentile(heatmap, 99)
        ys, xs = np.where(heatmap >= threshold)
        suggested_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if len(ys) > 0 else None

        return {
            "success": True, "method": "GuidedBackprop", "target_class": target_class,
            "original_image_size": {"width": original_size[0], "height": original_size[1]},
            "visualization_path": viz_path, "heatmap_path": heatmap_path,
            "suggested_bounding_box": suggested_bbox,
            "statistics": {"mean_importance": round(mean_val, 4), "top_importance_coords": top_coords},
            "description": f"Guided Backprop highlights high-resolution features for class {target_class}."
        }
    except Exception as e:
        import traceback
        return {"success": False, "error": str(e), "traceback": traceback.format_exc(), "method": "GuidedBackprop"}

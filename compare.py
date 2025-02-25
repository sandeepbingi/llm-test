def compare_lora_weights(base_model_weights, compare_model_weights, device_id, model_type):
    print(f"Comparing {model_type} model LoRA weights on GPU: {device_id}")

    for name, base_weight in base_model_weights.items():
        if name in compare_model_weights:
            compare_weight = compare_model_weights[name]

            # Ensure both are tensors
            if not isinstance(base_weight, torch.Tensor) or not isinstance(compare_weight, torch.Tensor):
                print(f"Skipping {name} as it is not a tensor ({type(base_weight)}, {type(compare_weight)})")
                continue

            # Ensure they have values and are not empty
            if base_weight.numel() == 0 or compare_weight.numel() == 0:
                print(f"Skipping {name} due to empty tensor (Base: {base_weight.shape}, Compare: {compare_weight.shape})")
                continue

            # Ensure they have the same shape
            if base_weight.shape != compare_weight.shape:
                print(f"Skipping {name} due to shape mismatch (Base: {base_weight.shape}, Compare: {compare_weight.shape})")
                continue

            # Compute difference
            diff_tensor = torch.abs(base_weight - compare_weight).mean()
            diff = diff_tensor.item() if isinstance(diff_tensor, torch.Tensor) else diff_tensor
            print(f"GPU: {device_id}, LoRA weight difference found for {name}")
            print(f"Average difference in {name}: {diff}")


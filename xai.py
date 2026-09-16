import torch
import numpy as np

def compute_seg_hires_gradcam(model, tensor_input):
    """
    Computes Seg-HiRes-Grad-CAM: Omits Global Average Pooling 
    to preserve spatial resolution at the bottleneck layer.
    """
    with torch.enable_grad():
        activations = []
        gradients = []

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        handle_fwd = model.enc2.register_forward_hook(forward_hook)
        handle_bwd = model.enc2.register_full_backward_hook(backward_hook)

        tensor_input = tensor_input.clone().detach().requires_grad_(True)
        model.zero_grad()

        try:
            output = model(tensor_input)
            target = output.sum()
            target.backward()

            act = activations[0].detach()  # Shape: [1, C, H, W]
            grad = gradients[0].detach()   # Shape: [1, C, H, W]
        finally:
            handle_fwd.remove()
            handle_bwd.remove()

        # Element-wise product (HiRes approach - No spatial reduction/pooling)
        cam = torch.sum(act * grad, dim=1, keepdim=True)
        cam = torch.relu(cam).squeeze().cpu().numpy()

        # Normalize map
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        return cam, output.squeeze().detach().cpu().numpy()

# Compatibility alias for legacy imports
generate_gradcam = compute_seg_hires_gradcam
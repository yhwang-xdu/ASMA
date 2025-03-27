import torch
from pytorch_msssim import ssim, ms_ssim

def calculate_metrics(X, Y):
    # Ensure that the input tensors are in the correct format (float32 and within the range [0, 1])
    X = X.float() 
    Y = Y.float() 
    # print("torchmin ", torch.min(X), torch.min(Y), torch.max(X), torch.max(Y))
    
    # SSIM (Structural Similarity Index)
    ssim_value = ssim(X, Y, data_range=1.0, size_average=True)
    
    # MS-SSIM (Multi-Scale Structural Similarity Index)
    ms_ssim_value = ms_ssim(X, Y, data_range=1.0, size_average=True)
    
    # PSNR (Peak Signal-to-Noise Ratio)
    mse_value = torch.mean((X - Y) ** 2)
    psnr_value = 20 * torch.log10(1.0 / torch.sqrt(mse_value))
    
    # MSE (Mean Squared Error)
    mse_value = torch.mean((X - Y) ** 2)
    
    # MAE (Mean Absolute Error)
    mae_value = torch.mean(torch.abs(X - Y))

    return ssim_value.item(), ms_ssim_value.item(), psnr_value.item(), mse_value.item(), mae_value.item()

# Example usage:
# X and Y should be tensors of shape (N, 3, H, W) with values in the range [0, 255]

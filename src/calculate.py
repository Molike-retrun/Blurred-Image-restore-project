
import torch

#psnr计算
def calculate_psnr(pred,taeget):

    mse = torch.mean(
        (pred - taeget) ** 2
    )

    if mse == 0:
        return torch.tensor(
            float("inf"),
            device=pred.device
        )

    psnr = 10 * torch.log10(1 / mse)
    return psnr

#每个batch中的psnr计算
def calculate_batch_psnr(pred, target):

    batch_psnr = []

    for i in range(pred.size(0)):

        mse = torch.mean(
            (pred[i] - target[i]) ** 2
        )

        if mse == 0:
            psnr = torch.tensor(
                float("inf"),
                device=pred.device
            )
        else:
            psnr = 10 * torch.log10(
                1.0 / mse
            )

        batch_psnr.append(psnr)

    return torch.stack(batch_psnr).mean()
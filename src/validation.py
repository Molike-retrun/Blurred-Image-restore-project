import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from src.calculate import calculate_batch_psnr

def validation(
        model,
        val_loader,
        criterion,
        device  
):
    
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0
    ).to(device)

    model.eval()

    val_total_loss = 0.0

    all_restored = []
    all_clean = []
    val_psnr_total = 0.0
    blurred_psnr_total = 0.0

    ssim_total = 0.0

    with torch.no_grad():

        for blurred,clean in val_loader:

            blurred = blurred.to(device)
            clean = clean.to(device)

            restored = model(blurred)


            loss = criterion(
                restored,
                clean
            )

            val_total_loss += loss.item()

            restored_eval = torch.clamp(
                restored,
                0,
                1
            )
            #计算SSIM
            ssim = ssim_metric(
                restored_eval,
                clean
            )

            ssim_total += ssim.item()
            #计算PSNR
            restored_psnr = calculate_batch_psnr(
                restored_eval,
                clean
            )

            blurred_psnr = calculate_batch_psnr(
                blurred,
                clean
            )
            
            val_psnr_total += restored_psnr.item()
            blurred_psnr_total += blurred_psnr.item()

            avg_val_psnr = (
                val_psnr_total / len(val_loader)
            )

            avg_blurred_psnr = (
                blurred_psnr_total / len(val_loader)
            )

    # restored 在 GPU 上，移动到 CPU 后可以减少 GPU 显存占用
    # detach() 的作用是切断 autograd 计算图，
    # 防止保存 restored 时继续保留反向传播所需的计算图
    #将输出的restored保存到all_restored列表中，clean保存到all_clean列表中
            all_restored.append(
                restored_eval.detach().cpu()
            )

                
            all_clean.append(
                clean.detach().cpu()
            )  

    #计算总的平均损失
        val_loss = val_total_loss/len(val_loader)

    #计算平均SSIM
        avg_ssim = ssim_total / len(val_loader)
    #将all_restored和all_clean拼接起来
        all_restored = torch.cat(
            all_restored,
            dim=0
        )

        all_clean = torch.cat(
            all_clean,  
            dim=0
        )

    #计算MAE和RMSE
        val_mae = torch.mean(
            torch.abs(
                all_restored - all_clean
            )
        )
    
        val_rmse = torch.sqrt(
            torch.mean(
                torch.pow(
                    all_restored - all_clean,
                    2
                )
            )
        ) 
    return val_loss, val_mae, val_rmse, avg_val_psnr, avg_blurred_psnr, avg_ssim
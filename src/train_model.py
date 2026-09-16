import torch
from torch import nn
from src.Train_one_epoch import train_one_epoch
from src.validation import validation
from config.config_loader import load_config
import logging

logger = logging.getLogger("ImageRestoration")

config = load_config("config/config.yaml")

def train_model(
        models,
        train_loader,
        val_loader,
        num_epochs,
        device,
        print_every,
        save_path
):
    model = models.to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config["TRAIN"]["LEARNING_RATE"]
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae": [],
        "val_rmse": [],
        "val_psnr": [],
        "val_ssim": [],
    }

    best_psnr = -float("inf")
    best_epoch = 0
    best_metrics = {}

    patience_counter = 0


    for epoch in range(num_epochs):

        train_loss = train_one_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer
        )
        val_loss, val_mae , val_rmse, avg_val_psnr, avg_blurred_psnr, avg_ssim = validation(
            model,
            val_loader,
            criterion,
            device  
        )

        if avg_val_psnr > best_psnr:    

            best_psnr = avg_val_psnr

            best_epoch = epoch + 1

            patience_counter = 0

            best_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae.item(),
                "val_rmse": val_rmse.item(),
                "val_psnr": avg_val_psnr,
                "blurred_psnr": avg_blurred_psnr,
                "val_ssim": avg_ssim,
                "best_epoch":epoch + 1
            }

            torch.save(
                model.state_dict(),
                save_path
            )

        else:
            patience_counter += 1

        history["train_loss"].append(train_loss)

        history["val_loss"].append(val_loss)

        history["val_mae"].append(
            val_mae.item()
        )

        history["val_rmse"].append(
            val_rmse.item()
        )

        history["val_psnr"].append(
            avg_val_psnr
        )

        history["val_ssim"].append(
            avg_ssim
        )  

        if patience_counter >= config["TRAIN"]["PATIENCE"]:
            logger.info(f"Early stopping triggered at epoch {epoch + 1}. Best PSNR: {best_psnr:.4f} at epoch {best_epoch}")
            break
        

        if (epoch + 1) % print_every == 0:
            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "                    
                f"Val MAE: {val_mae:.4f}, "
                f"Val RMSE: {val_rmse:.4f}, "                    
                f"Restored PSNR: {avg_val_psnr:.4f}, "
                f"Blurred PSNR: {avg_blurred_psnr:.4f}, "
                f"Relative SSIM: {avg_ssim:.4f},"
                f"Best PSNR: {best_psnr:.4f} at epoch {best_epoch}"
                )

    return model,history,best_metrics


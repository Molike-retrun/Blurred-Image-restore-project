import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("ImageRestoration")



def plot_loss(
        simple_history, 
        residual_history,
        UNet_history
    ):

    simple_epochs = range(1, len(simple_history["train_loss"]) + 1)
    residual_epochs = range(1, len(residual_history["train_loss"]) + 1)
    UNet_epochs = range(1, len(UNet_history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(
        simple_epochs,
        simple_history["train_loss"],
        label="Simple CNN Train Loss"
    )

    plt.plot(
        simple_epochs,
        simple_history["val_loss"],
        label="Simple CNN Val Loss"
    )

    plt.plot(
        residual_epochs,
        residual_history["train_loss"],
        label="Residual CNN Train Loss"
    )

    plt.plot(
        residual_epochs,
        residual_history["val_loss"],
        label="Residual CNN Val Loss"
    )

    plt.plot(
        UNet_epochs,
        UNet_history["train_loss"],
        label="UNet Train Loss"
    )

    plt.plot(
        UNet_epochs,
        UNet_history["val_loss"],       
        label="UNet Val Loss"
    )


    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.title("Training and Validation Loss")

    plt.legend()

    plt.grid()

    plt.show()

def plot_psnr(
        simple_history, 
        residual_history,
        UNet_history
    ):

    simple_epochs = range(1,len(simple_history["val_psnr"]) + 1)
    residual_epochs = range(1,len(residual_history["val_psnr"]) + 1)
    UNet_epochs = range(1,len(UNet_history["val_psnr"]) + 1)

    plt.figure(figsize = (10,5))

    plt.plot(
        simple_epochs,
        simple_history["val_psnr"],
        label = "simple CNN Val PSNR"
    )

    plt.plot(
        residual_epochs,
        residual_history["val_psnr"],
        label = "Residual CNN Val PSNR"
    )

    plt.plot(
        UNet_epochs,
        UNet_history["val_psnr"],
        label = "UNet Val PSNR"
    )

    plt.xlabel("Epoch")
    plt.ylabel("PSNR")

    plt.title("Validation PSNR")

    plt.legend()
    plt.grid()
    plt.show()



def plot_ssim(
        simple_history, 
        residual_history,
        UNet_history
    ):

    simple_epochs = range(1,len(simple_history["val_ssim"]) + 1)
    residual_epochs = range(1,len(residual_history["val_ssim"]) + 1)
    UNet_epochs = range(1,len(UNet_history["val_ssim"]) + 1)

    plt.plot(
        simple_epochs,
        simple_history["val_ssim"],
        label = "Simple CNN Val SSIM"
    )

    plt.plot(
        residual_epochs,
        residual_history["val_ssim"],
        label = "Residual CNN Val SSIM"
    )

    plt.plot(
        UNet_epochs,
        UNet_history["val_ssim"],
        label = "UNet Val SSIM" 
    )

    plt.xlabel("Epoch")
    plt.ylabel("SSIM")

    plt.title("Validation SSIM")

    plt.legend()
    plt.grid()
    plt.show()



def compare_models(
        simple_best_metrics,
        residual_best_metrics,
        unet_best_metrics
):

    logger.info("\nModel Comparison - Best Epoch")
    logger.info("-" * 75)

    logger.info(
        f"{'Metric':<15}"
        f"{'Simple CNN':<20}"
        f"{'Residual CNN':<20}"
        f"{'UNet':<20}"
    )

    logger.info("-" * 75)

    logger.info(
        f"{'Val Loss':<15}"
        f"{simple_best_metrics['val_loss']:<20.4f}"
        f"{residual_best_metrics['val_loss']:<20.4f}"
        f"{unet_best_metrics['val_loss']:<20.4f}"
    )

    logger.info(
        f"{'Val MAE':<15}"
        f"{simple_best_metrics['val_mae']:<20.4f}"
        f"{residual_best_metrics['val_mae']:<20.4f}"
        f"{unet_best_metrics['val_mae']:<20.4f}"
    )

    logger.info(
        f"{'Val RMSE':<15}"
        f"{simple_best_metrics['val_rmse']:<20.4f}"
        f"{residual_best_metrics['val_rmse']:<20.4f}"
        f"{unet_best_metrics['val_rmse']:<20.4f}"
    )

    logger.info(
        f"{'PSNR':<15}"
        f"{simple_best_metrics['val_psnr']:<20.4f}"
        f"{residual_best_metrics['val_psnr']:<20.4f}"
        f"{unet_best_metrics['val_psnr']:<20.4f}"
    )

    logger.info(
        f"{'SSIM':<15}"
        f"{simple_best_metrics['val_ssim']:<20.4f}"
        f"{residual_best_metrics['val_ssim']:<20.4f}"
        f"{unet_best_metrics['val_ssim']:<20.4f}"
    )

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from src.dataset import ImageRestorationDataset  
from src.model import (
    SimpleRestorationCNN,
    ResidualCNN,
    UNet
)
from src.train_model import train_model
from src.training_curve import plot_loss,plot_psnr,plot_ssim,compare_models
from config.config_loader import load_config
import os
from src.case_analysis import (
    analyze_cases,
    select_cases,
    print_case_results
)
from src.visualization import visualize_cases
from src.model_selection import select_best_model,rank_models
from src.utils import set_seed
from CV.logs.log_utils import setup_logger

def main():
    # =====================
    # 基础配置
    # =====================

    config = load_config("config/config.yaml")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    set_seed(config["TRAIN"]["SEED"])

    logger = setup_logger()

    # =====================
    # 数据准备Dataset
    # =====================

    logger.info("开始 Datasets and DataLoaders...")

    train_dataset = ImageRestorationDataset(
        config["PATH"]["TRAIN_DIR"]
    )
    val_dataset = ImageRestorationDataset(
        config["PATH"]["VAL_DIR"]
    )

    # =====================
    # 数据划分Dataloader
    # =====================

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = config["DATA"]["BATCH_SIZE"],
        shuffle = True
    )

    val_loader = DataLoader(
        dataset = val_dataset,
        batch_size = config["DATA"]["BATCH_SIZE"],
        shuffle = False
    )

    num_epochs = config["TRAIN"]["EPOCHS"]

    print_every = config["OUTPUT_EVERY"]

    os.makedirs(
        config["PATH"]["CHECKPOINT_DIR"],
        exist_ok=True
    )

    checkpoint_dir = Path(
        config["PATH"]["CHECKPOINT_DIR"]
    )

    # =====================
    # CNN训练及数据可视化
    # =====================
    logger.info("开始训练 SimpleRestorationCNN...")

    simple_checkpoint = (
        checkpoint_dir / "simple_cnn_best.pth"
    )

    simple_model,simple_history,simple_best_metrics = train_model(
        SimpleRestorationCNN(),
        train_loader,
        val_loader,
        num_epochs,
        device,
        print_every,
        simple_checkpoint
    )
    logger.info("训练完成 SimpleRestorationCNN")
    logger.info(
        f"model is {simple_model}\n"
        f"历史数据 {simple_history}\n"
        f"最佳指标 {simple_best_metrics}"
        )

    # =====================
    # Residual_CNN训练及数据可视化
    # =====================
    logger.info("开始训练 ResidualCNN...")

    simple_checkpoint = (
        checkpoint_dir / "residual_cnn_best.pth"
    )

    residual_model,residual_history,residual_best_metrics = train_model(
        ResidualCNN(),
        train_loader,
        val_loader,
        num_epochs,
        device,
        print_every,
        simple_checkpoint
    )
    logger.info("训练完成 ResidualCNN")
    logger.info(
        f"model is {residual_model}\n"
        f"历史数据 {residual_history}\n"
        f"最佳指标 {residual_best_metrics}"
        )

    # =====================
    # U-Net_CNN训练及数据可视化
    # =====================
    logger.info("开始训练 UNet...")

    simple_checkpoint = (
        checkpoint_dir / "unet_best.pth"
    )

    unet_model,unet_history,unet_best_metrics = train_model(
        UNet(),
        train_loader,
        val_loader,
        num_epochs,
        device,
        print_every,
        simple_checkpoint
    )

    logger.info("训练完成 UNet")
    logger.info(
        f"model is {unet_model}\n"
        f"历史数据 {unet_history}\n"
        f"最佳指标 {unet_best_metrics}"
        )

    # =====================
    # 最优模型选择
    # =====================
    logger.info("开始最优模型选择...")
    model_results = {

        "SimpleCNN": {
            "best_metrics": simple_best_metrics,
            "checkpoint": "checkpoints/simple_cnn_best.pth"
        },

        "ResidualCNN": {
            "best_metrics": residual_best_metrics,
            "checkpoint": "checkpoints/residual_cnn_best.pth"
        },

        "UNet": {
            "best_metrics": unet_best_metrics,
            "checkpoint": "checkpoints/unet_best.pth"
        }
    }

    ranking = rank_models(
        model_results,
        metric="val_psnr",
        higher_is_better=True
    )

    logger.info("\nModel Ranking")

    for rank, (model_name, result) in enumerate(
            ranking,
            start=1
    ):

        metrics = result["best_metrics"]

        logger.info(
            f"{rank}. {model_name} | "
            f"PSNR: {metrics['val_psnr']:.4f} | "
            f"SSIM: {metrics['val_ssim']:.4f}"
        )

    best_model_name, best_checkpoint, best_metrics = select_best_model(
        model_results,
        metric="val_psnr"
    )

    logger.info("\n" + "=" * 50)
    logger.info("Best Overall Model")
    logger.info("=" * 50)

    logger.info(
        f"Model: {best_model_name}"
    )

    logger.info(
        f"Checkpoint: {best_checkpoint}"
    )

    logger.info(
        f"PSNR: {best_metrics['val_psnr']:.4f}"
    )

    logger.info(
        f"SSIM: {best_metrics['val_ssim']:.4f}"
    )

    logger.info(
        f"Val Loss: {best_metrics['val_loss']:.4f}"
    )

    # =====================
    # 训练结果可视化
    # =====================
    logger.info("开始训练结果可视化...")
    plot_loss(
        simple_history,
        residual_history,
        unet_history
    )

    plot_psnr(
        simple_history,
        residual_history,
        unet_history
    )

    plot_ssim(
        simple_history,
        residual_history,
        unet_history
    )

    # =====================
    # 模型对比
    # =====================
    logger.info("开始模型对比...")
    compare_models(simple_best_metrics, residual_best_metrics, unet_best_metrics)

    # =========================
    # Load Best Residual CNN
    # =========================
    logger.info("开始加载最优模型...")
    # 建立“模型名 → 模型类”的映射
    MODEL_REGISTRY = {
        "ResidualCNN": ResidualCNN,
        "SimpleCNN": SimpleRestorationCNN,
        "UNet": UNet,
    }

    # 根据名字创建对应模型
    best_model = MODEL_REGISTRY[best_model_name]().to(device)

    #用path处理路径，保证取出来的模型是best_checkpoint
    checkpoint_path = Path(best_checkpoint)

    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path

    if not checkpoint_path.exists():
        logger.error(f"模型文件不存在: {checkpoint_path}")
        raise FileNotFoundError(f"模型文件不存在: {checkpoint_path}")

    best_model.load_state_dict(
        torch.load(
            checkpoint_path,#将路径转换为Path对象
            map_location=device
        )
    )

    best_model.eval()
    logger.info("模型加载完成")

    # =========================
    # Case Analysis
    # =========================

    logger.info("开始案例分析...")
    results = analyze_cases(
        best_model,
        val_loader,
        device
    )

    # =========================
    # Select Best / Normal / Bad
    # =========================

    best_case, normal_case, bad_case = select_cases(
        results
    )
    logger.info("案例分析完成")

    # =========================
    # Print Results
    # =========================
    logger.info("打印案例分析结果")
    print_case_results(
        best_case,
        normal_case,
        bad_case
    )

    visualize_cases(
        best_model,
        val_dataset,
        best_case,
        normal_case,
        bad_case,
        device
    )

if __name__ == "__main__":
    main()
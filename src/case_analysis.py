import torch

from src.calculate import calculate_psnr
from src.Image_analysis import(
    calculate_edge_strength,
    calculate_brightness
)
import logging

logger = logging.getLogger("ImageRestoration")


def analyze_cases(
    model,
    val_loader,
    device
):
    
    """
    使用 Best Model 对整个验证集进行逐图分析。

    对每张图片计算：
    1. Blurred PSNR
    2. Restored PSNR
    3. Improvement

    Improvement = Restored PSNR - Blurred PSNR
    """

    # =========================
    # 1. 切换到评估模式
    # =========================
    model.eval()

    # 保存所有图片的分析结果
    results = []

    # 记录当前图片在整个验证集中的索引
    image_index = 0

    # =========================
    # 2. 不需要计算梯度
    # =========================
    with torch.no_grad():

        # 遍历整个验证集
        for blurred, clean in val_loader:

            # =========================
            # 3. 将数据放到 GPU / CPU
            # =========================
            blurred = blurred.to(device)
            clean = clean.to(device)

            # =========================
            # 4. 使用 Best Model 推理
            # =========================
            restored = model(blurred)

            # =========================
            # 5. 将输出限制在 [0, 1]
            # =========================
            restored = torch.clamp(
                restored,
                0,
                1
            )

            # =========================
            # 6. 一个 batch 中逐张图片分析
            # =========================
            for i in range(blurred.size(0)):

                # 原始模糊图像的 PSNR
                blurred_psnr = calculate_psnr(
                    blurred[i],
                    clean[i]
                )

                # 模型恢复结果的 PSNR
                restored_psnr = calculate_psnr(
                    restored[i],
                    clean[i]
                )

                # 模型带来的 PSNR 提升
                improvement = (
                    restored_psnr - blurred_psnr
                )

                # ==========================
                # 图像内容特征
                # ==========================

                clean_edge = calculate_edge_strength(
                    clean[i]
                )

                clean_brightness = calculate_brightness(
                    clean[i]
                )

                blurred_edge = calculate_edge_strength(
                    blurred[i]
                )

                blurred_brightness = calculate_brightness(
                    blurred[i]
                )

                restored_edge = calculate_edge_strength(
                    restored[i]
                )

                restored_brightness = calculate_brightness(
                    restored[i]
                )

                blurred_edge_error = abs(
                    blurred_edge - clean_edge
                )

                restored_edge_error = abs(
                    restored_edge - clean_edge
                )

                blurred_brightness_error = abs(
                    blurred_brightness - clean_brightness
                )

                restored_brightness_error = abs(
                    restored_brightness - clean_brightness
                )


                # =========================
                # 7. 保存当前图片结果
                # =========================
                results.append({
                    "index": image_index,

                    "blurred_psnr":
                        blurred_psnr.item(),

                    "restored_psnr":
                        restored_psnr.item(),

                    "improvement":
                        improvement.item(),

                    "clean_edge":
                        clean_edge.item(),

                    "blurred_edge":
                        blurred_edge.item(),

                    "restored_edge":
                        restored_edge.item(),

                    "blurred_edge_error":
                        blurred_edge_error.item(),

                    "restored_edge_error":
                        restored_edge_error.item(),

                    "clean_brightness":
                        clean_brightness.item(),

                    "blurred_brightness":
                        blurred_brightness.item(),

                    "restored_brightness":
                        restored_brightness.item(),

                    "blurred_brightness_error":
                        blurred_brightness_error.item(),

                    "restored_brightness_error":
                        restored_brightness_error.item()
                })

                # 图片索引 +1
                image_index += 1

    return results


def select_cases(results):
    """
    根据 Improvement 自动选择：

    Best   : Improvement 最大
    Normal : Improvement 位于中间
    Bad    : Improvement 最小
    """

    # =========================
    # 1. 按 Improvement 从小到大排序，lambda代表根据什么排
    # =========================
    sorted_results = sorted(
        results,
        key=lambda x: x["improvement"]
    )

    # =========================
    # 2. Bad Case
    # =========================
    bad_case = sorted_results[0]

    # =========================
    # 3. Best Case
    # =========================
    best_case = sorted_results[-1]

    # =========================
    # 4. Normal Case 取中位数
    # =========================
    middle_index = len(sorted_results) // 2

    normal_case = sorted_results[middle_index]

    return (
        best_case,
        normal_case,
        bad_case
    )


def print_case_results(
    best_case,
    normal_case,
    bad_case
):
    """
    打印 Best / Normal / Bad Case 的结果。
    """

    logger.info("=" * 50)
    logger.info("Case Analysis")
    logger.info("=" * 50)

    logger.info("Best Case")
    logger.info(
        f"Image Index: {best_case['index']}"
    )
    logger.info(
        f"Blurred PSNR: "
        f"{best_case['blurred_psnr']:.4f}"
    )
    logger.info(
        f"Restored PSNR: "
        f"{best_case['restored_psnr']:.4f}"
    )
    logger.info(
        f"Improvement: "
        f"{best_case['improvement']:+.4f} dB"
    )

    logger.info(
        f"Edge Strength: "
        f"{best_case['clean_edge']:.2f}"
    )

    logger.info(
        f"Brightness: "
        f"{best_case['clean_brightness']:.4f}"
    )

    logger.info("Comparsion Results for Best Case")

    logger.info(
        f"edge_strength_error: "
        f"{best_case['blurred_edge_error']:.2f}"
    )

    logger.info(
        f"brightness_error: "
        f"{best_case['blurred_brightness_error']:.2f}"
    )

    logger.info(
        f"restored_edge_error: "
        f"{best_case['restored_edge_error']:.2f}"
    )

    logger.info(
        f"restored_brightness_error: "
        f"{best_case['restored_brightness_error']:.2f}"
    )

    logger.info(
        f"edge_recovery_gain: "
        f"{(
        best_case["blurred_edge_error"]
        - best_case["restored_edge_error"]):.2f}"
    )

    logger.info(
        f"brightness_recovery_gain: "
        f"{(
        best_case["blurred_brightness_error"]
        - best_case["restored_brightness_error"]):.2f}"
    )

    logger.info("\nNormal Case")
    logger.info(
        f"Image Index: {normal_case['index']}"
    )
    logger.info(
        f"Blurred PSNR: "
        f"{normal_case['blurred_psnr']:.4f}"
    )
    logger.info(
        f"Restored PSNR: "
        f"{normal_case['restored_psnr']:.4f}"
    )
    logger.info(
        f"Improvement: "
        f"{normal_case['improvement']:+.4f} dB"
    )

    logger.info(
        f"Edge Strength: "
        f"{normal_case['clean_edge']:.2f}"
    )

    logger.info(
        f"Brightness: "
        f"{normal_case['clean_brightness']:.4f}"
    )

    logger.info("Comparsion Results for Normal Case")
    
    logger.info(
        f"edge_strength_error: "
        f"{normal_case['blurred_edge_error']:.2f}"
        )
    
    logger.info(
        f"brightness_error: "
        f"{normal_case['blurred_brightness_error']:.2f}"
        )

    logger.info(
        f"edge_recovery_gain: "
        f"{(
        normal_case["blurred_edge_error"]
        - normal_case["restored_edge_error"]):.2f}"
    )

    logger.info(
        f"brightness_recovery_gain: "
        f"{(
        normal_case["blurred_brightness_error"]
        - normal_case["restored_brightness_error"]):.2f}"
    )

    logger.info("Bad Case")
    logger.info(
        f"Image Index: {bad_case['index']}"
    )
    logger.info(
        f"Blurred PSNR: "
        f"{bad_case['blurred_psnr']:.4f}"
    )
    logger.info(
        f"Restored PSNR: "
        f"{bad_case['restored_psnr']:.4f}"
    )
    logger.info(
        f"Improvement: "
        f"{bad_case['improvement']:+.4f} dB"
    )

    logger.info(
        f"Edge Strength: "
        f"{bad_case['clean_edge']:.2f}"
    )   
    logger.info(
        f"Brightness: "
        f"{bad_case['clean_brightness']:.4f}"
    )   

    logger.info("Comparsion Results for Bad Case")

    logger.info(
        f"edge_strength_error: "
        f"{bad_case['blurred_edge_error']:.2f}"
    )

    logger.info(
        f"brightness_error: "
        f"{bad_case['blurred_brightness_error']:.2f}"
    )   

    logger.info(
        f"edge_recovery_gain: "
        f"{(
        bad_case["blurred_edge_error"]
        - bad_case["restored_edge_error"]):.2f}"
    )

    logger.info(
        f"brightness_recovery_gain: "
        f"{(
        bad_case["blurred_brightness_error"]
        - bad_case["restored_brightness_error"]):.2f}"
    )

    logger.info( "=" * 50)

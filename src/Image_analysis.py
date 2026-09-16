import cv2
import numpy as np


def tensor_to_numpy(image):
    """
    Tensor:
    [C, H, W]

    ->
    
    numpy:
    [H, W, C]
    """

    image = image.permute(1, 2, 0).cpu().numpy()

    return image


def calculate_edge_strength(image):
    """
    使用 Laplacian 衡量图像边缘/高频信息强度
    """

    image = tensor_to_numpy(image)

    # [0,1] -> [0,255]
    image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F
    )

    edge_strength = laplacian.var()

    return edge_strength


def calculate_brightness(image):
    """
    计算平均亮度
    """

    image = tensor_to_numpy(image)

    image = (image * 255).astype(np.uint8)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2GRAY
    )

    brightness = gray.mean()

    return brightness
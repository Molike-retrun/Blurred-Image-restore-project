import torch
import matplotlib.pyplot as plt


def visualize_case(
        model,
        val_dataset,
        index,
        device
):

    #先打开评估，跑出对应图片
    model.eval()

    #从dataset中取出对应图片
    blurred, clean = val_dataset[index]

    #增加图片维度，将其从chw变为nchw，即替代dataloader
    blurred_input = blurred.unsqueeze(0).to(device)

    #跑模型
    with torch.no_grad():
        restored = model(blurred_input)

    #将batch维度去掉，恢复为chw
    restored = restored.squeeze(0).cpu()

    #限制到[0,1]
    restored = torch.clamp(restored,0,1)

    #绘图需要将格式转换回hwc
    blurred = blurred.permute(1,2,0)
    restored = restored.permute(1,2,0)
    clean = clean.permute(1,2,0)

    #绘图
    plt.figure(figsize = (12,4))

    plt.subplot(1,3,1)
    plt.imshow(blurred)
    plt.title("Blurred")
    plt.axis("off")


    plt.subplot(1,3,2)
    plt.imshow(restored)
    plt.title("Restored")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.imshow(clean)
    plt.title("Clean")
    plt.axis("off")

    plt.suptitle(
        f"Image_index:{index}",
        fontsize = 14
    )

    plt.tight_layout()
    plt.show()


def visualize_cases(
        model,
        val_dataset,
        best_case,
        normal_case,
        bad_case,
        device
):   

    #上一函数可视化模糊图，还原图，清晰图，这次可视化最优最劣
    cases = [
        ("best_case",best_case),
        ("normal_case",normal_case),
        ("bad_case",bad_case)
    ]

    model.eval()

    plt.figure(figsize = (12,12))

    #将cases打包成case_name和case形式，方便后续操作
    for row,(case_name,case) in enumerate(cases):

        """
        cases里面是之前保存的case，形式为
        results.append({
            "index": image_index,
            "blurred_psnr": blurred_psnr.item(),
            "restored_psnr": restored_psnr.item(),
            "improvement": improvement.item()
        })
        """

        index = case["index"]

        blurred,clean = val_dataset[index]

        blurred_input = blurred.unsqueeze(0).to(device)

        with torch.no_grad():
            restored = model(blurred_input)

        restored = restored.squeeze(0).cpu()
        restored = torch.clamp(restored,0,1)

        blurred = blurred.permute(1,2,0)
        restored = restored.permute(1,2,0)
        clean = clean.permute(1,2,0)

        #比如说取best_case中的blurred，row=0，则放在第一个位置
        #同理，若此时取restored_case，row=1，则1*3+1 = 4,第二行第一个
        plt.subplot(3,3,row*3+1)
        plt.imshow(blurred)
        plt.title(
            f"{case_name}\nBlurred"
        )
        plt.axis("off") 

        # -------------------------
        # Restored
        # -------------------------
        plt.subplot(3, 3, row * 3 + 2)
        plt.imshow(restored)
        plt.title(
            f"Restored\n"
            f"Improvement: {case['improvement']:+.2f} dB"
        )
        plt.axis("off")

        # -------------------------
        # Clean
        # -------------------------
        plt.subplot(3, 3, row * 3 + 3)
        plt.imshow(clean)
        plt.title("Clean")
        plt.axis("off")

    plt.tight_layout()
    plt.show()





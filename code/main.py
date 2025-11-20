# ======================================================
# main.py | Final Version — Self-Supervised Representation Learning Pipeline
# ======================================================

import os
import torch
import pandas as pd
from config_data import get_cifar100
from pretrain_mae import train_mae
from pretrain_rotation import train_rotation
from finetune_classification import finetune_vit
from tasks_multitask import run_multitask_finetune

from visualize_representation import visualize_tsne_umap
from evaluate_representation_metrics import evaluate_representation

from utils import measure_training_time, measure_inference_speed, timer, plot_data_scale


# ======================================================
# 自动创建输出目录
# ======================================================
os.makedirs("./outputs/checkpoints", exist_ok=True)
os.makedirs("./outputs/tensorboard", exist_ok=True)
os.makedirs("./outputs", exist_ok=True)


# ======================================================
# 主函数入口
# ======================================================
def main():
    print("\n🚀 Self-Supervised Representation Learning Pipeline Start\n")

    # ======================================================
    # 1️⃣ 加载 CIFAR-100 数据
    # ======================================================
    train_loader, val_loader, test_loader = get_cifar100(batch_size=256)
    print(f"✅ CIFAR-100 数据加载完成 | 训练样本: {len(train_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mae_ckpt = "./outputs/checkpoints/mae_vit.pth"
    rot_ckpt = "./outputs/checkpoints/rotation_vit.pth"

    # ======================================================
    # 2️⃣ 自监督预训练阶段 (MAE + Rotation)
    # ======================================================
    mae_pretrain_time = rot_pretrain_time = 0.0

    if not os.path.exists(mae_ckpt):
        print("\n🧩 开始 MAE 自监督预训练 (200 epochs)")
        _, mae_pretrain_time = measure_training_time(train_mae, train_loader, epochs=150, save_path=mae_ckpt)
    else:
        print("✅ 已检测到 MAE checkpoint，跳过预训练")

    if not os.path.exists(rot_ckpt):
        print("\n🔄 开始 Rotation 自监督预训练 (150 epochs)")
        _, rot_pretrain_time = measure_training_time(train_rotation, train_loader, epochs=120, save_path=rot_ckpt)
    else:
        print("✅ 已检测到 Rotation checkpoint，跳过预训练")

    # ======================================================
    # 3️⃣ 分类任务微调对比（Random / MAE / Rotation）
    # ======================================================
    print("\n🎯 分类任务微调对比实验")

    baseline_acc = 46.22   # 有监督ViT baseline
    baseline_time = 44.23  # baseline 训练时间 (min)
    results = [("Random (Baseline)", baseline_acc)]

    mae_acc, mae_finetune_time = measure_training_time(
        finetune_vit, train_loader, val_loader, epochs=50,
        pretrain_path=mae_ckpt,
        save_path="./outputs/checkpoints/ft_MAE.pth",
        log_dir="./outputs/tensorboard/ft_MAE"
    )

    rot_acc, rot_finetune_time = measure_training_time(
        finetune_vit, train_loader, val_loader, epochs=50,
        pretrain_path=rot_ckpt,
        save_path="./outputs/checkpoints/ft_Rotation.pth",
        log_dir="./outputs/tensorboard/ft_Rotation"
    )

    results += [("MAE", mae_acc), ("Rotation", rot_acc)]
    df_finetune = pd.DataFrame(results, columns=["Type", "ValAcc(%)"])
    df_finetune.to_csv("./outputs/finetune_compare.csv", index=False)
    print("✅ 分类任务结果已保存至 ./outputs/finetune_compare.csv")

    # ======================================================
    # 4️⃣ 表示可视化与定量评估
    # ======================================================
    print("\n📊 表示可视化 (Random / Supervised / MAE / Rotation)")
    
    visualize_tsne_umap(num_classes_to_show=100)
    print("\n📈 表示质量定量评估 (KNN / Alignment / Uniformity)")
    evaluate_representation()
    print("✅ 表示质量指标已完成并输出")

    # ======================================================
    # 5️⃣ 多任务下游任务微调（上色 + 去噪）
    # ======================================================
    print("\n🧠 下游任务：上色 + 去噪 (Colorization / Denoising)")
    for name, path in [
        ("MAE", mae_ckpt),
        ("Rotation", rot_ckpt)
    ]:
        run_multitask_finetune(pretrain_path=path,
                               output_dir=f"./outputs/{name}_tasks",
                               epochs=50)
    print("✅ 多任务下游任务完成")

    # ======================================================
    # 6️⃣ 数据规模影响实验 (0.5 × vs 1.0 ×)
    # ======================================================

    print("\n📦 数据规模影响实验启动")
    ratios = [0.3,0.6]

    scale_results = [(1.0, mae_acc)]  # baseline ratio=1 先加入

    for r in ratios:
        print(f"\n📉 当前数据比例: {r}")
        sub_ckpt = f"./outputs/checkpoints/mae_{int(r*100)}.pth"
        tr, vl, _ = get_cifar100(batch_size=128, subset_ratio=r, mode="mae")
           
        if not os.path.exists(sub_ckpt):
           train_mae(tr, epochs=50, save_path=sub_ckpt)

        acc, _ = measure_training_time(
            finetune_vit, tr, vl, epochs=50,
            pretrain_path=sub_ckpt,
            save_path=f"./outputs/checkpoints/ft_MAE_scale_{int(r*100)}.pth",
            log_dir=f"./outputs/tensorboard/ft_MAE_scale_{int(r*100)}"
        )
        scale_results.append((r, acc))

    df_scale = pd.DataFrame(scale_results, columns=["DataRatio", "Accuracy(%)"])
    df_scale.to_csv("./outputs/data_scale.csv", index=False)
    print("✅ 数据规模实验结果已保存至 ./outputs/data_scale.csv")

    # ======================================================
    # 7️⃣ 计算效率分析 (训练时间 / 推理速度 / 精度提升)
    # ======================================================

    print("\n🕒 计算效率分析")

    infer_mae = measure_inference_speed("./outputs/checkpoints/ft_MAE.pth", val_loader, device)
    infer_rot = measure_inference_speed("./outputs/checkpoints/ft_Rotation.pth", val_loader, device)

    infer_random = measure_inference_speed("/workspace/CIFAR_project/runs/cifar100_vit_tiny_aug_20251027_223443/best_model_vit_tiny.pth", val_loader, device)

    efficiency = pd.DataFrame([
        ["Random Init", None, baseline_time, infer_random, baseline_acc],
        ["MAE", mae_pretrain_time, mae_finetune_time, infer_mae, mae_acc],
        ["Rotation", rot_pretrain_time, rot_finetune_time, infer_rot, rot_acc],
    ], columns=["Model", "PretrainTime(min)", "FinetuneTime(min)", "InferTime(s/100)", "Accuracy(%)"])

    efficiency.to_csv("./outputs/efficiency_detail.csv", index=False)
    print("✅ 效率分析结果已保存至 ./outputs/efficiency_detail.csv")

    # ======================================================
    # 8️⃣ 绘制 数据规模 vs 分类精度 曲线
    # ======================================================
    print("\n📈 绘制 MAE 预训练数据规模 vs 分类精度 曲线")
    plot_data_scale("./outputs/data_scale.csv", "./outputs/plot_data_scale.png")

    # ======================================================
    # 9️⃣ 输出结果汇总
    # ======================================================
    print("\n✅ 全流程完成！主要输出文件:")
    print(" ├─ finetune_compare.csv")
    print(" ├─ data_scale.csv")
    print(" ├─ plot_data_scale.png")
    print(" ├─ efficiency_detail.csv")
    print(" ├─ representation_tsne_umap.png")
    print(" ├─ MAE_tasks/ & Rotation_tasks/ 图像输出")
    print(" └─ tensorboard/ 日志")


# ======================================================
# 程序入口
# ======================================================
if __name__ == "__main__":
    main()




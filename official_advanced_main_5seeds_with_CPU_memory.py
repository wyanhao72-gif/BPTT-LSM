"""
TCN、官方Transformer、官方Informer和官方Autoformer
在ETTh1、Lorenz、NARMA上的五随机种子实验。

种子：42、43、44、45、46，共5次。

本程序复用原工程：
    linear_main.net_train
    linear_main.net_test
    time_series_data_prediction.dataset_from

每个模型-任务-种子目录保存：
1. checkpoint_max.pth、checkpoint_latest.pth；
2. 每代训练/测试的NRMSE、RMSE、NMSE、MSE、MAE；
3. 每代训练wall-clock time；
4. 每代peak GPU memory；
5. 每代全局梯度L2范数；
6. grad.txt中各参数梯度最小值、最大值和均值。

每个模型-任务目录保存五种子汇总：
1. 每次实验总训练wall-clock time；
2. 每次实验训练峰值GPU显存；
3. 每次实验最优测试轮的五项性能指标；
4. 上述量在5个随机种子上的最小值、最大值、均值和方差。
"""

import datetime
import os
import random
import time
import psutil

import numpy as np
import openpyxl
import torch
from torch.cuda import amp

import linear_main
from time_series_data_prediction import dataset_from
from official_advanced_models import create_advanced_model


SEEDS = [1112]#, 1113, 1114, 1115, 1116
MODEL_NAMES = ["transformer"]#,,,,"informer","autoformer"，"tcn"

# =========================================================
# 完整10任务
# =========================================================
TASK_CONFIGS = [
{
    "task_name": "GR1Robot",
    "dataset_name": "GR1Robot",
    "lorenz_gap": 1,
    "narma_order": 20,
    "robot_gap": 1
}


]

METRIC_NAMES = ["NRMSE", "RMSE", "NMSE", "MSE", "MAE"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_float_array(values):
    return np.asarray(
        [
            value.detach().cpu().item()
            if torch.is_tensor(value)
            else float(value)
            for value in values
        ],
        dtype=np.float64
    )


def descriptive_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=0))
    }


def save_epoch_gradient_statistics(
    out_dir,
    iteration,
    net,
    gradient_l2_norm
):
    """
    保存本次运行中每一代、每个可训练参数梯度的：
    1. 最小值；
    2. 最大值；
    3. 平均值；
    4. 全网络梯度L2范数。
    """
    os.makedirs(out_dir, exist_ok=True)

    file_path = os.path.join(
        out_dir,
        "grad_statistics.txt"
    )

    with open(file_path, "a", encoding="utf-8") as file:
        file.write(f"######{iteration}\n")

        for name, parameter in net.named_parameters():
            if parameter.grad is None:
                continue

            grad = parameter.grad.detach()

            file.write(
                f"grad {name}: "
                f"{grad.min().item():.6e} ~ "
                f"{grad.max().item():.6e}\t"
                f"mean:{grad.mean().item():.6e}\n"
            )

        file.write(
            f"gradient global L2 norm: "
            f"{gradient_l2_norm:.6e}\n"
        )


def save_run_excel(
    file_path,
    train_result_list,
    test_result_list,
    efficiency_result_list,
    gradient_l2_list,
    completed_iterations,
    total_training_wall_time,
    run_peak_gpu_memory_mb,
    run_peak_cpu_memory_mb,
    best_iteration,
    best_test_metrics
):
    wb = openpyxl.Workbook()
    default_sheet = wb.active
    wb.remove(default_sheet)

    train_excel = wb.create_sheet("train")
    test_excel = wb.create_sheet("test")

    # 与原代码一致：一个指标占一行。
    for metric_index in range(
        train_result_list.shape[1]
    ):
        train_excel.append(
            train_result_list[
                :completed_iterations,
                metric_index
            ].tolist()
        )

        test_excel.append(
            test_result_list[
                :completed_iterations,
                metric_index
            ].tolist()
        )

    efficiency_excel = wb.create_sheet("efficiency")
    efficiency_excel.append(
        [
            "iteration",
            "train_wall_clock_time_s",
            "peak_gpu_memory_mb",
            "peak_cpu_memory_mb",
            "gradient_global_l2_norm"
        ]
    )

    for iteration in range(completed_iterations):
        efficiency_excel.append(
            [
                iteration,
                efficiency_result_list[iteration, 0],
                efficiency_result_list[iteration, 1],
                efficiency_result_list[iteration, 2],
                gradient_l2_list[iteration]
            ]
        )

    run_summary = wb.create_sheet("run_summary")
    run_summary.append(["item", "value"])

    run_summary.append(
        [
            "total_training_wall_clock_time_s",
            total_training_wall_time
        ]
    )

    run_summary.append(
        [
            "run_peak_gpu_memory_mb",
            run_peak_gpu_memory_mb
        ]
    )

    run_summary.append(
        [
            "run_peak_cpu_memory_mb",
            run_peak_cpu_memory_mb
        ]
    )

    run_summary.append(
        [
            "best_iteration_by_test_NRMSE",
            best_iteration
        ]
    )

    for name, value in zip(
        METRIC_NAMES,
        best_test_metrics
    ):
        run_summary.append(
            [
                f"best_test_{name}",
                value
            ]
        )

    wb.save(file_path)


def save_model_task_summary(
    summary_path,
    run_records
):
    wb = openpyxl.Workbook()

    ws_runs = wb.active
    ws_runs.title = "five_runs"

    headers = [
        "seed",
        "best_iteration",
        "total_training_wall_clock_time_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb"
    ] + [
        f"test_{name}"
        for name in METRIC_NAMES
    ]

    ws_runs.append(headers)

    for record in run_records:
        ws_runs.append(
            [
                record["seed"],
                record["best_iteration"],
                record["total_wall_time"],
                record["peak_gpu_memory_mb"],
                record["peak_cpu_memory_mb"],
                *record["best_test_metrics"]
            ]
        )

    ws_stats = wb.create_sheet("statistics")

    ws_stats.append(
        [
            "indicator",
            "min",
            "max",
            "mean",
            "variance"
        ]
    )

    indicator_values = {
        "total_training_wall_clock_time_s": [
            record["total_wall_time"]
            for record in run_records
        ],
        "peak_gpu_memory_mb": [
            record["peak_gpu_memory_mb"]
            for record in run_records
        ],
        "peak_cpu_memory_mb": [
            record["peak_cpu_memory_mb"]
            for record in run_records
        ]
    }

    for metric_index, metric_name in enumerate(
        METRIC_NAMES
    ):
        indicator_values[
            f"test_{metric_name}"
        ] = [
            record["best_test_metrics"][
                metric_index
            ]
            for record in run_records
        ]

    for indicator, values in (
        indicator_values.items()
    ):
        stats = descriptive_statistics(
            values
        )

        ws_stats.append(
            [
                indicator,
                stats["min"],
                stats["max"],
                stats["mean"],
                stats["variance"]
            ]
        )

    wb.save(summary_path)


def run_single_experiment(
    model_name,
    task,
    seed,
    device,
    hidden_dim=90,
    max_iteration=200,
    circle_iteration=100,
    opt="sgd",
    lr=5e-6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      ,
    momentum=0.9,
    use_amp=False
):
    set_seed(seed)

    task_name = task["task_name"]
    dataset_name = task["dataset_name"]

    input_dim, output_dim, train_set, test_set, predict_step = (
        dataset_from(
            name=dataset_name,
            device=device,
            random_bptt=False,
            narma_order=task["narma_order"],
            lorenz_gap=task["lorenz_gap"],
            robot_gap=task["robot_gap"]
        )
    )

    net = create_advanced_model(
        model_name=model_name,
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim
    ).to(device)

    if opt == "sgd":
        optimizer = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum
        )

    elif opt == "adam":
        optimizer = torch.optim.Adam(
            net.parameters(),
            lr=lr
        )

    else:
        raise NotImplementedError(opt)

    lr_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer,
            circle_iteration
        )
    )

    scaler = None

    output_root = (
        "./official_advanced_models_amp"
        if use_amp
        else "./official_advanced_models"
    )

    if use_amp:
        scaler = amp.GradScaler()

    # =====================================================
    # 关键：使用task_name作为目录，防止gap/order任务相互覆盖
    # =====================================================
    run_dir = os.path.join(
        output_root,
        model_name,
        task_name,
        f"seed_{seed}",
        f"t({max_iteration})_{task_name}_lr{lr}"
        f"({datetime.date.today()})"
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    print(
        f"Output directory: {run_dir}"
    )

    print(
        f"model={model_name}, "
        f"task={task_name}, "
        f"dataset={dataset_name}, "
        f"lorenz_gap={task['lorenz_gap']}, "
        f"narma_order={task['narma_order']}, "
        f"robot_gap={task['robot_gap']}, "
        f"predict_step={predict_step}"
    )

    min_test_nrmse = 100.0
    best_iteration = -1

    best_test_metrics = np.full(
        5,
        np.nan,
        dtype=np.float64
    )

    train_result_list = np.zeros(
        (max_iteration, 5),
        dtype=np.float64
    )

    test_result_list = np.zeros(
        (max_iteration, 5),
        dtype=np.float64
    )

    efficiency_result_list = np.zeros(
        (max_iteration, 3),
        dtype=np.float64
    )

    gradient_l2_list = np.zeros(
        max_iteration,
        dtype=np.float64
    )

    # 关闭原le_backward专用路径；
    # linear_main仍会保存grad.txt和全局L2。
    debug_grad = False

    process = psutil.Process(
        os.getpid()
    )

    for iteration in range(
        max_iteration
    ):
        start_time = time.time()
        draw_pic = False

        net.train()

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

            torch.cuda.reset_peak_memory_stats(
                device
            )

        peak_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        epoch_train_start_time = (
            time.perf_counter()
        )

        train_return = linear_main.net_train(
            device=device,
            dataset=train_set,
            optimizer=optimizer,
            net=net,
            scaler=scaler,
            out_dir=run_dir,
            iteration=iteration,
            draw_pic=draw_pic,
            debug_grad=debug_grad,
            debug_net=net
        )

        if device.type == "cuda":
            torch.cuda.synchronize(
                device
            )

        current_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        peak_cpu_memory_mb = max(
            peak_cpu_memory_mb,
            current_cpu_memory_mb
        )

        epoch_train_wall_time = (
            time.perf_counter()
            - epoch_train_start_time
        )

        if device.type == "cuda":
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(
                    device
                )
                / (1024 ** 2)
            )
        else:
            peak_gpu_memory_mb = 0.0

        if len(train_return) < 6:
            raise RuntimeError(
                "linear_main.net_train()必须返回6个值："
                "NRMSE、RMSE、NMSE、MSE、MAE、"
                "gradient_l2_norm。"
                f"当前只返回了{len(train_return)}个值。"
            )

        train_metrics = to_float_array(
            train_return[:5]
        )

        gradient_l2_norm = float(
            train_return[5]
        )

        train_result_list[
            iteration
        ] = train_metrics

        efficiency_result_list[
            iteration
        ] = [
            epoch_train_wall_time,
            peak_gpu_memory_mb,
            peak_cpu_memory_mb
        ]

        gradient_l2_list[
            iteration
        ] = gradient_l2_norm

        save_epoch_gradient_statistics(
            out_dir=run_dir,
            iteration=iteration,
            net=net,
            gradient_l2_norm=(
                gradient_l2_norm
            )
        )

        lr_scheduler.step()

        net.eval()

        test_return = linear_main.net_test(
            device=device,
            dataset=test_set,
            net=net,
            draw_pic=draw_pic
        )

        test_metrics = to_float_array(
            test_return[:5]
        )

        test_result_list[
            iteration
        ] = test_metrics

        test_nrmse = test_metrics[0]

        save_max = False

        if test_nrmse < min_test_nrmse:
            min_test_nrmse = (
                test_nrmse
            )

            best_iteration = iteration

            best_test_metrics = (
                test_metrics.copy()
            )

            save_max = True

        checkpoint = {
            "net": net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": (
                lr_scheduler.state_dict()
            ),
            "iteration": iteration,
            "min_test_nrmse": (
                min_test_nrmse
            ),
            "seed": seed,
            "model": model_name,
            "task_name": task_name,
            "dataset": dataset_name,
            "lorenz_gap": (
                task["lorenz_gap"]
            ),
            "narma_order": (
                task["narma_order"]
            ),
            "robot_gap": (
                task["robot_gap"]
            )
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    run_dir,
                    "checkpoint_max.pth"
                )
            )

        torch.save(
            checkpoint,
            os.path.join(
                run_dir,
                "checkpoint_latest.pth"
            )
        )

        completed_iterations = (
            iteration + 1
        )

        total_training_wall_time_so_far = float(
            np.sum(
                efficiency_result_list[
                    :completed_iterations,
                    0
                ]
            )
        )

        run_peak_gpu_memory_so_far = float(
            np.max(
                efficiency_result_list[
                    :completed_iterations,
                    1
                ]
            )
        )

        run_peak_cpu_memory_so_far = float(
            np.max(
                efficiency_result_list[
                    :completed_iterations,
                    2
                ]
            )
        )

        excel_path = os.path.join(
            run_dir,
            f"{model_name}_{task_name}_"
            f"{predict_step}_seed_{seed}_"
            f"({datetime.date.today()}).xlsx"
        )

        save_run_excel(
            file_path=excel_path,
            train_result_list=(
                train_result_list
            ),
            test_result_list=(
                test_result_list
            ),
            efficiency_result_list=(
                efficiency_result_list
            ),
            gradient_l2_list=(
                gradient_l2_list
            ),
            completed_iterations=(
                completed_iterations
            ),
            total_training_wall_time=(
                total_training_wall_time_so_far
            ),
            run_peak_gpu_memory_mb=(
                run_peak_gpu_memory_so_far
            ),
            run_peak_cpu_memory_mb=(
                run_peak_cpu_memory_so_far
            ),
            best_iteration=(
                best_iteration
            ),
            best_test_metrics=(
                best_test_metrics
            )
        )

        print(
            f"model={model_name}, "
            f"task={task_name}, "
            f"dataset={dataset_name}, "
            f"seed={seed}, "
            f"iteration={iteration}, "
            f"train_nrmse="
            f"{train_metrics[0]:.4f}, "
            f"test_nrmse="
            f"{test_nrmse:.4f}, "
            f"min_test_nrmse="
            f"{min_test_nrmse:.4f}"
        )

        print(
            f"train wall-clock="
            f"{epoch_train_wall_time:.4f}s, "
            f"peak GPU memory="
            f"{peak_gpu_memory_mb:.2f}MB, "
            f"peak CPU memory="
            f"{peak_cpu_memory_mb:.2f}MB, "
            f"gradient L2 norm="
            f"{gradient_l2_norm:.6e}"
        )

        print(
            "escape time = "
            f"{(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (max_iteration - iteration))).strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    total_training_wall_time = float(
        np.sum(
            efficiency_result_list[:, 0]
        )
    )

    run_peak_gpu_memory_mb = float(
        np.max(
            efficiency_result_list[:, 1]
        )
    )

    run_peak_cpu_memory_mb = float(
        np.max(
            efficiency_result_list[:, 2]
        )
    )

    return {
        "seed": seed,
        "best_iteration": (
            best_iteration
        ),
        "total_wall_time": (
            total_training_wall_time
        ),
        "peak_gpu_memory_mb": (
            run_peak_gpu_memory_mb
        ),
        "peak_cpu_memory_mb": (
            run_peak_cpu_memory_mb
        ),
        "best_test_metrics": (
            best_test_metrics.tolist()
        )
    }


def main():
    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Total tasks: "
        f"{len(TASK_CONFIGS)}"
    )

    for model_name in MODEL_NAMES:
        for task in TASK_CONFIGS:
            task_name = (
                task["task_name"]
            )

            run_records = []

            print()
            print("=" * 90)

            print(
                f"START "
                f"model={model_name}, "
                f"task={task_name}, "
                f"dataset="
                f"{task['dataset_name']}, "
                f"lorenz_gap="
                f"{task['lorenz_gap']}, "
                f"narma_order="
                f"{task['narma_order']}, "
                f"robot_gap="
                f"{task['robot_gap']}"
            )

            print("=" * 90)

            for seed in SEEDS:
                record = (
                    run_single_experiment(
                        model_name=model_name,
                        task=task,
                        seed=seed,
                        device=device
                    )
                )

                run_records.append(
                    record
                )

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_task_dir = os.path.join(
                "./official_advanced_models",
                model_name,
                task_name
            )

            os.makedirs(
                model_task_dir,
                exist_ok=True
            )

            summary_path = os.path.join(
                model_task_dir,
                f"summary_5seeds_"
                f"{model_name}_"
                f"{task_name}_"
                f"({datetime.date.today()}).xlsx"
            )

            save_model_task_summary(
                summary_path,
                run_records
            )

            print(
                f"Saved five-seed summary: "
                f"{summary_path}"
            )


if __name__ == "__main__":
    main()

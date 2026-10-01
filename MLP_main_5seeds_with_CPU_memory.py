"""
MLP线性网络时间序列预测主程序。

在原有MLP模型、dataset_from、SGD/Adam、CosineAnnealingLR、
200代训练、测试、checkpoint和Excel保存逻辑基础上，仅增加：

1. 每个任务使用随机种子42、43、44、45、46运行5次；
2. 每代保存训练/测试的NRMSE、RMSE、NMSE、MSE、MAE；
3. net_train返回第六位全局梯度L2范数；
4. 每代保存每个参数梯度的最小值、最大值、平均值；
5. 每代保存训练wall-clock时间和训练峰值GPU显存；
6. 每次运行保存总训练wall-clock时间和运行峰值GPU显存；
7. 每个任务完成5次后，保存总时间、峰值显存及5项测试指标的
   最小值、最大值、均值和方差。
"""

import datetime
import os
import random
import time
import psutil

import numpy as np
import openpyxl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp

import narma_estimate
import time_series_data_prediction as tp
import le_backward as lebw
from time_series_data_prediction import dataset_from


SEEDS = [1112]#, 1113, 1114, 1115, 1116
TASK_CONFIGS = [
    {"task_name": "Lorenz63_gap1",  "dataset_name": "Lorenz63", "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz63_gap5",  "dataset_name": "Lorenz63", "lorenz_gap": 5,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz63_gap10", "dataset_name": "Lorenz63", "lorenz_gap": 10, "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap1",  "dataset_name": "Lorenz84", "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap5",  "dataset_name": "Lorenz84", "lorenz_gap": 5,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap10", "dataset_name": "Lorenz84", "lorenz_gap": 10, "narma_order": 20, "robot_gap": 1},
    {"task_name": "NARMA10",        "dataset_name": "NARMA",    "lorenz_gap": 1,  "narma_order": 10, "robot_gap": 1},
    {"task_name": "NARMA20",        "dataset_name": "NARMA",    "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "ETTh1", "dataset_name": "ETTh1", "lorenz_gap": 1, "narma_order": 20, "robot_gap": 1},
    {"task_name": "GR1Robot", "dataset_name": "GR1Robot", "lorenz_gap": 1, "narma_order": 20, "robot_gap": 1},

]
METRIC_NAMES = ["NRMSE", "RMSE", "NMSE", "MSE", "MAE"]


class BatchNorm1dLN(nn.Module):
    """
    BatchNorm1d for input of shape (L, N, C)
    """

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        # x: (L, N, C)
        L, N, C = x.shape
        # -> (N, C, L)
        x = x.permute(1, 2, 0)
        x = self.bn(x)
        # -> (L, N, C)
        x = x.permute(2, 0, 1)
        return x


class three_linear_net(nn.Module):
    def __init__(
        self,
        in_dim,
        n_hidden,
        out_dim,
        history_step=1,
        for_lsm=False
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            BatchNorm1dLN(n_hidden),
            nn.LeakyReLU()
        )

        if for_lsm:
            self.layer2 = nn.Sequential(
                nn.Linear(
                    2 * history_step * n_hidden,
                    out_dim
                )
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(n_hidden, out_dim)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_float(value):
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def to_metric_array(values):
    return np.asarray(
        [to_float(value) for value in values],
        dtype=np.float64
    )


def get_loss(dataset, device, net):
    if type(dataset) is tuple:
        input_dataset, output_dataset = dataset
        input_dataset = input_dataset.to(device)
        output_dataset = output_dataset.to(device)
        s, d = input_dataset.shape
        input_dataset = input_dataset.reshape(s, 1, d)
        outputs = net.forward(
            input_dataset
        ).reshape(output_dataset.shape)
    else:
        outputs = torch.tensor([]).to(device)
        output_dataset_all = torch.tensor([]).to(device)

        for dataset_elem in dataset:
            input_dataset, output_dataset = dataset_elem
            s, d = input_dataset.shape
            input_dataset = input_dataset.reshape(s, 1, d)
            outputs_part = net.forward(
                input_dataset
            ).reshape(output_dataset.shape)
            outputs = torch.cat(
                [outputs, outputs_part],
                dim=0
            )
            output_dataset_all = torch.cat(
                [output_dataset_all, output_dataset],
                dim=0
            )

        output_dataset = output_dataset_all.clone()

    loss = F.mse_loss(outputs, output_dataset)
    return loss, outputs, output_dataset


def global_gradient_l2_norm(net):
    squared_sum = 0.0

    for parameter in net.parameters():
        if (
            parameter.requires_grad
            and parameter.grad is not None
        ):
            squared_sum += (
                parameter.grad.detach()
                .pow(2)
                .sum()
                .item()
            )

    return squared_sum ** 0.5


def collect_gradient_statistics(net):
    """
    收集当前一次backward后各参数梯度的min、max和mean。
    """
    statistics = {}

    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        statistics[name] = {
            "min": float(grad.min().item()),
            "max": float(grad.max().item()),
            "mean": float(grad.mean().item())
        }

    return statistics


def save_gradient_statistics(
    out_dir,
    iteration,
    gradient_statistics,
    gradient_l2_norm
):
    """
    每次运行单独目录，因此grad.txt不会与其他种子相互覆盖。
    """
    file_path = os.path.join(
        out_dir,
        "grad.txt"
    )

    with open(
        file_path,
        "a",
        encoding="utf-8"
    ) as file:
        file.write(f"######{iteration}\n")

        for name, values in gradient_statistics.items():
            file.write(
                f"grad {name}: "
                f"{values['min']:.6e} ~ "
                f"{values['max']:.6e}\t"
                f"mean:{values['mean']:.6e}\n"
            )

        file.write(
            f"gradient global L2 norm: "
            f"{gradient_l2_norm:.6e}\n"
        )


# 训练函数
def net_train(
    device,
    dataset,
    optimizer,
    net,
    scaler,
    out_dir=None,
    iteration=None,
    draw_pic=False,
    debug_grad=False,
    debug_net=None
):
    optimizer.zero_grad()

    do_debug = False
    do_print_grad = True
    do_grad_clip = True
    detect_grad_nan = True

    gradient_net = (
        debug_net
        if debug_net is not None
        else net
    )

    if scaler is not None:
        with amp.autocast():
            loss, outputs, output_dataset = get_loss(
                dataset=dataset,
                device=device,
                net=net
            )

        if debug_grad:
            lebw.test_grad_error(
                net=gradient_net,
                loss=loss,
                do_debug=do_debug,
                do_print_grad=do_print_grad,
                do_grad_clip=do_grad_clip,
                detect_grad_nan=detect_grad_nan,
                out_dir=out_dir,
                optimizer=optimizer,
                iteration=iteration,
                scaler=scaler
            )
        else:
            scaler.scale(loss).backward()

        # AMP下必须先unscale，才能得到真实梯度。
        scaler.unscale_(optimizer)

        gradient_l2_norm = global_gradient_l2_norm(
            gradient_net
        )
        gradient_statistics = collect_gradient_statistics(
            gradient_net
        )

        scaler.step(optimizer)
        scaler.update()

    else:
        loss, outputs, output_dataset = get_loss(
            dataset=dataset,
            device=device,
            net=net
        )

        if debug_grad:
            lebw.test_grad_error(
                net=gradient_net,
                loss=loss,
                do_debug=do_debug,
                do_print_grad=do_print_grad,
                do_grad_clip=do_grad_clip,
                detect_grad_nan=detect_grad_nan,
                out_dir=out_dir,
                optimizer=optimizer,
                iteration=iteration
            )
        else:
            loss.backward()

        gradient_l2_norm = global_gradient_l2_norm(
            gradient_net
        )
        gradient_statistics = collect_gradient_statistics(
            gradient_net
        )

        optimizer.step()

    nrmse, rmse, nmse, mse, mae = (
        narma_estimate.nrmse(
            outputs,
            output_dataset
        )
    )

    if draw_pic:
        tp.draw_lorenz(
            predict=outputs.cpu().detach(),
            true=output_dataset.cpu().detach()
        )

    # 前五位为性能指标，第六位为梯度L2。
    return (
        nrmse,
        rmse,
        nmse,
        mse,
        mae,
        gradient_l2_norm,
        gradient_statistics
    )


# 测试函数
def net_test(
    device,
    dataset,
    net,
    draw_pic=False
):
    with torch.no_grad():
        loss, outputs, output_dataset = get_loss(
            dataset=dataset,
            device=device,
            net=net
        )

    nrmse, rmse, nmse, mse, mae = (
        narma_estimate.nrmse(
            outputs,
            output_dataset
        )
    )

    if draw_pic:
        tp.draw_lorenz(
            predict=outputs.cpu().detach(),
            true=output_dataset.cpu().detach()
        )

    return nrmse, rmse, nmse, mse, mae


def descriptive_statistics(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        # 五次重复实验采用总体方差。
        "variance": float(np.var(values, ddof=0))
    }


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
    best_test_metrics,
    seed,
    dataset_name
):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    train_excel = wb.create_sheet("train")
    test_excel = wb.create_sheet("test")

    train_excel.append([
        "iteration",
        *METRIC_NAMES
    ])
    test_excel.append([
        "iteration",
        *METRIC_NAMES
    ])

    for iteration in range(completed_iterations):
        train_excel.append([
            iteration,
            *train_result_list[iteration].tolist()
        ])
        test_excel.append([
            iteration,
            *test_result_list[iteration].tolist()
        ])

    efficiency_excel = wb.create_sheet("efficiency")
    efficiency_excel.append([
        "iteration",
        "train_wall_clock_time_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "gradient_global_l2_norm"
    ])

    for iteration in range(completed_iterations):
        efficiency_excel.append([
            iteration,
            efficiency_result_list[iteration, 0],
            efficiency_result_list[iteration, 1],
            efficiency_result_list[iteration, 2],
            gradient_l2_list[iteration]
        ])

    run_summary = wb.create_sheet("run_summary")
    run_summary.append(["item", "value"])
    run_summary.append(["model", "MLP"])
    run_summary.append(["dataset", dataset_name])
    run_summary.append(["seed", seed])
    run_summary.append([
        "total_training_wall_clock_time_s",
        total_training_wall_time
    ])
    run_summary.append([
        "run_peak_gpu_memory_mb",
        run_peak_gpu_memory_mb
    ])
    run_summary.append([
        "run_peak_cpu_memory_mb",
        run_peak_cpu_memory_mb
    ])
    run_summary.append([
        "best_iteration_by_test_NRMSE",
        best_iteration
    ])

    for metric_name, value in zip(
        METRIC_NAMES,
        best_test_metrics
    ):
        run_summary.append([
            f"best_test_{metric_name}",
            float(value)
        ])

    wb.save(file_path)


def save_five_seed_summary(
    summary_path,
    run_records
):
    wb = openpyxl.Workbook()

    runs_ws = wb.active
    runs_ws.title = "five_runs"

    runs_ws.append([
        "seed",
        "best_iteration",
        "total_training_wall_clock_time_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "test_NRMSE",
        "test_RMSE",
        "test_NMSE",
        "test_MSE",
        "test_MAE"
    ])

    for record in run_records:
        runs_ws.append([
            record["seed"],
            record["best_iteration"],
            record["total_wall_time"],
            record["peak_gpu_memory_mb"],
            record["peak_cpu_memory_mb"],
            *record["best_test_metrics"]
        ])

    stats_ws = wb.create_sheet("statistics")
    stats_ws.append([
        "indicator",
        "min",
        "max",
        "mean",
        "variance"
    ])

    indicators = {
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
        indicators[f"test_{metric_name}"] = [
            record["best_test_metrics"][metric_index]
            for record in run_records
        ]

    for indicator, values in indicators.items():
        stats = descriptive_statistics(values)

        stats_ws.append([
            indicator,
            stats["min"],
            stats["max"],
            stats["mean"],
            stats["variance"]
        ])

    wb.save(summary_path)


def run_single_experiment(
    task,
    seed,
    device
):
    set_seed(seed)

    task_name = task["task_name"]
    dataset_name = task["dataset_name"]

    # 保持原逻辑：默认不恢复中断。
    resume = ""

    node_num = 90
    random_bptt = False

    (
        input_dim,
        output_dim,
        train_set,
        test_set,
        predict_step
    ) = dataset_from(
        name=dataset_name,
        device=device,
        random_bptt=random_bptt,
        narma_order=task["narma_order"],
        lorenz_gap=task["lorenz_gap"],
        robot_gap=task["robot_gap"]
    )

    # MLP模型结构保持不变。
    net = three_linear_net(
        in_dim=input_dim,
        n_hidden=node_num,
        out_dim=output_dim
    ).to(device)

    # 优化器设置保持不变。
    opt = "sgd"
    lr = 1e-2
    momentum = 0.9

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

    max_iteration = 200
    use_amp = True

    lr_scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            max_iteration
        )
    )

    output_root = "./linear1"
    scaler = None

    if use_amp:
        output_root += "_amp"
        scaler = amp.GradScaler()

    out_dir = os.path.join(
        output_root,
        "MLP",
        task_name,
        f"seed_{seed}",
        f"t({max_iteration})_{task_name}_lr{lr}"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    print(f"Output directory: {out_dir}")

    start_epoch = 0
    min_test_nrmse = 100.0
    best_iteration = -1
    best_test_metrics = np.full(
        5,
        np.nan,
        dtype=np.float64
    )

    if resume:
        checkpoint = torch.load(
            resume,
            map_location="cpu"
        )
        net.load_state_dict(checkpoint["net"])
        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )
        lr_scheduler.load_state_dict(
            checkpoint["lr_scheduler"]
        )
        start_epoch = (
            checkpoint["iteration"] + 1
        )
        min_test_nrmse = (
            checkpoint["min_test_nrmse"]
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

    process = psutil.Process(os.getpid())

    for iteration in range(
        start_epoch,
        max_iteration
    ):
        start_time = time.time()

        if iteration in [50, 100, 150, 199]:
            draw_pic = False
        else:
            draw_pic = False

        net.train()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        peak_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        epoch_train_start_time = (
            time.perf_counter()
        )

        train_result = net_train(
            device=device,
            dataset=train_set,
            optimizer=optimizer,
            net=net,
            scaler=scaler,
            out_dir=out_dir,
            iteration=iteration,
            draw_pic=draw_pic,
            debug_grad=False,
            debug_net=net
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)

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

        # 前五位：NRMSE、RMSE、NMSE、MSE、MAE。
        train_metrics = to_metric_array(
            train_result[:5]
        )

        # 第六位：梯度全局二范数。
        gradient_l2_norm = float(
            train_result[5]
        )

        # 第七位仅用于继续保存原有的min/max/mean。
        gradient_statistics = train_result[6]

        train_nrmse = train_metrics[0]
        train_result_list[iteration] = (
            train_metrics
        )
        gradient_l2_list[iteration] = (
            gradient_l2_norm
        )
        efficiency_result_list[iteration] = [
            epoch_train_wall_time,
            peak_gpu_memory_mb,
            peak_cpu_memory_mb
        ]

        save_gradient_statistics(
            out_dir=out_dir,
            iteration=iteration,
            gradient_statistics=gradient_statistics,
            gradient_l2_norm=gradient_l2_norm
        )

        lr_scheduler.step()

        net.eval()
        test_result = net_test(
            device=device,
            dataset=test_set,
            net=net,
            draw_pic=draw_pic
        )

        test_metrics = to_metric_array(
            test_result[:5]
        )
        test_nrmse = test_metrics[0]
        test_result_list[iteration] = (
            test_metrics
        )

        save_max = False

        if test_nrmse < min_test_nrmse:
            min_test_nrmse = test_nrmse
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
            "min_test_nrmse": min_test_nrmse,
            "seed": seed,
            "model": "MLP",
            "task_name": task_name,
            "dataset": dataset_name,
            "lorenz_gap": task["lorenz_gap"],
            "narma_order": task["narma_order"],
            "robot_gap": task["robot_gap"]
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    out_dir,
                    "checkpoint_max.pth"
                )
            )

        torch.save(
            checkpoint,
            os.path.join(
                out_dir,
                "checkpoint_latest.pth"
            )
        )

        completed_iterations = iteration + 1

        total_training_wall_time = float(
            np.sum(
                efficiency_result_list[
                    :completed_iterations,
                    0
                ]
            )
        )

        run_peak_gpu_memory_mb = float(
            np.max(
                efficiency_result_list[
                    :completed_iterations,
                    1
                ]
            )
        )

        run_peak_cpu_memory_mb = float(
            np.max(
                efficiency_result_list[
                    :completed_iterations,
                    2
                ]
            )
        )

        file_name = (
            f"ETT_{predict_step}_MLP_"
            f"{task_name}_seed_{seed}_"
            f"({datetime.date.today()}).xlsx"
        )

        save_run_excel(
            file_path=os.path.join(
                out_dir,
                file_name
            ),
            train_result_list=train_result_list,
            test_result_list=test_result_list,
            efficiency_result_list=(
                efficiency_result_list
            ),
            gradient_l2_list=gradient_l2_list,
            completed_iterations=(
                completed_iterations
            ),
            total_training_wall_time=(
                total_training_wall_time
            ),
            run_peak_gpu_memory_mb=(
                run_peak_gpu_memory_mb
            ),
            run_peak_cpu_memory_mb=(
                run_peak_cpu_memory_mb
            ),
            best_iteration=best_iteration,
            best_test_metrics=best_test_metrics,
            seed=seed,
            dataset_name=task_name
        )

        print(out_dir)
        print(
            f"dataset={dataset_name}, "
            f"seed={seed}, "
            f"iteration={iteration}, "
            f"train_nrmse={train_nrmse:.4f}, "
            f"test_nrmse={test_nrmse:.4f}, "
            f"min_test_nrmse={min_test_nrmse:.4f}"
        )
        print(
            f"train wall-clock="
            f"{epoch_train_wall_time:.4f}s, "
            f"peak GPU memory="
            f"{peak_gpu_memory_mb:.2f}MB, "
            f"peak CPU memory="
            f"{peak_cpu_memory_mb:.2f}MB, "
            f"gradient L2="
            f"{gradient_l2_norm:.6e}"
        )
        print(
            "escape time = "
            f"{(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (max_iteration - iteration))).strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    return {
        "seed": seed,
        "best_iteration": best_iteration,
        "total_wall_time": float(
            np.sum(
                efficiency_result_list[:, 0]
            )
        ),
        "peak_gpu_memory_mb": float(
            np.max(
                efficiency_result_list[:, 1]
            )
        ),
        "peak_cpu_memory_mb": float(
            np.max(
                efficiency_result_list[:, 2]
            )
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

    print(f"Device: {device}")
    print(f"Total tasks: {len(TASK_CONFIGS)}")

    for task in TASK_CONFIGS:
        task_name = task["task_name"]
        run_records = []

        print()
        print("=" * 90)
        print(
            f"START task={task_name}, "
            f"dataset={task['dataset_name']}, "
            f"lorenz_gap={task['lorenz_gap']}, "
            f"narma_order={task['narma_order']}, "
            f"robot_gap={task['robot_gap']}"
        )
        print("=" * 90)

        for seed in SEEDS:
            record = run_single_experiment(
                task=task,
                seed=seed,
                device=device
            )
            run_records.append(record)

            if device.type == "cuda":
                torch.cuda.empty_cache()

        task_dir = os.path.join(
            "./linear1_amp",
            "MLP",
            task_name
        )
        os.makedirs(
            task_dir,
            exist_ok=True
        )

        summary_path = os.path.join(
            task_dir,
            f"summary_5seeds_MLP_"
            f"{task_name}_"
            f"({datetime.date.today()}).xlsx"
        )

        save_five_seed_summary(
            summary_path,
            run_records
        )

        print(
            f"Saved five-seed summary: "
            f"{summary_path}"
        )


if __name__ == "__main__":
    main()

"""
脉冲时间依赖可塑性（STDP）训练主程序。

在原有STDP-LSM训练逻辑基础上保留：
- LSM初始化；
- STDP学习器；
- 梯度下降优化器Adam；
- STDP优化器SGD；
- 梯度显示与梯度裁剪；
- 两个余弦学习率调度器；
- 200代训练；
- 测试、网络重置和STDP学习器重置；
- checkpoint保存。

新增：
1. 每个任务使用随机种子42、43、44、45、46运行5次；
2. 每代训练输出前5位为NRMSE、RMSE、NMSE、MSE、MAE，
   第6位为梯度下降参数的全局梯度L2范数；
3. 每代保存原有梯度min/max/mean及第6位L2范数；
4. 每代保存完整训练阶段wall-clock时间及峰值GPU显存；
5. 每次运行保存总训练wall-clock时间及运行峰值GPU显存；
6. 每个任务完成5次后，对时间、显存和5项测试指标计算
   最小值、最大值、均值和方差；
7. 新增并保存两个严格区分的时间指标：
   - total_training_wall_clock_time_s：
     仅累计每代完整训练阶段，包括GD训练和STDP更新；
   - total_reservoir_preparation_cost_s：
     储层初始化时间与全部训练时间之和，不包含数据集读取、
     测试、绘图、checkpoint、Excel及日志保存。
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
from spikingjelly.activation_based import learning, functional
from torch.cuda import amp
from torch.optim import SGD, Adam

import le_backward as bkwd
import linear_main
import lsm_net
from ETD_predict_main import dataset_from
from main import init_popular


SEEDS = [1112]  # , 1113, 1114, 1115, 1116

TASK_CONFIGS = [
    {"task_name": "Lorenz63_gap1",  "dataset_name": "Lorenz63", "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz63_gap5",  "dataset_name": "Lorenz63", "lorenz_gap": 5,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz63_gap10", "dataset_name": "Lorenz63", "lorenz_gap": 10, "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap1",  "dataset_name": "Lorenz84", "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap5",  "dataset_name": "Lorenz84", "lorenz_gap": 5,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "Lorenz84_gap10", "dataset_name": "Lorenz84", "lorenz_gap": 10, "narma_order": 20, "robot_gap": 1},
    {"task_name": "NARMA10",        "dataset_name": "NARMA",    "lorenz_gap": 1,  "narma_order": 10, "robot_gap": 1},
    {"task_name": "NARMA20",        "dataset_name": "NARMA",    "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "ETTh1",          "dataset_name": "ETTh1",    "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
    {"task_name": "GR1Robot",       "dataset_name": "GR1Robot", "lorenz_gap": 1,  "narma_order": 20, "robot_gap": 1},
]

METRIC_NAMES = [
    "NRMSE",
    "RMSE",
    "NMSE",
    "MSE",
    "MAE"
]

MAX_ITERATION = 200
CIRCLE_ITERATION = 100
USE_AMP = False


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
        return float(
            value.detach().cpu().item()
        )
    return float(value)


def to_metric_array(values):
    return np.asarray(
        [to_float(value) for value in values],
        dtype=np.float64
    )


def gradient_l2_norm_from_parameters(parameters):
    """
    计算指定梯度下降参数集合的全局L2范数。

    这里统计params_gradient_descent，对应Adam优化器负责的参数，
    不把STDP专用突触参数重复计入。
    """
    squared_sum = 0.0

    for parameter in parameters:
        if parameter.grad is not None:
            squared_sum += float(
                parameter.grad.detach()
                .pow(2)
                .sum()
                .item()
            )

    return squared_sum ** 0.5


def collect_gradient_statistics(net):
    """
    读取当前各参数梯度的最小值、最大值和平均值。
    """
    result = {}

    for name, parameter in net.named_parameters():
        if parameter.grad is None:
            continue

        grad = parameter.grad.detach()

        result[name] = {
            "min": float(grad.min().item()),
            "max": float(grad.max().item()),
            "mean": float(grad.mean().item())
        }

    return result


def save_gradient_statistics(
    out_dir,
    iteration,
    gradient_statistics,
    gradient_l2_norm
):
    """
    每个种子使用独立目录，因此grad.txt不会相互覆盖。
    """
    path = os.path.join(
        out_dir,
        "grad.txt"
    )

    with open(
        path,
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
            f"gradient descent global L2 norm: "
            f"{gradient_l2_norm:.6e}\n"
        )


def descriptive_statistics(values):
    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        # 五次完整实验采用总体方差。
        "variance": float(
            np.var(values, ddof=0)
        )
    }


def save_run_excel(
    file_path,
    train_results,
    test_results,
    efficiency_results,
    gradient_l2_values,
    completed_iterations,
    total_training_wall_time,
    reservoir_initialization_wall_clock_time,
    total_reservoir_preparation_cost,
    run_peak_gpu_memory_mb,
    run_peak_cpu_memory_mb,
    best_iteration,
    best_test_metrics,
    seed,
    dataset_name
):
    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    train_ws = wb.create_sheet("train")
    test_ws = wb.create_sheet("test")

    train_ws.append([
        "iteration",
        *METRIC_NAMES
    ])
    test_ws.append([
        "iteration",
        *METRIC_NAMES
    ])

    for iteration in range(completed_iterations):
        train_ws.append([
            iteration,
            *train_results[iteration].tolist()
        ])
        test_ws.append([
            iteration,
            *test_results[iteration].tolist()
        ])

    efficiency_ws = wb.create_sheet(
        "efficiency"
    )
    efficiency_ws.append([
        "iteration",
        "train_wall_clock_time_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "gradient_descent_global_l2_norm"
    ])

    for iteration in range(completed_iterations):
        efficiency_ws.append([
            iteration,
            efficiency_results[iteration, 0],
            efficiency_results[iteration, 1],
            efficiency_results[iteration, 2],
            gradient_l2_values[iteration]
        ])

    summary_ws = wb.create_sheet(
        "run_summary"
    )
    summary_ws.append(["item", "value"])
    summary_ws.append(["model", "STDP-LSM"])
    summary_ws.append(["dataset", dataset_name])
    summary_ws.append(["seed", seed])
    summary_ws.append([
        "total_training_wall_clock_time_s",
        total_training_wall_time
    ])
    summary_ws.append([
        "reservoir_initialization_wall_clock_time_s",
        reservoir_initialization_wall_clock_time
    ])
    summary_ws.append([
        "total_reservoir_preparation_cost_s",
        total_reservoir_preparation_cost
    ])
    summary_ws.append([
        "run_peak_gpu_memory_mb",
        run_peak_gpu_memory_mb
    ])
    summary_ws.append([
        "run_peak_cpu_memory_mb",
        run_peak_cpu_memory_mb
    ])
    summary_ws.append([
        "best_iteration_by_test_NRMSE",
        best_iteration
    ])

    for metric_name, value in zip(
        METRIC_NAMES,
        best_test_metrics
    ):
        summary_ws.append([
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
        "reservoir_initialization_wall_clock_time_s",
        "total_reservoir_preparation_cost_s",
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
            record["total_training_wall_clock_time"],
            record["reservoir_initialization_wall_clock_time"],
            record["total_reservoir_preparation_cost"],
            record["peak_gpu_memory_mb"],
            record["peak_cpu_memory_mb"],
            *record["best_test_metrics"]
        ])

    stats_ws = wb.create_sheet(
        "statistics"
    )
    stats_ws.append([
        "indicator",
        "min",
        "max",
        "mean",
        "variance"
    ])

    indicators = {
        "total_training_wall_clock_time_s": [
            record["total_training_wall_clock_time"]
            for record in run_records
        ],
        "reservoir_initialization_wall_clock_time_s": [
            record["reservoir_initialization_wall_clock_time"]
            for record in run_records
        ],
        "total_reservoir_preparation_cost_s": [
            record["total_reservoir_preparation_cost"]
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


def build_stdp_model_and_optimizers(
    input_dim,
    output_dim,
    node_num,
    device,
    lr,
    momentum
):
    """
    完整保留原STDP模型和优化器划分逻辑。
    """
    v_th = 0.5

    (
        rc_weight_array,
        tau_reciprocal
    ) = init_popular(
        pop_num=1,
        node_num=node_num
    )[0]

    spike_v_init = True
    train_s_and_v = True

    net = lsm_net.Lsm(
        tau=1 / tau_reciprocal,
        v_threshold=v_th,
        lsm_node=node_num,
        init_neuron=spike_v_init,
        rc_spike_init=None,
        rc_v_init=None,
        device=device,
        in_features=input_dim,
        out_features=output_dim,
        train_s_and_v=train_s_and_v
    )

    net = bkwd.set_net_state(
        net=net,
        mode="test"
    )

    functional.set_step_mode(
        net=net,
        step_mode="m"
    )

    net.to(device)

    stdp_learners = []
    step_mode = "m"
    tau_pre = 2.0
    tau_post = 100.0

    def f_weight(x):
        return torch.clamp(
            x,
            -1,
            1.0
        )

    synapse_layer = [
        net.input_layer[0],
        net.rc[0].fc
    ]
    sn_layer = net.rc[0].sub_module

    params_stdp = []
    params_no_train = []

    for layer_index in range(
        len(synapse_layer)
    ):
        stdp_learners.append(
            learning.STDPLearner(
                step_mode=step_mode,
                synapse=synapse_layer[layer_index],
                sn=sn_layer,
                tau_pre=tau_pre,
                tau_post=tau_post,
                f_pre=f_weight,
                f_post=f_weight
            )
        )

        for parameter in (
            synapse_layer[layer_index]
            .parameters()
        ):
            params_stdp.append(parameter)
            params_no_train.append(parameter)

    # LIF膜时间常数不训练。
    for parameter in sn_layer.parameters():
        params_no_train.append(parameter)
        parameter.requires_grad = False

    params_no_train_set = set(
        params_no_train
    )

    params_gradient_descent = []

    for parameter in net.parameters():
        if parameter not in params_no_train_set:
            params_gradient_descent.append(
                parameter
            )

    optimizer_stdp = SGD(
        params_stdp,
        lr=lr * 0.01,
        momentum=momentum
    )

    optimizer_gd = Adam(
        params_gradient_descent,
        lr=lr
    )

    return (
        net,
        stdp_learners,
        params_gradient_descent,
        optimizer_stdp,
        optimizer_gd
    )


def run_single_experiment(
    task,
    seed,
    device
):
    set_seed(seed)

    task_name = task["task_name"]
    dataset_name = task["dataset_name"]
    lorenz_gap = task["lorenz_gap"]
    narma_order = task["narma_order"]
    robot_gap = task["robot_gap"]

    # 保留原默认逻辑：不恢复中断。
    resume = ""

    node_num = 90
    net_name = "lsm"

    (
        input_dim,
        output_dim,
        train_set,
        test_set,
        predict_step
    ) = dataset_from(
        name=dataset_name,
        device=device,
        random_bptt=(
            net_name == "rnn"
        ),
        narma_order=narma_order,
        lorenz_gap=lorenz_gap,
        robot_gap=robot_gap
    )

    opt = "sgd"
    lr = 5e-6
    momentum = 0.9
    debug_grad = False

    # ============================================================
    # Reservoir initialization wall-clock time
    #
    # 统计范围：
    #   init_popular
    #   LSM实例化与搬移到device
    #   STDP学习器构建
    #   Adam和SGD优化器构建
    #   两个学习率调度器构建
    #
    # 不包含：
    #   dataset_from及数据集读取
    #   输出目录构建
    #   测试、绘图与文件保存
    # ============================================================
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reservoir_initialization_start = time.perf_counter()

    (
        net,
        stdp_learners,
        params_gradient_descent,
        optimizer_stdp,
        optimizer_gd
    ) = build_stdp_model_and_optimizers(
        input_dim=input_dim,
        output_dim=output_dim,
        node_num=node_num,
        device=device,
        lr=lr,
        momentum=momentum
    )

    lr_scheduler_stdp = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer_stdp,
            CIRCLE_ITERATION
        )
    )

    lr_scheduler = (
        torch.optim.lr_scheduler
        .CosineAnnealingLR(
            optimizer_gd,
            CIRCLE_ITERATION
        )
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    reservoir_initialization_wall_clock_time = (
        time.perf_counter()
        - reservoir_initialization_start
    )

    output_root = "./stdp"
    scaler = None

    if USE_AMP:
        output_root += "_amp"
        scaler = amp.GradScaler()

    out_dir = os.path.join(
        output_root,
        net_name,
        task_name,
        f"seed_{seed}",
        f"t({MAX_ITERATION})_"
        f"{task_name}_lr{lr}"
        f"({datetime.date.today()})"
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    print(
        f"Output directory: {out_dir}"
    )

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
        net.load_state_dict(
            checkpoint["net"]
        )
        optimizer_stdp.load_state_dict(
            checkpoint["optimizer_stdp"]
        )
        optimizer_gd.load_state_dict(
            checkpoint["optimizer_gd"]
        )
        lr_scheduler.load_state_dict(
            checkpoint["lr_scheduler"]
        )
        lr_scheduler_stdp.load_state_dict(
            checkpoint["lr_scheduler_stdp"]
        )
        start_epoch = (
            checkpoint["iteration"] + 1
        )
        min_test_nrmse = (
            checkpoint["min_test_nrmse"]
        )

    train_result_list = np.zeros(
        (MAX_ITERATION, 5),
        dtype=np.float64
    )

    test_result_list = np.zeros(
        (MAX_ITERATION, 5),
        dtype=np.float64
    )

    efficiency_result_list = np.zeros(
        (MAX_ITERATION, 3),
        dtype=np.float64
    )

    gradient_l2_list = np.zeros(
        MAX_ITERATION,
        dtype=np.float64
    )

    process = psutil.Process(os.getpid())

    for iteration in range(
        start_epoch,
        MAX_ITERATION
    ):
        iteration_start_time = time.time()

        if iteration in [
            0,
            50,
            100,
            150,
            199
        ]:
            draw_pic = False
        else:
            draw_pic = False

        net.train()
        optimizer_stdp.zero_grad()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(
                device
            )

        peak_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        # STDP模型的一代训练时间包括：
        # GD反向传播与更新、STDP梯度生成、显示、裁剪及STDP更新。
        epoch_train_start = (
            time.perf_counter()
        )

        train_result = linear_main.net_train(
            device=device,
            dataset=train_set,
            optimizer=optimizer_gd,
            net=net,
            scaler=scaler,
            out_dir=out_dir,
            iteration=iteration,
            draw_pic=draw_pic,
            debug_grad=debug_grad,
            debug_net=net
        )

        if len(train_result) < 6:
            raise RuntimeError(
                "linear_main.net_train()必须返回6项："
                "NRMSE、RMSE、NMSE、MSE、MAE、"
                "gradient_l2_norm。"
            )

        # 前五位性能指标。
        train_metrics = to_metric_array(
            train_result[:5]
        )

        # 第六位优先采用linear_main返回的GD梯度L2。
        gradient_l2_norm = to_float(
            train_result[5]
        )

        # 防止旧linear_main计算了整个net而非明确GD参数集合，
        # 这里在STDP梯度写入前重新按Adam参数集合核验。
        gd_gradient_l2_norm = (
            gradient_l2_norm_from_parameters(
                params_gradient_descent
            )
        )

        if np.isfinite(
            gd_gradient_l2_norm
        ):
            gradient_l2_norm = (
                gd_gradient_l2_norm
            )

        # 此时保存的是GD反向传播完成后的梯度min/max/mean。
        gradient_statistics = (
            collect_gradient_statistics(net)
        )

        optimizer_stdp.zero_grad()

        for learner in stdp_learners:
            learner.step(on_grad=True)

        show_grad = True
        do_print_grad = show_grad
        do_grad_clip = True

        # 保留原梯度显示。
        if show_grad:
            bkwd.grad_show(
                net=net,
                out_dir=out_dir,
                do_print_grad=do_print_grad,
                iteration=iteration
            )

        # 保留原梯度裁剪。
        if do_grad_clip:
            for clip_parameter in filter(
                lambda parameter:
                parameter.requires_grad,
                net.parameters()
            ):
                nn.utils.clip_grad_norm_(
                    clip_parameter,
                    max_norm=(
                        0.1
                        * clip_parameter.data.numel()
                    ),
                    norm_type=2
                )

        optimizer_stdp.step()

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
            - epoch_train_start
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

        # 保存GD阶段梯度min/max/mean和第六位L2。
        save_gradient_statistics(
            out_dir=out_dir,
            iteration=iteration,
            gradient_statistics=(
                gradient_statistics
            ),
            gradient_l2_norm=(
                gradient_l2_norm
            )
        )

        lr_scheduler.step()
        lr_scheduler_stdp.step()

        net.eval()

        test_result = linear_main.net_test(
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

        functional.reset_net(net)

        for learner in stdp_learners:
            learner.reset()

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
            "optimizer_gd": (
                optimizer_gd.state_dict()
            ),
            "optimizer_stdp": (
                optimizer_stdp.state_dict()
            ),
            "lr_scheduler": (
                lr_scheduler.state_dict()
            ),
            "lr_scheduler_stdp": (
                lr_scheduler_stdp.state_dict()
            ),
            "iteration": iteration,
            "min_test_nrmse": (
                min_test_nrmse
            ),
            "seed": seed,
            "model": "STDP-LSM",
            "task_name": task_name,
            "dataset": dataset_name,
            "lorenz_gap": lorenz_gap,
            "narma_order": narma_order,
            "robot_gap": robot_gap
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

        completed_iterations = (
            iteration + 1
        )

        total_training_wall_time = float(
            np.sum(
                efficiency_result_list[
                    :completed_iterations,
                    0
                ]
            )
        )

        # Total cost of reservoir preparation:
        # 储层初始化 + 截至当前代的全部训练计算。
        # 测试、绘图、梯度日志、checkpoint和Excel保存不计入。
        total_reservoir_preparation_cost = (
            reservoir_initialization_wall_clock_time
            + total_training_wall_time
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
            f"ETT_{predict_step}_"
            f"STDP_{task_name}_"
            f"seed_{seed}_"
            f"({datetime.date.today()})"
            f".xlsx"
        )

        save_run_excel(
            file_path=os.path.join(
                out_dir,
                file_name
            ),
            train_results=(
                train_result_list
            ),
            test_results=(
                test_result_list
            ),
            efficiency_results=(
                efficiency_result_list
            ),
            gradient_l2_values=(
                gradient_l2_list
            ),
            completed_iterations=(
                completed_iterations
            ),
            total_training_wall_time=(
                total_training_wall_time
            ),
            reservoir_initialization_wall_clock_time=(
                reservoir_initialization_wall_clock_time
            ),
            total_reservoir_preparation_cost=(
                total_reservoir_preparation_cost
            ),
            run_peak_gpu_memory_mb=(
                run_peak_gpu_memory_mb
            ),
            run_peak_cpu_memory_mb=(
                run_peak_cpu_memory_mb
            ),
            best_iteration=(
                best_iteration
            ),
            best_test_metrics=(
                best_test_metrics
            ),
            seed=seed,
            dataset_name=task_name
        )

        print(out_dir)

        print(
            f"task={task_name}, dataset={dataset_name}, "
            f"seed={seed}, "
            f"iteration={iteration}, "
            f"train_nrmse="
            f"{train_nrmse:.4f}, "
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
            f"GD gradient L2="
            f"{gradient_l2_norm:.6e}"
        )

        estimated_finish = (
            datetime.datetime.now()
            + datetime.timedelta(
                seconds=(
                    time.time()
                    - iteration_start_time
                )
                * (
                    MAX_ITERATION
                    - iteration
                    - 1
                )
            )
        )

        print(
            "escape time = "
            f"{estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}\n"
        )

    final_total_training_wall_clock_time = float(
        np.sum(
            efficiency_result_list[:, 0]
        )
    )

    final_total_reservoir_preparation_cost = (
        reservoir_initialization_wall_clock_time
        + final_total_training_wall_clock_time
    )

    print(
        "\n========== Final Efficiency =========="
    )
    print(
        f"task={task_name}, dataset={dataset_name}, seed={seed}"
    )
    print(
        f"Total training wall-clock time = "
        f"{final_total_training_wall_clock_time:.6f} s"
    )
    print(
        f"Reservoir initialization wall-clock time = "
        f"{reservoir_initialization_wall_clock_time:.6f} s"
    )
    print(
        f"Total reservoir preparation cost = "
        f"{final_total_reservoir_preparation_cost:.6f} s"
    )

    return {
        "seed": seed,
        "best_iteration": best_iteration,
        "total_training_wall_clock_time":
            final_total_training_wall_clock_time,
        "reservoir_initialization_wall_clock_time":
            reservoir_initialization_wall_clock_time,
        "total_reservoir_preparation_cost":
            final_total_reservoir_preparation_cost,
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

    for task_index, task in enumerate(TASK_CONFIGS):
        task_name = task["task_name"]
        run_records = []

        print()
        print("=" * 90)
        print(
            f"TASK {task_index + 1}/{len(TASK_CONFIGS)}: "
            f"{task_name}"
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
            (
                "./stdp_amp"
                if USE_AMP
                else "./stdp"
            ),
            "lsm",
            task_name
        )

        os.makedirs(
            task_dir,
            exist_ok=True
        )

        summary_path = os.path.join(
            task_dir,
            f"summary_5seeds_"
            f"STDP_{task_name}_"
            f"({datetime.date.today()})"
            f".xlsx"
        )

        save_five_seed_summary(
            summary_path,
            run_records
        )

        print(
            f"Saved five-seed summary: "
            f"{summary_path}"
        )

    print()
    print("=" * 90)
    print("ALL 10 TASKS FINISHED")
    print("=" * 90)


if __name__ == "__main__":
    main()
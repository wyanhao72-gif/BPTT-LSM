"""
ETT时间序列预测主程序
本文件实现了ETT数据集的多种神经网络模型预测，包括RNN、LSTM、GRU和混合网络。
包含完整的训练、测试和结果评估流程。
"""

import datetime
import os
import time
import psutil

import numpy as np
import openpyxl
import torch
from torch.cuda import amp

import linear_main
import time_series_data_prediction as tp
from time_series_data_prediction import dataset_from
from linear_main import three_linear_net
from ETD_predict import RNN, LstmRNN, GRU, net_mix
import matplotlib
matplotlib.use('TkAgg')
import random

SEEDS = [1112]#, 1113, 1114, 1115, 1116
MODEL_NAMES = ['rnn', 'lstm', 'gru']
TASK_CONFIGS = [
    {'task_name': 'Lorenz63_gap1',  'dataset_name': 'Lorenz63', 'lorenz_gap': 1,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'Lorenz63_gap5',  'dataset_name': 'Lorenz63', 'lorenz_gap': 5,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'Lorenz63_gap10', 'dataset_name': 'Lorenz63', 'lorenz_gap': 10, 'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'Lorenz84_gap1',  'dataset_name': 'Lorenz84', 'lorenz_gap': 1,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'Lorenz84_gap5',  'dataset_name': 'Lorenz84', 'lorenz_gap': 5,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'Lorenz84_gap10', 'dataset_name': 'Lorenz84', 'lorenz_gap': 10, 'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'NARMA10',        'dataset_name': 'NARMA',    'lorenz_gap': 1,  'narma_order': 10, 'robot_gap': 1},
    {'task_name': 'NARMA20',        'dataset_name': 'NARMA',    'lorenz_gap': 1,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'ETTh1',          'dataset_name': 'ETTh1',    'lorenz_gap': 1,  'narma_order': 20, 'robot_gap': 1},
    {'task_name': 'GR1Robot',       'dataset_name': 'GR1Robot', 'lorenz_gap': 1,  'narma_order': 20, 'robot_gap': 1},
]
METRIC_NAMES = ['NRMSE', 'RMSE', 'NMSE', 'MSE', 'MAE']


def descriptive_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'variance': float(np.var(values, ddof=0))
    }


def save_five_seed_summary(summary_path, run_records):
    wb = openpyxl.Workbook()
    ws_runs = wb.active
    ws_runs.title = 'five_runs'
    ws_runs.append([
        'seed', 'best_iteration',
        'total_training_wall_clock_time_s',
        'peak_gpu_memory_mb',
        'peak_cpu_memory_mb',
        'test_NRMSE', 'test_RMSE', 'test_NMSE', 'test_MSE', 'test_MAE'
    ])
    for record in run_records:
        ws_runs.append([
            record['seed'],
            record['best_iteration'],
            record['total_wall_time'],
            record['peak_gpu_memory_mb'],
            record['peak_cpu_memory_mb'],
            *record['best_test_metrics']
        ])

    ws_stats = wb.create_sheet('statistics')
    ws_stats.append(['indicator', 'min', 'max', 'mean', 'variance'])

    values_dict = {
        'total_training_wall_clock_time_s': [
            r['total_wall_time'] for r in run_records
        ],
        'peak_gpu_memory_mb': [
            r['peak_gpu_memory_mb'] for r in run_records
        ],
        'peak_cpu_memory_mb': [
            r['peak_cpu_memory_mb'] for r in run_records
        ]
    }
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        values_dict[f'test_{metric_name}'] = [
            r['best_test_metrics'][metric_index] for r in run_records
        ]

    for indicator, values in values_dict.items():
        stats = descriptive_statistics(values)
        ws_stats.append([
            indicator, stats['min'], stats['max'],
            stats['mean'], stats['variance']
        ])
    wb.save(summary_path)


def run_single_experiment(net_name, task, seed):
    task_name = task['task_name']
    name = task['dataset_name']


    # 固定随机种子

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 保证CUDA结果可复现
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 加载中断文件
    # resume = f'./linear_amp/t(200)_sgd_lr0.01/checkpoint_max.pth'
    resume = ''

    node_num = 90  # 老传统90个隐藏层节点
    history_step = 1  # 历史节点
    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu'
    )
    debug_grad = net_name in ['mix_net', 'rnn','lstm', 'gru']
    random_bptt = False

    input_dim, output_dim, train_set, test_set, predict_step = (
        dataset_from(
            name=name,
            device=device,
            random_bptt=random_bptt,
            narma_order=task['narma_order'],
            lorenz_gap=task['lorenz_gap'],
            robot_gap=task['robot_gap']
        )
    )

    # 定义模型
    opt_net = None
    net = None

    if net_name == 'rnn':
        net = RNN(
            input_size=input_dim,
            hidden_size=node_num,
            output_size=output_dim
        )
        opt_net = net

    elif net_name == 'lstm':
        net = LstmRNN(
            input_size=input_dim,
            hidden_size=node_num,
            output_size=output_dim
        )
        opt_net = net

    elif net_name == 'gru':
        net = GRU(
            input_size=input_dim,
            hidden_size=node_num,
            output_size=output_dim
        )
        opt_net = net

    elif net_name == 'mix_net':
        net_linear = three_linear_net(
            in_dim=input_dim,
            n_hidden=node_num,
            out_dim=output_dim,
            for_lsm=True,
            history_step=history_step
        ).to(device)

        opt_net = net_linear

    if net_name != 'mix_net':
        net.to(device)

    if net_name == 'mix_net':
        qian_biao = '25-11-10'
        f_n_t = '(30_75)'
        f_n_lr = 5e-06

        rc_npy_dir = (
            f'./le_backward/'
            f'({qian_biao})t{f_n_t}_sgd_lr{f_n_lr}/'
        )

        data_dict = np.load(
            os.path.join(rc_npy_dir, 'le_bkwd.npy'),
            allow_pickle=True
        ).item()

        key = 'pop'
        pop_num_len_elem = 19

        weight_mat, tau_reciprocal = (
            data_dict[key][pop_num_len_elem - 1]
        )

        # 初始化网络
        v_th = 0.5

    # 定义损失函数和优化器
    opt = 'sgd'
    lr = 5e-6
    momentum = 0.9

    if opt == 'sgd':
        optimizer = torch.optim.SGD(
            opt_net.parameters(),
            lr=lr,
            momentum=momentum
        )

    elif opt == 'adam':
        optimizer = torch.optim.Adam(
            opt_net.parameters(),
            lr=lr
        )

    else:
        raise NotImplementedError(opt)

    # 定义超参数
    max_iteration = 200
    circle_iteration = 100
    use_amp = False

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        circle_iteration
    )

    # 设置文件输出位置
    out_dir = './mix_net_all'
    scaler = None

    if use_amp:
        out_dir += '_amp'
        scaler = amp.GradScaler()

    out_dir = os.path.join(
        out_dir,
        net_name,
        task_name,
        f'seed_{seed}',
        f't({max_iteration})_{task_name}_lr{lr}'
        f'({datetime.date.today()})'
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(f'EXdir {out_dir}.')

    start_epoch = 0
    min_test_nrmse = 100
    best_iteration = -1
    best_test_metrics = np.full(5, np.nan, dtype=np.float64)

    if resume:
        checkpoint = torch.load(
            resume,
            map_location='cpu'
        )

        opt_net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(
            checkpoint['lr_scheduler']
        )

        start_epoch = checkpoint['iteration'] + 1
        min_test_nrmse = checkpoint['min_test_nrmse']

    result_len = 5

    train_result_list = np.zeros(
        shape=[max_iteration, result_len]
    )

    test_result_list = np.zeros(
        shape=[max_iteration, result_len]
    )

    # =====================================================
    # ===== 新增：保存每轮训练时间和训练峰值GPU显存 =====
    # 第0列：每轮训练wall-clock时间，单位为秒
    # 第1列：每轮训练峰值GPU显存，单位为MB
    # 第2列：每轮训练峰值CPU内存，单位为MB
    # =====================================================
    efficiency_result_list = np.zeros(
        shape=[max_iteration, 3],
        dtype=np.float64
    )

    # net_train输出第六位：每代梯度全局L2范数
    gradient_l2_list = np.zeros(
        max_iteration,
        dtype=np.float64
    )

    process = psutil.Process(os.getpid())

    for iteration in range(start_epoch, max_iteration):
        start_time = time.time()

        if iteration in [0, 50, 100, 150, 199]:
            draw_pic = False
        else:
            draw_pic = False

        if net_name == 'mix_net':
            net_lsm, net_valid = tp.net_init(
                tau=1 / tau_reciprocal,
                v_threshold=v_th,
                lsm_node=node_num,
                init_neuron=True,
                weight_mat=weight_mat,
                le=None,
                device=device,
                input_dim=input_dim,
                output_dim=output_dim,
                reduction_ratio=0,
                train_s_and_v=True,
                input_weight_array=None
            )

            net = net_mix(
                input_layer=net_linear.layer1,
                rc=net_lsm.rc,
                output_layer=net_linear.layer2,
                history_step=history_step
            )

        opt_net.train()

        # =================================================
        # ===== 新增：训练开始前同步GPU并重置显存峰值 =====
        # =================================================
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        peak_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )

        epoch_train_start_time = time.perf_counter()

        train_result = linear_main.net_train(
            device=device,
            dataset=train_set,
            optimizer=optimizer,
            net=net,
            scaler=scaler,
            out_dir=out_dir,
            iteration=iteration,
            draw_pic=draw_pic,
            debug_grad=debug_grad,
            debug_net=opt_net
        )

        # =================================================
        # ===== 新增：训练结束后同步GPU并停止计时 =====
        # =================================================
        if device.type == 'cuda':
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
            time.perf_counter() - epoch_train_start_time
        )

        # =================================================
        # ===== 新增：读取训练阶段峰值GPU显存 =====
        # 此处位于测试之前，因此不包含测试阶段显存
        # =================================================
        if device.type == 'cuda':
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 2)
            )
        else:
            peak_gpu_memory_mb = 0.0

        efficiency_result_list[iteration, 0] = (
            epoch_train_wall_time
        )
        efficiency_result_list[iteration, 1] = (
            peak_gpu_memory_mb
        )
        efficiency_result_list[iteration, 2] = (
            peak_cpu_memory_mb
        )

        if len(train_result) < 6:
            raise RuntimeError(
                'linear_main.net_train()必须返回6个值：'
                'NRMSE, RMSE, NMSE, MSE, MAE, gradient_l2_norm。'
            )

        train_nrmse = train_result[0]
        train_result_list[iteration] = np.array(
            train_result[:5],
            dtype=np.float64
        )
        gradient_l2_list[iteration] = float(train_result[5])

        lr_scheduler.step()

        opt_net.eval()

        test_result = linear_main.net_test(
            device=device,
            dataset=test_set,
            net=net,
            draw_pic=draw_pic
        )

        test_nrmse = test_result[0]
        test_result_list[iteration] = np.array(
            test_result
        )

        if net_name in ['lstm', 'gru', 'rnn']:
            net.init()

        save_max = False

        if test_nrmse < min_test_nrmse:
            min_test_nrmse = test_nrmse
            best_iteration = iteration
            best_test_metrics = np.array(
                test_result[:5],
                dtype=np.float64
            )
            save_max = True

        checkpoint = {
            'net': opt_net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'iteration': iteration,
            'min_test_nrmse': min_test_nrmse
        }

        if save_max:
            torch.save(
                checkpoint,
                os.path.join(
                    out_dir,
                    'checkpoint_max.pth'
                )
            )

        torch.save(
            checkpoint,
            os.path.join(
                out_dir,
                'checkpoint_latest.pth'
            )
        )

        wb = openpyxl.Workbook()

        train_excel = wb.create_sheet('train')
        test_excel = wb.create_sheet('test')

        for evalu_para in range(
            train_result_list.shape[1]
        ):
            train_excel.append(
                train_result_list[
                    :,
                    evalu_para
                ].tolist()
            )

            test_excel.append(
                test_result_list[
                    :,
                    evalu_para
                ].tolist()
            )

        # =================================================
        # ===== 新增：保存时间和显存到Excel =====
        # =================================================
        efficiency_excel = wb.create_sheet('efficiency')

        efficiency_excel.append([
            'iteration',
            'train_wall_clock_time_s',
            'peak_gpu_memory_mb',
            'peak_cpu_memory_mb',
            'gradient_global_l2_norm'
        ])

        for i in range(iteration + 1):
            efficiency_excel.append([
                i,
                efficiency_result_list[i, 0],
                efficiency_result_list[i, 1],
                efficiency_result_list[i, 2],
                gradient_l2_list[i]
            ])

        file_name = (
            f'ETT_{predict_step}_{net_name}_{task_name}_seed_{seed}_'
            f'({datetime.date.today()}).xlsx'
        )

        wb.save(
            os.path.join(
                out_dir,
                file_name
            )
        )

        print(out_dir)

        print(
            f'iteration = {iteration}, '
            f'train_nrmse ={train_nrmse: .4f}, '
            f'test_nrmse ={test_nrmse: .4f}, '
            f'min_test_nrmse ={min_test_nrmse: .4f}'
        )

        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (max_iteration - iteration))).strftime("%Y-%m-%d %H:%M:%S")}\n')

        # =================================================
        # ===== 新增：打印当前轮训练时间和峰值显存 =====
        # =================================================
        print(
            f'train wall-clock time = '
            f'{epoch_train_wall_time:.4f} s, '
            f'peak GPU memory = '
            f'{peak_gpu_memory_mb:.2f} MB, '
            f'peak CPU memory = '
            f'{peak_cpu_memory_mb:.2f} MB, '
            f'gradient L2 norm = '
            f'{gradient_l2_list[iteration]:.6e}'
        )

    # =====================================================
    # ===== 新增：全部训练结束后的最终汇总 =====
    # =====================================================
    total_training_wall_time = np.sum(
        efficiency_result_list[
            start_epoch:max_iteration,
            0
        ]
    )

    total_peak_gpu_memory_mb = np.max(
        efficiency_result_list[
            start_epoch:max_iteration,
            1
        ]
    )

    total_peak_cpu_memory_mb = np.max(
        efficiency_result_list[
            start_epoch:max_iteration,
            2
        ]
    )

    print('\n========== Final Efficiency ==========')

    print(
        f'Total training wall-clock time = '
        f'{total_training_wall_time:.4f} s'
    )

    print(
        f'Peak GPU memory usage = '
        f'{total_peak_gpu_memory_mb:.2f} MB'
    )

    print(
        f'Peak CPU memory usage = '
        f'{total_peak_cpu_memory_mb:.2f} MB'
    )


    return {
        'seed': seed,
        'best_iteration': best_iteration,
        'total_wall_time': float(total_training_wall_time),
        'peak_gpu_memory_mb': float(total_peak_gpu_memory_mb),
        'peak_cpu_memory_mb': float(total_peak_cpu_memory_mb),
        'best_test_metrics': best_test_metrics.tolist()
    }


def main():
    for net_name in MODEL_NAMES:
        for task in TASK_CONFIGS:
            task_name = task['task_name']
            run_records = []

            print()
            print('=' * 90)
            print(
                f'START: model={net_name}, task={task_name}, '
                f'dataset={task["dataset_name"]}, '
                f'lorenz_gap={task["lorenz_gap"]}, '
                f'narma_order={task["narma_order"]}, '
                f'robot_gap={task["robot_gap"]}'
            )
            print('=' * 90)

            for seed in SEEDS:
                record = run_single_experiment(
                    net_name=net_name,
                    task=task,
                    seed=seed
                )
                run_records.append(record)

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            model_task_dir = os.path.join(
                './mix_net_all',
                net_name,
                task_name
            )
            os.makedirs(model_task_dir, exist_ok=True)

            summary_path = os.path.join(
                model_task_dir,
                f'summary_5seeds_{net_name}_{task_name}_'
                f'({datetime.date.today()}).xlsx'
            )

            save_five_seed_summary(
                summary_path,
                run_records
            )

            print(
                f'Saved five-seed summary: {summary_path}'
            )


if __name__ == '__main__':
    main()

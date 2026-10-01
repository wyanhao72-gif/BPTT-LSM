"""
遗传算法+液体状态机(LSM)主文件
本文件实现了基于遗传算法的液体状态机网络优化，用于寻找具有混沌边缘特性的网络权重。
主要功能包括种群初始化、适应度计算、选择、交叉和变异等遗传算法操作。

新增并保存两个严格区分的时间指标：
1. total_training_wall_clock_time_s：
   累计每代GA训练时间，包括适应度评估、选择、交叉、变异和精英保留；
2. total_reservoir_preparation_cost_s：
   初始种群生成时间与全部GA训练时间之和。

上述两个指标均不包含Numpy保存、Excel保存、日志写入和控制台输出。
"""

import argparse
import os
import sys
import time
import random
import psutil
import openpyxl

import numpy as np
import torch
from sklearn.cluster import KMeans
from spikingjelly.activation_based import functional

import lsm_net
import net_characteristic
import progress_bar


def init_popular(pop_num, node_num, for_train=True):
    """
    初始化种群（初始权重矩阵和膜时间常数）
    :param for_train: 用于反向传播训练
    :param pop_num: 初始种群的数量
    :param node_num: 神经元的数量
    :return: 种群的list
    """
    pop = []
    rc_rad_mean = 0.7
    rc_rad_std = 0.05
    tau_reciprocal_range = (0.07, 0.08)
    for pop_serial_num in range(pop_num):
        if for_train:
            connect_ratio = 1
            rc_rad = rc_rad_mean
        else:
            connect_ratio = np.random.rand()
            while connect_ratio == 0:  # 防止其为0
                connect_ratio = np.random.rand()

            rc_rad = np.random.normal(loc=rc_rad_mean, scale=rc_rad_std)
            while rc_rad <= 0:  # 防止其小于0
                rc_rad = np.random.normal(loc=rc_rad_mean, scale=rc_rad_std)

        rc_weight_array = lsm_net.rc_weight_init(node_num=node_num, connect_ratio=connect_ratio, rc_rad=rc_rad)
        rc_weight_array = rc_weight_array.numpy()
        tau_reciprocal = np.random.rand()*(tau_reciprocal_range[1] - tau_reciprocal_range[0]) + tau_reciprocal_range[0]
        while tau_reciprocal == 0:  # 防止其为0
            tau_reciprocal = np.random.rand()
        pop.append((rc_weight_array, tau_reciprocal))
    return pop


def kmeans_popular(pop, node_num, kinds_num):
    """
    对初始种群进行聚类，筛选中心种群
    :param pop: 初始种群
    :param node_num: 神经元数量，用于定义矩阵大小
    :param kinds_num: 聚类后的种类数量
    :return: 聚类后种群
    """
    pop_len = len(pop)
    individual_len = node_num ** 2 + 1  # 矩阵长度 + 膜时间常数长度
    pop_arr = np.zeros(shape=(pop_len, individual_len))
    for pop_len_ele in range(pop_len):  # 展成矩阵形式
        pop_arr[pop_len_ele, :-1] = pop[pop_len_ele][0].reshape(-1)
        pop_arr[pop_len_ele, -1] = pop[pop_len_ele][1]
    kmeans = KMeans(n_clusters=kinds_num)
    kmeans.fit(pop_arr)
    # 获取簇中心和簇分配结果
    centers = kmeans.cluster_centers_
    # labels = kmeans.labels_
    kmeans_pop = []
    for centers_ele in range(centers.shape[0]):  # 还原list形式
        kmeans_pop.append((centers[centers_ele, :-1].reshape(node_num, node_num), centers[centers_ele, -1].item()))
    return kmeans_pop


def get_fitness(pop, t_step, node_num, device, v_th):
    """
    获取种群的适应度
    :param pop: 种群本群
    :param t_step: 储层循环运行时间步长
    :param node_num: 储层节点数
    :param device: torch运行的设备'cpu' or 'cuda:0'
    :param v_th: 神经元阈值
    :return:各种指标
    """
    rc_input = torch.zeros(size=[t_step, 1, node_num]).to(device)  # 空的输入
    fire_ratio_denominator = 30  # 把1分成30份
    arr_shape = fire_ratio_denominator - 1  # 发放率从1/n 到 （n-1）/n

    pop_len = len(pop)
    le_mean_mem = np.zeros(shape=pop_len)
    le_std_mem = np.zeros(shape=pop_len)
    m_rad_mean_mem = np.zeros(shape=pop_len)
    m_rad_std_mem = np.zeros(shape=pop_len)
    m_det_mean_mem = np.zeros(shape=pop_len)
    m_det_std_mem = np.zeros(shape=pop_len)
    tau_corr_mean_mem = np.zeros(shape=pop_len)
    tau_corr_std_mem = np.zeros(shape=pop_len)
    without_weight_ave_cluster_mem = np.zeros(shape=pop_len)
    with_weight_ave_cluster_mem = np.zeros(shape=pop_len)
    rc_fro_norm_mem = np.zeros(shape=pop_len)
    rc_rad_mem = np.zeros(shape=pop_len)
    connect_ratio_mem = np.zeros(shape=pop_len)
    pos_neg_ratio_mem = np.zeros(shape=pop_len)

    start_time = time.time()
    ave_sum_le = 0
    ave_sum_m_rad = 0
    ave_sum_m_det = 0
    ave_tau_corr = 0
    for pop_counter in range(pop_len):
        first_recur = pop_counter / pop_len
        net = lsm_net.Lsm(tau=1 / pop[pop_counter][1], v_threshold=v_th, lsm_node=node_num,
                          init_neuron=True, rc_spike_init=None, rc_v_init=None, device=device
                          )
        net.rc[0].reg_grad = True  # 记录雅可比矩阵
        net.rc[0].t_start = 0  # 从0开始计算雅可比矩阵
        net.rc[0].mem_state = True  # 记录状态
        net.rc[0].reg_state = True  # 记录脉冲发放情况

        functional.set_step_mode(net=net, step_mode='m')  # 网络模式设置为多步
        net.rc_weight_init(rc_weight_array=pop[pop_counter][0])  # 初始化网络权重
        # 获取网络参数
        without_weight_ave_cluster, with_weight_ave_cluster, rc_fro_norm, rc_rad, connect_ratio, pos_neg_ratio = (
            net_characteristic.get_net_cha(pop[pop_counter][0]))
        net.to(device)

        le_arr = np.zeros(shape=arr_shape)  # arr = array 和 matrix意义相同
        le_real_arr = np.zeros(shape=arr_shape)  # 记录le的真实性
        m_rad_arr = np.zeros(shape=arr_shape)  # 记录分支比的谱半径
        m_det_arr = np.zeros(shape=arr_shape)  # 记录分支比的行列式
        tau_corr_arr = np.zeros(shape=arr_shape)  # 记录自相关时间常数
        # 遍历不同的脉冲发放率
        for fire_ratio_numerator in range(arr_shape):
            second_recur = ((fire_ratio_numerator + 1) / arr_shape) * (1 / pop_len)
            progress = (first_recur + second_recur)  # 任务进度
            fire_ratio = (fire_ratio_numerator + 1) / fire_ratio_denominator
            rc_spike_init, rc_v_init = lsm_net.init_spike_v(node_num, fire_ratio, v_th)
            rc_spike_init = torch.from_numpy(rc_spike_init).to(device)
            rc_v_init = torch.from_numpy(rc_v_init).to(device)
            # 初始化储层神经元的脉冲和膜电位
            net.rc[0].rc_spike_init = rc_spike_init
            net.rc[0].rc_v_init = rc_v_init

            net.rc(rc_input)
            # 获取储层的le和状态
            (le_real, le_g_list, le_lmt_list, ave_le_g, ave_le_lmt,
             rc_state_len, rc_state_mat, rc_state_mean, rc_state_err, _, _) = net.rc[0].get_le(store_le=False)

            if le_real:
                le_real_arr[fire_ratio_numerator] = 1
                # 计算分支比谱半径和自相关时间常数
                valid_step = net.rc[0].valid_step
                rc_state_mat = rc_state_mat[:valid_step].cpu().detach().numpy()
                m_rad, m_det = net_characteristic.branch_parameter(rc_state_mat)
                tau_corr = net_characteristic.corr_parameter(rc_state_mat)
                m_rad_arr[fire_ratio_numerator] = m_rad
                m_det_arr[fire_ratio_numerator] = m_det
                tau_corr_arr[fire_ratio_numerator] = tau_corr
            else:
                le_real_arr[fire_ratio_numerator] = 0

                m_rad_arr[fire_ratio_numerator] = 0
                m_det_arr[fire_ratio_numerator] = 0
                tau_corr_arr[fire_ratio_numerator] = 0
            le_arr[fire_ratio_numerator] = ave_le_g

            functional.reset_net(net)

            progress_bar.progress_bar_fun(start_time=start_time, progress=progress)
            print(f'\tle:{ave_sum_le:.4f}\tm_rad:{ave_sum_m_rad:.4f}\tm_det:{ave_sum_m_det:.4f}\t'
                  f'tau_corr={ave_tau_corr:.4f}', end='')
        # 计算上述各个脉冲发放率的平均值与方差
        if np.sum(le_real_arr) < 9:
            ave_sum_le = 128.0
            std_le = 0.
            ave_sum_m_rad = 0.
            std_m_rad = 0.
            ave_sum_m_det = 0.
            std_m_det = 0.
            ave_tau_corr = 0.
            std_tau_corr = 0.
        else:
            le_arr = le_arr[le_real_arr == 1]
            ave_sum_le = np.average(le_arr)
            std_le = np.std(le_arr)

            m_rad_arr = m_rad_arr[le_real_arr == 1]
            m_det_arr = m_det_arr[le_real_arr == 1]
            ave_sum_m_rad = np.average(m_rad_arr)
            std_m_rad = np.std(m_rad_arr)
            ave_sum_m_det = np.average(m_det_arr)
            std_m_det = np.std(m_det_arr)

            tau_corr_arr_real = tau_corr_arr[le_real_arr == 1]
            ave_tau_corr = np.average(tau_corr_arr_real)
            std_tau_corr = np.std(tau_corr_arr_real)
        le_mean_mem[pop_counter] = ave_sum_le
        le_std_mem[pop_counter] = std_le
        m_rad_mean_mem[pop_counter] = ave_sum_m_rad
        m_rad_std_mem[pop_counter] = std_m_rad
        m_det_mean_mem[pop_counter] = ave_sum_m_det
        m_det_std_mem[pop_counter] = std_m_det
        tau_corr_mean_mem[pop_counter] = ave_tau_corr
        tau_corr_std_mem[pop_counter] = std_tau_corr
        without_weight_ave_cluster_mem[pop_counter] = without_weight_ave_cluster
        with_weight_ave_cluster_mem[pop_counter] = with_weight_ave_cluster
        rc_fro_norm_mem[pop_counter] = rc_fro_norm
        rc_rad_mem[pop_counter] = rc_rad
        connect_ratio_mem[pop_counter] = connect_ratio
        pos_neg_ratio_mem[pop_counter] = pos_neg_ratio
    return (le_mean_mem, le_std_mem, m_rad_mean_mem, m_rad_std_mem, tau_corr_mean_mem, tau_corr_std_mem,
            without_weight_ave_cluster_mem, with_weight_ave_cluster_mem, rc_fro_norm_mem, rc_rad_mem,
            connect_ratio_mem, pos_neg_ratio_mem, m_det_mean_mem, m_det_std_mem)


def trans_dna(a: float, dna_bit: int = 8):
    """
    用于将浮点数转为dna_bit编码
    例如：
    000000001 = 1/（2**8） = 0.00390625
    :param a:
    :param dna_bit:
    :return:
    """
    dna = bin(int(a * (2 ** dna_bit)))[2:]
    while len(dna) < dna_bit:
        dna = '0' + dna
    dna = list(dna)
    return dna


def crossover_and_mutation(pop, new_pop_size=25, node_num=90, le_fitness=None, mutation_rate=0.003):
    """
    用于种群的交叉变异
    :param pop: 用于交叉变异的种群
    :param new_pop_size: 交叉出的种群数量（用于子代）
    :param node_num: 神经元数目（用于定义矩阵大小）
    :param le_fitness: 种群对应的le,用于计算交叉概率
    :param mutation_rate: 变异概率
    :return: 新种群
    """
    pop_size = len(pop)
    le_fitness = np.abs(le_fitness)
    dna_bit = 8
    new_pop = []
    for chile_serial_num in range(new_pop_size):
        father_idx = np.random.randint(pop_size)
        mother_idx = np.random.randint(pop_size)
        while father_idx == mother_idx:
            mother_idx = np.random.randint(pop_size)
        father = pop[father_idx]  # 选择父亲
        mother = pop[mother_idx]  # 再种群中选择另一个个体，并将该个体作为母亲
        child = []
        # 按照le的大小定义交叉概率，le越大，越要交叉
        crossover_rate = le_fitness[father_idx] / (le_fitness[father_idx] + le_fitness[mother_idx])
        # 防止交叉概率过大或者过小
        if crossover_rate < 0.005:
            crossover_rate = 0.005
        elif crossover_rate > 0.995:
            crossover_rate = 0.995
        crossover_mat = np.random.rand(node_num, node_num)  # 生成随机数矩阵用于选择交叉的位置
        crossover_index = crossover_mat < crossover_rate
        while np.all(crossover_index) or (not np.any(crossover_index)):  # 防止不交叉，浪费我时间
            crossover_mat = np.random.rand(node_num, node_num)
            crossover_index = crossover_mat < crossover_rate
        father[0][crossover_index] = mother[0][crossover_index]  # 交叉操作
        mutation_mat = np.random.rand(node_num, node_num)  # 生成随机数矩阵用于选择变异的位置
        mutation_index = np.argwhere(mutation_mat < mutation_rate)  # 获取符合变异条件的位置点
        if mutation_index.shape[0] != 0:  # 优化计算
            fa_mean = np.mean(father[0])
            fa_std = np.std(father[0])
            for mut_ind in mutation_index:
                if father[0][mut_ind[0], mut_ind[1]] == 0:
                    father[0][mut_ind[0], mut_ind[1]] = np.random.normal(loc=fa_mean, scale=fa_std)
                else:
                    father[0][mut_ind[0], mut_ind[1]] = 0
        child.append(father[0])
        father_tau_dna = trans_dna(father[1], dna_bit=dna_bit)  # 将膜时间常数的倒数转化为8bit
        mother_tau_dna = trans_dna(mother[1], dna_bit=dna_bit)
        crossover_dna_mat = np.random.rand(dna_bit)
        crossover_dna_index = crossover_dna_mat < crossover_rate  # 选择位置交叉
        mutation_dna_mat = np.random.rand(dna_bit)
        mutation_dna_index = np.argwhere(mutation_dna_mat < mutation_rate)  # 获取符合变异条件的位置点
        for dna_bit_elem in range(dna_bit):  # 交叉操作
            if crossover_dna_index[dna_bit_elem]:
                father_tau_dna[dna_bit_elem] = mother_tau_dna[dna_bit_elem]
        if mutation_dna_index.shape[0] != 0:  # 优化计算
            for dna_idx in mutation_dna_index:  # 变异操作
                if father_tau_dna[dna_idx.item()] == '1':
                    father_tau_dna[dna_idx.item()] = '0'
                else:
                    father_tau_dna[dna_idx.item()] = '1'
        father_tau_dna = "".join(father_tau_dna)
        child_tau = int(father_tau_dna, 2) / (2 ** dna_bit)
        if child_tau == 0:
            child_tau = 1 / (2 ** dna_bit)
        child.append(child_tau)
        new_pop.append(child)

    return new_pop


def select(pop, fitness, elite_num, pop_size: int):
    """
    遗传算法里面的选择策略，使用到精英策略和轮盘赌
    :param pop: 原始种群
    :param fitness: 适应度，采用le的绝对值的倒数
    :param elite_num: 精英数量
    :param pop_size: 选择的种群数量
    :return: 选择的种群（用于交叉变异）,精英种群下标,选择的种群下标
    """
    fit_sort = np.argsort(fitness)
    index_max = fit_sort[-elite_num:]
    pop_select = []
    # valid_size = int(np.sum(fitness > (1 / np.abs(128.0))))
    # fitness[fitness <= (1 / np.abs(128.0))] = 0.

    idx = np.random.choice(np.arange(len(pop)), size=pop_size, replace=False,
                           p=fitness / fitness.sum())
    for idx_elem in idx:
        pop_select.append(pop[idx_elem.item()])
    return pop_select, index_max, idx



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def descriptive_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=0))
    }


def save_five_seed_summary(path, records):
    wb = openpyxl.Workbook()

    runs_ws = wb.active
    runs_ws.title = "five_runs"
    runs_ws.append([
        "seed",
        "total_training_wall_clock_time_s",
        "reservoir_initialization_wall_clock_time_s",
        "total_reservoir_preparation_cost_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "total_fitness_evaluations"
    ])

    for record in records:
        runs_ws.append([
            record["seed"],
            record["total_training_wall_clock_time"],
            record["reservoir_initialization_wall_clock_time"],
            record["total_reservoir_preparation_cost"],
            record["peak_gpu_memory_mb"],
            record["peak_cpu_memory_mb"],
            record["total_fitness_evaluations"]
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
            record["total_training_wall_clock_time"]
            for record in records
        ],
        "reservoir_initialization_wall_clock_time_s": [
            record["reservoir_initialization_wall_clock_time"]
            for record in records
        ],
        "total_reservoir_preparation_cost_s": [
            record["total_reservoir_preparation_cost"]
            for record in records
        ],
        "peak_gpu_memory_mb": [
            record["peak_gpu_memory_mb"]
            for record in records
        ],
        "peak_cpu_memory_mb": [
            record["peak_cpu_memory_mb"]
            for record in records
        ]
    }

    for indicator, values in indicators.items():
        stats = descriptive_statistics(values)
        stats_ws.append([
            indicator,
            stats["min"],
            stats["max"],
            stats["mean"],
            stats["variance"]
        ])

    wb.save(path)


def save_run_efficiency(
    path,
    efficiency_results,
    valid_individual_counts,
    fitness_evaluation_counts,
    completed_generations,
    seed,
    reservoir_initialization_wall_clock_time,
    total_reservoir_preparation_cost
):
    wb = openpyxl.Workbook()

    efficiency_ws = wb.active
    efficiency_ws.title = "efficiency"
    efficiency_ws.append([
        "generation",
        "training_wall_clock_time_s",
        "peak_gpu_memory_mb",
        "peak_cpu_memory_mb",
        "gradient_min",
        "gradient_max",
        "gradient_mean",
        "gradient_global_l2_norm",
        "valid_individual_count",
        "fitness_evaluation_count"
    ])

    for generation in range(completed_generations):
        efficiency_ws.append([
            generation,
            efficiency_results[generation, 0],
            efficiency_results[generation, 1],
            efficiency_results[generation, 2],
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            int(valid_individual_counts[generation]),
            int(fitness_evaluation_counts[generation])
        ])

    summary_ws = wb.create_sheet("run_summary")
    summary_ws.append(["item", "value"])
    summary_ws.append(["seed", seed])
    summary_ws.append([
        "total_training_wall_clock_time_s",
        float(np.sum(
            efficiency_results[:completed_generations, 0]
        ))
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
        float(np.max(
            efficiency_results[:completed_generations, 1]
        ))
    ])
    summary_ws.append([
        "run_peak_cpu_memory_mb",
        float(np.max(
            efficiency_results[:completed_generations, 2]
        ))
    ])
    summary_ws.append([
        "total_fitness_evaluations",
        int(np.sum(
            fitness_evaluation_counts[:completed_generations]
        ))
    ])
    summary_ws.append([
        "gradient_metrics",
        "N/A: GA-LSM has no loss.backward() or gradient descent"
    ])

    wb.save(path)


def run_single_seed(args, seed):

    set_seed(seed)
    out_dir = os.path.join(args.out_dir, f'seed_{seed}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    print(f'dir={out_dir}.')  # 建立并保存文件
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))

    node_num = args.node_num
    zero_len = 125
    # 创建各种记录列表，以便写入字典
    ever_ite_le_mean = []
    ever_ite_le_std = []
    ever_ite_m_rad_mean = []
    ever_ite_m_rad_std = []
    ever_ite_tau_corr_mean = []
    ever_ite_tau_corr_std = []
    ever_ite_woc = []
    ever_ite_wic = []
    ever_ite_rc_norm = []
    ever_ite_rc_rad = []
    ever_ite_con_ratio = []
    ever_ite_p_n_ratio = []
    ever_ite_m_det_mean = []
    ever_ite_m_det_std = []
    pop_list = []
    data_dict = {}  # 创建存储字典
    idx_max = 0

    # ============================================================
    # Reservoir initialization wall-clock time
    #
    # GA-LSM中，储层初始化对应初始种群的生成：
    #   - 生成每个个体的储层权重矩阵；
    #   - 生成每个个体的膜时间常数。
    #
    # 不包含输出目录、args.txt和后续适应度评估。
    # ============================================================
    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    reservoir_initialization_start = time.perf_counter()

    pop = init_popular(
        args.pop_num,
        node_num
    )

    # 如果启用KMeans初始种群筛选，则应将该行取消注释。
    # 聚类属于初始化阶段，因此自然计入初始化时间。
    # pop = kmeans_popular(
    #     pop=pop,
    #     node_num=node_num,
    #     kinds_num=args.kinds_num
    # )

    if args.device.startswith('cuda') and torch.cuda.is_available():
        torch.cuda.synchronize()
    reservoir_initialization_wall_clock_time = (
        time.perf_counter()
        - reservoir_initialization_start
    )

    efficiency_results = np.zeros((args.ga_epoch, 3), dtype=np.float64)
    valid_individual_counts = np.zeros(args.ga_epoch, dtype=np.int64)
    fitness_evaluation_counts = np.zeros(args.ga_epoch, dtype=np.int64)
    process = psutil.Process(os.getpid())

    for ite_num in range(args.ga_epoch):
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        peak_cpu_memory_mb = (
            process.memory_info().rss / (1024 ** 2)
        )

        pop_list.append(pop)
        data_dict['pop'] = pop_list

        # ========================================================
        # GA training wall-clock，第一部分：适应度评估
        # ========================================================
        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        fitness_start_time = time.perf_counter()

        if ite_num == 0:
            (le_mean_mem, le_std_mem, m_rad_mean_mem, m_rad_std_mem, tau_corr_mean_mem, tau_corr_std_mem,
             without_weight_ave_cluster_mem, with_weight_ave_cluster_mem, rc_fro_norm_mem, rc_rad_mem,
             connect_ratio_mem, pos_neg_ratio_mem, m_det_mean_mem, m_det_std_mem) = (
                get_fitness(pop=pop, t_step=zero_len, node_num=node_num, device=args.device, v_th=args.v_th))
        else:
            (le_mean_mem, le_std_mem, m_rad_mean_mem, m_rad_std_mem, tau_corr_mean_mem, tau_corr_std_mem,
             without_weight_ave_cluster_mem, with_weight_ave_cluster_mem, rc_fro_norm_mem, rc_rad_mem,
             connect_ratio_mem, pos_neg_ratio_mem, m_det_mean_mem, m_det_std_mem) = (
                get_fitness(pop=pop[:args.new_pop_num], t_step=zero_len, node_num=node_num, device=args.device,
                            v_th=args.v_th))
            le_mean_mem = np.append(le_mean_mem, data_dict['le_mean'][ite_num - 1][idx_max].reshape(-1))
            le_std_mem = np.append(le_std_mem, data_dict['le_std'][ite_num - 1][idx_max].reshape(-1))
            m_rad_mean_mem = np.append(m_rad_mean_mem, data_dict['m_rad_mean'][ite_num - 1][idx_max].reshape(-1))
            m_rad_std_mem = np.append(m_rad_std_mem, data_dict['m_rad_std'][ite_num - 1][idx_max].reshape(-1))
            tau_corr_mean_mem = np.append(tau_corr_mean_mem,
                                          data_dict['tau_corr_mean'][ite_num - 1][idx_max].reshape(-1))
            tau_corr_std_mem = np.append(tau_corr_std_mem, data_dict['tau_corr_std'][ite_num - 1][idx_max].reshape(-1))
            without_weight_ave_cluster_mem = np.append(without_weight_ave_cluster_mem,
                                                       data_dict['woc'][ite_num - 1][idx_max].reshape(-1))
            with_weight_ave_cluster_mem = np.append(with_weight_ave_cluster_mem,
                                                    data_dict['wic'][ite_num - 1][idx_max].reshape(-1))
            rc_fro_norm_mem = np.append(rc_fro_norm_mem, data_dict['rc_norm'][ite_num - 1][idx_max].reshape(-1))
            rc_rad_mem = np.append(rc_rad_mem, data_dict['rc_rad'][ite_num - 1][idx_max].reshape(-1))
            connect_ratio_mem = np.append(connect_ratio_mem, data_dict['con_ratio'][ite_num - 1][idx_max].reshape(-1))
            pos_neg_ratio_mem = np.append(pos_neg_ratio_mem, data_dict['p_n_ratio'][ite_num - 1][idx_max].reshape(-1))
            m_det_mean_mem = np.append(m_det_mean_mem, data_dict['m_det_mean'][ite_num - 1][idx_max].reshape(-1))
            m_det_std_mem = np.append(m_det_std_mem, data_dict['m_det_std'][ite_num - 1][idx_max].reshape(-1))

        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
        fitness_evaluation_wall_time = (
            time.perf_counter() - fitness_start_time
        )

        if ite_num == 0:
            fitness_evaluation_counts[ite_num] = len(pop)
        else:
            fitness_evaluation_counts[ite_num] = args.new_pop_num

        ever_ite_le_mean.append(le_mean_mem)
        ever_ite_le_std.append(le_std_mem)
        ever_ite_m_rad_mean.append(m_rad_mean_mem)
        ever_ite_m_rad_std.append(m_rad_std_mem)
        ever_ite_tau_corr_mean.append(tau_corr_mean_mem)
        ever_ite_tau_corr_std.append(tau_corr_std_mem)
        ever_ite_woc.append(without_weight_ave_cluster_mem)
        ever_ite_wic.append(with_weight_ave_cluster_mem)
        ever_ite_rc_norm.append(rc_fro_norm_mem)
        ever_ite_rc_rad.append(rc_rad_mem)
        ever_ite_con_ratio.append(connect_ratio_mem)
        ever_ite_p_n_ratio.append(pos_neg_ratio_mem)
        ever_ite_m_det_mean.append(m_det_mean_mem)
        ever_ite_m_det_std.append(m_det_std_mem)

        data_dict['le_mean'] = ever_ite_le_mean
        data_dict['le_std'] = ever_ite_le_std
        data_dict['m_rad_mean'] = ever_ite_m_rad_mean
        data_dict['m_rad_std'] = ever_ite_m_rad_std
        data_dict['tau_corr_mean'] = ever_ite_tau_corr_mean
        data_dict['tau_corr_std'] = ever_ite_tau_corr_std
        data_dict['woc'] = ever_ite_woc
        data_dict['wic'] = ever_ite_wic
        data_dict['rc_norm'] = ever_ite_rc_norm
        data_dict['rc_rad'] = ever_ite_rc_rad
        data_dict['con_ratio'] = ever_ite_con_ratio
        data_dict['p_n_ratio'] = ever_ite_p_n_ratio
        data_dict['m_det_mean'] = ever_ite_m_det_mean
        data_dict['m_det_std'] = ever_ite_m_det_std

        np.save(os.path.join(out_dir, 'data_dict.npy'), data_dict)
        # new_dict = np.load('file.npy', allow_pickle=True)    # 输出即为Dict 类型

        # ========================================================
        # GA training wall-clock，第二部分：
        # 选择、交叉、变异和精英保留
        # ========================================================
        evolutionary_start_time = time.perf_counter()

        pop_select, idx_max, idx_select = select(
            pop=pop,
            fitness=1 / np.abs(le_mean_mem),
            elite_num=args.elite_num,
            pop_size=args.select_num
        )
        pop_new = crossover_and_mutation(pop=pop_select, new_pop_size=args.new_pop_num, node_num=args.node_num,
                                         le_fitness=le_mean_mem[idx_select], mutation_rate=args.mutation_rate)
        for idx_max_elem in idx_max:  # 将精英加入下一代
            pop_new.append(pop[idx_max_elem.item()])
        pop = pop_new[:]

        evolutionary_wall_time = (
            time.perf_counter() - evolutionary_start_time
        )

        generation_training_wall_time = (
            fitness_evaluation_wall_time
            + evolutionary_wall_time
        )

        le_valid_index = le_mean_mem != 128.
        # 打印关键数据
        if not np.max(le_valid_index):
            this_le = 0
            this_m_rad = 0
            this_m_det = 0
            this_tau_corr = 0
            this_woc = 0
            this_wic = 0
            this_rc_rad = 0
        else:
            this_le = np.mean(np.abs(le_mean_mem[le_valid_index]))
            this_m_rad = np.mean(m_rad_mean_mem[le_valid_index])
            this_m_det = np.mean(m_det_mean_mem[le_valid_index])
            this_tau_corr = np.mean(np.abs(tau_corr_mean_mem[le_valid_index]))
            this_woc = np.mean(without_weight_ave_cluster_mem[le_valid_index])
            this_wic = np.mean(with_weight_ave_cluster_mem[le_valid_index])
            this_rc_rad = np.mean(rc_rad_mem[le_valid_index])
        print(f'')
        print(f'iteration:{ite_num}\tle:{this_le}\tm_rad:{this_m_rad}\tm_det:{this_m_det}\ttau_corr:{this_tau_corr}\t'
              f'woc:{this_woc}\twic:{this_wic}\trc_rad={this_rc_rad}\tvalid_num={np.sum(le_valid_index)}')

        valid_individual_counts[ite_num] = int(np.sum(le_valid_index))

        if args.device.startswith('cuda') and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak_gpu_memory_mb = (
                torch.cuda.max_memory_allocated()
                / (1024 ** 2)
            )
        else:
            peak_gpu_memory_mb = 0.0

        current_cpu_memory_mb = (
            process.memory_info().rss
            / (1024 ** 2)
        )
        peak_cpu_memory_mb = max(
            peak_cpu_memory_mb,
            current_cpu_memory_mb
        )

        efficiency_results[ite_num] = [
            generation_training_wall_time,
            peak_gpu_memory_mb,
            peak_cpu_memory_mb
        ]

        total_training_wall_clock_time_so_far = float(
            np.sum(
                efficiency_results[
                    :ite_num + 1,
                    0
                ]
            )
        )

        total_reservoir_preparation_cost_so_far = (
            reservoir_initialization_wall_clock_time
            + total_training_wall_clock_time_so_far
        )

        save_run_efficiency(
            path=os.path.join(
                out_dir,
                'training_efficiency.xlsx'
            ),
            efficiency_results=efficiency_results,
            valid_individual_counts=valid_individual_counts,
            fitness_evaluation_counts=fitness_evaluation_counts,
            completed_generations=ite_num + 1,
            seed=seed,
            reservoir_initialization_wall_clock_time=(
                reservoir_initialization_wall_clock_time
            ),
            total_reservoir_preparation_cost=(
                total_reservoir_preparation_cost_so_far
            )
        )

        print(
            f'generation training wall-clock='
            f'{generation_training_wall_time:.4f}s, '
            f'cumulative reservoir preparation cost='
            f'{total_reservoir_preparation_cost_so_far:.4f}s, '
            f'peak GPU memory='
            f'{peak_gpu_memory_mb:.2f}MB, '
            f'peak CPU memory='
            f'{peak_cpu_memory_mb:.2f}MB'
        )

    total_training_wall_clock_time = float(
        np.sum(efficiency_results[:, 0])
    )

    total_reservoir_preparation_cost = (
        reservoir_initialization_wall_clock_time
        + total_training_wall_clock_time
    )

    run_peak_gpu_memory_mb = float(
        np.max(efficiency_results[:, 1])
    )
    run_peak_cpu_memory_mb = float(
        np.max(efficiency_results[:, 2])
    )
    total_fitness_evaluations = int(
        np.sum(fitness_evaluation_counts)
    )

    print('\n========== Final Efficiency ==========')
    print(f'seed = {seed}')
    print(
        f'Total training wall-clock time = '
        f'{total_training_wall_clock_time:.6f} s'
    )
    print(
        f'Reservoir initialization wall-clock time = '
        f'{reservoir_initialization_wall_clock_time:.6f} s'
    )
    print(
        f'Total reservoir preparation cost = '
        f'{total_reservoir_preparation_cost:.6f} s'
    )
    print(
        f'Peak GPU memory usage = '
        f'{run_peak_gpu_memory_mb:.2f} MB'
    )
    print(
        f'Peak CPU memory usage = '
        f'{run_peak_cpu_memory_mb:.2f} MB'
    )

    return {
        'seed': seed,
        'total_training_wall_clock_time':
            total_training_wall_clock_time,
        'reservoir_initialization_wall_clock_time':
            reservoir_initialization_wall_clock_time,
        'total_reservoir_preparation_cost':
            total_reservoir_preparation_cost,
        'peak_gpu_memory_mb': run_peak_gpu_memory_mb,
        'peak_cpu_memory_mb': run_peak_cpu_memory_mb,
        'total_fitness_evaluations':
            total_fitness_evaluations
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=float, default=2.0, help='膜时间常数')
    parser.add_argument('--v_th', type=float, default=0.5, help='阈值电压')
    parser.add_argument('--node_num', type=int, default=90, help='神经元数量')
    parser.add_argument('--init_neuron', type=bool, default=True, help='初始化神经元')
    parser.add_argument('--pop_num', type=int, default=200, help='种群数量')
    parser.add_argument('--kinds_num', type=int, default=80, help='聚类后种群数量')
    parser.add_argument('--select_num', type=int, default=20, help='选择数量')
    parser.add_argument('--elite_num', type=int, default=5, help='精英数量')
    parser.add_argument('--new_pop_num', type=int, default=35, help='新产生数量')
    parser.add_argument('--ga_epoch', type=int, default=40, help='迭代次数')
    parser.add_argument('--crossover_rate', type=float, default=0.5, help='交叉概率')
    parser.add_argument('--mutation_rate', type=float, default=0.003, help='变异概率')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备')
    parser.add_argument('--out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    args = parser.parse_args()

    seeds = [42, 43, 44, 45, 46]
    records = []

    for seed in seeds:
        records.append(
            run_single_seed(args, seed)
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    os.makedirs(args.out_dir, exist_ok=True)
    save_five_seed_summary(
        os.path.join(
            args.out_dir,
            'GA_LSM_5seeds_efficiency_summary.xlsx'
        ),
        records
    )


if __name__ == '__main__':
    main()

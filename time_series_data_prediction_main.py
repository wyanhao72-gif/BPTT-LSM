"""
时间序列预测主程序
本文件实现了基于液体状态机的时间序列预测任务，支持Lorenz、ETT、NARMA等多种数据集。
包含网络初始化、训练、测试和结果评估的完整流程。
"""

import os
import time

import numpy as np
import openpyxl
import torch

import progress_bar
import time_series_data_prediction as tp
from linear_main import three_linear_net
import matplotlib
matplotlib.use("TkAgg")

# 该文件是用来进行时序预测的主文件

def main():
    out_dir = None
    data_dict = None
    seed = None
    pop_from = 'bkwd'  # 'bkwd' 或 'ga' : 反向传播 或 遗传算法
    if pop_from == 'bkwd':
        qian_biao = '25-12-8'
        f_n_t = '(25_25)'
        f_n_lr = 0.0001
        out_dir = f'./le_backward/({qian_biao})t{f_n_t}_sgd_lr{f_n_lr}/'
        data_dict = np.load(os.path.join(out_dir, f'le_bkwd.npy'), allow_pickle=True).item()  # 输出即为Dict 类型
    elif pop_from == 'ga':
        # 读文件操作
        seed = 1112
        date = '12_30'
        out_dir = f'./logs/seed_{seed}({date})'
        aval = '_aval0'
        data_dict = np.load(os.path.join(out_dir, f'data_dict.npy'), allow_pickle=True).item()  # 输出即为Dict 类型

    v_th = 0.5  # 神经元阈值电压
    node_num = 90  # 神经元个数
    spike_v_init = True  # 初始化膜电位以及脉冲（热启动）
    device = 'cuda:0'  # 运行设备
    seri_data_name = ['lorenz', 'ETT', 'NARMA']  # 进行任务的序列名称
    select_name = 'lorenz'
    train_len = None
    test_len = None
    predict_step = None
    history_step = 1
    if select_name == seri_data_name[0]:
        train_len = 10000  # 训练序列长度
        test_len = 1000  # 测试序列长度
        predict_step = 10  # gap,提前预测步长
    elif select_name == seri_data_name[1]:
        # total_len = 17420  # 总共拥有数据长度
        predict_step = 24  # 提前预测步长，预测一天的变化
        train_len = 24 * 365  # 训练序列长度
        test_len = 24 * 90  # 测试序列长度

        # train_len = total_len
        # test_len = 24
    elif select_name == seri_data_name[2]:
        train_len = 3000  # 训练序列长度
        test_len = 1000  # 测试序列长度
        predict_step = 1  # gap,提前预测步长

    data_len = train_len + test_len + predict_step  # 总长度
    bench_num = 1  # 批次数，1就行了
    lorenz_dim = 3  # 洛伦兹维度，3维就行
    target_row = None  # 预测结果对应的列，洛伦兹是全预测所以是[0, 1 ,2]或者None
    if select_name == seri_data_name[0]:
        target_row = None  # 预测结果对应的列，洛伦兹是全预测所以是[0, 1 ,2]或者None
    elif select_name == seri_data_name[1]:
        target_row = [6]  # 预测结果对应的列，洛伦兹是全预测所以是[0, 1 ,2]或者None
    elif select_name == seri_data_name[2]:
        target_row = None  # 预测结果对应的列，洛伦兹是全预测所以是[0, 1 ,2]或者None
    washout = 200  # 洛伦兹序列冲洗长度，以确保回到混沌轨迹
    # reduction_ratio = 0.05  # 输入层权重缩放比例，理论上是微扰，所以应该小
    train_s_and_v = True  # 将脉冲和电位一起作为rc输出

    pop_len = 0
    if pop_from == 'bkwd':
        pop_len = 1  # 获取迭代次数
    elif pop_from == 'ga':
        pop_len = len(data_dict['pop'])  # 获取迭代次数

    start_pop = [1, 20]  # 从哪个种群开始

    reduction_ratio_list_len = 5  # 设置不同的输入衰减,并且和每次迭代的储池循环次数也有关
    fixed_reduction = False
    reduction_bias = 6  # 意思是从1/(2 ** reduction_bias)到1/(2 ** (reduction_bias+reduction_ratio_list_len))，
    # 当设置输入层的时候无效

    le_list = []  # 记录李雅普诺夫指数
    error_data_list = []  # 记录误差数据
    error_data_name = ['nmse', 'nrmse', 'rmse', 'mse', 'mae']
    # error_data_dim = ['x', 'y', 'z', 'sum']
    if select_name == seri_data_name[0]:
        error_data_dim = ['x', 'y', 'z', 'sum']
    else:
        error_data_dim = ['sum']
    norm_mode = 'norm'  # 归一化方式norm是归一化，std是标准化
    dim = 0  # None是全维度一起归一化，0是各个维度单独归一化
    output_ori = True  # 是否输出真实值还是归一化值
    NARMA_order = None  # NARMA阶数
    if select_name == seri_data_name[0]:
        # 生成洛伦兹序列
        dataset_norm = tp.lorenz_data_creat(sample_len=data_len, n_samples=1, washout=washout,
                                            norm_mode=norm_mode, dim=dim)
        input_dim = dataset_norm.shape[-1]
        output_dim = dataset_norm.shape[-1]
        dataset = dataset_norm
    elif select_name == seri_data_name[1]:
        # 生成预测序列
        name = f'ETTh1'
        dataset = np.load(f'./dataset/{name}.npy')
        dataset = dataset[:train_len + test_len + predict_step]
        dataset = torch.Tensor(dataset)
        dataset_norm = tp.normalize_sample(sample=dataset, mode=norm_mode, dim=dim)
        input_dim = dataset_norm.shape[-1]
        output_dim = len(target_row)
        if not output_ori:
            dataset = dataset_norm
    elif select_name == seri_data_name[2]:
        # 生成NARMA10序列
        NARMA_order = 10  # NARMA阶数
        dataset_norm, dataset = tp.NARMA_data_creat(sample_len=data_len, n_samples=1, system_order=NARMA_order)
        dataset_norm = torch.cat((dataset_norm, dataset), dim=1)  # 在1维进行拼接
        # dataset = tp.normalize_sample(sample=dataset, mode=norm_mode, dim=dim)
        dataset_norm = tp.normalize_sample(sample=dataset_norm, mode=norm_mode, dim=dim)
        input_dim = dataset_norm.shape[-1]
        output_dim = dataset.shape[-1]
    else:
        raise ValueError(f'do not have {select_name}')

    # 载入输入层模型
    use_input_layer = False
    if use_input_layer and (select_name == seri_data_name[1]):
        net_linear = three_linear_net(in_dim=input_dim, n_hidden=node_num, out_dim=output_dim
                                      , history_step=1, for_lsm=True).to(device)
        resume = f'./mix_net_amp/mix_net/(25-11-11)t(1000)_sgd_lr5e-06/checkpoint_max.pth'
        checkpoint = torch.load(resume, map_location=device)
        net_linear.load_state_dict(checkpoint['net'])
        input_layer = net_linear.layer1
    else:
        input_layer = None

    if select_name == seri_data_name[2]:
        give_up_len = NARMA_order
    else:
        give_up_len = 0

    for j in range(len(error_data_name)):
        error_data_list.append([])
        for k in range(len(error_data_dim)):
            error_data_list[j].append([])
            for i in range(reduction_ratio_list_len):
                error_data_list[j][k].append([])
    # 设置记录储层脉冲发放的list
    spike_sum_list = []
    for i in range(reduction_ratio_list_len):
        spike_sum_list.append([])

    pop_num_len = None
    le = None
    for pop_len_elem in range(pop_len):
        # pop_len_elem = 29  # 方便快速跳转
        start_time = time.time()  # 记录开始时间
        print(f'\niteration={pop_len_elem}')  # 输出当前代数

        if pop_len_elem < (start_pop[0] - 1):
            continue
        if pop_from == 'bkwd':
            pop_num_len = len(data_dict['pop']) - 1
        elif pop_from == 'ga':
            pop_num_len = len(data_dict['pop'][pop_len_elem])

        for pop_num_len_elem in range(0,int(pop_num_len/2),5):
            if pop_len_elem == (start_pop[0] - 1):
                if pop_num_len_elem < (start_pop[1] - 1):
                    continue

            first_recur = pop_num_len_elem / pop_num_len  # 进度第一部分
            if pop_from == 'bkwd':
                le = data_dict['le_mean'][pop_num_len_elem]
            elif pop_from == 'ga':
                le = data_dict['mle'][pop_len_elem][pop_num_len_elem]

            if le != 128 and (pop_len_elem == 0 or pop_num_len - pop_num_len_elem > 5):
                weight_mat = None
                tau_reciprocal = None
                for key in data_dict:
                    if key == 'pop':  # 获取权重矩阵和膜时间常数
                        if pop_from == 'bkwd':
                            weight_mat, tau_reciprocal = data_dict[key][pop_num_len_elem + 1]
                        elif pop_from == 'ga':
                            weight_mat, tau_reciprocal = data_dict[key][pop_len_elem][pop_num_len_elem]

                # if select_name == seri_data_name[0]:
                #     # 生成洛伦兹序列
                #     dataset_norm = tp.lorenz_data_creat(sample_len=data_len, n_samples=1, washout=washout,
                #                                         norm_mode=norm_mode, dim=dim)
                #     input_dim = dataset_norm.shape[-1]
                #     output_dim = dataset_norm.shape[-1]

                # 设置不同的输入衰减进行测试
                reduction_ratio_list = []
                for reduction_ratio_denominator in range(reduction_ratio_list_len):
                    if (reduction_ratio_list_len == 1) or fixed_reduction:
                        reduction_ratio = 1 / (2 ** reduction_bias)  # 综合考虑
                    else:
                        reduction_ratio = 1 / (2 ** (reduction_ratio_denominator + reduction_bias))
                    reduction_ratio_list.append(reduction_ratio)
                    # 初始化网络
                    net, net_valid = tp.net_init(tau=1 / tau_reciprocal, v_threshold=v_th, lsm_node=node_num,
                                                 init_neuron=spike_v_init, weight_mat=weight_mat, le=le,
                                                 device=device,
                                                 input_dim=input_dim, output_dim=output_dim,
                                                 reduction_ratio=reduction_ratio,
                                                 train_s_and_v=train_s_and_v, input_weight_array=None)

                    # 训练测试获得数据
                    nmse_list, nrmse_list, rmse_list, spike_sum, mse_list, mae_list = (
                        tp.get_test_result(net=net, input_layer=input_layer, seri_data=[dataset, dataset_norm],
                                           train_len=train_len,
                                           test_len=test_len,
                                           bench_num=bench_num,
                                           target_row=target_row,
                                           node_num=node_num, device=device,
                                           le=le, predict_step=predict_step, give_up_len=give_up_len,
                                           draw_pic=False, reduction_ratio=reduction_ratio, history_step=history_step))
                    # print(f'net_valid={net_valid}\treduction_ratio={reduction_ratio}')
                    # print(f'nmse={nmse_list}\n'
                    #       f'nrmse={nrmse_list}\n'
                    #       f'rmse={rmse_list}\n')

                    for j in range(len(error_data_name)):  # 两层循环分别记录各个维度的各种误差
                        for k in range(len(error_data_dim)):
                            if error_data_name[j] == 'nmse':
                                error_data_list[j][k][reduction_ratio_denominator].append(nmse_list[k])
                            elif error_data_name[j] == 'nrmse':
                                error_data_list[j][k][reduction_ratio_denominator].append(nrmse_list[k])
                            elif error_data_name[j] == 'rmse':
                                error_data_list[j][k][reduction_ratio_denominator].append(rmse_list[k])
                            elif error_data_name[j] == 'mse':
                                error_data_list[j][k][reduction_ratio_denominator].append(mse_list[k])
                            elif error_data_name[j] == 'mae':
                                error_data_list[j][k][reduction_ratio_denominator].append(mae_list[k])
                    spike_sum_list[reduction_ratio_denominator].append(spike_sum)

                le_list.append(le)  # 记录李雅普诺夫指数
                wb = openpyxl.Workbook()  # 写进excel
                for j in range(len(error_data_name)):
                    for k in range(len(error_data_dim)):
                        ws = wb.create_sheet(f'{error_data_name[j]}_{error_data_dim[k]}')
                        ws.append(le_list)
                        for i in range(reduction_ratio_list_len):
                            ws.append(error_data_list[j][k][i])

                wm = wb.create_sheet(f'spike_sum')
                wm.append(le_list)
                for i in range(reduction_ratio_list_len):
                    wm.append(spike_sum_list[i])
                wm.append(reduction_ratio_list)
                file_name = None
                if pop_from == 'bkwd':
                    file_name = f'./nrmse_sum_{predict_step}_{select_name}{NARMA_order}.xlsx'
                elif pop_from == 'ga':
                    file_name = f'./nrmse({seed})_sum_{predict_step}_{select_name}{NARMA_order}.xlsx'

                # while os.path.exists(os.path.join(out_dir, file_name)):
                #     file_name.removesuffix('.xlsx')
                #     file_name += '_c'
                #     file_name += '.xlsx'
                wb.save(os.path.join(out_dir, file_name))

                torch.cuda.empty_cache()
                pass

            else:
                pass

            progress_bar.progress_bar_fun(start_time=start_time, progress=first_recur)
            print(f'le={le:.4f}', end='')


if __name__ == '__main__':
    main()

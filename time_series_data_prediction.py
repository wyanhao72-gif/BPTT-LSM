"""
时间序列预测核心模块
本模块实现了时间序列预测的核心功能，包括网络初始化、训练、评估和可视化。
支持液体状态机网络、岭回归等多种模型，适用于混沌系统预测任务。
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from echotorch.data.datasets.LorenzAttractor import LorenzAttractor
from echotorch.data.datasets.NARMADataset import NARMADataset
from sklearn.linear_model import RidgeCV
from spikingjelly.activation_based import functional

import lsm_net
import narma_estimate
import le_backward as bkwd


def net_init(tau, v_threshold, lsm_node, init_neuron, weight_mat, le, device, input_dim, output_dim,
             reduction_ratio=0.1, train_s_and_v=True, input_weight_array=None):
    """

    :param input_weight_array: 初始网络的输入层
    :param tau: 膜时间常数
    :param v_threshold: 阈值电压
    :param lsm_node: 液体层节点
    :param init_neuron: 是否随机化神经元
    :param weight_mat: 矩阵权重
    :param le: 李指数
    :param device: 部署设备
    :param input_dim: 输入维数
    :param output_dim: 输出维数
    :param reduction_ratio: 输入层权重缩放系数
    :param train_s_and_v: 液体层输出是否为脉冲和电压的组合
    :return: 网络, 网络的有效性
    """
    node_num = lsm_node  # 一个东西不同名称
    preheat_step = 20
    bench_num = 1
    rc_input = torch.zeros(size=[preheat_step, bench_num, node_num]).to(device)  # 空的输入
    net = lsm_net.Lsm(tau=tau, v_threshold=v_threshold, lsm_node=lsm_node,
                      init_neuron=init_neuron, rc_spike_init=None, rc_v_init=None, device=device,
                      in_features=input_dim, out_features=output_dim, train_s_and_v=train_s_and_v
                      )
    net = bkwd.set_net_state(net=net, mode='test')
    functional.set_step_mode(net=net, step_mode='m')  # 网络模式设置为多步
    net.input_weight_init(reduction_ratio=reduction_ratio, input_weight_array=input_weight_array)  # 初始化输入权重
    net.rc_weight_init(rc_weight_array=weight_mat)  # 初始化网络权重
    net.to(device)

    rc_out = torch.zeros(size=(lsm_node,))  # 初始化储层输出
    rc_fire_rate = torch.sum(rc_out) / lsm_node  # 计算储层脉冲发放率
    fire_ratio = 0.5  # 设置第一次储层脉冲发放率，以便寻找合适的脉冲发放率使得储层处于持续运行状态
    cycle_num = 0  # 循环次数
    cycle_num_max = 100  # 循环最大次数
    while (rc_fire_rate <= 0 or rc_fire_rate >= 0.9) and cycle_num < cycle_num_max:
        functional.reset_net(net)
        rc_spike_init, rc_v_init = lsm_net.init_spike_v(node_num=lsm_node, fire_ratio=fire_ratio, v_th=v_threshold
                                                        , rc_weight=weight_mat)
        rc_spike_init = torch.Tensor(rc_spike_init).to(device)
        rc_v_init = torch.Tensor(rc_v_init).to(device)
        # 初始化储层神经元的脉冲和膜电位
        net.rc[0].rc_spike_init = rc_spike_init
        net.rc[0].rc_v_init = rc_v_init

        rc_out = net.rc(rc_input)
        # rc_out_sum = torch.sum(rc_out, dim=2)  # 对储层脉冲发放进行记录，以便断点查看
        rc_fire_rate = torch.sum(rc_out[-1]) / lsm_node  # 计算储层脉冲发放率

        fire_ratio += 0.1
        fire_ratio, _ = math.modf(fire_ratio)
        # if rc_fire_rate <= 0:
        #     fire_ratio += 0.1
        # elif rc_fire_rate >= 0.9:
        #     fire_ratio -= 0.1
        cycle_num += 1

        # if le < 0:  # le小于0本身就不好自激发
        #     break
    if rc_fire_rate <= 0 or rc_fire_rate >= 0.9:
        net_valid = False
    else:
        net_valid = True
    return net, net_valid


def lorenz_data_creat(sample_len, n_samples, xyz=None, sigma=10, b=8 / 3, r=28, dt=0.01, washout=0, norm_mode='norm',
                      dim=0):
    """
    用来创建lorenz混沌序列的函数
    :param dim: 归一化的方式None全维度， 0每个维度单独归一化
    :param sample_len: 序列长度
    :param n_samples: 序列个数
    :param xyz: 初始值
    :param sigma: 第一行公式参数
    :param b: 第三行公式参数
    :param r: 第二行公式参数
    :param dt: 单位时间间隔
    :param washout: 冲洗长度
    :param norm_mode: norm：归一化，std：标准化
    :return: 洛伦兹序列
    """
    if xyz is None:  # 初始值
        xyz = [12, 12, 9]

    # 获取洛伦兹数据
    lorenz_attractor = LorenzAttractor(sample_len=sample_len, n_samples=n_samples,
                                       xyz=xyz, sigma=sigma, b=b, r=r, dt=dt,
                                       washout=washout, normalize=True, seed=None, norm_mode=norm_mode, dim=dim)
    lorenz_data = lorenz_attractor.outputs
    return lorenz_data[0][-sample_len:]  # 返回洛伦兹数据，单个


def NARMA_data_creat(sample_len, n_samples, system_order=10):
    """
    NARMA序列生成
    :param sample_len: 序列长度
    :param n_samples: 序列个数
    :param system_order: 序列阶数
    :return: 序列
    """
    narma10_train_dataset = NARMADataset(sample_len=sample_len,
                                         n_samples=n_samples, system_order=system_order)

    inputs = narma10_train_dataset.inputs[0][-sample_len:]
    outputs = narma10_train_dataset.outputs[0][-sample_len:]
    return inputs, outputs

def lorenz84_data_creat(sample_len, xyz=None, a=0.25, b=4.0, F=8.0, G=1.0,
                         dt=0.01, washout=200, norm_mode='norm', dim=0):
    """
    生成 Lorenz-84 三维混沌时间序列。

    Lorenz-84 方程：
        dX/dt = -Y^2 - Z^2 - aX + aF
        dY/dt = XY - bXZ - Y + G
        dZ/dt = bXY + XZ - Z

    :param sample_len: 最终保留的序列长度
    :param xyz: 初始状态 [X0, Y0, Z0]
    :param a: Lorenz-84 参数a
    :param b: Lorenz-84 参数b
    :param F: Lorenz-84 外部强迫参数F
    :param G: Lorenz-84 非对称热力强迫参数G
    :param dt: 采样时间间隔
    :param washout: 舍弃的初始过渡步数
    :param norm_mode: 'norm'为Min-Max归一化，'std'为标准化
    :param dim: None表示全部维度共同归一化，0表示各维度分别归一化
    :return: 形状为(sample_len, 3)的torch.Tensor
    """
    if sample_len <= 0:
        raise ValueError('sample_len must be positive')
    if washout < 0:
        raise ValueError('washout must be non-negative')
    if dt <= 0:
        raise ValueError('dt must be positive')

    if xyz is None:
        # 常用非平衡初始状态；大写X/Y/Z只是状态变量记号，不是矩阵
        xyz = [1.0, 1.0, 1.0]

    total_steps = sample_len + washout
    t_eval = np.arange(total_steps, dtype=np.float64) * dt

    def lorenz84_equation(_t, state):
        X, Y, Z = state
        dX = -Y ** 2 - Z ** 2 - a * X + a * F
        dY = X * Y - b * X * Z - Y + G
        dZ = b * X * Y + X * Z - Z
        return [dX, dY, dZ]

    solution = solve_ivp(
        fun=lorenz84_equation,
        t_span=(t_eval[0], t_eval[-1]),
        y0=np.asarray(xyz, dtype=np.float64),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-9,
        atol=1e-11
    )

    if not solution.success:
        raise RuntimeError(
            f'Lorenz-84 integration failed: {solution.message}'
        )

    lorenz84_data = torch.tensor(
        solution.y.T,
        dtype=torch.float32
    )

    # 舍弃初始过渡段，再保留sample_len个采样点
    lorenz84_data = lorenz84_data[washout:washout + sample_len]

    # 与现有Lorenz-63数据保持相同的预处理接口
    lorenz84_data = normalize_sample(
        sample=lorenz84_data,
        mode=norm_mode,
        dim=dim
    )

    return lorenz84_data

def get_test_result(net, input_layer, seri_data, train_len, test_len, bench_num=1, target_row=None, node_num=90,
                    device='cuda:0',
                    le=0, predict_step=1, give_up_len=0, draw_pic: bool = False, reduction_ratio=1,
                    history_step=1):
    """
    获取训练后的测试结果
    :param history_step: 历史步数用于推理
    :param reduction_ratio: 输入到液体层前的缩放系数
    :param input_layer: 输入层
    :param draw_pic: 是否画图
    :param give_up_len: NARMA中需要舍弃前十项
    :param target_row: 预测目标在数据集中的列
    :param predict_step: 预测步长
    :param le: 李雅普诺夫指数
    :param net: 网络
    :param seri_data: 序列数据
    :param train_len: 训练长度
    :param test_len: 测试长度
    :param bench_num: 批次
    :param node_num: 节点个数
    :param device: 设备'cpu' 或者 'cuda:0'
    :return: nmse_list, nrmse_list, rmse_list
    """
    # 训练集，测试集输入输出
    seri_data_ori = torch.Tensor(seri_data[0])  # 未归一化或者标准化的原数据
    seri_data = torch.Tensor(seri_data[1])  # 归一化或者标准化之后的原数据
    input_set, output_set = (seri_data[:train_len + test_len],
                             seri_data_ori[predict_step:train_len + test_len + predict_step])
    train_set_input, test_set_input = input_set[:train_len], input_set[train_len:train_len + test_len]
    if target_row is None:
        output_dim = seri_data_ori.shape[-1]
        train_set_output, test_set_output = (output_set[:train_len],
                                             output_set[train_len:train_len + test_len])
    else:
        output_dim = len(target_row)
        train_set_output, test_set_output = (output_set[:train_len, target_row],
                                             output_set[train_len:train_len + test_len, target_row])

    input_dim = seri_data.shape[-1]
    # train_set_input = train_set_input.reshape(train_len, bench_num, input_dim).to(device)  # 整形
    # test_set_input = test_set_input.reshape(test_len, bench_num, input_dim).to(device)  # 整形
    train_set_input = train_set_input.to(device)
    test_set_input = test_set_input.to(device)
    # 训练
    train_s_and_v = True
    s_v_combine = False
    # train_one_step = True
    if input_layer is None:
        rc_out_train = net(train_set_input.reshape(train_len, bench_num, input_dim))
        rc_out_test = net(test_set_input.reshape(test_len, bench_num, input_dim))
    else:
        input_layer_out_train = input_layer(train_set_input).reshape(train_len, bench_num, node_num) * reduction_ratio
        rc_out_train = net.rc(input_layer_out_train)  # 网络必须设置成多步

        input_layer_out_test = input_layer(test_set_input).reshape(test_len, bench_num, node_num) * reduction_ratio
        rc_out_test = net.rc(input_layer_out_test)  # 网络必须设置成多步

    if train_s_and_v:
        if s_v_combine:
            rc_out = net.rc[0].rc_out_v_s.reshape(-1, 1, node_num)  # 取网络的状态（s, v）

        else:
            rc_out = net.rc[0].rc_state.reshape(-1, 1, 2 * node_num)  # 取网络的状态（s, v）
            # rc_out = net.rc[0].rc_state_s_h.reshape(-1, 1, 2 * node_num)  # 取网络的状态（s, v）

        new_rc_out_set = torch.tensor([]).to(rc_out)
        for x_len_para in range(rc_out.shape[0]):
            new_rc_out = torch.tensor([]).to(rc_out)
            for history_step_para in range(history_step):
                head_index = x_len_para - history_step_para
                if head_index < 0:
                    new_rc_out = torch.cat([new_rc_out, torch.zeros(size=(1, 1, rc_out.shape[-1])).to(rc_out)], dim=2)
                else:
                    new_rc_out = torch.cat([new_rc_out, rc_out[head_index: head_index + 1]], dim=2)
            new_rc_out_set = torch.cat([new_rc_out_set, new_rc_out], dim=0)

        rc_out_train = new_rc_out_set[-train_len + give_up_len - test_len: -test_len]  # 由于在层函数中这些状态都是连续存储的所以需要取末尾

        rc_out_test = new_rc_out_set[-test_len:]  # 由于在层函数中这些状态都是连续存储的所以需要取末尾
    # rc_out_sum = torch.sum(rc_out, dim=2)  # 对储层脉冲发放进行记录，以便断点查看
    # for i in range(rc_out_sum.shape[0]):
    #     print(f'{rc_out_sum[i]}')
    X = rc_out_train.reshape(-1, rc_out_train.shape[-1])  # 岭回归的预测数据由(train_len,1 ,node_num)整成(train_len, node_num)
    X_test = rc_out_test.reshape(-1, rc_out_test.shape[-1])
    X = X.cpu().detach().numpy()
    X_test = X_test.cpu().detach().numpy()
    y = train_set_output[give_up_len:].numpy()  # 岭回归实际数据
    y_test = test_set_output.numpy()

    test_predict = None
    w_out = None
    bias_out = None
    train_predict = None

    alphas = [1e-11, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]  # 通常1是最好的，容我明天研究一下
    min_nmse = np.inf
    best_test_predict = None
    # for alpha in alphas:
    ridge_cv = RidgeCV(alphas=alphas)
    ridge_cv.fit(X, y)
    # score = ridge_cv.score(X=X_test, y=y_test)

    # score = ridge_cv.score(rc_out, label_onehot)
    # decision = ridge_cv.decision_function(rc_out)
    train_predict = ridge_cv.predict(X).reshape(-1, y.shape[-1])
    test_predict = ridge_cv.predict(X_test).reshape(-1, y.shape[-1])
    nmse_train = narma_estimate.nmse(prediction=train_predict, ground_truth=y)
    # u = np.sum((test_predict - y_test) ** 2)
    # v = np.sum((y_test - y_test.mean()) ** 2)
    # r = 1 - u/v
    # print(f'alpha = {alpha}, nmse = {nmse}')
    if nmse_train < min_nmse:
        min_nmse = nmse_train
        best_test_predict = test_predict
        # train_predict = ridge_cv.predict(X)
        # test_predict = ridge_cv.predict(X_test)
        w_out = ridge_cv.coef_  # 岭回归计算得权重 size=(3,90)
        bias_out = ridge_cv.intercept_  # 岭回归计算偏置

    if draw_pic:
        draw_lorenz(predict=train_predict, true=y, le=le)  # 画图
        draw_lorenz(predict=test_predict, true=y_test, le=le)  # 画图

    # 测试------
    net.output_weight_init(output_weight_array=w_out, output_bias_array=bias_out)  # 把岭回归权重赋值给网络
    # lsm_out_test = None
    # predict_set = torch.tensor([]).to(device)  # 记录预测值
    # spike_sum = 0

    # lsm_out_test = net.output_layer[0](rc_out_test)  # (input_len, bench, out_features)
    spike_sum = net.rc[0].rc_fire.reshape(-1, node_num)[-test_len:]
    spike_sum = torch.sum(spike_sum)
    # if train_one_step:  # 单步或者多步预测
    #     if input_layer is None:
    #         rc_out = net(test_set_input.reshape(test_len, bench_num, input_dim))
    #     else:
    #         input_layer_out = input_layer(test_set_input).reshape(test_len, bench_num, node_num) * reduction_ratio
    #         rc_out = net.rc(input_layer_out)  # 网络必须设置成多步
    #
    #     if train_s_and_v:
    #         if s_v_combine:
    #             rc_out = net.rc[0].rc_out_v_s.reshape(-1, 1, node_num)  # 取网络的状态（s, v）
    #         else:
    #             rc_out = net.rc[0].rc_state.reshape(-1, 1, 2 * node_num)  # 取网络的状态（s, v）
    #         rc_out = rc_out[-test_len:]  # 由于在层函数中这些状态都是连续存储的所以需要取末尾
    #     lsm_out_test = net.output_layer[0](rc_out)  # (input_len, bench, out_features)
    #     spike_sum = net.rc[0].rc_fire.reshape(-1, node_num)[-test_len:]
    #     spike_sum = torch.sum(spike_sum)
    #     predict_set = lsm_out_test
    # else:  # 首尾相连的模式
    #     for test_len_elem in range(test_len):
    #         if test_len_elem == 0:
    #             rc_out_test = net(test_set_input[-1:])  # 网络必须设置成多步
    #         else:
    #             rc_out_test = net(lsm_out_test)
    #
    #         if train_s_and_v:
    #             if s_v_combine:
    #                 rc_out_test = net.rc[0].rc_out_v_s.reshape(-1, 1, node_num)  # 取网络的状态（s, v）
    #             else:
    #                 rc_out_test = net.rc[0].rc_state.reshape(-1, 1, 2 * node_num)  # 取网络的状态（s, v）
    #             rc_out_test = rc_out_test[-1:]
    #         lsm_out_test = net.output_layer[0](rc_out_test)  # (input_len, bench, out_features)
    #         predict_set = torch.cat([predict_set, lsm_out_test], dim=0)  # (test_len, bench,3)预测数据拼接
    #         spike_sum += torch.sum(net.rc[0].y)
    # print(f'spike_sum={spike_sum}')
    spike_sum = spike_sum.cpu().detach().numpy().item()

    # test_set_input = test_set_input.cpu().detach().numpy()[:, 0, :]

    nmse_list = []
    nrmse_list = []
    rmse_list = []
    mse_list = []
    mae_list = []
    if output_dim == 1:
        range_output_dim = 1
    else:
        range_output_dim = output_dim + 1
    for lorenz_dim_elem in range(range_output_dim):  # 分别计算x,y,z的误差数据以及总维度
        if lorenz_dim_elem == range_output_dim - 1:  # 计算总维度的误差
            # 计算相关误差
            nrmse, rmse, nmse, mse, mae = narma_estimate.nrmse(best_test_predict, test_set_output)
        else:
            # 计算相关误差
            nrmse, rmse, nmse, mse, mae = \
                (narma_estimate.nrmse(best_test_predict[:, lorenz_dim_elem], test_set_output[:, lorenz_dim_elem]))
        nmse_list.append(nmse)
        nrmse_list.append(nrmse)
        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)
    return nmse_list, nrmse_list, rmse_list, spike_sum, mse_list, mae_list


def draw_lorenz(predict, true, le=None):
    """
    画洛伦兹序列图
    :param le: LE
    :param predict: 预测值
    :param true: 真实值
    :return:
    """
    # predict = predict.cpu().detach()
    # true = true.cpu().detach()
    predict = np.array(predict).reshape((predict.shape[0], -1))
    true = np.array(true)
    name = ['x', 'y', 'z']
    for i in range(predict.shape[-1]):  # 分别画x,y,z三个序列的真实和预测
        plt.figure(num=i + 1, figsize=(12, 6))
        plt.rc('font', family=['Times New Roman', 'SimSun'])
        plt.title(f"lorenz({name[i]})--le={le}", family=['Times New Roman', 'SimSun'])
        plt.xlabel("step", size=8)
        plt.ylabel("number", size=8)
        plt.plot(true[:, i], linewidth=1, label=f'true', color='green')
        plt.plot(predict[:, i], linewidth=1, label=f'predict', color='lime')
        plt.plot(true[:, i] - predict[:, i], linewidth=1, label=f'error', color='red')
        plt.plot(np.zeros(shape=true[:, i].shape), linewidth=1)
        plt.legend()
        # plt.tight_layout()
    if predict.shape[-1] >= 3:
        plt.figure(num=4, figsize=(12, 6))
        ax = plt.axes(projection='3d')  # 画洛伦兹3d图
        ax.plot3D(true[:, 0], true[:, 1], true[:, 2])
        ax.plot3D(predict[:, 0], predict[:, 1], predict[:, 2])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel(f'Z\nle={le}')

    plt.show()


def normalize_sample(sample, mode='norm', dim=0):
    total_size = sample.shape[0]
    if mode == 'norm':
        if dim is None:
            maxval = torch.max(sample)
            minval = torch.min(sample)
            sample = (sample - minval.repeat(total_size, 1)) / (maxval - minval)
        else:
            maxval, _ = torch.max(sample, dim=dim)
            minval, _ = torch.min(sample, dim=dim)
            sample = torch.mm((sample - minval.repeat(total_size, 1)), torch.inverse(torch.diag(maxval - minval)))
    elif mode == 'std':
        if dim is None:
            meanval = torch.mean(sample)
            stdval = torch.std(sample)
        else:
            meanval = torch.mean(sample, dim=dim)
            stdval = torch.std(sample, dim=dim)
        sample = (sample - meanval) / stdval
    return sample


def main():
    lorenz_data = lorenz_data_creat(sample_len=11000, n_samples=1)
    name = ['x', 'y', 'z']
    for i in range(lorenz_data.size(1)):
        plt.figure(num=i + 1, figsize=(12, 6))
        plt.rc('font', family=['Times New Roman', 'SimSun'])
        plt.title(f"lorenz_{name[i]}", family=['Times New Roman', 'SimSun'])
        plt.xlabel("step", size=8)
        plt.ylabel("number", size=8)
        plt.plot(lorenz_data[:, i],
                 linewidth=1,
                 label=f'{name[i]}')
        plt.legend()
        plt.tight_layout()
    plt.show()
    pass


def dataset_from(
        name,
        device,
        random_bptt=False,
        narma_order=20,
        lorenz_gap=10,
        robot_gap=1,
        robot_train_ratio=0.8
):
    """
    统一读取和构造时间序列预测任务。

    支持：
        ETTh1
        Lorenz63
        Lorenz84
        NARMA
        GR1Robot

    GR1Robot任务定义：
        observation.state(t)
            ->
        action(t + robot_gap)

    参数
    ----------
    name : str
        数据集名称。

    device : str 或 torch.device
        运行设备，例如'cpu'或'cuda:0'。

    random_bptt : bool
        是否将训练序列随机切成20~50步的BPTT片段。

    narma_order : int
        NARMA系统阶数。

    lorenz_gap : int
        Lorenz提前预测步数。

    robot_gap : int
        GR1机器人动作提前预测步数。

    robot_train_ratio : float
        按Episode划分训练集的比例。
    """

    output_ori = True
    norm_mode = 'norm'
    dim = 0

    dataset_names = [
        'ETTh1',
        'Lorenz63',
        'Lorenz84',
        'NARMA',
        'GR1Robot'
    ]

    # ==========================================================
    # ETTh1
    # ==========================================================
    if name == 'ETTh1':
        predict_step = 24
        train_len = 24 * 365
        test_len = 24 * 90
        target_row = [6]

        data_dir = './dataset/ETTh1.npy'

        dataset = np.load(data_dir)
        dataset = dataset[
            :train_len + test_len + predict_step
        ]

        dataset = torch.tensor(
            dataset,
            dtype=torch.float32
        )

        dataset_norm = normalize_sample(
            sample=dataset,
            mode=norm_mode,
            dim=dim
        )

        if not output_ori:
            dataset = dataset_norm

        input_set = dataset_norm[
            :train_len + test_len
        ]

        output_set = dataset[
            predict_step:
            train_len + test_len + predict_step
        ]

        input_dim = input_set.shape[-1]
        output_dim = len(target_row)

        train_set = (
            input_set[:train_len].to(device),
            output_set[:train_len, target_row].to(device)
        )

        test_set = (
            input_set[
                train_len:
                train_len + test_len
            ].to(device),

            output_set[
                train_len:
                train_len + test_len,
                target_row
            ].to(device)
        )

    # ==========================================================
    # Lorenz-63
    # ==========================================================
    elif name in ['Lorenz', 'Lorenz63']:
        predict_step = lorenz_gap
        train_len = 10000
        test_len = 1000
        target_row = None

        data_len = (
            train_len
            + test_len
            + predict_step
        )

        dataset_norm = lorenz_data_creat(
            sample_len=data_len,
            n_samples=1,
            washout=200,
            norm_mode=norm_mode,
            dim=dim
        )

        dataset = dataset_norm

        input_set = dataset_norm[
            :train_len + test_len
        ]

        output_set = dataset[
            predict_step:
            train_len + test_len + predict_step
        ]

        input_dim = input_set.shape[-1]
        output_dim = output_set.shape[-1]

        train_set = (
            input_set[:train_len].to(device),
            output_set[:train_len].to(device)
        )

        test_set = (
            input_set[
                train_len:
                train_len + test_len
            ].to(device),

            output_set[
                train_len:
                train_len + test_len
            ].to(device)
        )

    # ==========================================================
    # Lorenz-84
    # ==========================================================
    elif name == 'Lorenz84':
        predict_step = lorenz_gap
        train_len = 10000
        test_len = 1000
        target_row = None

        data_len = (
            train_len
            + test_len
            + predict_step
        )

        dataset_norm = lorenz84_data_creat(
            sample_len=data_len,
            xyz=[1.0, 1.0, 1.0],
            a=0.25,
            b=4.0,
            F=8.0,
            G=1.0,
            dt=0.01,
            washout=200,
            norm_mode=norm_mode,
            dim=dim
        )

        dataset = dataset_norm

        input_set = dataset_norm[
            :train_len + test_len
        ]

        output_set = dataset[
            predict_step:
            train_len + test_len + predict_step
        ]

        input_dim = input_set.shape[-1]
        output_dim = output_set.shape[-1]

        train_set = (
            input_set[:train_len].to(device),
            output_set[:train_len].to(device)
        )

        test_set = (
            input_set[
                train_len:
                train_len + test_len
            ].to(device),

            output_set[
                train_len:
                train_len + test_len
            ].to(device)
        )

    # ==========================================================
    # NARMA
    # ==========================================================
    elif name == 'NARMA':
        predict_step = 1
        train_len = 3000
        test_len = 1000
        target_row = None

        data_len = (
            train_len
            + test_len
            + predict_step
        )

        narma_input, narma_output = NARMA_data_creat(
            sample_len=data_len,
            n_samples=1,
            system_order=narma_order
        )

        # 输入由外部驱动u(t)和当前输出y(t)共同组成
        dataset_norm = torch.cat(
            (narma_input, narma_output),
            dim=1
        )

        dataset_norm = normalize_sample(
            sample=dataset_norm,
            mode=norm_mode,
            dim=dim
        )

        dataset = narma_output

        input_set = dataset_norm[
            :train_len + test_len
        ]

        output_set = dataset[
            predict_step:
            train_len + test_len + predict_step
        ]

        input_dim = input_set.shape[-1]
        output_dim = output_set.shape[-1]

        train_set = (
            input_set[:train_len].to(device),
            output_set[:train_len].to(device)
        )

        test_set = (
            input_set[
                train_len:
                train_len + test_len
            ].to(device),

            output_set[
                train_len:
                train_len + test_len
            ].to(device)
        )

    # ==========================================================
    # GR1机器人动作预测
    # ==========================================================
    elif name == 'GR1Robot':
        predict_step = robot_gap

        if predict_step < 1:
            raise ValueError(
                'robot_gap must be greater than or equal to 1'
            )

        if not 0 < robot_train_ratio < 1:
            raise ValueError(
                'robot_train_ratio must be between 0 and 1'
            )

        state_path = './dataset/GR1_state.npy'
        action_path = './dataset/GR1_action.npy'
        episode_path = './dataset/GR1_episode_index.npy'

        state = np.load(state_path)
        action = np.load(action_path)
        selected_dims = [0, 1, 2]

        state = state[:, selected_dims]
        action = action[:, selected_dims]
        episode_index = np.load(episode_path)

        state = torch.tensor(
            state,
            dtype=torch.float32
        )

        action = torch.tensor(
            action,
            dtype=torch.float32
        )

        episode_index = torch.tensor(
            episode_index,
            dtype=torch.long
        )

        # ------------------------------------------------------
        # 数据合法性检查
        # ------------------------------------------------------
        if state.ndim != 2:
            raise ValueError(
                f'GR1 state should be two-dimensional, '
                f'but got shape {state.shape}'
            )

        if action.ndim != 2:
            raise ValueError(
                f'GR1 action should be two-dimensional, '
                f'but got shape {action.shape}'
            )

        if episode_index.ndim != 1:
            raise ValueError(
                f'GR1 episode_index should be one-dimensional, '
                f'but got shape {episode_index.shape}'
            )

        if not (
                state.shape[0]
                == action.shape[0]
                == episode_index.shape[0]
        ):
            raise ValueError(
                'GR1 state, action and episode_index '
                'must have the same number of rows'
            )

        if not torch.isfinite(state).all():
            raise ValueError(
                'GR1 state contains NaN or Inf'
            )

        if not torch.isfinite(action).all():
            raise ValueError(
                'GR1 action contains NaN or Inf'
            )

        # ------------------------------------------------------
        # 只取排序后的前30个Episode
        # ------------------------------------------------------
        all_unique_episodes = torch.unique(
            episode_index,
            sorted=True
        )

        max_episode_num =50

        if all_unique_episodes.numel() < max_episode_num:
            raise ValueError(
                f'GR1Robot requires at least {max_episode_num} episodes, '
                f'but only {all_unique_episodes.numel()} episodes were found'
            )

        unique_episodes = all_unique_episodes[
                          :max_episode_num
                          ]

        num_episodes = unique_episodes.numel()

        if num_episodes < 2:
            raise ValueError(
                'GR1Robot requires at least two episodes'
            )

        # ------------------------------------------------------
        # 按Episode划分训练集和测试集
        # 前80%训练，后20%测试
        #
        # 当前固定使用30个Episode：
        # 24个训练Episode
        # 6个测试Episode
        # ------------------------------------------------------
        num_train_episodes = int(
            num_episodes * robot_train_ratio
        )

        num_train_episodes = max(
            1,
            min(
                num_train_episodes,
                num_episodes - 1
            )
        )

        train_episode_ids = unique_episodes[
                            :num_train_episodes
                            ]

        test_episode_ids = unique_episodes[
                           num_train_episodes:
                           ]

        # ------------------------------------------------------
        # 在每个Episode内部构造：
        # state(t) -> action(t + predict_step)
        # ------------------------------------------------------
        train_input_parts = []
        train_output_parts = []
        train_episode_parts = []

        test_input_parts = []
        test_output_parts = []
        test_episode_parts = []

        for episode_id in unique_episodes:
            episode_mask = (
                    episode_index == episode_id
            )

            episode_state = state[
                episode_mask
            ]

            episode_action = action[
                episode_mask
            ]

            episode_length = episode_state.shape[0]

            if episode_length <= predict_step:
                print(
                    f'Warning: episode {episode_id.item()} '
                    f'is shorter than or equal to '
                    f'predict_step={predict_step}, skipped'
                )
                continue

            episode_input = episode_state[
                            :-predict_step
                            ]

            episode_output = episode_action[
                             predict_step:
                             ]

            episode_labels = torch.full(
                size=(episode_input.shape[0],),
                fill_value=int(episode_id.item()),
                dtype=torch.long
            )

            if torch.any(
                    train_episode_ids == episode_id
            ):
                train_input_parts.append(
                    episode_input
                )

                train_output_parts.append(
                    episode_output
                )

                train_episode_parts.append(
                    episode_labels
                )

            else:
                test_input_parts.append(
                    episode_input
                )

                test_output_parts.append(
                    episode_output
                )

                test_episode_parts.append(
                    episode_labels
                )

        if not train_input_parts:
            raise RuntimeError(
                'No valid GR1 training episodes remain'
            )

        if not test_input_parts:
            raise RuntimeError(
                'No valid GR1 testing episodes remain'
            )

        train_input_raw = torch.cat(
            train_input_parts,
            dim=0
        )

        train_output_raw = torch.cat(
            train_output_parts,
            dim=0
        )

        train_sample_episode = torch.cat(
            train_episode_parts,
            dim=0
        )

        test_input_raw = torch.cat(
            test_input_parts,
            dim=0
        )

        test_output_raw = torch.cat(
            test_output_parts,
            dim=0
        )

        test_sample_episode = torch.cat(
            test_episode_parts,
            dim=0
        )

        # ------------------------------------------------------
        # 归一化只使用训练集统计量
        # 防止测试集信息泄漏
        # ------------------------------------------------------
        input_min = train_input_raw.min(
            dim=0
        ).values

        input_max = train_input_raw.max(
            dim=0
        ).values

        input_scale = (
                input_max - input_min
        ).clamp_min(1e-8)

        train_input = (
                              train_input_raw - input_min
                      ) / input_scale

        test_input = (
                             test_input_raw - input_min
                     ) / input_scale

        if output_ori:
            train_output = train_output_raw
            test_output = test_output_raw

        else:
            output_min = train_output_raw.min(
                dim=0
            ).values

            output_max = train_output_raw.max(
                dim=0
            ).values

            output_scale = (
                    output_max - output_min
            ).clamp_min(1e-8)

            train_output = (
                                   train_output_raw - output_min
                           ) / output_scale

            test_output = (
                                  test_output_raw - output_min
                          ) / output_scale

        input_dim = train_input.shape[-1]
        output_dim = train_output.shape[-1]

        # ------------------------------------------------------
        # 返回格式和ETTh保持一致
        # ------------------------------------------------------
        train_set = (
            train_input.to(device),
            train_output.to(device)
        )

        test_set = (
            test_input.to(device),
            test_output.to(device)
        )

        if random_bptt:
            train_set = random_bptt_cut_by_episode(
                input_data=train_set[0],
                output_data=train_set[1],
                episode_index=train_sample_episode
            )

            test_set = random_bptt_cut_by_episode(
                input_data=test_set[0],
                output_data=test_set[1],
                episode_index=test_sample_episode
            )

        print('\nGR1Robot dataset loaded')
        print('=' * 60)

        print(
            f'Available episodes:     '
            f'{len(all_unique_episodes)}'
        )

        print(
            f'Used episodes:          '
            f'{len(unique_episodes)}'
        )

        print(
            f'Input dimension:        '
            f'{input_dim}'
        )

        print(
            f'Output dimension:       '
            f'{output_dim}'
        )

        print(
            f'Prediction step:        '
            f'{predict_step}'
        )

        print(
            f'Train episodes:         '
            f'{len(train_episode_ids)}'
        )

        print(
            f'Test episodes:          '
            f'{len(test_episode_ids)}'
        )

        print(
            f'Train samples:          '
            f'{train_input.shape[0]}'
        )

        print(
            f'Test samples:           '
            f'{test_input.shape[0]}'
        )

        print('=' * 60)

    else:
        raise ValueError(
            f'do not have dataset: {name}; '
            f'supported datasets are {dataset_names} '
            f'and Lorenz alias'
        )

    # ==========================================================
    # 统一随机BPTT切分
    # GR1Robot已在自身分支中处理
    # ==========================================================
    if random_bptt and name != 'GR1Robot':
        train_set = random_bptt_cut(
            dataset=train_set
        )

        test_set = random_bptt_cut(
            dataset=test_set
        )

    return (
        input_dim,
        output_dim,
        train_set,
        test_set,
        predict_step
    )


def random_bptt_cut(dataset):
    set_start = 0
    train_list = []
    while set_start < dataset[0].shape[0]:
        this_part_len = np.random.randint(low=20, high=50)
        set_end = min(set_start + this_part_len, dataset[0].shape[0])
        part_set = (dataset[0][set_start:set_end], dataset[1][set_start:set_end])
        set_start = set_end
        train_list.append(part_set)
    return train_list
def random_bptt_cut_by_episode(
        input_data,
        output_data,
        episode_index,
        min_length=20,
        max_length=50
):
    """
    在各Episode内部随机切分BPTT片段。

    不允许一个训练片段跨越两个不同的Episode。
    """

    if input_data.shape[0] != output_data.shape[0]:
        raise ValueError(
            'input_data and output_data must have '
            'the same sequence length'
        )

    if input_data.shape[0] != episode_index.shape[0]:
        raise ValueError(
            'episode_index length does not match data length'
        )

    dataset_parts = []

    unique_episodes = torch.unique(
        episode_index,
        sorted=True
    )

    for episode_id in unique_episodes:
        episode_mask = (
            episode_index == episode_id
        )

        episode_input = input_data[
            episode_mask
        ]

        episode_output = output_data[
            episode_mask
        ]

        part_start = 0
        episode_length = episode_input.shape[0]

        while part_start < episode_length:
            part_length = np.random.randint(
                low=min_length,
                high=max_length
            )

            part_end = min(
                part_start + part_length,
                episode_length
            )

            part_input = episode_input[
                part_start:
                part_end
            ]

            part_output = episode_output[
                part_start:
                part_end
            ]

            dataset_parts.append((
                part_input,
                part_output
            ))

            part_start = part_end

    return dataset_parts

if __name__ == '__main__':
    main()

"""
线性网络时间序列预测主程序
本文件实现了基于线性网络的时间序列预测任务，支持多种数据集（ETTh1, Lorenz, NARMA等）。
包含网络定义、训练、测试和结果保存功能。
"""

import datetime
import os
import time

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
    def __init__(self, in_dim, n_hidden, out_dim, history_step=1, for_lsm=False):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden),
            BatchNorm1dLN(n_hidden),
            nn.LeakyReLU()
        )

        if for_lsm:
            self.layer2 = nn.Sequential(
                nn.Linear(2 * history_step * n_hidden, out_dim)
            )
        else:
            self.layer2 = nn.Sequential(
                nn.Linear(n_hidden, out_dim)
            )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


def get_loss(dataset, device, net):
    if type(dataset) is tuple:
        input_dataset, output_dataset = dataset
        input_dataset = input_dataset.to(device)
        output_dataset = output_dataset.to(device)
        s, d = input_dataset.shape
        input_dataset = input_dataset.reshape(s, 1, d)
        outputs = net.forward(input_dataset).reshape(output_dataset.shape)
    else:
        outputs = torch.tensor([]).to(device)
        output_dataset_all = torch.tensor([]).to(device)
        for dataset_elem in dataset:
            input_dataset, output_dataset = dataset_elem
            s, d = input_dataset.shape
            input_dataset = input_dataset.reshape(s, 1, d)
            outputs_part = net.forward(input_dataset).reshape(output_dataset.shape)
            outputs = torch.cat([outputs, outputs_part], dim=0)
            output_dataset_all = torch.cat([output_dataset_all, output_dataset], dim=0)
        output_dataset = output_dataset_all.clone()
    loss = F.mse_loss(outputs, output_dataset)
    return loss, outputs, output_dataset

def global_gradient_l2_norm(net):
    squared_sum = 0.0

    for parameter in net.parameters():
        if parameter.requires_grad and parameter.grad is not None:
            squared_sum += (
                parameter.grad.detach().pow(2).sum().item()
            )

    return squared_sum ** 0.5
# 训练函数
def net_train(device, dataset, optimizer, net, scaler, out_dir=None, iteration=None, draw_pic=False, debug_grad=False,
              debug_net=None):
    optimizer.zero_grad()

    do_debug = False  # 检测梯度
    do_print_grad = True  # 查看梯度
    do_grad_clip = True  # 梯度裁剪
    detect_grad_nan = True  # 检测梯度是否为NAN,并剔除

    if scaler is not None:
        with amp.autocast():
            loss, outputs, output_dataset = get_loss(dataset=dataset, device=device, net=net)

        if debug_grad:
            lebw.test_grad_error(net=debug_net, loss=loss, do_debug=do_debug, do_print_grad=do_print_grad,
                                 do_grad_clip=do_grad_clip,
                                 detect_grad_nan=detect_grad_nan, out_dir=out_dir, optimizer=optimizer,
                                 iteration=iteration, scaler=scaler)
        else:
            scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss, outputs, output_dataset = get_loss(dataset=dataset, device=device, net=net)
        if debug_grad:
            lebw.test_grad_error(net=debug_net, loss=loss, do_debug=do_debug, do_print_grad=do_print_grad,
                                 do_grad_clip=do_grad_clip,
                                 detect_grad_nan=detect_grad_nan, out_dir=out_dir, optimizer=optimizer,
                                 iteration=iteration)
        else:
            loss.backward()
        gradient_l2_norm = global_gradient_l2_norm(
            debug_net if debug_net is not None else net
        )
        optimizer.step()

    nrmse, rmse, nmse, mse, mae = narma_estimate.nrmse(outputs, output_dataset)
    if draw_pic:
        tp.draw_lorenz(predict=outputs.cpu().detach(), true=output_dataset.cpu().detach())
    return nrmse, rmse, nmse, mse, mae,gradient_l2_norm


# 测试函数
def net_test(device, dataset, net, draw_pic=False):
    with torch.no_grad():
        loss, outputs, output_dataset = get_loss(dataset=dataset, device=device, net=net)

    nrmse, rmse, nmse, mse, mae = narma_estimate.nrmse(outputs, output_dataset)
    if draw_pic:
        tp.draw_lorenz(predict=outputs.cpu().detach(), true=output_dataset.cpu().detach())
    return nrmse, rmse, nmse, mse, mae


def main():
    # 加载中断文件
    # resume = f'./linear_amp/t(200)_sgd_lr0.01/checkpoint_max.pth'  # 是否从中断的次数开始，是则赋文件地址 like:'./checkpoint/'
    resume = ''

    node_num = 90  # 老传统90个隐藏层节点
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    random_bptt = False  # (net_name == 'rnn')

    dataset_names = ['ETTh1', 'Lorenz', 'NARMA']
    name = dataset_names[2]
    input_dim, output_dim, train_set, test_set, predict_step = dataset_from(name=name, device=device,
                                                                            random_bptt=random_bptt)  # (net_name == 'rnn')

    # 定义模型
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = three_linear_net(in_dim=input_dim, n_hidden=node_num, out_dim=output_dim).to(device)

    # 定义损失函数和优化器
    opt = 'sgd'
    lr = 1e-2  # 学习率
    momentum = 0.9  # 梯度下降算法惯性

    if opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    elif opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    else:
        raise NotImplementedError(opt)

    # 定义超参数
    max_iteration = 200
    use_amp = True  # 是否使用混合精度训练

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max_iteration)  # 余弦降低的学习率

    # 设置文件输出位置
    out_dir = './linear'
    scaler = None
    if use_amp:  # 添加混合精度计算
        out_dir += '_amp'
        scaler = amp.GradScaler()
    out_dir = os.path.join(out_dir, f't({max_iteration})_{name}_lr{lr}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    else:
        print(f'EXdir {out_dir}.')  # 建立并保存文件

    start_epoch = 0
    min_test_nrmse = 100
    if resume:  # 加载中断数据
        checkpoint = torch.load(resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['iteration'] + 1
        min_test_nrmse = checkpoint['min_test_nrmse']

    result_len = 5  # 保存的参数量(nrmse nmse mse mae rmse)
    train_result_list = np.zeros(shape=[max_iteration, result_len])
    test_result_list = np.zeros(shape=[max_iteration, result_len])
    for iteration in range(start_epoch, max_iteration):
        start_time = time.time()

        if iteration in [50, 100, 150, 199]:
            draw_pic = True
        else:
            draw_pic = False

        net.train()
        train_result = (
            net_train(device=device, dataset=train_set, optimizer=optimizer, net=net, scaler=scaler,
                      out_dir=out_dir, iteration=iteration, draw_pic=draw_pic, debug_grad=False))
        train_nrmse = train_result[0]
        train_result_list[iteration] = np.array(train_result)
        lr_scheduler.step()

        net.eval()
        test_result = net_test(device=device, dataset=test_set, net=net, draw_pic=draw_pic)
        test_nrmse = test_result[0]
        test_result_list[iteration] = np.array(test_result)

        save_max = False
        if test_nrmse < min_test_nrmse:
            min_test_nrmse = test_nrmse
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'iteration': iteration,
            'min_test_nrmse': min_test_nrmse
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
        # 保存数据
        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))
        wb = openpyxl.Workbook()  # 写进excel
        train_excel = wb.create_sheet(f'train')
        test_excel = wb.create_sheet(f'test')
        for evalu_para in range(train_result_list.shape[1]):
            train_excel.append(train_result_list[:, evalu_para].tolist())
            test_excel.append(test_result_list[:, evalu_para].tolist())
        file_name = f'./ETT_{predict_step}_({datetime.date.today()}).xlsx'
        wb.save(os.path.join(out_dir, file_name))

        print(out_dir)
        print(
            f'iteration = {iteration}, train_nrmse ={train_nrmse: .4f}, test_nrmse ={test_nrmse: .4f}, min_test_nrmse ={min_test_nrmse: .4f}')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (max_iteration - iteration))).strftime("%Y-%m-%d %H:%M:%S")}\n')


if __name__ == '__main__':
    main()

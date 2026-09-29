"""
时间序列预测评估指标模块
本模块实现了多种时间序列预测评估指标，包括NRMSE、RMSE、NMSE、MSE、MAE等。
主要用于NARMA等时间序列预测任务的性能评估。
"""

import torch


def nrmse(prediction, ground_truth):
    # 确保输入是Tensor
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32)
    # 获取输入数据维度
    dim = len(prediction.shape)
    # 计算误差的平方
    if dim == 1:
        squared_error = torch.pow(prediction - ground_truth, 2)
        absolute_error = torch.abs(prediction - ground_truth)
    else:
        squared_error = torch.pow(prediction - ground_truth, 2)
        squared_error = torch.sum(squared_error, dim=1)
        absolute_error = torch.abs(prediction - ground_truth)
        absolute_error = torch.sum(absolute_error, dim=1)
    # 计算MAE
    mae_value = torch.sum(absolute_error) / prediction.shape[0]
    # 计算MSE
    mse_value = torch.sum(squared_error) / prediction.shape[0]
    # 计算RMSE
    rmse_value = torch.sqrt(mse_value)
    if dim == 1:
        # 计算真实值范围
        den = torch.max(ground_truth) - torch.min(ground_truth)
        # 计算真实值方差
        den_nmse = ground_truth.var()
        std = torch.sqrt(den_nmse)
    else:
        ground_truth_len = torch.sqrt(torch.sum(torch.pow(ground_truth, 2), dim=1))
        den = torch.max(ground_truth_len) - torch.min(ground_truth_len)
        # 计算真实值方差
        den_nmse = ground_truth_len.var()
        std = torch.sqrt(den_nmse)
    den = torch.where(den == 0, torch.tensor(1.0), den)  # 防止分母为0
    den_nmse = torch.where(den_nmse == 0, torch.tensor(1.0), den_nmse)  # 防止分母为0
    std = torch.where(std == 0, torch.tensor(1.0), std)  # 防止分母为0

    # 计算NRMSE
    nrmse_value = rmse_value / den
    # nrmse_value = rmse_value / std
    # print(f'den:{nrmse_value.item()}\tstd={nrmse_}')
    # 计算NMSE
    nmse_value = mse_value / den_nmse

    return nrmse_value.item(), rmse_value.item(), nmse_value.item(), mse_value.item(), mae_value.item()
    # 转换为Python标量


def nmse(prediction, ground_truth):
    # 确保输入是Tensor
    prediction = torch.as_tensor(prediction, dtype=torch.float32)
    ground_truth = torch.as_tensor(ground_truth, dtype=torch.float32)
    dim = len(prediction.shape)

    if dim == 1:  # 计算误差的平方
        squared_error = torch.pow(prediction - ground_truth, 2)
    else:
        squared_error = torch.pow(prediction - ground_truth, 2)
        squared_error = torch.sum(squared_error, dim=1)

    # 计算NRMSE
    mse_value = torch.sum(squared_error) / prediction.numel()
    # 计算真实值方差
    den = ground_truth.var()
    den = torch.where(den == 0, torch.tensor(1.0), den)  # 防止分母为0
    nmse_value = mse_value / den

    return nmse_value.item()  # 转换为Python标量


if __name__ == '__main__':
    truth = torch.rand(size=(10, 1))
    predict = torch.rand(size=(10, 1))
    for i in range(truth.shape[0]):
        print(f'{(truth[i] - predict[i]) ** 2}')
    nrmse(prediction=predict, ground_truth=truth)

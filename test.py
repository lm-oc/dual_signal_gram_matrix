import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F

if __name__ == '__main__':
    # 从文件加载数据
    average_losses_pandora = np.load('./log/average_Pandora18k_losses.npy').tolist()
    average_losses_wikiart = np.load('./log/cross1_losses.npy').tolist()
    average_losses_flickr = np.load('./log/Flickr6_losses.npy').tolist()

    plt.figure()

    # 绘制 WikiArt 损失曲线（虚线）
    plt.plot(range(1, len(average_losses_wikiart) + 1), average_losses_wikiart, linestyle='-', label='WikiArt', color='blue')
    # 绘制 Flickr 损失曲线（波浪线）
    plt.plot(range(1, len(average_losses_flickr) + 1), average_losses_flickr, linestyle='-.', label='Flickr', color='green')
    # 绘制 Pandora 损失曲线（实线）
    plt.plot(range(1, len(average_losses_pandora) + 1), average_losses_pandora, linestyle=':', label='Pandora18k',
             color='red')

    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    # 保存图像到文件
    plt.savefig('./log/loss.png', format='png', dpi=300)
    plt.show()  # 显示图像
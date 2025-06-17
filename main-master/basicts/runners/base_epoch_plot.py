import matplotlib.pyplot as plt
import re
import numpy as np
import os


# 读取日志文件内容
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        log_content = file.read()
    return log_content

# 提取训练、验证和测试的指标
def extract_metrics(log_content):
    # 初始化双重字典
    metrics = {
        key: {metric: [] for metric in ['loss', 'mae', 'rmse']}
        for key in ['train', 'val', 'test']
    }

    def regular_expression(flag, metric):
        return float(re.search(r'{}/{}: (\d+\.\d+)'.format(flag, metric), line).group(1))

    for line in log_content.splitlines():
        if "bts" in line or "zh" in line:
            continue
        if "Result <train>: [train/time" in line:
            for metric in ['loss', 'MAE', 'RMSE']:
                metrics['train'][str.lower(metric)].append(regular_expression('train', metric))
        elif "Result <val>: [val/time" in line:
            for metric in ['loss', 'MAE', 'RMSE']:
                metrics['val'][str.lower(metric)].append(regular_expression('val', metric))
        elif "Result <test>: [test/time" in line:
            for metric in ['loss', 'MAE', 'RMSE']:
                metrics['test'][str.lower(metric)].append(regular_expression('test', metric))

    # 确保所有列表长度相同
    min_length = min([len(metrics[i]['loss']) for i in ['train', 'val', 'test'] if len(metrics[i]['loss'])>0])

    for key in ['train', 'val', 'test']:
        for metric in ['loss', 'mae', 'rmse']:
            metrics[key][metric] = metrics[key][metric][:min_length]

    return metrics

# 绘制单个指标曲线
def plot_one(ax, tra, val, tes, flag='loss'):
    epochs = range(1, len(tra) + 1)

    # 绘制曲线
    ax.plot(epochs, tra, label=f'Training {flag}', color='b', linestyle='-', linewidth=3.0)
    ax.plot(epochs, val, label=f'Validation {flag}', color='g', linestyle='-', linewidth=3.0)
    if len(tes)>0:
        ax.plot(epochs, tes, label=f'Test {flag}', color='r', linestyle='-', linewidth=3.0)

    # 找到每条曲线的最小值点
    for i, c, v, h in zip([tra, val, tes], ['b', 'g', 'r'], ['top', 'bottom', 'bottom'], ['lef', 'right', 'center']):
        if len(i)>0:
            min_value = np.min(i)
            min_epoch = np.argmin(i) + 1
            ax.scatter([min_epoch], [min_value], color=c, s=50, zorder=5)  # 标点
            ax.text(min_epoch, min_value, f'{min_value:.2f}', color=c, ha='center', va=v, fontsize=20, fontweight='bold')  # 加文本

    ax.tick_params(axis='x', labelsize=20)  # 仅x轴刻度字体大小
    ax.tick_params(axis='y', labelsize=20)  # 仅y轴刻度字体大小
    ax.set_xlabel('Epochs')
    ax.set_ylabel(flag, fontsize=20)
    ax.legend(fontsize=20)
    ax.grid(True)

# 绘制指标曲线
def plot_metrics(metrics,title,fig_file):
    plt.figure(figsize=(12, 18))
    plt.suptitle(title, fontsize=20, fontweight='bold')

    # 绘制Loss曲线
    plt.subplot(3, 1, 1)
    plot_one(plt.gca(), metrics['train']['loss'], metrics['val']['loss'], metrics['test']['loss'], flag='Loss')

    # 绘制MAE曲线
    plt.subplot(3, 1, 2)
    plot_one(plt.gca(), metrics['train']['mae'], metrics['val']['mae'], metrics['test']['mae'], flag='MAE')

    # 绘制RMSE曲线
    plt.subplot(3, 1, 3)
    plot_one(plt.gca(), metrics['train']['rmse'], metrics['val']['rmse'], metrics['test']['rmse'], flag='RMSE')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(fig_file)
    # plt.show()
    plt.close()


def epoch_plot(log_file,method_idx=1,len_idx=3):
    dir = os.path.dirname(log_file)
    fname = log_file.split("/")
    title = "{}_{}".format(fname[method_idx], fname[len_idx])
    fig_file = os.path.join(dir, "{}.png".format(title))

    log_content = read_log_file(log_file)
    metrics = extract_metrics(log_content)
    plot_metrics(metrics,title,fig_file)
    return fig_file


# 主函数
if __name__ == "__main__":
    flog = ""
    fig_file = epoch_plot(flog,method_idx=4,len_idx=6)
    print(fig_file)
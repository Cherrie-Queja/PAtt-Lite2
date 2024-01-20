import torch
from thop import profile
from models.model_patt_lite import Model
import seaborn as sns
import matplotlib.pyplot as plt


def cal_flops():
    model = Model()
    input = torch.Tensor(1, 3, 224, 224)
    macs, params = profile(model, inputs=(input,))
    from thop import clever_format

    macs, params = clever_format([macs, params], "%.3f")
    print(macs, params)


cal_flops()


def draw_confusion_matrix():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    classes = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']
    confusion_matrix = np.array([
        # [295, 9, 3, 4, 0, 3, 15],
        # [9, 52, 1, 2, 8, 0, 2],
        # [2, 1, 121, 9, 10, 7, 10],
        # [3, 1, 5, 1149, 3, 2, 22],
        # [0, 1, 6, 6, 435, 1, 29],
        # [2, 0, 8, 4, 1, 142, 5],
        # [8, 0, 2, 10, 21, 0, 639]#raf

        # [340, 4, 0, 7, 0, 3, 4],
        # [12, 34, 0, 0, 1, 3, 0],
        # [0, 0, 4, 1, 1, 1, 0],
        # [5, 0, 0, 865, 3, 3, 11],
        # [1, 1, 1, 8, 247, 3, 49],
        # [1, 2, 1, 3, 2, 213, 6],
        # [15, 1, 2, 18, 24, 4, 957]#ferplus
        #
        [312, 78, 18, 34, 16, 9, 33],
        [69, 330, 23, 9, 41, 18, 10],
        [10, 20, 304, 22, 37, 85, 22],
        [19, 2, 9, 443, 2, 3, 22, ],
        [15, 16, 23, 10, 342, 43, 51],
        [21, 22, 55, 6, 21, 328, 47],
        [56, 7, 17, 29, 51, 49, 291]  # affect7
    ], dtype=np.int)  # 输入特征矩阵

    proportion = []
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)
    # print(np.sum(confusion_matrix[0]))
    # print(proportion)
    pshow = []
    for i in proportion:
        pt = "%.2f" % i
        pshow.append(pt)
    proportion = np.array(proportion).reshape(7, 7)  # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(7, 7)
    # print(pshow)
    config = {
        "font.family": 'Times New Roma',  # 设置字体类型
    }
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
    # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
    # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
    # plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=14, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=14)

    thresh = confusion_matrix.max() / 2.
    # iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
    # ij配对，遍历矩阵迭代器
    iters = np.reshape([[[i, j] for j in range(7)] for i in range(7)], (confusion_matrix.size, 2))
    for i, j in iters:
        if (i == j):
            # plt.text(j, i - 0.06, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=12, color='white',
            #          weight=5)  # 显示对应的数字
            plt.text(j, i + 0.06, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
        else:
            # plt.text(j, i - 0.06, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=6)  # 显示对应的数字
            plt.text(j, i + 0.06, pshow[i, j], va='center', ha='center', fontsize=12)

    plt.ylabel('True label', fontsize=14)
    plt.xlabel('Predict label', fontsize=14)
    plt.tight_layout()
    # plt.margins(0., 0.)
    plt.savefig('./c_aff.pdf', bbox_inches='tight', pad_inches=0.0)
    plt.show()

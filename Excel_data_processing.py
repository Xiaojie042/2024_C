import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def read_and_plot_excel(file_path):
    # 读取Excel文件，第一行为标题
    df = pd.read_excel(file_path, header=0)

    # 选择第二行的第五列至第1029列的数据（index=1，列index从4开始）
    data = df.iloc[1, 4:1029]  # 选择第二行，第五列至第1029列

    # 绘制数据
    plt.figure(figsize=(12, 8))
    plt.plot(range(1024), data.values, marker='.', linestyle='-', color='r',linewidth=0.1, markersize=1)  # 0-1024为横轴
    plt.title('waveform')
    plt.xlabel('Sample Points (0-1024)')
    plt.ylabel('Variable Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def read_and_plot_excels(file_path):
    # 读取Excel文件，第一行为标题
    df = pd.read_excel(file_path, header=0,sheet_name='材料1正弦')

    # 获取从第二行到最后一行的数据（index从1开始）
    data = df.iloc[1:, 4:1029]  # 从第二行开始，第五列至第1029列

    plt.figure(figsize=(12, 8))
    num_lines = data.shape[0]  # 行数
    colors = plt.cm.viridis(np.linspace(0, 1, num_lines))  # 生成不同的颜色

    for i in range(num_lines):
        #if i>=1 and i<=10:
            plt.plot(range(1024), data.iloc[i], marker='.', linestyle='-',
                 color=colors[i], linewidth= 0.5,markersize=0.5)  # 线宽依行数增加

    plt.title('Plot of Data from Rows 2 to Last, 5th to 1029th Columns')
    plt.xlabel('Sample Points (0-1024)')
    plt.ylabel('Variable Values')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    # 使用该函数
    file_path = './data_ex.xlsx'  # 替换为你的Excel文件路径
    #read_and_plot_excel(file_path)
    read_and_plot_excels(file_path)


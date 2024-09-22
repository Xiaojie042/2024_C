import Excel_data_processing as dp

import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from scipy.stats import spearmanr,pearsonr
import matplotlib.pyplot as plt
from scipy.linalg import lstsq




def store_data_by_attributes(analyzer):
    """根据属性组合存储数据"""
    stored_data = {}

    # 遍历每个数据项
    for i in range(len(analyzer.temperature)):
        waveform = analyzer.waveform_type[i]
        temp = analyzer.temperature[i]
        #freq = analyzer.frequency[i]
        mat = analyzer.material[i]

        # 创建命名规则
        #key = f"{waveform}_tem{temp}"
        key = f"{waveform}_tem{temp}_mat{mat}"

        # 将数据存储到相应的数组
        if key not in stored_data:
            stored_data[key] = []
        stored_data[key].append([analyzer.frequency[i],
                                 analyzer.flux_density_peak[i],
                                 analyzer.core_loss[i]])

    return stored_data


def Extracting_2D_array(stored_data, key):
    # 提取特定键对应的二维数组
    if key in stored_data:
        data = np.array(stored_data[key])

        # 提取 x, y, z 数据
        x_data = data[:, 0]
        y_data = data[:, 1]
        z_data = data[:, 2]
    return x_data,y_data,z_data




def filter_and_store(analyzer):
    stored_data = {}

    for wave in analyzer.waveform_type:
        for temp in analyzer.temperature:
            #for mat in analyzer.material:
                # 筛选数据
                indices = [i for i in range(len(analyzer.frequency))
                           if analyzer.temperature[i] == temp
                           and analyzer.waveform_type[i] == wave]

                           # and analyzer.material[i] == mat]

                # 存储数据
                if indices:  # 如果有符合条件的数据
                    key = f"{wave}_tem{temp}"
                    #key = f"{wave}_tem{temp}_mat{mat}"
                    stored_data[key] = {
                        'frequency': [analyzer.frequency[i] for i in indices],
                        'flux_density_peak': [analyzer.flux_density_peak[i] for i in indices],
                        'core_loss': [analyzer.core_loss[i] for i in indices]
                    }

    return stored_data

def Fitting_Gaussian_model(core_loss,encoded_temperatures,encoded_wave,encoded_material):
    # 确保输入为 numpy 数组
    T = np.array(encoded_temperatures)
    W = np.array(encoded_wave)
    M = np.array(encoded_material)
    P = np.array(core_loss)

    # 构建设计矩阵 X
    X = np.column_stack((
        np.ones(len(T)),  # a0
        T,                # a1 * T
        W,                # a2 * W
        M,                # a3 * M
        T * W,           # a12 * T * W
        T * M,           # a13 * T * M
        W * M,           # a23 * W * M
        T * W * M        # a123 * T * W * M
    ))

    coefficients, residuals, rank, s = lstsq(X, P)

    return coefficients



if __name__ == '__main__':
    # 使用该函数
    file_path = './train_data.xlsx'

    analyzer = dp.MagneticCoreAnalyzer(file_path,mult_material=True)
    analyzer.read_train_data()

    #valid_temperatures = [25,50]
    #valid_frequency = [50020]


    #stored_data = filter_and_store(analyzer)
    # 进行波形类型和温度过滤
    #analyzer.filter_waveform(valid_waveforms)
    #analyzer.filter_temperature(valid_temperatures)
    #analyzer.filter_frequency(valid_frequency)
    #analyzer.filter_flux_density_peak(threshold_low=0.1)
    # valid_temperatures = [25,50,70,90]
    # analyzer.filter_temperature(valid_temperatures)

    encoded_temperatures = analyzer.target_encoding_tem()
    encoded_wave = analyzer.target_encoding_wave()
    encoded_material = analyzer.target_encoding_material()
    # print(set(encoded_material))
    #
    # correlation, p_value = spearmanr(encoded_material, analyzer.core_loss)
    #
    # print("Spearman Correlation Coefficient:", correlation)
    # print("P-value:", p_value)
    #
    # correlation_p, p_value_p = pearsonr(encoded_temperatures, analyzer.core_loss)
    #
    # print("pearman Correlation Coefficient:", correlation_p)
    # print("P-value:", correlation_p)

    coeffs = Fitting_Gaussian_model(encoded_temperatures=encoded_temperatures,
                                    encoded_wave=encoded_wave,
                                    encoded_material=encoded_material,
                                    core_loss=analyzer.core_loss)

    # 输出结果
    a0, a1, a2, a3, a12, a13, a23, a123 = coeffs
    print(f"Coefficients:\na0: {a0}\na1: {a1}\na2: {a2}\na3: {a3}\na12: {a12}\na13: {a13}\na23: {a23}\na123: {a123}")

    # fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    #
    # # 绘制第一个图像
    # axs[0,0].plot(stored_data['正弦波_tem25'], color='blue', label='sin(x)')
    # axs[0,0].set_title('sin')
    # axs[0,0].set_xlabel('x')
    # axs[0,0].set_ylabel('sin')
    # axs[0,0].legend()
    #
    # # 绘制第二个图像
    # axs[0,1].plot(stored_data['三角波_tem25'], color='orange', label='cos(x)')
    # axs[0,1].set_title('tri')
    # axs[0,1].set_xlabel('x')
    # axs[0,1].set_ylabel('tri')
    # axs[0,1].legend()
    #
    #
    # axs[1,0].plot(stored_data['梯形波_tem25'], color='orange', label='cos(x)')
    # axs[1,0].set_title('tra')
    # axs[1,0].set_xlabel('x')
    # axs[1,0].set_ylabel('tra')
    # axs[1,0].legend()
    #
    # # axs[1,1].plot(stored_data['梯形波_tem90'], color='orange', label='cos(x)')
    # # axs[1,1].set_title('tem90')
    # # axs[1,1].set_xlabel('x')
    # # axs[1,1].set_ylabel('90')
    # # axs[1,1].legend()
    #
    #
    #
    # plt.tight_layout()
    # plt.show()

    # 创建三维折线图
    # stored_data = store_data_by_attributes(analyzer)
    # fig = plt.figure(figsize=(15, 10))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # fr_x,fl_y,co_z = Extracting_2D_array(stored_data=stored_data,key='正弦波_tem25')
    # fr_x_tri,fl_y_tri,co_z_tri = Extracting_2D_array(stored_data=stored_data,key='三角波_tem25')
    # fr_x_tra,fl_y_tra,co_z_tra = Extracting_2D_array(stored_data=stored_data,key='梯形波_tem25')
    #
    #
    # # 绘制折线图
    # ax.plot(fr_x, fl_y,co_z,
    #         color='black', marker='o',linewidth=1.5,markersize=3)
    #
    # ax.plot(fr_x_tri, fl_y_tri, co_z_tri,
    #         color='r', marker='o',linewidth=1.5,markersize=3)
    #
    # ax.plot( fr_x_tra,fl_y_tra,co_z_tra,
    #         color='b', marker='o',linewidth=1.5,markersize=3)
    #
    #
    # # 设置轴标签
    # ax.set_xlabel('freq(Hz)')
    # ax.set_ylabel('flux_peak(T)')
    # ax.set_zlabel('core_loss(W)')
    #
    # # 设置标题
    # ax.set_title('3d:freq vs flux_peak vs core_loss')
    #
    # # 显示图形
    # plt.show()









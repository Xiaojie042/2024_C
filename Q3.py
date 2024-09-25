import Excel_data_processing as dp

import numpy as np
import pandas as pd
from scipy.optimize import minimize
# from sklearn.linear_model import LinearRegression
from scipy.spatial import Delaunay
from scipy.stats import spearmanr,pearsonr
import matplotlib.pyplot as plt
from scipy.linalg import lstsq

from scipy.optimize import curve_fit

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
        key = f"{waveform}_tem{temp}_{mat}"

        # 将数据存储到相应的数组
        if key not in stored_data:
            stored_data[key] = []
        stored_data[key].append([analyzer.frequency[i],
                                 analyzer.flux_density_peak[i],
                                 analyzer.core_loss[i],
                                 ])
                                 #analyzer.material[i]])

    return stored_data


def Extracting_2D_array(stored_data, key):
    # 提取特定键对应的二维数组
    if key in stored_data:
        data = np.array(stored_data[key])

        # 提取 x, y, z 数据
        first_column = [row[0] for row in data]
        s_col = [row[1] for row in data]
        t_col = [row[2] for row in data]
    return first_column,s_col,t_col

def filter_and_store(analyzer):
    stored_data = {}

    for wave in analyzer.waveform_type:
        for temp in analyzer.temperature:
            for mat in analyzer.material:
                # 筛选数据
                indices = [i for i in range(len(analyzer.frequency))
                           if analyzer.temperature[i] == temp
                           and analyzer.waveform_type[i] == wave

                           and analyzer.material[i] == mat]

                # 存储数据
                if indices:  # 如果有符合条件的数据
                    #key = f"{wave}_tem{temp}"
                    key = f"{wave}_tem{temp}_mat{mat}"
                    stored_data[key] = {
                        'frequency': [analyzer.frequency[i] for i in indices],
                        'flux_density_peak': [analyzer.flux_density_peak[i] for i in indices],
                        'core_loss': [analyzer.core_loss[i] for i in indices]
                    }

    return stored_data
def gaussian_model(T, W, M, a0, a1, a2, a3, a12, a13, a23, a123):
    W = np.array(W)

    M = np.array(M)
    T = np.array(T)



    return (a0 +
            a1 * T +
            a2 * W +
            a3 * M +
            a12 * T * W +
            a13 * T * M +
            a23 * W * M +
            a123 * T * W * M)

def Fitting_Gaussian_model_curve(core_loss, encoded_temperatures, encoded_wave, encoded_material):
    # 确保输入为 numpy 数组
    T = np.array(encoded_temperatures)
    W = np.array(encoded_wave)
    M = np.array(encoded_material)
    P = np.array(core_loss)

    # 使用 curve_fit 进行拟合
    popt, pcov = curve_fit(
        lambda x, a0, a1, a2, a3, a12, a13, a23, a123: gaussian_model(x[0], x[1], x[2], a0, a1, a2, a3, a12, a13, a23, a123),
        (T, W, M),  # 输入自变量
        P,          # 目标变量
        bounds=([0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],  # 下界
                [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])   # 上界
    )

    return popt

def custom_model(W, F, B, T, M, a0, a1, a2, a3, a4, a5, a6, a7, a8):
    W = np.array(W)
    F = np.array(F)
    M = np.array(M)
    T = np.array(T)
    B = np.array(B)
    return a0 * ((W**a1) + a2) * (F**a3) * (B**a4) * ((a5 / T) + a6) * (M**a7 + a8)

def fitting_custom_model(core_loss, encoded_wave, encoded_frequency, encoded_material, encoded_temperature,flux_peak):
    # 确保输入为 numpy 数组
    W = np.array(encoded_wave)
    F = np.array(encoded_frequency)
    M = np.array(encoded_material)
    T = np.array(encoded_temperature)
    P = np.array(core_loss)
    B = np.array(flux_peak)

    initial_params = [1, 1, 1, 1, 1, 1, 1, 1,1]
    # 使用 curve_fit 进行拟合
    popt, pcov = curve_fit(
        lambda x, a0, a1, a2, a3, a4, a5, a6, a7, a8: custom_model(x[0], x[1], x[2], x[3], x[4], a0, a1, a2, a3, a4, a5, a6, a7, a8),
        (W, F, B, T, M),
        P,
        initial_params,
        maxfev=100000
    )

    return popt

# 示例调用
# core_loss, encoded_wave, encoded_frequency, encoded_material, encoded_temperature 需要定义
# coefficients = fitting_custom_model(core_loss, encoded_wave, encoded_frequency, encoded_material, encoded_temperature)

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



def calculate_differences_and_stats(array1, array2):
    """
    计算两个数组索引一一对应的差值，并计算两个数组的方差和平均值。

    参数:
    array1 -- 第一个数组
    array2 -- 第二个数组

    返回:
    differences -- 两个数组的差值
    stats -- 一个字典，包含两个数组的平均值和方差
    """
    # 确保输入是 numpy 数组
    percentage_differences_filter = []
    arr1 = np.asarray(array1)
    arr2 = np.asarray(array2)

    # 计算差值，以数据值为分母
    differences = abs(arr1 - arr2)
    percentage_differences = (differences / arr1) * 100
    # 计算差值，以计算值为分母
    # percentage_differences = (differences / arr2) * 100
    for i in range(len(percentage_differences)):
        if percentage_differences[i] <100:
            percentage_differences_filter.append(percentage_differences[i])


    return differences,percentage_differences_filter



if __name__ == '__main__':
    # 使用该函数
    file_path = './train_data.xlsx'
    #file_path = './data_test.xlsx'

    analyzer = dp.MagneticCoreAnalyzer(file_path,mult_material=True)
    analyzer.read_train_data()

    valid_temperatures = [90]
    #valid_frequency = [50020]
    valid_waveforms = ['正弦波']
    valid_material = ['材料4']


    #analyzer.filter_material(valid_material)

    stored_data = store_data_by_attributes(analyzer)
    # # 进行波形类型和温度过滤
    # analyzer.filter_waveform(valid_waveforms)
    # analyzer.filter_temperature(valid_temperatures)
    #analyzer.filter_frequency(valid_frequency)
    #analyzer.filter_flux_density_peak(threshold_low=0.1)
    # valid_temperatures = [25,50,70,90]
    # analyzer.filter_temperature(valid_temperatures)
    #print(np.max(analyzer.frequency))

    encoded_temperatures = analyzer.target_encoding_tem()
    encoded_wave = analyzer.target_encoding_wave()
    encoded_material = analyzer.target_encoding_material()
    # print(set(encoded_material))
    # print(set(encoded_wave))
    # print(set(encoded_temperatures))
    #
    # correlation, p_value = spearmanr(encoded_material, analyzer.core_loss)

    # print("M-Spearman Correlation Coefficient:", correlation)
    # print("P-value:", p_value)
    #
    # correlation_p, p_value_p = spearmanr(encoded_wave, analyzer.core_loss)
    #
    # print("T-encode-Spearman Correlation Coefficient:", correlation_p)
    # print("P-value:", correlation_p)
    #
    # correlation_p, p_value_p = spearmanr(analyzer.temperature, analyzer.core_loss)
    #
    # print("T-Spearman Correlation Coefficient:", correlation_p)
    # print("P-value:", correlation_p)
    #
    # correlation_p, p_value_p = spearmanr(encoded_wave, analyzer.core_loss)
    #
    # print("W-spearman Correlation Coefficient:", correlation_p)
    # print("P-value:", correlation_p)

    # coeffs = Fitting_Gaussian_model_curve(encoded_temperatures=analyzer.temperature,
    #                                 encoded_wave=encoded_wave,
    #                                 encoded_material=encoded_material,
    #                                 core_loss=analyzer.core_loss)
    # coeffs = fitting_custom_model(encoded_temperature=encoded_temperatures,
    #                                 encoded_wave=encoded_wave,
    #                                 encoded_material=encoded_material,
    #                                 core_loss=analyzer.core_loss,
    #                               encoded_frequency=analyzer.frequency,
    #                               flux_peak=analyzer.flux_density_peak)
    # #
    # # 输出结果
    # a0, a1, a2, a3, a12, a13, a23 ,a123,a01234= coeffs
    # print(f"Coefficients:\na0: {a0}\na1: {a1}\na2: {a2}\na3: {a3}\na12: {a12}\na13: {a13}\na23: {a23}\na123: {a123}\na123: {a01234}")

    # P_gs_cal = gaussian_model(W=encoded_wave,
    #                           T=encoded_temperatures,
    #                           M=encoded_material,
    #                             a0=1.3528560979328668,
    #                             a1=0.19660513484446449,
    #                             a2= 0.021876019740737492,
    #                             a3= -0.09846427944118223,
    #                             a12= -1.0986410451368416e-06,
    #                             a13= -4.946936411433923e-07,
    #                             a23= -1.843527485508973e-07,
    #                             a123= 3.121294904352256e-11
    # )
    #
    # per_gs, per_dif_tem_gs = calculate_differences_and_stats(array2=P_gs_cal,array1=analyzer.core_loss)
    # print(len(per_gs), len(per_dif_tem_gs))
    # print(np.max(per_dif_tem_gs))
    # print(np.mean(per_dif_tem_gs))
    # print(np.var(per_dif_tem_gs))
    #
    #
    # P_gs_no_enc_cal = gaussian_model(W=encoded_wave,
    #                           T=analyzer.temperature,
    #                           M=encoded_material,
    #                                 a0=48254.573385202544,
    #                                 a1=-158.1033157791698,
    #                                 a2= -0.2519928611756105,
    #                                 a3= -0.21735803059244949,
    #                                 a12= 0.0009538193597101108,
    #                                 a13= 0.00037482194799932266,
    #                                 a23= 7.76314001039132e-06,
    #                                 a123= -2.9958200442442035e-08
    # )
    #
    # per_gs_no, per_dif_tem_gs_no = calculate_differences_and_stats(array2=P_gs_no_enc_cal,array1=analyzer.core_loss)
    # print(len(per_gs_no), len(per_dif_tem_gs_no))
    # print(np.max(per_dif_tem_gs_no))
    # print(np.mean(per_dif_tem_gs_no))
    # print(np.var(per_dif_tem_gs_no))

    P_cal = custom_model(W=encoded_wave,
                         F=analyzer.frequency,
                         B=analyzer.flux_density_peak,
                         T=analyzer.temperature,
                         M=encoded_material,
                        a0=0.002539227305716865,
                        a1=0.0014445108854908575,
                        a2=-1.007350084588356,
                        a3=1.3314822318918627,
                        a4=2.0707236371873523,
                        a5=-1925.1004499526096,
                        a6=0.018620155914894392,
                        a7=1.2344098617520405,
                        a8=10065457.42370792
                         )
    dif_tem, per_dif_tem = calculate_differences_and_stats(array2=P_cal,array1=analyzer.core_loss)
    print(len(dif_tem), len(per_dif_tem))
    print(np.max(per_dif_tem))
    print(np.mean(per_dif_tem))
    print(np.var(per_dif_tem))
    #
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # axs[0].plot(per_dif_tem_gs_no, color='blue', label='sin(x)')
    # axs[0].set_title('origin_eq')
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('sin(x)')
    # axs[0].legend()

    # 绘制第二个图像
    axs[1].plot(per_dif_tem, color='orange', label='cos(x)')
    axs[1].set_title('t1_tem_eq')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('cos(x)')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

    # fig, axs = plt.subplots(2, 2, figsize=(15, 8))
    # # print(stored_data['正弦波_tem25_材料1'])
    # # third_elements = [row[2] for row in stored_data['正弦波_tem25_材料1']]
    # # 绘制第一个图像
    # axs[0,0].plot([row1[2] for row1 in stored_data['正弦波_tem90_材料1']], color='blue', label='sin(x)')
    # axs[0,0].set_title('sin')
    # axs[0,0].set_xlabel('x')
    # axs[0,0].set_ylabel('sin')
    # axs[0,0].legend()
    #
    # # 绘制第二个图像
    # axs[0,1].plot([row2[2] for row2 in stored_data['正弦波_tem90_材料2']], color='orange', label='cos(x)')
    # axs[0,1].set_title('tri')
    # axs[0,1].set_xlabel('x')
    # axs[0,1].set_ylabel('tri')
    # axs[0,1].legend()
    #
    #
    # axs[1,0].plot([row3[2] for row3 in stored_data['正弦波_tem90_材料3']], color='orange', label='cos(x)')
    # axs[1,0].set_title('tra')
    # axs[1,0].set_xlabel('x')
    # axs[1,0].set_ylabel('tra')
    # axs[1,0].legend()
    #
    # axs[1,1].plot([row4[2] for row4 in stored_data['正弦波_tem90_材料4']], color='orange', label='cos(x)')
    # axs[1,1].set_title('tem90')
    # axs[1,1].set_xlabel('x')
    # axs[1,1].set_ylabel('90')
    # axs[1,1].legend()
    #
    #
    #
    # plt.tight_layout()
    # plt.show()

    #创建三维折线图
    #stored_data = store_data_by_attributes(analyzer)
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    #fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    fr_x,fl_y,co_z = Extracting_2D_array(stored_data=stored_data,key='正弦波_tem90_材料1')
    # fr_x_tri,fl_y_tri,co_z_tri = Extracting_2D_array(stored_data=stored_data,key='三角波_tem25')
    # fr_x_tra,fl_y_tra,co_z_tra = Extracting_2D_array(stored_data=stored_data,key='梯形波_tem25')
    #
    fr_x = np.array(fr_x)
    fl_y = np.array(fl_y)
    x = fr_x*fl_y
    # 绘制折线图
    ax.plot(x,co_z,
            color='black', marker='o',linewidth=1.5,markersize=3)

    # # ax.plot(fr_x_tri, fl_y_tri, co_z_tri,
    # #         color='r', marker='o',linewidth=1.5,markersize=3)
    # #
    # # ax.plot( fr_x_tra,fl_y_tra,co_z_tra,
    # #         color='b', marker='o',linewidth=1.5,markersize=3)
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

    # fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    #
    # axs[0].plot(fr_x*fl_y,co_z, color='blue', label='sin(x)')
    # axs[0].set_title('origin_eq')
    # axs[0].set_xlabel('x')
    # axs[0].set_ylabel('sin(x)')
    # axs[0].legend()
    #
    # # 绘制第二个图像
    # # axs[1].plot(per_dif_tem,co_z color='orange', label='cos(x)')
    # # axs[1].set_title('t1_tem_eq')
    # # axs[1].set_xlabel('x')
    # # axs[1].set_ylabel('cos(x)')
    # # axs[1].legend()
    #
    # plt.tight_layout()
    # plt.show()



    for i in range(len(fr_x)):
        if fr_x[i] * fl_y[i] > 43740 and co_z[i] < 139250:
            min_f = fr_x[i]
            min_B = fl_y[i]
            print(i)
            print(min_f*min_B)
            print(min_f)
            print(min_B)
            print(co_z[i])







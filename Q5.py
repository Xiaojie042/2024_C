import numpy as np

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


cal_coeff = False

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

    return popt  # 返回拟合参数



def custom_model_line(W, F, B, T, M, a0, a1, a2, a3, a4, a5, a6, a7):
    W = np.array(W)
    F = np.array(F)
    M = np.array(M)
    T = np.array(T)
    B = np.array(B)
    return a0 * ((W*a1) + a2) * (F**a3) * (B**a4) * (T**a5) * (M*a6 + a7)

def fitting_custom_model_line(core_loss, encoded_wave, encoded_frequency, encoded_material, encoded_temperature,flux_peak):
    # 确保输入为 numpy 数组
    W = np.array(encoded_wave)
    F = np.array(encoded_frequency)
    M = np.array(encoded_material)
    T = np.array(encoded_temperature)
    P = np.array(core_loss)
    B = np.array(flux_peak)

    initial_params = [1, 1, 1, 1, 1, 1, 1, 1]
    # 使用 curve_fit 进行拟合
    popt, pcov = curve_fit(
        lambda x, a0, a1, a2, a3, a4, a5, a6, a7: custom_model_line(x[0], x[1], x[2], x[3], x[4], a0, a1, a2, a3, a4, a5, a6, a7),
        (W, F, B, T, M),
        P,
        initial_params,
        maxfev=100000
    )

    return popt  # 返回拟合参数






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


# 示例离散数据
# data = np.array([0, 1, 4, 9, 16, 25])  # y 值
# n = len(data)
#
# # 计算导数（差分）
# derivatives = np.diff(data)
#
# # 计算导数的平方
# derivatives_squared = derivatives ** 2
#
# # 计算定积分（使用梯形法则）
# # 由于我们只知道数据点，不知道时间间隔，可以假设时间间隔为1
# integral_value = np.trapz(derivatives_squared)
#
# print("导数的平方在一个周期内的定积分值:", integral_value)




if __name__ == '__main__':
    # 使用该函数
    file_path = './test3.xlsx'
    #file_path = './data_test.xlsx'

    analyzer = dp.MagneticCoreAnalyzer(file_path,mult_material=True)
    #analyzer.read_train_data()
    analyzer.read_test3_data()
    #valid_temperatures = [25,50]
    #valid_frequency = [50020]
    #valid_waveforms = ['正弦波']
    #valid_material = ['材料1']


    #analyzer.filter_material(valid_material)

    #stored_data = store_data_by_attributes(analyzer)


    # analyzer.filter_waveform(valid_waveforms)
    #analyzer.filter_temperature(valid_temperatures)
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


    # # coeffs = Fitting_Gaussian_model_curve(encoded_temperatures=analyzer.temperature,
    #                                 encoded_wave=encoded_wave,
    #                                 encoded_material=encoded_material,
    #                                 core_loss=analyzer.core_loss)
    if cal_coeff:
        coeffs = fitting_custom_model_line(encoded_temperature=analyzer.temperature,
                                        encoded_wave=encoded_wave,
                                        encoded_material=encoded_material,
                                        core_loss=analyzer.core_loss,
                                      encoded_frequency=analyzer.frequency,
                                      flux_peak=analyzer.flux_density_peak)
        #
        # 输出结果
        a0, a1, a2, a3, a12, a13, a23 ,a123= coeffs
        print(f"Coefficients:\na0={a0}\na1={a1}\na2={a2}\na3={a3}\na12={a12}\na13={a13}\na23={a23}\na123={a123}\n")



    P_cal = custom_model(W=encoded_wave,
                         F=analyzer.frequency,
                         B=analyzer.flux_density_peak,
                         T=encoded_temperatures,
                         M=encoded_material,
                        a0=0.002539227305716865,
                        a1=0.0014445108854908575,
                        a2=-1.007350084588356,
                        a3=1.3314822318918627,
                        a4=2.0707236371873523,
                        a5=-1925.1004499526096,
                        a6=0.018620155914894392,
                        a7=1.2344098617520405,
                        a8=10065457.42370792)


    dif_tem, per_dif_tem = calculate_differences_and_stats(array2=P_cal,array1=analyzer.core_loss)
    print(len(dif_tem), len(per_dif_tem))
    print(np.max(per_dif_tem))
    print(np.mean(per_dif_tem))
    print(np.var(per_dif_tem))




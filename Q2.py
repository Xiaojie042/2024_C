import Excel_data_processing as dp

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


compare = True
def stanley_metz_objective_residual(variables, f_data, B_data, p_data):
    x, y, z = variables
    # 计算残差,通过残差的方法拟合出来一个近似值
    residuals = p_data - (x * (f_data ** y) * (B_data ** z))
    return np.sum(residuals ** 2)  # 返回残差平方和

def solve_stanley_metz_minimize(f_data, B_data, p_data, initial_guess=(1, 1, 1)):
    # 使用 minimize 求解方程
    # bounds = [(0, None), (1, 3), (2, 3)]
    result = minimize(solve_stanley_metz, initial_guess, args=(f_data, B_data, p_data))
    return result.x  # 返回优化后的变量值


def prepare_data(f_data, B_data):
    # 对输入数据进行对数变换
    log_f = np.log(f_data)
    log_B = np.log(B_data)
    return log_f, log_B


def solve_stanley_metz(f_data, B_data, p_data):
    # 准备数据
    log_f, log_B = prepare_data(f_data, B_data)

    # 构建特征矩阵，包含对数变换后的频率和磁通密度
    X = np.column_stack((log_f, log_B))

    # 目标数据
    y = np.log(p_data)

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef = model.coef_  # 拟合的y和z系数

    return x, y_coef, z_coef

def prepare_data_tem(f_data, B_data, T_data):
    """
    对输入的频率、磁通密度和温度进行对数变换。

    参数:
    f_data -- 频率数据
    B_data -- 磁通密度数据
    T_data -- 温度数据

    返回:
    log_f -- 频率的对数值
    log_B -- 磁通密度的对数值
    T_data -- 温度数据
    """
    log_f = np.log(f_data)
    log_B = np.log(B_data)
    return log_f, log_B, T_data


def solve_stanley_metz_tem_t0_sum_t1T(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, T_data = prepare_data_tem(f_data, B_data, T_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度和温度
    X = np.column_stack((log_f, log_B, T_data, np.ones(len(T_data))))  # 添加常数项

    # 目标数据
    y = np.log(p_data)

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t_coef = model.coef_[:-1]  # 拟合的y、z系数
    t0_coef = model.intercept_  # 拟合的t0系数
    t1_coef = model.coef_[-1]  # 拟合的t1系数

    return x, y_coef, z_coef, t0_coef, t1_coef

def solve_stanley_metz_t0_dif_t1T(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, T_data = prepare_data_tem(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度和温度
    X = np.column_stack((log_f, log_B, T_data))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef = model.coef_  # 拟合的y、z系数

    # 计算 t0 系数
    t0_coef = np.exp(y - (x * (f_data ** y_coef) * (B_data ** z_coef) * (1 - t1_coef * T_data)))

    return x, y_coef, z_coef, t0_coef.mean(), t1_coef



def solve_stanley_metz_t0_dif_t1T_sum_t2T2(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, T_data = prepare_data_tem(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度、温度和温度平方
    T_squared = T_data ** 2
    X = np.column_stack((log_f, log_B, T_data, T_squared))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef, t2_coef = model.coef_  # 拟合的y、z、t1、t2系数

    # t0 为模型的常数项
    t0 = np.exp(model.intercept_)  # 拟合的t0系数

    return x, y_coef, z_coef, t0, t1_coef, t2_coef

def solve_stanley_metz_t0_sum_t1T_sum_t2T2(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, T_data = prepare_data_tem(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度、温度和温度平方
    T_squared = T_data ** 2
    X = np.column_stack((log_f, log_B, T_data, T_squared))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef, t2_coef = model.coef_  # 拟合的y、z、t1、t2系数

    # t0 为模型的常数项
    t0 = np.exp(model.intercept_)  # 拟合的t0系数

    return x, y_coef, z_coef, t0, t1_coef, t2_coef

def solve_stanley_metz_t0_sum_T_div_t1(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, T_data = prepare_data_tem(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度和温度的倒数
    T_inv = 1 / T_data
    X = np.column_stack((log_f, log_B, T_inv))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef = model.coef_  # 拟合的y、z、t1系数

    # 计算t0
    t0 = np.exp(model.intercept_) - x * np.mean((f_data ** y_coef) * (B_data ** z_coef) * (T_data / t1_coef))

    return x, y_coef, z_coef, t0, t1_coef


def prepare_data_all_index(f_data, B_data, T_data):
    """
    对输入的频率、磁通密度和温度进行处理。

    参数:
    f_data -- 频率数据
    B_data -- 磁通密度数据
    T_data -- 温度数据

    返回:
    log_f -- 频率的对数值
    log_B -- 磁通密度的对数值
    log_T -- 温度的对数值
    """
    log_f = np.log(f_data)
    log_B = np.log(B_data)
    log_T = np.log(T_data)
    return log_f, log_B, log_T


def solve_stanley_metz_mult_T_index(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, log_T = prepare_data_all_index(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度和温度
    X = np.column_stack((log_f, log_B, log_T))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef = model.coef_  # 拟合的y、z、t1系数

    # 计算t0
    t0 = np.exp(model.intercept_) / (np.mean(f_data ** y_coef) * np.mean(B_data ** z_coef) * np.mean(T_data ** t1_coef))

    return x, y_coef, z_coef, t0, t1_coef


def solve_stanley_metz_only_mult_Tindex(f_data, B_data, p_data, T_data):
    # 准备数据
    log_f, log_B, log_T = prepare_data_all_index(f_data, B_data, T_data)

    # 目标数据
    y = np.log(p_data)

    # 构建特征矩阵，包含对数变换后的频率、磁通密度和温度
    X = np.column_stack((log_f, log_B, log_T))

    # 使用线性回归进行拟合
    model = LinearRegression()
    model.fit(X, y)

    # 提取拟合参数
    x = np.exp(model.intercept_)  # 拟合的x系数
    y_coef, z_coef, t1_coef = model.coef_  # 拟合的y、z、t1系数

    return x, y_coef, z_coef, t1_coef





def cal_core_loss(x,y,z,f,B):
    """
    根据输入计算 P 值。
    如果 f 和 B 是数组，则逐个计算。

    参数:
    x -- 系数 x
    y -- 系数 y
    z -- 系数 z
    f -- 输入频率，可以是数字或数组
    B -- 输入磁通密度，可以是数字或数组

    返回:
    计算得到的 P 值，数字或数组形式
    """
    # 确保 f 和 B 是 numpy 数组
    f_array = np.asarray(f)
    B_array = np.asarray(B)

    # 计算 P 值
    P = x * (f_array ** y) * (B_array ** z)

    return P

def cal_core_loss_tem(x,y,z,f,B,T,t0 = 0.0,t1 = 0.0,t2 = 0.0,select = 0):
    f_array = np.asarray(f)
    B_array = np.asarray(B)
    T_array = np.asarray(T)

    if select == 0:
        return x * (f_array ** y) * (B_array ** z) * (t0 - t1 *T_array)
    elif select == 1:
        return x * (f_array ** y) * (B_array ** z) * (t0 + t1 * T_array + t2*(T_array**2))
    elif select ==2:
        return x * (f_array ** y) * (B_array ** z) * (t0 + T_array / t1)
    elif select ==3:
        return x * (f_array ** y) * (B_array ** z) * (T_array**t1)

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

    # 计算差值
    differences = abs(arr1 - arr2)
    percentage_differences = (differences / arr1) * 100
    for i in range(len(percentage_differences)):
        if percentage_differences[i] <100:
            percentage_differences_filter.append(percentage_differences[i])

    return differences,percentage_differences_filter


if __name__ == '__main__':
    # 使用该函数
    file_path = './data_ex.xlsx'

    analyzer = dp.MagneticCoreAnalyzer(file_path,material=1)
    analyzer.read_train_data()
    # valid_temperatures = [90]
    # valid_frequency = [112160]
    # valid_flux_density_peak = [0.2]
    peak_mean = np.mean(analyzer.flux_density_peak)

    # analyzer.filter_temperature(valid_temperatures)
    #analyzer.filter_frequency(valid_frequency)
    # analyzer.filter_flux_density_peak(threshold_low=0.2,threshold_high=0.5)
    if compare:
        cal_core_loss_array = cal_core_loss(x=2.574190557505583,
                                      y=1.2668979257012518,
                                      z=2.0366904854100527,
                                      f=analyzer.frequency,
                                      B=analyzer.flux_density_peak
                                      )
        # cal_core_loss_min_array = cal_core_loss(x=3.2251,
        #                               y=1.3606,
        #                               z=2.2869,
        #                               f=analyzer.frequency,
        #                               B=analyzer.flux_density_peak
        #                               )
        cal_core_loss_tem_array = cal_core_loss_tem(x=20.845014314426773,
                                      y=1.302887917402657,
                                      z=2.0548809285088545,
                                      t1=-0.6198901519350334,
                                      f=analyzer.frequency,
                                      B=analyzer.flux_density_peak,
                                      T=analyzer.temperature,
                                      select=3
                                      )


        dif, per_dif = calculate_differences_and_stats(array2=cal_core_loss_array,array1=analyzer.core_loss)
        # dif_min, per_dif_min = calculate_differences_and_stats(array2=cal_core_loss_min_array,array1=analyzer.core_loss)
        print(len(dif), len(per_dif))
        print(np.max(per_dif))
        print(np.mean(per_dif))
        print(np.var(per_dif))

        dif_tem, per_dif_tem = calculate_differences_and_stats(array2=cal_core_loss_tem_array,array1=analyzer.core_loss)
        # dif_min, per_dif_min = calculate_differences_and_stats(array2=cal_core_loss_min_array,array1=analyzer.core_loss)
        print(len(dif_tem), len(per_dif_tem))
        print(np.max(per_dif_tem))
        print(np.mean(per_dif_tem))
        print(np.var(per_dif_tem))

        # dif_tem2, per_dif_tem2 = calculate_differences_and_stats(array2=cal_core_loss_tem2_array,array1=analyzer.core_loss)
        # # dif_min, per_dif_min = calculate_differences_and_stats(array2=cal_core_loss_min_array,array1=analyzer.core_loss)
        # print(len(dif_tem2), len(per_dif_tem2))
        # print(np.max(per_dif_tem2))


        # print(len(dif), len(per_dif))
        # print(np.max(per_dif))
        # np.savetxt('./dif_per.txt', per_dif, delimiter=',', fmt='%s')

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))

        # 绘制第一个图像
        axs[0].plot(per_dif, color='blue', label='sin(x)')
        axs[0].set_title('origin_eq')
        axs[0].set_xlabel('x')
        axs[0].set_ylabel('sin(x)')
        axs[0].legend()

        #绘制第二个图像
        axs[1].plot(per_dif_tem, color='orange', label='cos(x)')
        axs[1].set_title('t1_tem_eq')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('cos(x)')
        axs[1].legend()

        # 绘制第三个图像
        # axs[2].plot(per_dif_tem2, color='green', label='sin(x) * cos(x)')
        # axs[2].set_title('Sine * Cosine Function')
        # axs[2].set_xlabel('x')
        # axs[2].set_ylabel('sin(x) * cos(x)')
        # axs[2].legend()

        # 调整布局
        plt.tight_layout()
        plt.show()


        # plt.figure(figsize=(10, 8))
        # plt.plot(per_dif_tem,
        #          marker='.', linestyle='-', color='r',linewidth=1, markersize=10)
        # plt.title('waveform')
        # plt.xlabel('Sample Points (0-1024)')
        # plt.ylabel('Variable Values')
        # plt.grid(True)
        # plt.tight_layout()
        # plt.show()



    f_data = np.array(analyzer.frequency)  # f 的实验数据
    B_data = np.array(analyzer.flux_density_peak)  # B 的实验数据
    p_data = np.array(analyzer.core_loss)  # P 的实验数据
    T_data = np.array(analyzer.temperature)  # P 的实验数据

    solution = solve_stanley_metz_tem_t0_sum_t1T(f_data, B_data, p_data,T_data)
    k, a, b, t0 ,t1 = solution

    # solution = solve_stanley_metz_t0_dif_t1T_sum_t2T2(f_data, B_data, p_data,T_data)
    # k_1, a_1, b_1, t0_1 ,t1_1 ,t2_1= solution

    solution = solve_stanley_metz_only_mult_Tindex(f_data, B_data, p_data,T_data)
    k_1, a_1, b_1,t1_1 = solution

    print(f"解得: k = {k}, a = {a}, b = {b}, t0 = {t0}, t1 = {t1}")
    #print(f"解得: k = {k_1}, a = {a_1}, b = {b_1}, t0 = {t0_1}, t1 = {t1_1}, t2 = {t2_1}")
   # print(f"解得: k = {k_1}, a = {a_1}, b = {b_1}, t0 = {t0_1}, t1 = {t1_1},")
    print(f"解得: k = {k_1}, a = {a_1}, b = {b_1}, t1 = {t1_1},")

    print("done")
import Excel_data_processing as dp

import numpy as np
from scipy.optimize import minimize

def stanley_metz_objective(variables, f_data, B_data, p_data):
    x, y, z = variables
    # 计算残差,通过残差的方法拟合出来一个近似值
    residuals = p_data - (x * (f_data ** y) * (B_data ** z))
    return np.sum(residuals ** 2)  # 返回残差平方和

def solve_stanley_metz(f_data, B_data, p_data, initial_guess=(1, 1, 1)):
    # 使用 minimize 求解方程
    result = minimize(stanley_metz_objective, initial_guess, args=(f_data, B_data, p_data))
    return result.x  # 返回优化后的变量值

if __name__ == '__main__':
    # 使用该函数
    file_path = './data_ex.xlsx'  # 替换为你的Excel文件路径

    analyzer = dp.MagneticCoreAnalyzer(file_path,material=1)
    analyzer.read_train_data()
    valid_temperatures = [90]
    analyzer.filter_temperature(valid_temperatures)



    f_data = np.array(analyzer.frequency)  # f 的实验数据
    B_data = np.array(analyzer.flux_density_peak)  # B 的实验数据
    p_data = np.array(analyzer.core_loss)  # P 的实验数据

    solution = solve_stanley_metz(f_data, B_data, p_data)
    k, a, b = solution

    print(f"解得: k = {k}, a = {a}, b = {b}")

    print("done")
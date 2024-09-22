import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MagneticCoreAnalyzer:
    def __init__(self, file_path,material = 1):
        self.file_path = file_path
        self.material = material
        self.test_material = []
        self.temperature = []
        self.frequency = []
        self.core_loss = []
        self.waveform_type = []
        self.serial_number = []
        self.flux_density = []
        self.flux_density_peak = []

    def read_train_data(self):
        """读取训练Excel表格中的数据，初始化类的各个属性"""
        # 读取Excel文件，第一行为表头
        df = pd.read_excel(self.file_path, header=0,
                           sheet_name=f'材料{self.material}')

        # 获取相应列的数据
        self.temperature = df.iloc[:, 0].values  # 第一列: 温度
        self.frequency = df.iloc[:, 1].values  # 第二列: 频率
        self.core_loss = df.iloc[:, 2].values  # 第三列: 磁芯损耗值
        self.waveform_type = df.iloc[:, 3].values  # 第四列: 励磁波形

        # 第五列到第1029列是磁通密度的序列
        self.flux_density = df.iloc[:, 4:1029].values
        # 计算磁通密度峰值（每一行中的最大绝对值）
        self.flux_density_peak = np.max(np.abs(self.flux_density), axis=1)
    def read_test2_data(self):
        """读取测试集1(附件2)Excel表格中的数据，初始化类的各个属性"""
        # 读取Excel文件，第一行为表头
        df = pd.read_excel(self.file_path, header=0)

        # 获取相应列的数据
        self.serial_number = df.iloc[:, 0].values  # 第一列: 样本序号
        self.temperature = df.iloc[:, 1].values  # 第二列: 温度
        self.frequency = df.iloc[:, 2].values  # 第三列: 频率
        self.test_material = df.iloc[:, 3].values  # 第四列: 磁芯材料

        # 第五列到第1029列是磁通密度的序列
        self.flux_density = df.iloc[:, 4:1029].values
        # 计算磁通密度峰值（每一行中的最大绝对值）
        self.flux_density_peak = np.max(np.abs(self.flux_density), axis=1)
    def read_test3_data(self):
        """读取测试集2(附件3)Excel表格中的数据，初始化类的各个属性"""
        # 读取Excel文件，第一行为表头
        df = pd.read_excel(self.file_path, header=0)

        # 获取相应列的数据
        self.serial_number = df.iloc[:, 0].values  # 第一列: 样本序号
        self.frequency = df.iloc[:, 1].values  # 第二列: 温度
        self.core_loss = df.iloc[:, 2].values  # 第三列: 频率
        self.test_material = df.iloc[:, 3].values  # 第四列: 磁芯材料
        self.waveform_type = df.iloc[:, 4].values  # 第四列: 波形种类

        # 第五列到第1029列是磁通密度的序列
        self.flux_density = df.iloc[:, 4:1030].values
        # 计算磁通密度峰值（每一行中的最大绝对值）
        self.flux_density_peak = np.max(np.abs(self.flux_density), axis=1)

    def display_results(self):
        """显示读取和处理后的数据摘要"""
        for i in range(len(self.temperature)):
            print(f"Temperature: {self.temperature[i]}°C, Frequency: {self.frequency[i]} Hz")
            print(f"Core Loss: {self.core_loss[i]}, Exciting waveform_type: {self.exciting_waveform_type[i]}")
            print(f"Peak Flux Density: {self.flux_density_peak[i]}")
            print("-" * 50)


    def filter_waveform(self, valid_waveforms,train_data = True):
        """根据给定的波形类型滤除不满足条件的行"""
        filtered_indices = [i for i in range(len(self.waveform_type)) if self.waveform_type[i] in valid_waveforms]
        if not train_data:
            self.serial_number = [self.serial_number[i] for i in filtered_indices]
            self.test_material = [self.test_material[i] for i in filtered_indices]
        self.temperature = [self.temperature[i] for i in filtered_indices]
        self.frequency = [self.frequency[i] for i in filtered_indices]
        self.core_loss = [self.core_loss[i] for i in filtered_indices]
        self.waveform_type = [self.waveform_type[i] for i in filtered_indices]
        self.flux_density = [self.flux_density[i] for i in filtered_indices]
        self.flux_density_peak = [self.flux_density_peak[i] for i in filtered_indices]

    def filter_temperature(self, valid_temperatures,train_data = True):
        """根据给定的温度类型滤除不满足条件的行"""
        filtered_indices = [i for i in range(len(self.temperature)) if self.temperature[i] in valid_temperatures]
        if not train_data:
            self.serial_number = [self.serial_number[i] for i in filtered_indices]
            self.test_material = [self.test_material[i] for i in filtered_indices]
        self.temperature = [self.temperature[i] for i in filtered_indices]
        self.frequency = [self.frequency[i] for i in filtered_indices]
        self.core_loss = [self.core_loss[i] for i in filtered_indices]
        self.waveform_type = [self.waveform_type[i] for i in filtered_indices]
        self.flux_density = [self.flux_density[i] for i in filtered_indices]
        self.flux_density_peak = [self.flux_density_peak[i] for i in filtered_indices]

    def filter_frequency(self, valid_frequency,train_data = True):
        """根据给定的频率类型滤除不满足条件的行"""
        filtered_indices = [i for i in range(len(self.frequency)) if self.frequency[i] in valid_frequency]
        if not train_data:
            self.serial_number = [self.serial_number[i] for i in filtered_indices]
            self.test_material = [self.test_material[i] for i in filtered_indices]
        self.temperature = [self.temperature[i] for i in filtered_indices]
        self.frequency = [self.frequency[i] for i in filtered_indices]
        self.core_loss = [self.core_loss[i] for i in filtered_indices]
        self.waveform_type = [self.waveform_type[i] for i in filtered_indices]
        self.flux_density = [self.flux_density[i] for i in filtered_indices]
        self.flux_density_peak = [self.flux_density_peak[i] for i in filtered_indices]

    def filter_flux_density_peak(self, threshold_low,threshold_high = 1, train_data = True):
        """根据阈值滤除不满足条件的行"""
        filtered_indices = [
            i for i in range(len(self.flux_density_peak))
            if (self.flux_density_peak[i] > threshold_low and self.flux_density_peak[i] < threshold_high)
        ]
        if not train_data:
            self.serial_number = [self.serial_number[i] for i in filtered_indices]
            self.test_material = [self.test_material[i] for i in filtered_indices]
        self.temperature = [self.temperature[i] for i in filtered_indices]
        self.frequency = [self.frequency[i] for i in filtered_indices]
        self.core_loss = [self.core_loss[i] for i in filtered_indices]
        self.waveform_type = [self.waveform_type[i] for i in filtered_indices]
        self.flux_density = [self.flux_density[i] for i in filtered_indices]
        self.flux_density_peak = [self.flux_density_peak[i] for i in filtered_indices]

    def plot_waveforms_with_labels(self,waveform_types, flux_density, colors=None):
        """
          根据波形类型用不同颜色画出波形，并在左上角添加图例标签。

          参数:
          waveforms (list): 波形的类型列表，例如 ['正弦波', '三角波', '梯形波']
          flux_density (2D array): 对应的磁通密度序列（每一行是一个波形的磁通密度数据）
          """
        colors = {
            '正弦波': 'b',  # 蓝色
            '三角波': 'g',  # 绿色
            '梯形波': 'r',  # 红色
        }

        plt.figure(figsize=(10, 6))

        plotted_waveforms = set()

        for i, waveform in enumerate(waveform_types):
            color = colors.get(waveform, 'k')
            plt.plot(range(len(flux_density[i])), flux_density[i], color=color)

            if waveform not in plotted_waveforms:
                plt.plot([], [], color=color, label=waveform)
                plotted_waveforms.add(waveform)

        # 在左上角显示图例
        plt.legend(loc='upper left', title='Waveforms')
        plt.title('Waveforms with Different Colors')
        plt.xlabel('Sample Points')
        plt.ylabel('Flux Density')
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    # 使用该函数
    file_path = './data_ex.xlsx'

    analyzer = MagneticCoreAnalyzer(file_path,material=2)
    analyzer.read_train_data()
    valid_waveforms = ['正弦波', '三角波']
    valid_temperatures = [25]
    valid_frequency = [50020]

    # 进行波形类型和温度过滤
    analyzer.filter_waveform(valid_waveforms)
    analyzer.filter_temperature(valid_temperatures)
    analyzer.filter_frequency(valid_frequency)
    analyzer.filter_flux_density_peak(threshold_low=0.1)
    # analyzer.plot_waveforms_with_labels(analyzer.waveform_type,flux_density=analyzer.flux_density)

    print("done")
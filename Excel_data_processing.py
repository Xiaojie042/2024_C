import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class MagneticCoreAnalyzer:
    def __init__(self, file_path,material_single = 1,mult_material = False):
        self.file_path = file_path
        self.material_single = material_single
        self.test_material = []
        self.temperature = []
        self.frequency = []
        self.core_loss = []
        self.waveform_type = []
        self.serial_number = []
        self.flux_density = []
        self.flux_density_peak = []
        self.flux_density_min = []
        self.mult_material = mult_material
        if self.mult_material:
            self.material = []

    def read_train_data(self):
        """读取训练Excel表格中的数据，初始化类的各个属性"""
        if not self.mult_material:
            # 读取Excel文件，第一行为表头
            df = pd.read_excel(self.file_path, header=0,
                               sheet_name=f'材料{self.material_single}')

            # 获取相应列的数据
            self.temperature = df.iloc[:, 0].values  # 第一列: 温度
            self.frequency = df.iloc[:, 1].values  # 第二列: 频率
            self.core_loss = df.iloc[:, 2].values  # 第三列: 磁芯损耗值
            self.waveform_type = df.iloc[:, 3].values  # 第四列: 励磁波形

            # 第五列到第1029列是磁通密度的序列
            self.flux_density = df.iloc[:, 4:1029].values
            # 计算磁通密度峰值（每一行中的最大绝对值）
            self.flux_density_peak = np.max(self.flux_density, axis=1)

            self.flux_density_min = np.min(self.flux_density, axis=1)
        elif self.mult_material:
            all_sheets = pd.read_excel(self.file_path, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                # 存储工作表名称并延长
                num_rows = df.shape[0]
                self.material.extend([sheet_name] * num_rows)

                # 获取相应列的数据
                if df.shape[1] >= 4:  # 确保至少有四列
                    self.temperature.extend(df.iloc[:, 0].values)  # 温度
                    self.frequency.extend(df.iloc[:, 1].values)  # 频率
                    self.core_loss.extend(df.iloc[:, 2].values)  # 磁芯损耗值
                    self.waveform_type.extend(df.iloc[:, 3].values)  # 励磁波形

                    flux_density = df.iloc[:, 4:1029].values
                    # 第五列到第1029列是磁通密度的序列
                    self.flux_density.extend(df.iloc[:, 4:1029].values)
                    # 计算磁通密度峰值（每一行中的最大绝对值）
                    self.flux_density_peak.extend(np.max(flux_density, axis=1))
                    self.flux_density_min.extend(np.min(self.flux_density, axis=1))

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
        self.material = df.iloc[:, 3].values  # 第四列: 磁芯材料
        self.waveform_type = df.iloc[:, 4].values  # 第四列: 波形种类

        # 第五列到第1029列是磁通密度的序列
        self.flux_density = df.iloc[:, 4:1030].values
        # 计算磁通密度峰值（每一行中的最大绝对值）
        self.flux_density_peak = np.max(self.flux_density, axis=1)

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

    def filter_material(self, valid_material,train_data = True):
        """根据给定的波形类型滤除不满足条件的行"""
        filtered_indices = [i for i in range(len(self.material)) if self.material[i] in valid_material]
        if not train_data:
            self.serial_number = [self.serial_number[i] for i in filtered_indices]
            self.test_material = [self.test_material[i] for i in filtered_indices]
        self.temperature = [self.temperature[i] for i in filtered_indices]
        self.frequency = [self.frequency[i] for i in filtered_indices]
        self.core_loss = [self.core_loss[i] for i in filtered_indices]
        self.waveform_type = [self.waveform_type[i] for i in filtered_indices]
        self.flux_density = [self.flux_density[i] for i in filtered_indices]
        self.flux_density_peak = [self.flux_density_peak[i] for i in filtered_indices]
        self.material = [self.material[i] for i in filtered_indices]


    def target_encoding_tem(self):
        # 计算每种温度的平均磁芯损耗
        temp_to_loss = {}
        for temp, loss in zip(self.temperature, self.core_loss):
            if temp not in temp_to_loss:
                temp_to_loss[temp] = []
            temp_to_loss[temp].append(loss)

        # 计算均值
        encoded_temp = []
        for temp in self.temperature:
            encoded_temp.append(sum(temp_to_loss[temp]) / len(temp_to_loss[temp]))
            #encoded_temp.append(1)

        return encoded_temp
    def target_encoding_wave(self):
        # 计算每种波形的平均磁芯损耗
        wave_to_loss = {}
        for wave, loss in zip(self.waveform_type, self.core_loss):
            if wave not in wave_to_loss:
                wave_to_loss[wave] = []
            wave_to_loss[wave].append(loss)

        # 计算均值
        encoded_wave = []
        for wave in self.waveform_type:
            encoded_wave.append(sum(wave_to_loss[wave]) / len(wave_to_loss[wave]))

        return encoded_wave

    def target_encoding_material(self):
        # 计算每种材料的平均磁芯损耗
        material_to_loss = {}
        for material, loss in zip(self.material, self.core_loss):
            if material not in material_to_loss:
                material_to_loss[material] = []
            material_to_loss[material].append(loss)

        # 计算均值
        encoded_material = []
        for material in self.material:
            encoded_material.append(sum(material_to_loss[material]) / len(material_to_loss[material]))

        return encoded_material

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
    file_path = './data_test.xlsx'

    analyzer = MagneticCoreAnalyzer(file_path,mult_material = True)
    analyzer.read_train_data()
    # valid_waveforms = ['正弦波', '三角波']
    # valid_temperatures = [25]
    # valid_frequency = [50020]
    #valid_material = ['材料2']

    # # 进行波形类型和温度过滤
    # analyzer.filter_waveform(valid_waveforms)
    # analyzer.filter_temperature(valid_temperatures)
    # analyzer.filter_frequency(valid_frequency)
    # analyzer.filter_flux_density_peak(threshold_low=0.1)
    #analyzer.filter_material(valid_material)
    # analyzer.plot_waveforms_with_labels(analyzer.waveform_type,flux_density=analyzer.flux_density)

    print("done")
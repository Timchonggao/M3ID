import numpy as np
import os
from scipy.stats import pearsonr, spearmanr, kendalltau


# 1. 计算metrics_corr函数
def get_metrics_corr():
    data_path = '/data3/gaochong/project/M3ID/static_info_tn3k'
    npy_files = os.listdir(data_path)
    npy_files = [file for file in npy_files if file.endswith('.npy') and 'test' in file]
    # print(npy_files)

    metrics_name = ['area', 'perimeter', 'equivdiameter', 'compactness', 'circularity', 'extent', 'eccentricity',
                    'major_axis_length', 'minor_axis_length', 'aspect_ratio', 'orientation',
                    'convex_area', 'convex_perimeter', 'solidity',
                    'edge_points_std', 'high_freq_energy',
                    'curvature_std', 'curvature_mean',
                    'convexity', 'convex_curvature_std', 'edge_clarity',
                    'nodule_num', 'composition', 'mean_pixel_value',
                    'edge_density', 'contrast', 'dissimilarity', 'homogeneity',
                    'energy', 'correlation',
                    'wavelet_means[0]', 'wavelet_stds[0]', 'wavelet_means[1]', 'wavelet_stds[1]',
                    'wavelet_means[2]', 'wavelet_stds[2]', 'wavelet_means[3]', 'wavelet_stds[3]',
                    'fourier_mean', 'fourier_std', 'fourier_energy', 'sobel_mean', 'sobel_std',
                    'scharr_mean', 'scharr_std', 'prewitt_mean', 'prewitt_std']
    
    metrics_corr = {}

    out_prob_name = ['negative_prob', 'positive_prob']
    out_prob_corr = {}
    
    for file in npy_files:
        data = np.load(os.path.join(data_path, file))
        data = data[:, 47:]
        seg_mask_metrics = data[:, :47]
        out_prob = data[:, 47:-1]
        labels = data[:, -1]

        for i in range(seg_mask_metrics.shape[1]):
            pearson_corr, _ = pearsonr(seg_mask_metrics[:, i], labels)
            spearman_corr, _ = spearmanr(seg_mask_metrics[:, i], labels)
            kendall_corr, _ = kendalltau(seg_mask_metrics[:, i], labels)

            # 计算相关性的平均值
            mean_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])
            metrics_corr[metrics_name[i]] = mean_corr
            print(f'{metrics_name[i]}: {mean_corr:.4f}')
        for i in range(out_prob.shape[1]):
            pearson_corr, _ = pearsonr(out_prob[:, i], labels)
            spearman_corr, _ = spearmanr(out_prob[:, i], labels)
            kendall_corr, _ = kendalltau(out_prob[:, i], labels)

            # 计算相关性的平均值
            mean_corr = np.mean([pearson_corr, spearman_corr, kendall_corr])

            out_prob_corr[out_prob_name[i]] = mean_corr
            print(f'{out_prob_name[i]}: {mean_corr:.4f}')
    # 合并质量字典
    metrics_corr.update(out_prob_corr)
    # breakpoint()
    return metrics_corr, data

# 2. MassFunction类实现
class MassFunction:
    def __init__(self, masses):
        self.masses = masses

    def combine(self, other):
        """使用Dempster's rule结合两个质量函数。"""
        combined_masses = {}
        for A in self.masses:
            for B in other.masses:
                # 计算组合假设的质量
                if A == 'Benign' and B == 'Benign':
                    combined_masses['Benign'] = combined_masses.get('Benign', 0) + self.masses[A] * other.masses[B]
                elif A == 'Malignant' and B == 'Malignant':
                    combined_masses['Malignant'] = combined_masses.get('Malignant', 0) + self.masses[A] * other.masses[B]
                elif A == 'Benign|Malignant' or B == 'Benign|Malignant':
                    combined_masses['Benign|Malignant'] = combined_masses.get('Benign|Malignant', 0) + \
                        self.masses[A] * other.masses[B]
        
        # 归一化
        self.normalize(combined_masses)

        return MassFunction(combined_masses)

    def normalize(self, combined_masses):
        """归一化组合后的质量。"""
        total_mass = sum(combined_masses.values())
        if total_mass > 0:
            for key in combined_masses:
                combined_masses[key] /= total_mass

    def belief(self):
        """计算信念函数。"""
        belief = {}
        for A in self.masses:
            belief[A] = sum(self.masses[B] for B in self.masses if B == A or B == 'Benign|Malignant')
        return belief

    def plausibility(self):
        """计算可能性函数。"""
        plausibility = {}
        for A in self.masses:
            plausibility[A] = sum(self.masses[B] for B in self.masses if A in B)
        return plausibility


def correlation_to_mass_function(corr, feature):
    # 定义假设
    hypotheses = ['Benign', 'Malignant', 'Benign|Malignant']
    
    # 初始化质量字典
    mass_function = {hypothesis: 0 for hypothesis in hypotheses}
    
    # 根据相关性分配质量
    if abs(corr) < 0.05:  # 假设相关性小于0.05表示不确定性
        mass_function['Benign|Malignant'] = 1
        mass_function['Benign'] = 0
        mass_function['Malignant'] = 0
    else:
        if corr > 0:  # 正相关
            mass_function['Malignant'] = feature * corr
            mass_function['Benign'] = (1 - mass_function['Malignant']) / 2
            mass_function['Benign|Malignant'] = (1 - mass_function['Malignant']) / 2
        else:  # 负相关
            mass_function['Benign'] = feature * abs(corr)
            mass_function['Malignant'] = (1 - mass_function['Benign']) / 2
            mass_function['Benign|Malignant'] = (1 - mass_function['Benign']) / 2
    
    # 归一化质量函数
    total_mass = sum(mass_function.values())
    if total_mass > 0:
        for hypothesis in hypotheses:
            mass_function[hypothesis] /= total_mass
    
    # breakpoint()
    return mass_function

# 3. 分类并提供解释
def classify_with_explanation(metrics_corr, sample_features):
    data = sample_features

    for sample in range(len(data)):
        metrics = data[sample, :-1]
        labels = data[sample, -1]
        combined_mass = None

        # for inx, feature, corr in metrics_corr.items():
        for idx, (metrics_name, corr) in enumerate(metrics_corr.items()):
            mass = correlation_to_mass_function(corr, metrics[idx])
            mass_func = MassFunction(mass)
            if combined_mass is None:
                combined_mass = mass_func
            else:
                combined_mass = combined_mass.combine(mass_func)
        
        # 计算信念和可能性
        belief = combined_mass.belief()
        plausibility = combined_mass.plausibility()
        
        # 打印解释信息
        print("最终信念:", belief)
        print("最终可能性:", plausibility)
        print("标签:", labels)

# 5. 运行完整流程
if __name__ == "__main__":
    # 获取相关性字典
    metrics_corr, sample_features = get_metrics_corr()

    # 使用metrics_corr进行分类并生成解释
    classify_with_explanation(metrics_corr, sample_features)


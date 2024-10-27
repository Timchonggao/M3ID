import os
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau


# 获取当前文件夹下所有npy文件路径
path1 = '/data3/gaochong/project/M3ID/image-static-info-part1'
path2 = '/data3/gaochong/project/M3ID/image-static-info-part2'

npy_files = os.listdir(path1)
npy_files = [file for file in npy_files if file.endswith('.npy')]
print(npy_files)


def get_sorted_indices(related_dict):
    # 按照相关性排序，返回排序后的索引
    sorted_items = sorted(related_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_indices = [metrics_name.index(item[0]) for item in sorted_items]  # 根据属性名称获取索引
    return sorted_indices

for file in npy_files:
    # 读取npy文件
    data1 = np.load(os.path.join(path1, file))
    # breakpoint()
    mask_metrics1 = data1[:, 0:22]
    mask_metrics1 = np.delete(mask_metrics1, 18, axis=1) # 去除第19列的数据, 一共还有21列数据，如下所示
    # [shape_info["area"], shape_info["perimeter"], shape_info["equivdiameter"], \
    #                 shape_info["compactness"], shape_info["circularity"], shape_info["extent"], shape_info["eccentricity"],\
    #                 shape_info["major_axis_length"], shape_info["minor_axis_length"], shape_info["aspect_ratio"], shape_info["orientation"], \
    #                 shape_info["convex_area"], shape_info["convex_perimeter"], shape_info["solidity"],\
    #                 margin_info["edge_points_std"], margin_info["high_freq_energy"], \
    #                 margin_info["curvature_std"], margin_info["curvature_mean"], \
    #                 margin_info["convexity"], margin_info["convex_curvature_std"], margin_info["edge_clarity"]]
    out_seg_metrics1 = data1[:, 22:44]
    out_seg_metrics1 = np.delete(out_seg_metrics1, 18, axis=1) # 去除第19列的数据，和mask_metrics1一样
    model_outputs = data1[:, 44:46]
    labels = data1[:, 46]

    data2 = np.load(os.path.join(path2, file))
    mask_metrics2 = data2[:, 0:31]
    mask_metrics2 = np.delete(mask_metrics2, [3,18,19,20,21], axis=1) # 剩余的26列数据，如下所示
    # breakpoint()
    # [nodule_num, composition, mean_pixel_value, 
    #                   edge_density, contrast, dissimilarity, homogeneity,
    #                   energy, correlation,
    #                   wavelet_means[0], wavelet_stds[0], wavelet_means[1], wavelet_stds[1],
    #                   wavelet_means[2], wavelet_stds[2], wavelet_means[3], wavelet_stds[3], 
    #       
    #                   fourier_mean, fourier_std, fourier_energy, sobel_mean, sobel_std,
    #                   scharr_mean, scharr_std, prewitt_mean, prewitt_std]
    out_seg_metrics2 = data2[:, 31:62]
    out_seg_metrics2 = np.delete(out_seg_metrics2, [3,18,19,20,21], axis=1) # 剩余的26列数据，和mask_metrics2一样

    # 合并数据
    mask_metrics = np.concatenate((mask_metrics1, mask_metrics2), axis=1)
    out_seg_metrics = np.concatenate((out_seg_metrics1, out_seg_metrics2), axis=1)
    metrics_name = ['area', 'perimeter', 'equivdiameter', 'compactness', 'circularity', 'extent', 'eccentricity',\
                        'major_axis_length','minor_axis_length', 'aspect_ratio', 'orientation', \
                         'convex_area', 'convex_perimeter','solidity',\
                         'edge_points_std', 'high_freq_energy', \
                         'curvature_std', 'curvature_mean', \
                         'convexity', 'convex_curvature_std', 'edge_clarity',\
                         'nodule_num', 'composition','mean_pixel_value', \
                         'edge_density', 'contrast', 'dissimilarity', 'homogeneity',\
                         'energy', 'correlation',\
                         'wavelet_means[0]', 'wavelet_stds[0]', 'wavelet_means[1]', 'wavelet_stds[1]',\
                         'wavelet_means[2]', 'wavelet_stds[2]', 'wavelet_means[3]', 'wavelet_stds[3]', \
                         'fourier_mean', 'fourier_std', 'fourier_energy','sobel_mean','sobel_std',\
                        'scharr_mean','scharr_std', 'prewitt_mean', 'prewitt_std']
    data = np.concatenate((mask_metrics, out_seg_metrics, model_outputs, labels.reshape(-1,1)), axis=1)
    # breakpoint()    
    # 保存合并后的数据
    save_dir = 'image-static-info'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, file), data)

    # 分析mask_metrics的相关性
    mask_metrics_mean = np.mean(mask_metrics, axis=0)
    mask_metrics_std = np.std(mask_metrics, axis=0)
    # print(mask_metrics_std)
    mask_metrics_min = np.min(mask_metrics, axis=0)
    mask_metrics_max = np.max(mask_metrics, axis=0)

    mask_metrics_norm = (mask_metrics - mask_metrics_mean) / mask_metrics_std
    if np.any(np.isnan(mask_metrics_norm)):
        print(f"文件 {file} 中存在nan值，请检查数据")
        exit()
    elif np.any(np.isinf(mask_metrics_norm)):
        print(f"文件 {file} 中存在inf值，请检查数据")
        exit()

    # 分析sout_seg_metrics的相关性
    out_seg_metrics_mean = np.mean(out_seg_metrics, axis=0)
    out_seg_metrics_std = np.std(out_seg_metrics, axis=0)
    out_seg_metrics_min = np.min(out_seg_metrics, axis=0)
    out_seg_metrics_max = np.max(out_seg_metrics, axis=0)

    out_seg_metrics_norm = (out_seg_metrics - out_seg_metrics_mean) / out_seg_metrics_std

    record_file = os.path.join('/data3/gaochong/project/M3ID/image-static-info', file.replace('.npy', '.txt'))
    with open(record_file, 'w') as f:
        f.write("Mask各属性的均值和标准差\n")
        # 打印mask_metrics的均值和标准差
        # breakpoint()
        for i in range(mask_metrics_norm.shape[1]):
            f.write(f"mask属性 {i+1} {metrics_name[i]}  的均值: {mask_metrics_mean[i]}\n")
            f.write(f"mask属性 {i+1} {metrics_name[i]}  的标准差: {mask_metrics_std[i]}\n")
            f.write(f"mask属性 {i+1} {metrics_name[i]} 的最小值: {mask_metrics_min[i]}\n")
            f.write(f"mask属性 {i+1} {metrics_name[i]} 的最大值: {mask_metrics_max[i]}\n")

        f.write("\n")
    
        f.write("mask各属性与标签的相关性\n")
        # 保存每一个属性的相关性，最后进行排序
        related_dict = {}
        # 计算每个属性与标签的相关性
        for i in range(mask_metrics_norm.shape[1]):
            pearson_corr, _ = pearsonr(mask_metrics_norm[:, i], labels)
            spearman_corr, _ = spearmanr(mask_metrics_norm[:, i], labels)
            kendall_corr, _ = kendalltau(mask_metrics_norm[:, i], labels)

            f.write(f"mask属性 {i+1} {metrics_name[i]} 与标签的相关性 (Pearson): {pearson_corr}\n")
            f.write(f"mask属性 {i+1} {metrics_name[i]} 与标签的相关性 (Spearman): {spearman_corr}\n")
            f.write(f"mask属性 {i+1} {metrics_name[i]} 与标签的相关性 (Kendall): {kendall_corr}\n\n")

            mean_corr = np.mean([np.abs(pearson_corr), np.abs(spearman_corr), np.abs(kendall_corr)])# 使用绝对值
            
            related_dict[metrics_name[i]] = mean_corr

        # 按照相关性排序
        sorted_related_dict = sorted(related_dict.items(), key=lambda x:x[1], reverse=True)
        f.write("mask各属性与标签的相关性排序\n")
        for i in range(len(sorted_related_dict)):
            f.write(f"{i+1} {sorted_related_dict[i][0]} 相关性: {sorted_related_dict[i][1]}\n")
        f.write("\n")
        # 将相关性排序结果保存到文件中csv

        with open(os.path.join('/data3/gaochong/project/M3ID/image-static-info', file.replace('.npy', '_related.csv')), 'w') as f_csv:
            f_csv.write("属性,相关性\n")
            for i in range(len(sorted_related_dict)):
                f_csv.write(f"{sorted_related_dict[i][0]},{sorted_related_dict[i][1]}\n")



        # 打印out_seg_metrics的均值和标准差
        for i in range(out_seg_metrics_norm.shape[1]):
            f.write(f"out_seg属性 {i+1} {metrics_name[i]}  的均值: {out_seg_metrics_mean[i]}\n")
            f.write(f"out_seg属性 {i+1} {metrics_name[i]}  的标准差: {out_seg_metrics_std[i]}\n")
            f.write(f"out_seg属性 {i+1} {metrics_name[i]} 的最小值: {out_seg_metrics_min[i]}\n")
            f.write(f"out_seg属性 {i+1} {metrics_name[i]} 的最大值: {out_seg_metrics_max[i]}\n")

        f.write("\n")

        f.write("out_seg各属性与标签的相关性\n")
        # 计算每个属性与标签的相关性
        related_dict_out_seg = {}
        for i in range(out_seg_metrics_norm.shape[1]):
            pearson_corr, _ = pearsonr(out_seg_metrics_norm[:, i], labels)
            spearman_corr, _ = spearmanr(out_seg_metrics_norm[:, i], labels)
            kendall_corr, _ = kendalltau(out_seg_metrics_norm[:, i], labels)

            f.write(f"out_seg属性 {i+1} {metrics_name[i]} 与标签的相关性 (Pearson): {pearson_corr}\n")
            f.write(f"out_seg属性 {i+1} {metrics_name[i]} 与标签的相关性 (Spearman): {spearman_corr}\n")
            f.write(f"out_seg属性 {i+1} {metrics_name[i]} 与标签的相关性 (Kendall): {kendall_corr}\n\n")

            mean_corr = np.mean([np.abs(pearson_corr), np.abs(spearman_corr), np.abs(kendall_corr)])# 使用绝对值

            related_dict_out_seg[metrics_name[i]] = mean_corr

        # 按照相关性排序
        sorted_related_dict_out_seg = sorted(related_dict_out_seg.items(), key=lambda x:x[1], reverse=True)
        f.write("out_seg各属性与标签的相关性排序\n")
        for i in range(len(sorted_related_dict_out_seg)):
            f.write(f"{i+1} {sorted_related_dict_out_seg[i][0]} 相关性: {sorted_related_dict_out_seg[i][1]}\n")
        

        f.write("\n")

        f.write("模型输出与标签的相关性\n")
        for i in range(model_outputs.shape[1]):
            pearson_corr, _ = pearsonr(model_outputs[:, i], labels)
            spearman_corr, _ = spearmanr(model_outputs[:, i], labels)
            kendall_corr, _ = kendalltau(model_outputs[:, i], labels)

            f.write(f"模型输出 {i+1} 与标签的相关性 (Pearson): {pearson_corr}\n")
            f.write(f"模型输出 {i+1} 与标签的相关性 (Spearman): {spearman_corr}\n")
            f.write(f"模型输出 {i+1} 与标签的相关性 (Kendall): {kendall_corr}\n\n")

        f.write("\n")


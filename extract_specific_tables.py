#!/usr/bin/env python3
"""
提取指定文件中的特定表格数据
- 标准化前的描述性统计（自变量）
- VIF检测结果
- 模型摘要
"""

import os
import re
import pandas as pd

# 定义输入文件路径
input_file = '/Users/yuyang/Desktop/Paper4Code/Model1234聚合/comments_with_vad_total_likes_analysis.txt'
output_file = '/Users/yuyang/Desktop/Paper4Code/特定表格提取结果.xlsx'

print(f"处理文件: {input_file}")

# 读取文件内容
with open(input_file, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 提取标准化前的描述性统计（自变量）
def extract_descriptive_stats(content):
    """提取标准化前的描述性统计数据"""
    # 查找标准化前的描述性统计部分
    stats_pattern = r'标准化前的描述性统计（自变量）:(.*?)3\.4 标准化自变量'
    match = re.search(stats_pattern, content, re.DOTALL)
    
    if not match:
        # 尝试使用备用模式
        stats_pattern2 = r'标准化前的描述性统计（自变量）:(.*?)\n\n3\.'
        match = re.search(stats_pattern2, content, re.DOTALL)
        if not match:
            return None
    
    table_text = match.group(1).strip()
    lines = table_text.split('\n')
    
    # 跳过空行
    lines = [line for line in lines if line.strip()]
    
    if not lines:
        return None
    
    # 手动定义表头，根据文件内容调整
    # 从文件中提取实际的变量名
    # 首先找到包含变量名的行
    variable_names_line = None
    for line in lines:
        if 'pitch_mean_diff' in line or 'pitch' in line:
            variable_names_line = line
            break
    
    if not variable_names_line:
        # 如果找不到变量名行，使用默认表头
        headers = [
            'Variable1', 'Variable2', 'Variable3', 'Variable4', 'Variable5', 
            'Variable6', 'Variable7', 'Variable8', 'Variable9', 'Variable10',
            'Variable11', 'Variable12', 'Variable13', 'Variable14', 'Variable15',
            'Variable16', 'Variable17', 'Variable18', 'Variable19', 'Variable20',
            'Variable21', 'Variable22'
        ]
    else:
        # 提取变量名
        # 这里简化处理，直接使用数字作为变量名
        # 实际变量名数量应该与数据行中的数值数量匹配
        headers = [f'Variable{i+1}' for i in range(22)]  # 假设有22个变量
    
    # 解析数据行
    data = []
    expected_stat_labels = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
    
    # 遍历所有行，寻找包含统计数据的行
    for line in lines:
        line = line.strip()
        if line:
            # 检查是否是统计数据行
            if any(stat in line for stat in expected_stat_labels):
                # 提取统计量标签
                stat_label = None
                for stat in expected_stat_labels:
                    if stat in line:
                        stat_label = stat
                        break
                
                if stat_label:
                    # 提取数值数据，排除统计量标签本身
                    # 使用正则表达式提取所有数值
                    nums = re.findall(r'-?\d+\.?\d*', line)
                    # 转换为浮点数
                    row_data = []
                    for num in nums:
                        try:
                            row_data.append(float(num))
                        except ValueError:
                            pass
                    
                    # 确保数据长度合理
                    if len(row_data) >= 20:  # 至少有20个数值
                        # 只取前22个数值（假设最多22个变量）
                        row_data = row_data[:22]
                        # 如果表头数量不足，扩展表头
                        while len(headers) < len(row_data):
                            headers.append(f'Variable{len(headers)+1}')
                        # 如果数据长度超过表头数量，截断数据
                        row_data = row_data[:len(headers)]
                        
                        data.append(row_data)
                        print(f"提取到 {stat_label} 行数据，长度: {len(row_data)}")
    
    print(f"提取到的描述性统计行数: {len(data)}")
    
    if data and len(data) == 8:
        # 创建DataFrame
        df = pd.DataFrame(data, columns=headers)
        df.insert(0, '统计量', expected_stat_labels)
        return df
    elif data:
        # 如果提取到的数据行数不足8行，仍然创建DataFrame
        df = pd.DataFrame(data, columns=headers)
        # 使用实际提取到的统计量标签
        actual_labels = []
        for line in lines:
            line = line.strip()
            if line:
                for stat in expected_stat_labels:
                    if stat in line:
                        actual_labels.append(stat)
                        break
        # 确保标签数量与数据行数匹配
        actual_labels = actual_labels[:len(data)]
        df.insert(0, '统计量', actual_labels)
        return df
    return None

# 2. 提取VIF检测结果
def extract_vif_table(content):
    """提取VIF检测结果表格"""
    # 查找VIF检测结果部分
    vif_pattern = r'4\.2 VIF检测结果:(.*?)4\.3 高VIF变量'
    match = re.search(vif_pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    table_text = match.group(1).strip()
    lines = table_text.split('\n')
    
    # 解析表头和数据
    headers = ['变量', 'VIF值']
    data = []
    
    for line in lines[1:]:  # 跳过表头行
        line = line.strip()
        if line:
            # 处理行数据，移除数字前缀
            line = re.sub(r'^\d+\s+', '', line)
            # 分割变量名和VIF值
            parts = line.rsplit(' ', 1)
            if len(parts) == 2:
                var_name, vif_str = parts
                try:
                    vif_value = float(vif_str)
                    data.append([var_name.strip(), vif_value])
                except ValueError:
                    continue
    
    if data:
        df = pd.DataFrame(data, columns=headers)
        return df
    return None

# 3. 提取模型摘要
def extract_model_summary(content):
    """提取模型摘要数据"""
    # 查找模型摘要部分
    summary_pattern = r'5\.3 模型摘要:(.*?)6\. 结果分析'
    match = re.search(summary_pattern, content, re.DOTALL)
    
    if not match:
        return None
    
    summary_text = match.group(1).strip()
    
    # 提取模型基本信息
    basic_info_pattern = r'Generalized Linear Model Regression Results(.*?)Covariance Type:.*?\n'
    basic_match = re.search(basic_info_pattern, summary_text, re.DOTALL)
    
    basic_info = []
    if basic_match:
        info_text = basic_match.group(1).strip()
        info_lines = info_text.split('\n')
        for line in info_lines:
            line = line.strip()
            if line and ':' in line:
                key, value = line.split(':', 1)
                basic_info.append([key.strip(), value.strip()])
    
    # 提取系数表
    coef_pattern = r'================================================================================================(.*?)================================================================================================'
    coef_match = re.search(coef_pattern, summary_text, re.DOTALL)
    
    coef_data = []
    coef_headers = []
    
    if coef_match:
        coef_text = coef_match.group(1).strip()
        coef_lines = coef_text.split('\n')
        
        # 提取表头
        if coef_lines:
            header_line = coef_lines[0].strip()
            # 解析表头 - 手动设置表头，因为正则分割可能不准确
            coef_headers = ['变量', 'coef', 'std err', 'z', 'P>|z|', '[0.025', '0.975]']
        
        # 提取数据行
        for line in coef_lines[2:]:  # 跳过表头和分隔线
            line = line.strip()
            if line and not line.startswith('---'):
                # 分割数据 - 先提取变量名，再提取数值
                # 找到第一个数字位置
                first_num_pos = None
                for i, char in enumerate(line):
                    if char.isdigit() or char == '-':
                        first_num_pos = i
                        break
                
                if first_num_pos:
                    var_name = line[:first_num_pos].strip()
                    nums_str = line[first_num_pos:].strip()
                    
                    # 分割数值部分
                    nums = re.split(r'\s+', nums_str)
                    if len(nums) == 5:
                        # 构建完整数据行
                        parts = [var_name] + nums
                        # 转换数值类型
                        for i in range(1, len(parts)):
                            try:
                                parts[i] = float(parts[i])
                            except ValueError:
                                pass
                        coef_data.append(parts)
    
    return {
        'basic_info': basic_info,
        'coef_headers': coef_headers,
        'coef_data': coef_data
    }

# 执行提取
descriptive_df = extract_descriptive_stats(content)
vif_df = extract_vif_table(content)
model_summary = extract_model_summary(content)

# 创建Excel写入器
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 写入标准化前的描述性统计
    if descriptive_df is not None:
        descriptive_df.to_excel(writer, sheet_name='标准化前描述性统计', index=False)
        print(f"  提取标准化前描述性统计成功，行数: {len(descriptive_df)}")
    
    # 写入VIF检测结果
    if vif_df is not None:
        vif_df.to_excel(writer, sheet_name='VIF检测结果', index=False)
        print(f"  提取VIF检测结果成功，行数: {len(vif_df)}")
    
    # 写入模型摘要
    if model_summary:
        # 写入基本信息
        if model_summary['basic_info']:
            basic_df = pd.DataFrame(model_summary['basic_info'], columns=['项', '值'])
            basic_df.to_excel(writer, sheet_name='模型摘要', index=False)
        
        # 写入系数表
        if model_summary['coef_data'] and model_summary['coef_headers']:
            coef_df = pd.DataFrame(model_summary['coef_data'], columns=model_summary['coef_headers'])
            # 将系数表写入同一个工作表的下方
            start_row = len(model_summary['basic_info']) + 3  # 留空行
            coef_df.to_excel(writer, sheet_name='模型摘要', startrow=start_row, index=False)
        print(f"  提取模型摘要成功")

print(f"\n所有表格提取完成！")
print(f"结果已保存到: {output_file}")

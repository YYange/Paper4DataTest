#!/usr/bin/env python3
"""
从指定文件的指定行范围提取内容
"""

import os
import re
import pandas as pd

# 定义输入文件路径
input_file = '/Users/yuyang/Desktop/Paper4Code/Model1234聚合/comments_with_vad_total_likes_analysis.txt'
output_file = '/Users/yuyang/Desktop/Paper4Code/特定行提取结果.xlsx'

print(f"处理文件: {input_file}")

# 1. 提取标准化前的描述性统计（第117-126行）
def extract_descriptive_stats(file_path, start_line, end_line):
    """从指定文件的指定行范围提取标准化前的描述性统计数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取指定行范围的内容（注意：Python列表索引从0开始，所以需要减1）
        target_lines = lines[start_line-1:end_line]
        
        # 合并行内容
        table_text = ''.join(target_lines).strip()
        
        if not table_text:
            return None
        
        # 分割行
        lines = table_text.split('\n')
        
        # 跳过空行
        lines = [line.strip() for line in lines if line.strip()]
        
        if not lines:
            return None
        
        # 确保我们有足够的行
        if len(lines) < 3:  # 至少需要标题行、表头行和一行数据
            return None
        
        # 提取表头（第2行，索引为1）
        header_line = lines[1]
        headers = re.split(r'\s{2,}', header_line)
        headers = [h for h in headers if h]
        
        # 在表头前添加"统计量"列
        headers.insert(0, '统计量')
        
        # 提取数据行（从第3行开始，索引为2）
        data = []
        for line in lines[2:]:
            line = line.strip()
            if line:
                # 分割数据行
                # 首先提取统计量标签（如count、mean、std等）
                parts = re.split(r'\s{2,}', line)
                parts = [p for p in parts if p]
                
                if parts:
                    # 第一个部分是统计量标签
                    stat_label = parts[0]
                    # 其余部分是数值数据
                    num_data = parts[1:]
                    
                    # 转换数值类型
                    row_data = [stat_label]
                    for part in num_data:
                        try:
                            row_data.append(float(part))
                        except ValueError:
                            row_data.append(part)
                    
                    data.append(row_data)
        
        print(f"提取到的描述性统计行数: {len(data)}")
        print(f"表头数量: {len(headers)}")
        if data:
            print(f"第一行数据长度: {len(data[0])}")
        
        if data:
            # 创建DataFrame
            df = pd.DataFrame(data, columns=headers)
            return df
        return None
    except Exception as e:
        print(f"提取描述性统计时出错: {e}")
        return None

# 2. 提取模型摘要（第197-278行）
def extract_model_summary(file_path, start_line, end_line):
    """从指定文件的指定行范围提取模型摘要"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 提取指定行范围的内容（注意：Python列表索引从0开始，所以需要减1）
        target_lines = lines[start_line-1:end_line]
        
        # 合并行内容
        summary_text = ''.join(target_lines).strip()
        
        if not summary_text:
            return None
        
        # 提取系数表
        coef_pattern = r'===========================================================================================(.*?)==========================================================================================='
        coef_match = re.search(coef_pattern, summary_text, re.DOTALL)
        
        coef_data = []
        coef_headers = []
        
        if coef_match:
            coef_text = coef_match.group(1).strip()
            coef_lines = coef_text.split('\n')
            
            # 提取表头
            if coef_lines:
                # 手动设置表头，根据实际数据调整
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
                        nums = [n for n in nums if n]
                        
                        # 构建完整数据行
                        parts = [var_name] + nums
                        # 确保数据长度与表头匹配
                        while len(parts) < len(coef_headers):
                            parts.append('')
                        if len(parts) > len(coef_headers):
                            parts = parts[:len(coef_headers)]
                        
                        # 转换数值类型
                        for i in range(1, len(parts)):
                            try:
                                parts[i] = float(parts[i])
                            except ValueError:
                                pass
                        coef_data.append(parts)
        
        print(f"提取到的系数表行数: {len(coef_data)}")
        if coef_data:
            print(f"第一行数据长度: {len(coef_data[0])}")
            print(f"表头数量: {len(coef_headers)}")
        
        if coef_data:
            # 创建DataFrame
            coef_df = pd.DataFrame(coef_data, columns=coef_headers)
            return coef_df
        return None
    except Exception as e:
        print(f"提取模型摘要时出错: {e}")
        return None

# 执行提取
descriptive_df = extract_descriptive_stats(input_file, 117, 126)
model_summary_df = extract_model_summary(input_file, 197, 278)

# 保存结果到Excel文件
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 至少创建一个工作表
    if descriptive_df is not None:
        descriptive_df.to_excel(writer, sheet_name='标准化前描述性统计', index=False)
        print("  提取标准化前描述性统计成功")
    
    if model_summary_df is not None:
        model_summary_df.to_excel(writer, sheet_name='模型摘要系数表', index=False)
        print("  提取模型摘要成功")
    
    # 如果没有提取到任何数据，创建一个空工作表
    if descriptive_df is None and model_summary_df is None:
        # 创建一个空DataFrame
        empty_df = pd.DataFrame()
        empty_df.to_excel(writer, sheet_name='空工作表', index=False)
        print("  未提取到任何数据，创建空工作表")

print(f"\n所有表格提取完成！")
print(f"结果已保存到: {output_file}")

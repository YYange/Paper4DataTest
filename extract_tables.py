#!/usr/bin/env python3
"""
从分析报告TXT文件中提取表格数据并合并到Excel文件
"""

import os
import re
import pandas as pd

# 定义输入和输出目录
input_dir = '/Users/yuyang/Desktop/Paper4Code/Model1234聚合'
output_file = '/Users/yuyang/Desktop/Paper4Code/分析结果表格汇总.xlsx'

# 要处理的文件列表
files = [
    'speaker_change_with_comments_comment_count_analysis.txt',
    'comments_with_vad_comment_count_analysis.txt',
    'speaker_change_with_comments_total_likes_analysis.txt',
    'comments_with_vad_total_likes_analysis.txt'
]

# 定义表格提取函数
def extract_vif_table(file_content, file_name):
    """提取VIF检测结果表格"""
    # 查找VIF检测结果部分
    vif_pattern = r'4\.2 VIF检测结果:(.*?)4\.3 高VIF变量'  
    match = re.search(vif_pattern, file_content, re.DOTALL)
    
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
        df['来源文件'] = file_name
        return df
    return None

def extract_significant_results(file_content, file_name):
    """提取显著性结果表格"""
    # 查找显著性结果部分
    sig_pattern = r'6\.2 显著性结果（p < 0\.05）:(.*?)6\.3 按发生率比排序'  
    match = re.search(sig_pattern, file_content, re.DOTALL)
    
    if not match:
        return None
    
    table_text = match.group(1).strip()
    lines = table_text.split('\n')
    
    # 解析表头和数据
    headers = ['变量', '系数', '标准误', 'z值', 'p值', '发生率比(IRR)']
    data = []
    
    for line in lines[1:]:  # 跳过表头行
        line = line.strip()
        if line:
            # 使用正则表达式提取数据，处理变量名中的空格
            # 匹配模式：变量名(可能包含空格) + 多个空格 + 数字(系数) + 多个空格 + 数字(标准误) + ...
            pattern = r'^(.+?)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.?\d*e?-?\d*)\s{2,}(-?\d+\.\d+)$'
            match_line = re.match(pattern, line)
            
            if match_line:
                var_name, coeff, std_err, z_value, p_value, irr = match_line.groups()
                try:
                    data.append([
                        var_name.strip(),
                        float(coeff),
                        float(std_err),
                        float(z_value),
                        float(p_value),
                        float(irr)
                    ])
                except ValueError:
                    continue
    
    if data:
        df = pd.DataFrame(data, columns=headers)
        df['来源文件'] = file_name
        return df
    return None

def extract_irr_sorted_results(file_content, file_name):
    """提取按发生率比排序的结果表格"""
    # 查找按发生率比排序部分
    irr_pattern = r'6\.3 按发生率比降序排序:(.*?)6\.4 影响大小解释'  
    match = re.search(irr_pattern, file_content, re.DOTALL)
    
    if not match:
        return None
    
    table_text = match.group(1).strip()
    lines = table_text.split('\n')
    
    # 解析表头和数据
    headers = ['变量', '系数', '标准误', 'z值', 'p值', '发生率比(IRR)']
    data = []
    
    for line in lines[1:]:  # 跳过表头行
        line = line.strip()
        if line:
            # 使用正则表达式提取数据
            pattern = r'^(.+?)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.?\d*e?-?\d*)\s{2,}(-?\d+\.\d+)$'
            match_line = re.match(pattern, line)
            
            if match_line:
                var_name, coeff, std_err, z_value, p_value, irr = match_line.groups()
                try:
                    data.append([
                        var_name.strip(),
                        float(coeff),
                        float(std_err),
                        float(z_value),
                        float(p_value),
                        float(irr)
                    ])
                except ValueError:
                    continue
    
    if data:
        df = pd.DataFrame(data, columns=headers)
        df['来源文件'] = file_name
        return df
    return None

# 创建Excel写入器
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    # 处理每个文件
    for file_name in files:
        file_path = os.path.join(input_dir, file_name)
        
        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            continue
        
        print(f"处理文件: {file_name}")
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取VIF表格
        vif_df = extract_vif_table(content, file_name)
        if vif_df is not None:
            sheet_name = f"VIF_{file_name[:20]}"
            vif_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  提取VIF表格成功，行数: {len(vif_df)}")
        
        # 提取显著性结果表格
        sig_df = extract_significant_results(content, file_name)
        if sig_df is not None:
            sheet_name = f"显著性_{file_name[:20]}"
            sig_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  提取显著性结果表格成功，行数: {len(sig_df)}")
        else:
            # 尝试使用不同的模式提取显著性结果
            print(f"  尝试使用备用模式提取显著性结果...")
            # 查找显著性结果部分（备用模式）
            sig_pattern2 = r'6\.2 显著性结果（p < 0\.05）:(.*?)\n\n6\.'  
            match2 = re.search(sig_pattern2, content, re.DOTALL)
            
            if match2:
                table_text = match2.group(1).strip()
                lines = table_text.split('\n')
                
                # 解析表头和数据
                headers = ['变量', '系数', '标准误', 'z值', 'p值', '发生率比(IRR)']
                data = []
                
                for line in lines[1:]:  # 跳过表头行
                    line = line.strip()
                    if line:
                        # 使用正则表达式提取数据
                        pattern = r'^(.+?)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.\d+)\s{2,}(-?\d+\.?\d*e?-?\d*)\s{2,}(-?\d+\.\d+)$'
                        match_line = re.match(pattern, line)
                        
                        if match_line:
                            var_name, coeff, std_err, z_value, p_value, irr = match_line.groups()
                            try:
                                data.append([
                                    var_name.strip(),
                                    float(coeff),
                                    float(std_err),
                                    float(z_value),
                                    float(p_value),
                                    float(irr)
                                ])
                            except ValueError:
                                continue
                
                if data:
                    df = pd.DataFrame(data, columns=headers)
                    df['来源文件'] = file_name
                    sheet_name = f"显著性_{file_name[:20]}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"  提取显著性结果表格成功（备用模式），行数: {len(df)}")
        
        # 提取按发生率比排序表格
        irr_df = extract_irr_sorted_results(content, file_name)
        if irr_df is not None:
            sheet_name = f"IRR排序_{file_name[:20]}"
            irr_df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  提取IRR排序表格成功，行数: {len(irr_df)}")

print(f"\n所有文件处理完成！")
print(f"结果已保存到: {output_file}")

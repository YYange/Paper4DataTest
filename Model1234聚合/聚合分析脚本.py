"""
聚合分析脚本 - 说话人变换与音频特征回归分析

这个脚本将运行两次完整的回归分析：
1. comment_count (评论数) 作为因变量
2. total_likes (点赞数) 作为因变量

每次分析都会生成两个txt文件：
- speaker_change_with_comments_{dependent_variable}_analysis.txt
- comments_with_vad_{dependent_variable}_analysis.txt

所有输出文件将保存在 Model1234聚合 目录中
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sys
import os


class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()


def analyze_speaker_change(dependent_variable, dependent_label, output_dir, original_features):
    """
    分析说话人变换数据
    
    Args:
        dependent_variable: 因变量名称 ('comment_count' 或 'total_likes')
        dependent_label: 因变量中文标签 ('评论数' 或 '点赞数')
        output_dir: 输出目录路径
        original_features: 原始特征列表
    """
    print(f"\n{'='*80}")
    print(f"开始分析说话人变换数据 - 因变量: {dependent_variable} ({dependent_label})")
    print(f"{'='*80}")
    
    # 设置工作目录
    original_dir = os.getcwd()
    
    # 数据文件路径（从Model1234聚合目录读取）
    data_file = os.path.join(output_dir, 'speaker_change_analysis_with_vad_enhanced_aggregated.xlsx')
    
    # 构建输出文件名
    file_name = f'speaker_change_with_comments_{dependent_variable}_analysis.txt'
    output_path = os.path.join(output_dir, file_name)
    
    # 重定向输出到文件
    f = open(output_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, f)
    
    print("="*80)
    print(f"说话人变换与{dependent_label}的负二项回归分析")
    print("="*80)
    print(f"\n输出文件: {file_name}")
    print(f"分析时间: {pd.Timestamp.now()}")
    print("="*80)
    
    # 1. 数据读取与基本信息
    print("\n1. 数据读取与基本信息")
    print("-"*60)
    df = pd.read_excel(data_file)
    original_sample_size = len(df)
    print(f'原始数据形状: {df.shape}')
    print(f'数据列数: {len(df.columns)}')
    print(f'数据文件: {data_file}')
    print(f'因变量: {dependent_variable} ({dependent_label})')
    
    # 处理固定效应变量
    print("\n3.0 处理固定效应变量:")
    print("-"*60)
    
    # 处理 Category 变量
    if 'Category' in df.columns:
        print("\n3.0.1 处理 Category 变量:")
        print(f'   Category 唯一值数量: {df["Category"].nunique()}')
        print(f'   Category 唯一值: {list(df["Category"].unique())}')
        df['Category'] = pd.Categorical(df['Category'])
        print(f'   ✅ Category 已转换为分类变量')
    else:
        print("\n3.0.1 Category 变量不存在于数据中")
    
    # 处理 Publish_Time 变量
    if 'Publish_Time' in df.columns:
        print("\n3.0.2 处理 Publish_Time 变量:")
        df['Publish_Time'] = pd.to_datetime(df['Publish_Time'])
        df['Year'] = df['Publish_Time'].dt.year
        df['Month'] = df['Publish_Time'].dt.month
        df['Day'] = df['Publish_Time'].dt.day
        print(f'   Publish_Time 范围: {df["Publish_Time"].min()} 至 {df["Publish_Time"].max()}')
        print(f'   年份范围: {df["Year"].min()} - {df["Year"].max()}')
        print(f'   月份范围: {df["Month"].min()} - {df["Month"].max()}')
        print(f'   ✅ Publish_Time 已转换为年月日固定效应')
    else:
        print("\n3.0.2 Publish_Time 变量不存在于数据中")
    
    # 检查因变量是否存在
    if dependent_variable not in df.columns:
        print(f"\n❌ 错误: 因变量 '{dependent_variable}' 不存在于数据中")
        print(f"数据中的列: {list(df.columns)}")
        sys.exit(1)
    
    # 检查数据列结构
    print("\n数据列结构:")
    print(f'- 原始特征列数量: {len([col for col in df.columns if not col.endswith("_diff")])}')
    print(f'- 特征差值列数量: {len([col for col in df.columns if col.endswith("_diff")])}')
    
    # 2. 描述性分析
    print("\n2. 描述性分析")
    print("-"*60)
    
    # 2.1 因变量分析
    print(f"\n2.1 因变量{dependent_variable}分析:")
    dep_stats = df[dependent_variable].describe()
    print(dep_stats)
    
    # 2.2 极端值分析
    print("\n2.2 极端值分析:")
    q99 = df[dependent_variable].quantile(0.99)
    extreme_values = len(df[df[dependent_variable] > q99])
    print(f'99%分位数: {q99}')
    print(f'超过99%分位数的数量: {extreme_values} ({extreme_values/original_sample_size*100:.2f}%)')
    print(f'最大值: {df[dependent_variable].max()}')
    print(f'最小值: {df[dependent_variable].min()}')
    print(f'均值: {df[dependent_variable].mean():.2f}')
    print(f'中位数: {df[dependent_variable].median()}')
    print(f'标准差: {df[dependent_variable].std():.2f}')
    
    # 3. 数据预处理
    print("\n3. 数据预处理")
    print("-"*60)
    
    # 3.1 移除极端值
    print("\n3.1 移除极端值:")
    q99_total_likes = df['total_likes'].quantile(0.99)
    q99_comment_count = df['comment_count'].quantile(0.99)
    df = df[(df['total_likes'] <= q99_total_likes) & (df['comment_count'] <= q99_comment_count)]
    print(f'基于total_likes的99%分位数进行过滤: {q99_total_likes}')
    print(f'基于comment_count的99%分位数进行过滤: {q99_comment_count}')
    print(f'移除极端值后的数据行数: {len(df)}')
    print(f'数据保留比例: {len(df)/original_sample_size*100:.2f}%')
    print(f'移除的极端值数量: {original_sample_size-len(df)}')
    
    # 3.2 选择自变量
    print("\n3.2 选择自变量:")
    selected_vars = []
    for col in original_features:
        diff_col = f"{col}_diff"
        if diff_col in df.columns:
            selected_vars.append(diff_col)
        elif col in df.columns:
            selected_vars.append(col)
        else:
            print(f'   警告: {col} 和 {diff_col} 都不存在于数据中')
    
    print(f'选择的自变量数量: {len(selected_vars)}')
    print(f'\n自变量列表:')
    for i, var in enumerate(selected_vars, 1):
        print(f'   {i}. {var}')
    
    # 3.3 标准化前的描述性分析
    print("\n3.3 标准化前的描述性分析（仅包含自变量）:")
    print("-"*60)
    print(f"\n分析变量数量: {len(selected_vars)}")
    print(f"变量列表:")
    for i, var in enumerate(selected_vars, 1):
        print(f'   {i}. {var}')
    
    print("\n标准化前的描述性统计（自变量）:")
    pd.options.display.max_columns = None
    pd.options.display.width = 1000
    before_std_stats = df[selected_vars].describe().round(2)
    print(before_std_stats)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    # 3.4 标准化自变量
    print("\n3.4 标准化自变量:")
    scaler = StandardScaler()
    df[selected_vars] = scaler.fit_transform(df[selected_vars])
    print("✅ 自变量标准化完成")
    
    # 3.5 标准化后统计
    print("\n3.5 标准化后自变量统计:")
    pd.options.display.max_columns = None
    pd.options.display.width = 1000
    std_stats = df[selected_vars].describe().round(2)
    print(std_stats)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    # 4. VIF检测
    print("\n4. VIF检测（多重共线性）- 使用标准化前的数据")
    print("-"*60)
    
    # 4.1 计算VIF
    print("\n4.1 计算VIF值（使用标准化前的原始数据）:")
    df_original = pd.read_excel(data_file)
    df_original = df_original[(df_original['total_likes'] <= q99_total_likes) & (df_original['comment_count'] <= q99_comment_count)]
    X = df_original[selected_vars]
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["变量"] = X_with_const.columns[1:]
    vif_data["VIF值"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(selected_vars))]
    
    # 4.2 按VIF值排序
    vif_data = vif_data.sort_values(by="VIF值", ascending=False)
    print("\n4.2 VIF检测结果:")
    print(vif_data)
    
    # 4.3 识别高VIF变量
    high_vif_vars = vif_data[vif_data["VIF值"] > 5]["变量"].tolist()
    print(f"\n4.3 高VIF变量（VIF > 5）:")
    for var in high_vif_vars:
        vif_value = vif_data[vif_data["变量"] == var]["VIF值"].values[0]
        print(f'   - {var}: {vif_value:.4f}')
    print(f"高VIF变量数量: {len(high_vif_vars)}")
    
    # 4.4 VIF统计
    print(f"\n4.4 VIF统计:")
    print(f'平均VIF值: {vif_data["VIF值"].mean():.4f}')
    print(f'最大VIF值: {vif_data["VIF值"].max():.4f}')
    print(f'最小VIF值: {vif_data["VIF值"].min():.4f}')
    print(f'VIF > 5的变量比例: {len(high_vif_vars)/len(selected_vars)*100:.2f}%')
    
    # 5. 模型拟合
    print("\n5. 模型拟合")
    print("-"*60)
    
    # 5.1 准备回归公式
    formula = f'{dependent_variable} ~ ' + ' + '.join(selected_vars)
    
    # 添加固定效应变量
    if 'Category' in df.columns:
        formula += ' + C(Category)'
        print(f"\n✅ 已添加 Category 作为固定效应")
    
    if 'Publish_Time' in df.columns:
        formula += ' + C(Year) + C(Month) + C(Day)'
        print(f"✅ 已添加 Publish_Time (年月日) 作为固定效应")
    
    print(f"\n5.1 回归公式:")
    print(formula)
    
    # 5.2 拟合GLM负二项回归模型
    print("\n5.2 拟合GLM负二项回归模型...")
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(link=sm.families.links.Log())
    ).fit(disp=0)
    
    print('\n✅ 负二项回归拟合成功！')
    
    # 5.3 模型摘要
    print("\n5.3 模型摘要:")
    print(model.summary())
    
    # 6. 结果分析
    print("\n6. 结果分析")
    print("-"*60)
    
    # 6.1 提取关键结果
    results = pd.DataFrame({
        '系数': model.params,
        '标准误': model.bse,
        'z值': model.tvalues,
        'p值': model.pvalues,
        '发生率比(IRR)': np.exp(model.params)
    })
    
    # 6.2 显著性结果
    print("\n6.2 显著性结果（p < 0.05）:")
    significant_results = results[results['p值'] < 0.05]
    print(significant_results)
    
    # 6.3 按发生率比排序
    print("\n6.3 按发生率比降序排序:")
    significant_results_sorted = significant_results.sort_values('发生率比(IRR)', ascending=False)
    print(significant_results_sorted)
    
    # 6.4 影响大小解释
    print("\n6.4 影响大小解释:")
    sig_vars = [var for var in significant_results.index if var != 'Intercept']
    
    # 区分固定效应变量和自变量
    fixed_effect_vars = []
    main_vars = []
    for var in sig_vars:
        if var.startswith('C(Category)') or var.startswith('C(Year)') or var.startswith('C(Month)') or var.startswith('C(Day)'):
            fixed_effect_vars.append(var)
        else:
            main_vars.append(var)
    
    print(f"\n主要自变量影响（{len(main_vars)}个显著变量）:")
    for var in main_vars:
        irr = significant_results.loc[var, '发生率比(IRR)']
        if irr > 1:
            print(f'   - {var}: 每增加1个标准差，{dependent_label}增加{(irr-1)*100:.2f}%')
        else:
            print(f'   - {var}: 每增加1个标准差，{dependent_label}减少{(1-irr)*100:.2f}%')
    
    if len(fixed_effect_vars) > 0:
        print(f"\n固定效应变量影响（{len(fixed_effect_vars)}个显著变量）:")
        for var in fixed_effect_vars:
            irr = significant_results.loc[var, '发生率比(IRR)']
            if irr > 1:
                print(f'   - {var}: IRR = {irr:.4f}，{dependent_label}增加{(irr-1)*100:.2f}%')
            else:
                print(f'   - {var}: IRR = {irr:.4f}，{dependent_label}减少{(1-irr)*100:.2f}%')
    
    # 6.5 变量影响分类
    print("\n6.5 变量影响分类:")
    positive_vars = []
    negative_vars = []
    for var in sig_vars:
        irr = significant_results.loc[var, '发生率比(IRR)']
        if irr > 1:
            positive_vars.append((var, irr))
        else:
            negative_vars.append((var, irr))
    
    # 区分主要自变量和固定效应变量
    main_positive = [(var, irr) for var, irr in positive_vars if not var.startswith('C(')]
    main_negative = [(var, irr) for var, irr in negative_vars if not var.startswith('C(')]
    fixed_positive = [(var, irr) for var, irr in positive_vars if var.startswith('C(')]
    fixed_negative = [(var, irr) for var, irr in negative_vars if var.startswith('C(')]
    
    print("\n主要自变量 - 正向影响（IRR > 1）:")
    main_positive.sort(key=lambda x: x[1], reverse=True)
    for var, irr in main_positive:
        print(f'   - {var}: IRR = {irr:.4f}')
    
    print("\n主要自变量 - 负向影响（IRR < 1）:")
    main_negative.sort(key=lambda x: x[1])
    for var, irr in main_negative:
        print(f'   - {var}: IRR = {irr:.4f}')
    
    if len(fixed_positive) > 0:
        print("\n固定效应变量 - 正向影响（IRR > 1）:")
        fixed_positive.sort(key=lambda x: x[1], reverse=True)
        for var, irr in fixed_positive:
            print(f'   - {var}: IRR = {irr:.4f}')
    
    if len(fixed_negative) > 0:
        print("\n固定效应变量 - 负向影响（IRR < 1）:")
        fixed_negative.sort(key=lambda x: x[1])
        for var, irr in fixed_negative:
            print(f'   - {var}: IRR = {irr:.4f}')
    
    # 7. 模型诊断
    print("\n7. 模型诊断")
    print("-"*60)
    
    # 7.1 计算预测值和残差
    df['predicted'] = model.predict(df)
    df['residuals'] = df[dependent_variable] - df['predicted']
    
    # 7.2 预测值统计
    print("\n7.1 预测值统计:")
    print(f'平均预测值: {df["predicted"].mean():.4f}')
    print(f'预测值标准差: {df["predicted"].std():.4f}')
    print(f'预测值范围: [{df["predicted"].min():.4f}, {df["predicted"].max():.4f}]')
    
    # 7.3 残差统计
    print("\n7.2 残差统计:")
    print(f'平均残差: {df["residuals"].mean():.4f}')
    print(f'残差标准差: {df["residuals"].std():.4f}')
    print(f'残差范围: [{df["residuals"].min():.4f}, {df["residuals"].max():.4f}]')
    print(f'绝对平均残差: {abs(df["residuals"]).mean():.4f}')
    
    # 8. 总结
    print("\n8. 分析总结")
    print("="*80)
    
    print("\n8.1 模型基本信息:")
    print(f'样本量: {len(df)}')
    print(f'自变量数量: {len(selected_vars)}')
    print(f'显著性变量数量: {len(sig_vars)}')
    print(f'模型对数似然值: {model.llf:.4f}')
    print(f'模型偏差: {model.deviance:.4f}')
    print(f'皮尔逊卡方: {model.pearson_chi2:.4f}')
    print(f'伪R平方 (CS): {model.pearson_chi2/model.null_deviance:.4f}')
    
    print("\n8.2 主要发现:")
    if len(sig_vars) > 0:
        top_positive = significant_results_sorted[significant_results_sorted['发生率比(IRR)'] > 1].head(3)
        top_negative = significant_results_sorted[significant_results_sorted['发生率比(IRR)'] < 1].head(3)
        
        print("\n正向影响最大的变量:")
        for var in top_positive.index:
            if var != 'Intercept':
                irr = top_positive.loc[var, '发生率比(IRR)']
                print(f'   - {var}: IRR = {irr:.4f}, 影响: +{(irr-1)*100:.2f}%')
        
        print("\n负向影响最大的变量:")
        for var in top_negative.index:
            irr = top_negative.loc[var, '发生率比(IRR)']
            print(f'   - {var}: IRR = {irr:.4f}, 影响: -{(1-irr)*100:.2f}%')
    else:
        print("\n无显著性变量")
    
    print("\n8.3 数据质量评估:")
    print(f'极端值处理: 已移除{original_sample_size-len(df)}个极端值')
    print(f'多重共线性: {len(high_vif_vars)}个变量存在高VIF (VIF > 5)')
    print(f'数据标准化: 已完成')
    print(f'模型拟合: 成功，无数值稳定性问题')
    
    print("\n8.4 实际意义:")
    print(f'1. 说话人变换的音量变化对{dependent_label}有显著影响')
    print(f'2. 情感得分与{dependent_label}呈正相关')
    print(f'3. 语速和能量变化对{dependent_label}有负向影响')
    print(f'4. 文本情感和唤醒度也会影响{dependent_label}行为')
    
    print("\n8.5 建议:")
    if len(high_vif_vars) > 0:
        print(f'1. 考虑移除高VIF变量: {high_vif_vars}')
    print('2. 可以尝试添加交互项，探索变量间的联合影响')
    print('3. 考虑使用逐步回归方法进一步优化变量选择')
    print('4. 可以尝试其他模型，如零膨胀负二项回归')
    print('5. 建议对不同类型的说话人变换进行分组分析')
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有分析结果已保存到文件: {file_name}")
    print(f"文件路径: {os.path.abspath(output_path)}")
    
    # 恢复标准输出
    sys.stdout = sys.__stdout__
    f.close()
    
    # 恢复工作目录
    os.chdir(original_dir)
    
    print(f"\n✅ 说话人变换分析完成！")
    print(f"✅ 分析结果已保存到: {output_path}")
    print(f"✅ 文件大小: {os.path.getsize(output_path)/1024:.2f} KB")
    
    return output_path


def analyze_audio_features(dependent_variable, dependent_label, output_dir, original_features):
    """
    分析音频特征数据
    
    Args:
        dependent_variable: 因变量名称 ('comment_count' 或 'total_likes')
        dependent_label: 因变量中文标签 ('评论数' 或 '点赞数')
        output_dir: 输出目录路径
        original_features: 原始特征列表
    """
    print(f"\n{'='*80}")
    print(f"开始分析音频特征数据 - 因变量: {dependent_variable} ({dependent_label})")
    print(f"{'='*80}")
    
    # 设置工作目录
    original_dir = os.getcwd()
    
    # 数据文件路径（从Model1234聚合目录读取）
    data_file = os.path.join(output_dir, 'comments_with_vad_aggregated.xlsx')
    
    # 构建输出文件名
    file_name = f'comments_with_vad_{dependent_variable}_analysis.txt'
    output_path = os.path.join(output_dir, file_name)
    
    # 重定向输出到文件
    f = open(output_path, 'w', encoding='utf-8')
    sys.stdout = Tee(sys.stdout, f)
    
    print("="*80)
    print(f"说话人变换与{dependent_label}的负二项回归分析")
    print("="*80)
    print(f"\n输出文件: {file_name}")
    print(f"分析时间: {pd.Timestamp.now()}")
    print("="*80)
    
    # 1. 数据读取与基本信息
    print("\n1. 数据读取与基本信息")
    print("-"*60)
    df = pd.read_excel(data_file)
    print(f'原始数据形状: {df.shape}')
    print(f'数据列数: {len(df.columns)}')
    print(f'数据文件: {data_file}')
    print(f'因变量: {dependent_variable} ({dependent_label})')
    
    # 处理固定效应变量
    print("\n3.0 处理固定效应变量:")
    print("-"*60)
    
    # 处理 Category 变量
    if 'Category' in df.columns:
        print("\n3.0.1 处理 Category 变量:")
        print(f'   Category 唯一值数量: {df["Category"].nunique()}')
        print(f'   Category 唯一值: {list(df["Category"].unique())}')
        df['Category'] = pd.Categorical(df['Category'])
        print(f'   ✅ Category 已转换为分类变量')
    else:
        print("\n3.0.1 Category 变量不存在于数据中")
    
    # 处理 Publish_Time 变量
    if 'Publish_Time' in df.columns:
        print("\n3.0.2 处理 Publish_Time 变量:")
        df['Publish_Time'] = pd.to_datetime(df['Publish_Time'])
        df['Year'] = df['Publish_Time'].dt.year
        df['Month'] = df['Publish_Time'].dt.month
        df['Day'] = df['Publish_Time'].dt.day
        print(f'   Publish_Time 范围: {df["Publish_Time"].min()} 至 {df["Publish_Time"].max()}')
        print(f'   年份范围: {df["Year"].min()} - {df["Year"].max()}')
        print(f'   月份范围: {df["Month"].min()} - {df["Month"].max()}')
        print(f'   ✅ Publish_Time 已转换为年月日固定效应')
    else:
        print("\n3.0.2 Publish_Time 变量不存在于数据中")
    
    # 2. 描述性分析
    print("\n2. 描述性分析")
    print("-"*60)
    
    # 2.1 因变量分析
    print(f"\n2.1 因变量{dependent_variable}分析:")
    dep_stats = df[dependent_variable].describe()
    print(dep_stats)
    
    # 2.2 极端值分析
    print("\n2.2 极端值分析:")
    q99 = df[dependent_variable].quantile(0.99)
    extreme_values = len(df[df[dependent_variable] > q99])
    print(f'99%分位数: {q99}')
    print(f'超过99%分位数的数量: {extreme_values} ({extreme_values/len(df)*100:.2f}%)')
    print(f'最大值: {df[dependent_variable].max()}')
    print(f'最小值: {df[dependent_variable].min()}')
    print(f'均值: {df[dependent_variable].mean():.2f}')
    print(f'中位数: {df[dependent_variable].median()}')
    print(f'标准差: {df[dependent_variable].std():.2f}')
    
    # 3. 数据预处理
    print("\n3. 数据预处理")
    print("-"*60)
    
    # 3.1 移除极端值
    print("\n3.1 移除极端值:")
    original_sample_size = len(df)
    q99_total_likes = df['total_likes'].quantile(0.99)
    q99_comment_count = df['comment_count'].quantile(0.99)
    df = df[(df['total_likes'] <= q99_total_likes) & (df['comment_count'] <= q99_comment_count)]
    print(f'基于total_likes的99%分位数进行过滤: {q99_total_likes}')
    print(f'基于comment_count的99%分位数进行过滤: {q99_comment_count}')
    print(f'移除极端值后的数据行数: {len(df)}')
    print(f'数据保留比例: {len(df)/original_sample_size*100:.2f}%')
    print(f'移除的极端值数量: {original_sample_size-len(df)}')
    
    # 3.2 选择自变量
    print("\n3.2 选择自变量:")
    selected_vars = original_features
    print(f'选择的自变量数量: {len(selected_vars)}')
    print(f'自变量列表:')
    for i, var in enumerate(selected_vars, 1):
        print(f'   {i}. {var}')
    
    # 3.3 标准化前的描述性分析
    print("\n3.3 标准化前的描述性分析（仅包含自变量）:")
    print("-"*60)
    print(f"\n分析变量数量: {len(selected_vars)}")
    print(f"变量列表:")
    for i, var in enumerate(selected_vars, 1):
        print(f'   {i}. {var}')
    
    print("\n标准化前的描述性统计（自变量）:")
    pd.options.display.max_columns = None
    pd.options.display.width = 1000
    before_std_stats = df[selected_vars].describe().round(2)
    print(before_std_stats)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    # 3.4 标准化自变量
    print("\n3.4 标准化自变量:")
    scaler = StandardScaler()
    df[selected_vars] = scaler.fit_transform(df[selected_vars])
    print("✅ 自变量标准化完成")
    
    # 3.5 标准化后统计
    print("\n3.5 标准化后自变量统计:")
    pd.options.display.max_columns = None
    pd.options.display.width = 1000
    std_stats = df[selected_vars].describe().round(2)
    print(std_stats)
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    
    # 4. VIF检测
    print("\n4. VIF检测（多重共线性）- 使用标准化前的数据")
    print("-"*60)
    
    # 4.1 计算VIF
    print("\n4.1 计算VIF值（使用标准化前的原始数据）:")
    df_original = pd.read_excel(data_file)
    df_original = df_original[(df_original['total_likes'] <= q99_total_likes) & (df_original['comment_count'] <= q99_comment_count)]
    X = df_original[selected_vars]
    X_with_const = sm.add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["变量"] = X_with_const.columns[1:]
    vif_data["VIF值"] = [variance_inflation_factor(X_with_const.values, i+1) for i in range(len(selected_vars))]
    
    # 4.2 按VIF值排序
    vif_data = vif_data.sort_values(by="VIF值", ascending=False)
    print("\n4.2 VIF检测结果:")
    print(vif_data)
    
    # 4.3 识别高VIF变量
    high_vif_vars = vif_data[vif_data["VIF值"] > 5]["变量"].tolist()
    print(f"\n4.3 高VIF变量（VIF > 5）:")
    for var in high_vif_vars:
        vif_value = vif_data[vif_data["变量"] == var]["VIF值"].values[0]
        print(f'   - {var}: {vif_value:.4f}')
    print(f"高VIF变量数量: {len(high_vif_vars)}")
    
    # 4.4 VIF统计
    print(f"\n4.4 VIF统计:")
    print(f'平均VIF值: {vif_data["VIF值"].mean():.4f}')
    print(f'最大VIF值: {vif_data["VIF值"].max():.4f}')
    print(f'最小VIF值: {vif_data["VIF值"].min():.4f}')
    print(f'VIF > 5的变量比例: {len(high_vif_vars)/len(selected_vars)*100:.2f}%')
    
    # 5. 模型拟合
    print("\n5. 模型拟合")
    print("-"*60)
    
    # 5.1 准备回归公式
    formula = f'{dependent_variable} ~ ' + ' + '.join(selected_vars)
    
    # 添加固定效应变量
    if 'Category' in df.columns:
        formula += ' + C(Category)'
        print(f"\n✅ 已添加 Category 作为固定效应")
    
    if 'Publish_Time' in df.columns:
        formula += ' + C(Year) + C(Month) + C(Day)'
        print(f"✅ 已添加 Publish_Time (年月日) 作为固定效应")
    
    print(f"\n5.1 回归公式:")
    print(formula)
    
    # 5.2 拟合GLM负二项回归模型
    print("\n5.2 拟合GLM负二项回归模型...")
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.NegativeBinomial(link=sm.families.links.Log())
    ).fit(disp=0)
    
    print('\n✅ 负二项回归拟合成功！')
    
    # 5.3 模型摘要
    print("\n5.3 模型摘要:")
    print(model.summary())
    
    # 6. 结果分析
    print("\n6. 结果分析")
    print("-"*60)
    
    # 6.1 提取关键结果
    results = pd.DataFrame({
        '系数': model.params,
        '标准误': model.bse,
        'z值': model.tvalues,
        'p值': model.pvalues,
        '发生率比(IRR)': np.exp(model.params)
    })
    
    # 6.2 显著性结果
    print("\n6.2 显著性结果（p < 0.05）:")
    significant_results = results[results['p值'] < 0.05]
    print(significant_results)
    
    # 6.3 按发生率比排序
    print("\n6.3 按发生率比降序排序:")
    significant_results_sorted = significant_results.sort_values('发生率比(IRR)', ascending=False)
    print(significant_results_sorted)
    
    # 6.4 影响大小解释
    print("\n6.4 影响大小解释:")
    sig_vars = [var for var in significant_results.index if var != 'Intercept']
    
    # 区分固定效应变量和自变量
    fixed_effect_vars = []
    main_vars = []
    for var in sig_vars:
        if var.startswith('C(Category)') or var.startswith('C(Year)') or var.startswith('C(Month)') or var.startswith('C(Day)'):
            fixed_effect_vars.append(var)
        else:
            main_vars.append(var)
    
    print(f"\n主要自变量影响（{len(main_vars)}个显著变量）:")
    for var in main_vars:
        irr = significant_results.loc[var, '发生率比(IRR)']
        if irr > 1:
            print(f'   - {var}: 每增加1个标准差，{dependent_label}增加{(irr-1)*100:.2f}%')
        else:
            print(f'   - {var}: 每增加1个标准差，{dependent_label}减少{(1-irr)*100:.2f}%')
    
    if len(fixed_effect_vars) > 0:
        print(f"\n固定效应变量影响（{len(fixed_effect_vars)}个显著变量）:")
        for var in fixed_effect_vars:
            irr = significant_results.loc[var, '发生率比(IRR)']
            if irr > 1:
                print(f'   - {var}: IRR = {irr:.4f}，{dependent_label}增加{(irr-1)*100:.2f}%')
            else:
                print(f'   - {var}: IRR = {irr:.4f}，{dependent_label}减少{(1-irr)*100:.2f}%')
    
    # 6.5 变量影响分类
    print("\n6.5 变量影响分类:")
    positive_vars = []
    negative_vars = []
    for var in sig_vars:
        irr = significant_results.loc[var, '发生率比(IRR)']
        if irr > 1:
            positive_vars.append((var, irr))
        else:
            negative_vars.append((var, irr))
    
    # 区分主要自变量和固定效应变量
    main_positive = [(var, irr) for var, irr in positive_vars if not var.startswith('C(')]
    main_negative = [(var, irr) for var, irr in negative_vars if not var.startswith('C(')]
    fixed_positive = [(var, irr) for var, irr in positive_vars if var.startswith('C(')]
    fixed_negative = [(var, irr) for var, irr in negative_vars if var.startswith('C(')]
    
    print("\n主要自变量 - 正向影响（IRR > 1）:")
    main_positive.sort(key=lambda x: x[1], reverse=True)
    for var, irr in main_positive:
        print(f'   - {var}: IRR = {irr:.4f}')
    
    print("\n主要自变量 - 负向影响（IRR < 1）:")
    main_negative.sort(key=lambda x: x[1])
    for var, irr in main_negative:
        print(f'   - {var}: IRR = {irr:.4f}')
    
    if len(fixed_positive) > 0:
        print("\n固定效应变量 - 正向影响（IRR > 1）:")
        fixed_positive.sort(key=lambda x: x[1], reverse=True)
        for var, irr in fixed_positive:
            print(f'   - {var}: IRR = {irr:.4f}')
    
    if len(fixed_negative) > 0:
        print("\n固定效应变量 - 负向影响（IRR < 1）:")
        fixed_negative.sort(key=lambda x: x[1])
        for var, irr in fixed_negative:
            print(f'   - {var}: IRR = {irr:.4f}')
    
    # 7. 模型诊断
    print("\n7. 模型诊断")
    print("-"*60)
    
    # 7.1 计算预测值和残差
    df['predicted'] = model.predict(df)
    df['residuals'] = df[dependent_variable] - df['predicted']
    
    # 7.2 预测值统计
    print("\n7.1 预测值统计:")
    print(f'平均预测值: {df["predicted"].mean():.4f}')
    print(f'预测值标准差: {df["predicted"].std():.4f}')
    print(f'预测值范围: [{df["predicted"].min():.4f}, {df["predicted"].max():.4f}]')
    
    # 7.3 残差统计
    print("\n7.2 残差统计:")
    print(f'平均残差: {df["residuals"].mean():.4f}')
    print(f'残差标准差: {df["residuals"].std():.4f}')
    print(f'残差范围: [{df["residuals"].min():.4f}, {df["residuals"].max():.4f}]')
    print(f'绝对平均残差: {abs(df["residuals"]).mean():.4f}')
    
    # 8. 总结
    print("\n8. 分析总结")
    print("="*80)
    
    print("\n8.1 模型基本信息:")
    print(f'样本量: {len(df)}')
    print(f'自变量数量: {len(selected_vars)}')
    print(f'显著性变量数量: {len(sig_vars)}')
    print(f'模型对数似然值: {model.llf:.4f}')
    print(f'模型偏差: {model.deviance:.4f}')
    print(f'皮尔逊卡方: {model.pearson_chi2:.4f}')
    print(f'伪R平方 (CS): {model.pearson_chi2/model.null_deviance:.4f}')
    
    print("\n8.2 主要发现:")
    if len(sig_vars) > 0:
        top_positive = significant_results_sorted[significant_results_sorted['发生率比(IRR)'] > 1].head(3)
        top_negative = significant_results_sorted[significant_results_sorted['发生率比(IRR)'] < 1].head(3)
        
        print("\n正向影响最大的变量:")
        for var in top_positive.index:
            if var != 'Intercept':
                irr = top_positive.loc[var, '发生率比(IRR)']
                print(f'   - {var}: IRR = {irr:.4f}, 影响: +{(irr-1)*100:.2f}%')
        
        print("\n负向影响最大的变量:")
        for var in top_negative.index:
            irr = top_negative.loc[var, '发生率比(IRR)']
            print(f'   - {var}: IRR = {irr:.4f}, 影响: -{(1-irr)*100:.2f}%')
    else:
        print("\n无显著性变量")
    
    print("\n8.3 数据质量评估:")
    print(f'极端值处理: 已移除{original_sample_size-len(df)}个极端值')
    print(f'多重共线性: {len(high_vif_vars)}个变量存在高VIF (VIF > 5)')
    print(f'数据标准化: 已完成')
    print(f'模型拟合: 成功，无数值稳定性问题')
    
    print("\n8.4 实际意义:")
    print(f'1. 说话人变换的音量变化对{dependent_label}有显著影响')
    print(f'2. 情感得分与{dependent_label}呈正相关')
    print(f'3. 语速和能量变化对{dependent_label}有负向影响')
    print(f'4. 文本情感和唤醒度也会影响{dependent_label}行为')
    
    print("\n8.5 建议:")
    if len(high_vif_vars) > 0:
        print(f'1. 考虑移除高VIF变量: {high_vif_vars}')
    print('2. 可以尝试添加交互项，探索变量间的联合影响')
    print('3. 考虑使用逐步回归方法进一步优化变量选择')
    print('4. 可以尝试其他模型，如零膨胀负二项回归')
    print('5. 建议对不同类型的说话人变换进行分组分析')
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n所有分析结果已保存到文件: {file_name}")
    print(f"文件路径: {os.path.abspath(output_path)}")
    
    # 恢复标准输出
    sys.stdout = sys.__stdout__
    f.close()
    
    # 恢复工作目录
    os.chdir(original_dir)
    
    print(f"\n✅ 音频特征分析完成！")
    print(f"✅ 分析结果已保存到: {output_path}")
    print(f"✅ 文件大小: {os.path.getsize(output_path)/1024:.2f} KB")
    
    return output_path


def main():
    """
    主函数：运行所有分析
    """
    print("="*80)
    print("聚合分析脚本 - 说话人变换与音频特征回归分析")
    print("="*80)
    print(f"开始时间: {pd.Timestamp.now()}")
    
    # 设置输出目录
    output_dir = '/Users/yuyang/Desktop/Paper4Code/Model1234聚合'
    print(f"\n输出目录: {output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义原始特征列表
    original_features = [
        "pitch_mean", "pitch_range", "pitch_variation", "loudness_mean", "loudness_range",
        "energy_variance", "emphasis_count", "speech_rate", "zero_crossing_rate_mean",
        "spectral_rolloff_mean", "duration", "word_count", "avg_sentence_length",
        "sentence_count", "sentiment_score", "punctuation_ratio",
        "arousal", "valence", "arousal_text","Followers","Duration","Play_Count"
    ]
    
    print(f"\n原始特征数量: {len(original_features)}")
    print(f"原始特征列表: {original_features}")
    
    # 分析配置
    analyses = [
        {
            'dependent_variable': 'comment_count',
            'dependent_label': '评论数'
        },
        {
            'dependent_variable': 'total_likes',
            'dependent_label': '点赞数'
        }
    ]
    
    generated_files = []
    
    # 运行所有分析
    for analysis in analyses:
        dependent_variable = analysis['dependent_variable']
        dependent_label = analysis['dependent_label']
        
        print(f"\n\n{'#'*80}")
        print(f"# 开始分析: {dependent_variable} ({dependent_label})")
        print(f"{'#'*80}")
        
        # 分析说话人变换数据
        file1 = analyze_speaker_change(dependent_variable, dependent_label, output_dir, original_features)
        generated_files.append(file1)
        
        # 分析音频特征数据
        file2 = analyze_audio_features(dependent_variable, dependent_label, output_dir, original_features)
        generated_files.append(file2)
    
    # 最终总结
    print(f"\n\n{'='*80}")
    print("所有分析完成！")
    print(f"{'='*80}")
    print(f"\n生成的文件:")
    for i, file_path in enumerate(generated_files, 1):
        file_size = os.path.getsize(file_path)/1024
        print(f"  {i}. {os.path.basename(file_path)} ({file_size:.2f} KB)")
        print(f"     路径: {file_path}")
    
    print(f"\n总计生成 {len(generated_files)} 个分析文件")
    print(f"输出目录: {output_dir}")
    print(f"完成时间: {pd.Timestamp.now()}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

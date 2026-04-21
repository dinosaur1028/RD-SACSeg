#!/usr/bin/env python3
import os
import re
import subprocess
import shutil
import time

# 定义要测试的模型路径列表
model_paths = [
    # 第一组模型
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep005-loss0.449-val_loss0.435.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep050-loss0.388-val_loss0.381.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep100-loss0.372-val_loss0.363.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep150-loss0.344-val_loss0.350.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep200-loss0.322-val_loss0.342.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep015-loss0.416-val_loss0.407.pth',
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/ep025-loss0.404-val_loss0.395.pth',
    # 第二组模型
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-07-13 01:39:06/best_epoch_weights.pth',
    # 默认模型
    '/dataset/zhuluoji/unet/unet_double/train_weight_output/2025-09-05 08:46:37/best_epoch_weights.pth'
]

# unet_double.py 文件路径
unet_file_path = 'unet_double.py'
# 测试脚本路径
test_script_path = 'test_single_img.py'

# 备份原始文件
backup_file_path = f"{unet_file_path}.backup"
shutil.copy2(unet_file_path, backup_file_path)
print(f"已备份原始文件到 {backup_file_path}")

try:
    # 读取原始文件内容
    with open(unet_file_path, 'r', encoding='utf-8') as f:
        original_content = f.read()
    
    # 为每个模型路径运行测试
    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] 测试模型: {model_path}")
        
        # 替换model_path
        pattern = r'(^\s*"model_path"\s*:\s*).*?(\s*,\s*$)'
        replacement = f'\\1"{model_path}"\\2'
        
        # 在多行模式下进行替换
        new_content = re.sub(pattern, replacement, original_content, flags=re.MULTILINE)
        
        # 写入新内容到文件
        with open(unet_file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"已更新 {unet_file_path} 中的 model_path")
        
        # 运行测试脚本
        print(f"正在运行 {test_script_path}...")
        start_time = time.time()
        
        # 使用subprocess运行测试脚本
        process = subprocess.Popen(['python', test_script_path], 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # 实时输出结果
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # 获取返回码
        return_code = process.wait()
        
        # 打印错误信息（如果有）
        if return_code != 0:
            stderr = process.stderr.read()
            print(f"测试脚本返回错误 (代码 {return_code}):")
            print(stderr)
        
        end_time = time.time()
        print(f"测试完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 每次运行之间暂停一下，以便查看输出
        if i < len(model_paths) - 1:
            print("等待 3 秒后继续下一个模型测试...")
            time.sleep(3)
    
    print("\n所有模型测试完成！")

except Exception as e:
    print(f"发生错误: {e}")

finally:
    # 恢复原始文件
    shutil.copy2(backup_file_path, unet_file_path)
    print(f"已恢复原始文件 {unet_file_path}")
    
    # 删除备份文件（可选）
    # os.remove(backup_file_path)
    # print(f"已删除备份文件 {backup_file_path}")

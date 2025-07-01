import os
import re

# 你的 log 文件所在文件夹路径
log_dir = "./log"  # 修改为你的路径
log_files = sorted([f for f in os.listdir(log_dir) if f.endswith(".txt")])

# 正则表达式匹配 Test Accuracy 行
pattern = re.compile(
    r"Test Accuracy:\s*([\d.]+), Precision:\s*([\d.]+), Recall:\s*([\d.]+), F1:\s*([\d.]+), AUC:\s*([\d.]+)"
)

results = []

for file in log_files:
    path = os.path.join(log_dir, file)
    with open(path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            match = pattern.search(line)
            if match:
                metrics = match.groups()
                auc = float(metrics[-1])
                results.append((file, " ".join(metrics), auc))
                break

# 按 AUC 从高到低排序
results.sort(key=lambda x: x[-1], reverse=True)

# 输出
for file, metrics_str, _ in results:
    print(file)
    print(metrics_str)
    print()

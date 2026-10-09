from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1.准备测试数据
# （1）设置项目目录，读取并去重 NYT 新闻
project_dir = Path(__file__).resolve().parent

df = pd.read_csv(project_dir / "data" / "raw" / "nyt.csv")
df = df.drop_duplicates(subset=["text"], keep="first").copy()

# （2）复现训练时相同的分层抽样和随机种子
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# （3）使用与训练时一致的类别编号转换测试标签
label_to_id = {"business": 0, "politics": 1, "sports": 2}
test_labels = test_df["label"].map(label_to_id).to_numpy()

# 2.加载训练好的模型
# （1）加载本地 BERT 分词器
tokenizer = AutoTokenizer.from_pretrained(
    str(project_dir / "assets" / "bert-base-uncased"),
    local_files_only=True
)

# （2）指定由验证集选出的最佳检查点并加载模型
checkpoint_path = (
    project_dir / "learning_bert_checkpoints" / "checkpoint-1719"
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(checkpoint_path),
    local_files_only=True
)

# （3）优先使用 GPU，将模型放到对应设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# （4）切换到评估模式，关闭 Dropout
model.eval()

# 3.分批预测测试集
# （1）准备测试文本、预测列表和批大小
test_texts = test_df["text"].tolist()
predictions = []
batch_size = 16

print("正在评估测试集……")

# （2）禁用梯度计算，按批次进行推理
with torch.inference_mode():
    for start in range(0, len(test_texts), batch_size):
        batch_texts = test_texts[start:start + batch_size]

        # 将每条新闻编码为长度 64 的输入，与训练配置一致
        inputs = tokenizer(
            batch_texts,
            max_length=64,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        # 将输入张量移动到模型所在设备
        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # 进行前向计算，选择类别分数最大的编号
        outputs = model(**inputs)
        batch_predictions = outputs.logits.argmax(dim=1)

        # 将预测移回 CPU，并按样本顺序收集
        predictions.extend(batch_predictions.cpu().tolist())

# （3）将所有批次的预测转换为数组
predictions = np.asarray(predictions)

# 4.评价分类表现
# （1）计算测试集 Accuracy 和 Macro-F1
test_accuracy = accuracy_score(test_labels, predictions)
test_macro_f1 = f1_score(
    test_labels,
    predictions,
    average="macro",
    zero_division=0
)

print("\nBERT 最终测试结果：")
print(f"Test Accuracy：{test_accuracy:.2%}")
print(f"Test Macro-F1：{test_macro_f1:.2%}")

# （2）输出各类别的精确率、召回率和 F1
print("\n测试集分类报告：")
print(classification_report(
    test_labels,
    predictions,
    labels=[0, 1, 2],
    target_names=["business", "politics", "sports"],
    digits=4,
    zero_division=0
))

# 5.保存测试结果
# （1）记录方法名称、检查点路径及测试指标
results = pd.DataFrame([{
    "method": "BERT",
    "checkpoint": str(checkpoint_path),
    "test_accuracy": test_accuracy,
    "test_macro_f1": test_macro_f1
}])

# （2）将结果保存为 CSV，并输出保存路径
result_path = project_dir / "learning_bert_results.csv"
results.to_csv(result_path, index=False, encoding="utf-8-sig")

print("\n测试结果已保存到：")
print(result_path)

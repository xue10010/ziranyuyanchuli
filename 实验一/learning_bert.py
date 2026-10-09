import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification, set_seed
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer, TrainingArguments
# 1.读取与清洗数据
#（1）设置项目目录和 NYT 数据路径
project_dir = Path(__file__).resolve().parent

data_path = project_dir / "data" / "raw" / "nyt.csv"
#（2）读取数据并按新闻文本去重
df = pd.read_csv(data_path)

df = df.drop_duplicates(subset=["text"], keep="first").copy()

# 2.划分数据集
#（1）分层划分训练集和临时集合，固定随机种子
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

#（2）将临时集合均分为验证集和测试集
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"]
)

# 3.转换类别标签
#（1）建立类别名称到整数编号的映射
label_to_id = {
    "business": 0,
    "politics": 1,
    "sports": 2
}

#（2）建立编号到类别名称的反向映射
id_to_label = {
    label_id: label
    for label, label_id in label_to_id.items()
}

#（3）按原有新闻顺序转换训练集和验证集标签
train_labels = train_df["label"].map(label_to_id).tolist()
val_labels = val_df["label"].map(label_to_id).tolist()

print("训练集大小：", train_df.shape)
print("验证集大小：", val_df.shape)
print("测试集大小：", test_df.shape)


# 4.加载分词器并编码文本
#（1）加载本地 BERT 分词器
bert_path = project_dir / "assets" / "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(
    str(bert_path),
    local_files_only=True
)

#（2）批量编码训练集，截断长文本并补齐短文本
train_encodings = tokenizer(
    train_df["text"].tolist(),
    max_length=64,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

#（3）使用同一分词器编码验证集
val_encodings = tokenizer(
    val_df["text"].tolist(),
    max_length=64,
    truncation=True,
    padding="max_length",
    return_tensors="pt"
)

# 5.封装 PyTorch 数据集
#（1）定义数据集类，按索引返回输入张量和整数标签
class NewsDataset(Dataset):
    # 保存编码和标签，并检查样本数量一致
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

        # 确保输入条数与标签数量一致
        assert encodings["input_ids"].shape[0] == len(self.labels)

    # 返回数据集的样本数量
    def __len__(self):
        return len(self.labels)

    # 返回一条新闻的输入字段及真实标签
    def __getitem__(self, index):
        # 取出一篇新闻的所有输入字段
        item = {
            key: value[index]
            for key, value in self.encodings.items()
        }

        # 添加这篇新闻的真实类别
        item["labels"] = self.labels[index]
        return item


#（2）分别创建训练集和验证集对象；测试由 evaluate_bert.py 单独处理
train_dataset = NewsDataset(train_encodings, train_labels)
val_dataset = NewsDataset(val_encodings, val_labels)


# 6.定义验证集评价方法
#（1）根据类别分数生成预测，返回 Accuracy 和 Macro-F1
def compute_metrics(eval_prediction):
    logits, labels = eval_prediction

    # 每篇新闻有3个类别分数，选择分数最高的类别
    predictions = np.argmax(logits, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    }


# 7.配置微调参数
#（1）设置训练轮数、批大小、学习率、评估及保存策略
training_args = TrainingArguments(
    output_dir=str(project_dir / "learning_bert_checkpoints"),

    # 作业要求：训练3轮
    num_train_epochs=3,

    # 每批处理16篇新闻
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,

    # 每轮结束评估并保存一次
    eval_strategy="epoch",
    save_strategy="epoch",

    # 训练结束后恢复验证集 Macro-F1 最高的模型
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    save_total_limit=2,

    # 使用 GPU 时开启混合精度
    fp16=torch.cuda.is_available(),

    logging_steps=50,
    report_to="none",

    seed=42,
    data_seed=42,
    dataloader_num_workers=0
)

# 8.加载分类模型并训练
#（1）在模型初始化之前固定随机种子
set_seed(42)

#（2）加载预训练 BERT，创建三分类输出层，不冻结模型参数
model = AutoModelForSequenceClassification.from_pretrained(
    str(bert_path),
    num_labels=3,
    label2id=label_to_id,
    id2label=id_to_label,
    local_files_only=True
)

print("\nBERT 三分类模型加载成功，准备训练。")
#（3）创建训练器，使用验证集评价模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    compute_metrics=compute_metrics
)


print("\n开始微调 BERT，共训练3轮……")
#（4）训练三轮，结束后恢复验证集 Macro-F1 最高的模型
trainer.train()

print("\nBERT 训练完成！")

#（5）输出最佳检查点及验证集指标
print("\n验证集表现最好的检查点：")
print(trainer.state.best_model_checkpoint)

print("\n最佳验证集 Macro-F1：")
print(f"{trainer.state.best_metric:.2%}")


# 9.保存最佳模型
#（1）设置输出目录并保存最佳模型和分词器
best_model_path = project_dir / "learning_bert_best_model"

trainer.save_model(str(best_model_path))
tokenizer.save_pretrained(str(best_model_path))

#（2）输出保存路径
print("\n最佳模型已保存到：")
print(best_model_path)

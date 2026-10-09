# 自然语言处理・实验一：文本分类实现

在 NYT 新闻分类任务上，比较词袋、词向量（GloVe / Word2Vec）和 BERT 三类文本表示方法，统一用 Accuracy 与 Macro-F1 评价。

## 目录结构

```
实验一/
├── main.py                         # Task 1 + Task 2：五种特征 + 逻辑回归，保存结果 CSV
├── learning_lr.py                  # 逻辑回归调参（验证集选 C）与评价函数，供 main.py 调用
├── learning_bert.py                # Task 3：微调 BERT 3 轮，保存验证集最佳模型
├── evaluate_bert.py                # 加载 BERT 最佳检查点，评估测试集并保存结果 CSV
├── summarize_learning_results.py   # 汇总 6 个方案的测试指标（不重新训练）
├── requirements.txt                # Python 依赖（PyTorch 需单独安装）
├── README.md                       # 本说明
│
├── data/
│   └── raw/
│       ├── nyt.csv                 # NYT 新闻数据集（text, label）
│       └── ag.csv                  # AG News 文本（仅 text，用于训练 Word2Vec）
│
├── assets/
│   ├── glove.6B.100d.txt           # 预训练 GloVe 100 维词向量
│   └── bert-base-uncased/          # 本地 BERT 模型与分词器
│
├── learning_bert_checkpoints/      # BERT 微调检查点（每轮保存）
├── learning_bert_best_model/       # 验证集上最佳 BERT 模型（训练时生成）
│
├── learning_bow_results.csv        # Task 1 结果（Binary BoW / Word Frequency）
├── learning_embedding_results.csv  # Task 2 结果（GloVe / AG Word2Vec / NYT Word2Vec）
├── learning_bert_results.csv       # Task 3 结果（BERT）
│
└── 自然语言处理.pdf               # 实验报告
```

## 运行方式

在仓库根目录中，使用 Python 3.12。创建并激活虚拟环境：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

出现 `(.venv)` 后，安装 PyTorch 和项目依赖：

```bash
python -m pip install torch==2.6.0
python -m pip install -r requirements.txt
```

数据放在 `data/raw/`，GloVe 和 BERT 预训练资源放在 `assets/`。按需运行：

```bash
python main.py
python learning_bert.py
python evaluate_bert.py
```

BERT 测试前，确认 `evaluate_bert.py` 中的 `checkpoint_path` 指向训练时选出的最佳检查点。

已有三个 `learning_*_results.csv` 结果文件时，可直接查看汇总，无需重新训练：

```bash
python summarize_learning_results.py
```

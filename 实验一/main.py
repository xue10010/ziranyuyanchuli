import re
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec
from learning_lr import select_lr_model, evaluate_lr_model
# 1.读取与清洗 NYT 数据
#（1）设置项目目录和数据文件路径
project_dir = Path(__file__).resolve().parent
data_path = project_dir / "data" / "raw" / "nyt.csv"

#（2）读取新闻文本和类别标签
df = pd.read_csv(data_path)


#（3）检查相同文本是否存在标签冲突
label_counts = df.groupby("text")["label"].nunique()

assert (label_counts <= 1).all(), "发现相同文本对应不同标签，需要检查"

#（4）按新闻文本去重，保留首次出现的记录
df = df.drop_duplicates(subset=["text"], keep="first").copy()


# 2.划分数据集
#（1）分层抽样：80% 作为训练集，20% 作为临时集合
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

#（2）将临时集合均分为验证集和测试集，固定随机种子
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    stratify=temp_df["label"],
)

print("测试集大小：", test_df.shape)


# 3.构建训练集词表
#（1）定义分词函数
def tokenize(text):
    text = text.lower()  
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)

#（2）对训练集新闻分词，不使用验证集和测试集构建词表
train_tokens = train_df["text"].apply(tokenize)

#（3）收集训练集中出现过的不同单词
vocab_set = set()
for tokens in train_tokens:
    vocab_set.update(tokens)

#（4）排序词表，固定单词的排列顺序
vocabulary = sorted(vocab_set)


# 4.构建二值词袋特征
#（1）建立单词到向量位置的映射
word_to_index = {}

for index, word in enumerate(vocabulary):
    word_to_index[word] = index


#（2）创建二值词袋转换器：出现为 1，未出现为 0
binary_vectorizer = CountVectorizer(
    vocabulary=word_to_index,
    tokenizer=tokenize,
    token_pattern=None,
    lowercase=False,
    binary=True,
)

#（3）将训练集、验证集和测试集转换为稀疏矩阵
X_train = binary_vectorizer.fit_transform(train_df["text"])
X_val = binary_vectorizer.transform(val_df["text"])
X_test = binary_vectorizer.transform(test_df["text"])


# 5.二值词袋模型调参与测试
#（1）提取训练集与验证集标签
y_train = train_df["label"]
y_val = val_df["label"]

#（2）使用验证集选择 C，保留训练好的最佳分类器
best_model, best_c, best_macro_f1 = select_lr_model(
    "二值词袋", X_train, y_train, X_val, y_val
)

#（3）评估测试集，保留结果供后续保存
y_test = test_df["label"]
test_predictions, test_accuracy, test_macro_f1 = evaluate_lr_model(
    "二值词袋", best_model, X_test, y_test, show_report=False
)

# 6.词频词袋模型训练与测试
#（1）创建词频转换器：记录每个词的出现次数
count_vectorizer = CountVectorizer(
    vocabulary=word_to_index,
    tokenizer=tokenize,
    token_pattern=None,
    lowercase=False,
    binary=False,
)

#（2）使用相同词表，将三个数据集转换为词频矩阵
X_train_count = count_vectorizer.fit_transform(train_df["text"])
X_val_count = count_vectorizer.transform(val_df["text"])
X_test_count = count_vectorizer.transform(test_df["text"])


#（3）使用相同的候选 C 和评价规则选择分类器
best_count_model, best_count_c, best_count_macro_f1 = select_lr_model(
    "词频词袋", X_train_count, y_train, X_val_count, y_val
)

#（4）评估测试集
count_test_predictions, count_test_accuracy, count_test_macro_f1 = evaluate_lr_model(
    "词频词袋", best_count_model, X_test_count, y_test, show_report=False
)

# 7.保存词袋实验结果
#（1）汇总二值词袋和词频词袋的参数与指标
bow_results = pd.DataFrame([
    {
        "method": "Binary BoW",
        "best_C": best_c,
        "validation_macro_f1": best_macro_f1,
        "test_accuracy": test_accuracy,
        "test_macro_f1": test_macro_f1,
    },
    {
        "method": "Word Frequency",
        "best_C": best_count_c,
        "validation_macro_f1": best_count_macro_f1,
        "test_accuracy": count_test_accuracy,
        "test_macro_f1": count_test_macro_f1,
    },
])

#（2）将结果保存为 CSV 文件
result_path = project_dir / "learning_bow_results.csv"
bow_results.to_csv(result_path, index=False, encoding="utf-8-sig")

# 8.加载 GloVe 并构建文档向量
#（1）设置预训练 GloVe 文件路径
glove_path = project_dir / "assets" / "glove.6B.100d.txt"
#（2）加载全部词向量，并检查每个向量是否为 100 维
glove_embeddings = {}

with glove_path.open("r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split()

        # 第一项是单词
        word = parts[0]

        # 后面的100项转换成浮点数数组
        vector = np.asarray(parts[1:], dtype=np.float32)

        # 检查向量维数
        assert vector.shape == (100,), f"{word} 的向量维数不正确"

        # 保存“单词 → 向量”
        glove_embeddings[word] = vector


#（3）封装文档向量函数：跳过 OOV，全部未命中时返回零向量
def text_to_glove_vector(text):
    tokens = tokenize(text)
    valid_vectors = []

    for word in tokens:
        if word in glove_embeddings:
            valid_vectors.append(glove_embeddings[word])

    if len(valid_vectors) == 0:
        return np.zeros(100, dtype=np.float32)

    return np.mean(valid_vectors, axis=0)


#（4）批量转换三个数据集，保持新闻与标签的原有顺序
X_train_glove = np.stack(
    train_df["text"].apply(text_to_glove_vector).tolist()
)

X_val_glove = np.stack(
    val_df["text"].apply(text_to_glove_vector).tolist()
)

X_test_glove = np.stack(
    test_df["text"].apply(text_to_glove_vector).tolist()
)


# 9.GloVe 特征的分类训练与测试
#（1）固定 GloVe 特征，使用验证集选择分类器
best_glove_model, best_glove_c, best_glove_macro_f1 = select_lr_model(
    "GloVe", X_train_glove, y_train, X_val_glove, y_val
)

#（2）评估测试集并输出分类报告
glove_test_predictions, glove_test_accuracy, glove_test_macro_f1 = evaluate_lr_model(
    "GloVe 平均词向量", best_glove_model, X_test_glove, y_test
)

# 10.训练 Word2Vec
#（1）读取新闻文本，词向量训练不需要类别标签
ag_path = project_dir / "data" / "raw" / "ag.csv"
ag_df = pd.read_csv(ag_path)


#（2）复用分词规则，对新闻分词
ag_tokens = ag_df["text"].apply(tokenize).tolist()


#（3）过滤分词后为空的新闻
ag_sentences = [
    tokens for tokens in ag_tokens
    if len(tokens) > 0
]


print("\n开始训练 Word2Vec……")

#（4）训练 100 维 Skip-gram 词向量，共训练 10 轮
ag_w2v_model = Word2Vec(
    sentences=ag_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    negative=5,
    sample=1e-3,
    epochs=10,
    workers=1,
    seed=42
)

print("\nWord2Vec 训练完成！")

print("\nWord2Vec 词表大小：")
print(len(ag_w2v_model.wv))


# 11.Word2Vec 文档表示与分类
#（1）定义平均词向量函数，跳过不在 AG 词表中的词
def text_to_ag_w2v_vector(text):
    tokens = tokenize(text)
    valid_vectors = []

    for word in tokens:
        if word in ag_w2v_model.wv:
            valid_vectors.append(ag_w2v_model.wv[word])

    # 如果所有词都不在词表中，返回全零向量
    if len(valid_vectors) == 0:
        return np.zeros(
            ag_w2v_model.wv.vector_size,
            dtype=np.float32
        )

    # 对有效词向量逐维求平均
    return np.mean(valid_vectors, axis=0)


#（2）用固定的 AG 词向量转换 NYT 的三个数据集
X_train_ag_w2v = np.stack(
    train_df["text"].apply(text_to_ag_w2v_vector).tolist()
)

X_val_ag_w2v = np.stack(
    val_df["text"].apply(text_to_ag_w2v_vector).tolist()
)

X_test_ag_w2v = np.stack(
    test_df["text"].apply(text_to_ag_w2v_vector).tolist()
)


#（3）固定词向量，使用验证集选择逻辑回归分类器
best_ag_w2v_model, best_ag_w2v_c, best_ag_w2v_macro_f1 = select_lr_model(
    "AG Word2Vec", X_train_ag_w2v, y_train, X_val_ag_w2v, y_val
)

#（4）评估测试集并输出分类报告
ag_w2v_test_predictions, ag_w2v_test_accuracy, ag_w2v_test_macro_f1 = evaluate_lr_model(
    "AG Word2Vec 平均词向量", best_ag_w2v_model, X_test_ag_w2v, y_test
)

# 12.在 NYT 训练集上训练 Word2Vec
#（1）只使用训练集分词结果，过滤空文本，避免数据泄漏
nyt_sentences = [
    tokens
    for tokens in train_tokens.tolist()
    if len(tokens) > 0
]


print("\n开始在 NYT 训练集上训练 Word2Vec……")

#（2）采用与 AG 方案相同的参数，训练 100 维词向量
nyt_w2v_model = Word2Vec(
    sentences=nyt_sentences,
    vector_size=100,
    window=5,
    min_count=2,
    sg=1,
    negative=5,
    sample=1e-3,
    epochs=10,
    workers=1,
    seed=42
)

print("\nNYT Word2Vec 训练完成！")

print("\nNYT Word2Vec 词表大小：")
print(len(nyt_w2v_model.wv))


# 13.NYT Word2Vec 文档表示与分类
#（1）定义文档平均词向量函数，全部 OOV 时返回零向量
def text_to_nyt_w2v_vector(text):
    tokens = tokenize(text)
    valid_vectors = []

    for word in tokens:
        if word in nyt_w2v_model.wv:
            valid_vectors.append(nyt_w2v_model.wv[word])

    # 所有词都找不到向量时，返回全零向量
    if len(valid_vectors) == 0:
        return np.zeros(
            nyt_w2v_model.wv.vector_size,
            dtype=np.float32
        )

    return np.mean(valid_vectors, axis=0)


#（2）用固定的 NYT 词向量转换三个数据集
X_train_nyt_w2v = np.stack(
    train_df["text"].apply(text_to_nyt_w2v_vector).tolist()
)

X_val_nyt_w2v = np.stack(
    val_df["text"].apply(text_to_nyt_w2v_vector).tolist()
)

X_test_nyt_w2v = np.stack(
    test_df["text"].apply(text_to_nyt_w2v_vector).tolist()
)


#（3）固定词向量，使用验证集选择逻辑回归分类器
best_nyt_w2v_model, best_nyt_w2v_c, best_nyt_w2v_macro_f1 = select_lr_model(
    "NYT Word2Vec", X_train_nyt_w2v, y_train, X_val_nyt_w2v, y_val
)

#（4）评估测试集并输出分类报告
nyt_w2v_test_predictions, nyt_w2v_test_accuracy, nyt_w2v_test_macro_f1 = evaluate_lr_model(
    "NYT Word2Vec 平均词向量", best_nyt_w2v_model, X_test_nyt_w2v, y_test
)

# 14.保存三个词向量方案的实验结果
#（1）汇总最佳 C、验证集 Macro-F1 和测试集指标
embedding_results = pd.DataFrame([
    {
        "method": "GloVe + LR",
        "best_C": best_glove_c,
        "validation_macro_f1": best_glove_macro_f1,
        "test_accuracy": glove_test_accuracy,
        "test_macro_f1": glove_test_macro_f1
    },
    {
        "method": "AG Word2Vec + LR",
        "best_C": best_ag_w2v_c,
        "validation_macro_f1": best_ag_w2v_macro_f1,
        "test_accuracy": ag_w2v_test_accuracy,
        "test_macro_f1": ag_w2v_test_macro_f1
    },
    {
        "method": "NYT Word2Vec + LR",
        "best_C": best_nyt_w2v_c,
        "validation_macro_f1": best_nyt_w2v_macro_f1,
        "test_accuracy": nyt_w2v_test_accuracy,
        "test_macro_f1": nyt_w2v_test_macro_f1
    }
])

#（2）设置保存路径并写入 CSV 文件
embedding_result_path = (
    project_dir / "learning_embedding_results.csv"
)

embedding_results.to_csv(
    embedding_result_path,
    index=False,
    encoding="utf-8-sig"
)

#（3）输出结果汇总表和文件保存路径
print("\n三个词向量方案的实验结果：")
print(embedding_results.to_string(index=False))

print("\n结果已保存到：")
print(embedding_result_path)

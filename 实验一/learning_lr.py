"""五种文本特征共用的逻辑回归调参与评价函数。"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report


# 1.根据验证集选择逻辑回归分类器
def select_lr_model(method, X_train, y_train, X_val, y_val):
    #（1）设置候选 C 和最佳模型记录
    best_model = None
    best_c = None
    best_macro_f1 = -1.0

    #（2）每个候选值分别训练，只使用验证集进行选择
    for c in [0.1, 1.0, 10.0]:
        model = LogisticRegression(C=c, max_iter=3000, solver="lbfgs")
        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
        macro_f1 = f1_score(
            y_val, predictions, average="macro", zero_division=0
        )
        print(
            f"{method} C={c}：验证集准确率={accuracy:.2%}，"
            f"Macro-F1={macro_f1:.2%}"
        )

        # 分数相同时保留先选中的 C
        if macro_f1 > best_macro_f1:
            best_model = model
            best_c = c
            best_macro_f1 = macro_f1

    #（3）返回训练好的最佳分类器、C 和验证集分数
    print(f"\n{method} 选中的 C：{best_c}")
    print(f"对应的验证集 Macro-F1：{best_macro_f1:.2%}")
    return best_model, best_c, best_macro_f1


# 2.评估已选定的分类器
def evaluate_lr_model(method, model, X_test, y_test, show_report=True):
    #（1）只预测测试集，不重新训练或选择参数
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(
        y_test, predictions, average="macro", zero_division=0
    )

    #（2）输出总体指标，按需输出分类报告
    print(f"\n{method} + Logistic Regression 的最终测试结果：")
    print(f"选中的 C：{model.C}")
    print(f"Test Accuracy：{accuracy:.2%}")
    print(f"Test Macro-F1：{macro_f1:.2%}")
    if show_report:
        print("\n测试集分类报告：")
        print(classification_report(
            y_test, predictions, digits=4, zero_division=0
        ))

    #（3）返回结果，供主脚本保存到 CSV
    return predictions, accuracy, macro_f1

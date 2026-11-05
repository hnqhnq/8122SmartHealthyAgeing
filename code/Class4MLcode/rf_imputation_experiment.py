"""
Random Forest 缺失值插补实验

比较四种常见插补方法：
1. Mean：训练集均值插补
2. Median：训练集中位数插补
3. MostFrequent：训练集众数插补
4. KNN：标准化后使用 5 个近邻插补

安装依赖：
    python -m pip install pandas numpy scikit-learn tqdm

使用方法：
1. 修改下方 CSV_PATH，直接填入需要分析的 CSV 文件路径。
2. 运行脚本：
    python rf_imputation_experiment.py

实验原则：
- Diagnosis 是分类标签，其余列是特征。
- 所有方法共用同一次 80%/20% 分层划分。
- 插补器只在训练集上拟合，避免测试集信息泄漏。
- 所有方法使用完全相同的 Random Forest 参数。
- 每次训练均显示真实的树生成进度条。
- 最终只保存 rf_imputation_comparison.csv。
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from tqdm.auto import tqdm


RANDOM_STATE = 42
TEST_SIZE = 0.20
RF_TREES = 300
RF_BATCH_SIZE = 10

CSV_PATH = Path(
    r"C:\Users\booming\Desktop\新建文件夹\demo_mask.csv"
)

# 结果自动保存在 CSV 文件同目录下的 rf_imputation_outputs 文件夹。
OUTPUT_DIR = Path(
    r"C:\Users\booming\Desktop\新建文件夹\rf_imputation_outputs"
)
LABEL_COLUMN = "Diagnosis"

def load_raw_data(data_path: Path, label_column: str):
    """读取数据，把非数值和无穷值统一转换为缺失值。"""
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    data = pd.read_csv(data_path, encoding="utf-8-sig")
    data.columns = data.columns.astype(str).str.strip()

    if label_column not in data.columns:
        raise ValueError(
            f"数据中没有标签列 {label_column!r}。实际列名：{data.columns.tolist()}"
        )

    raw_labels = data[label_column]
    if raw_labels.isna().any():
        missing_count = int(raw_labels.isna().sum())
        raise ValueError(f"标签列有 {missing_count} 个缺失值，请先处理标签缺失。")

    labels = raw_labels.astype(str).str.strip()
    if (labels == "").any():
        raise ValueError("标签列包含空字符串，请先处理标签缺失。")

    X_raw = data.drop(columns=[label_column]).copy()
    if X_raw.shape[1] == 0:
        raise ValueError("除标签列外没有可用特征。")

    X_numeric = X_raw.apply(pd.to_numeric, errors="coerce")
    nonnumeric_cells = int((X_numeric.isna() & X_raw.notna()).sum().sum())
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    all_missing_columns = X_numeric.columns[X_numeric.isna().all()].tolist()
    if all_missing_columns:
        raise ValueError(
            "以下特征整列缺失，无法完成插补：" + ", ".join(all_missing_columns)
        )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    class_counts = labels.value_counts()
    if len(label_encoder.classes_) < 2:
        raise ValueError("Diagnosis 只有一个类别，无法训练分类模型。")
    if class_counts.min() < 2:
        raise ValueError("最少类别的样本数小于 2，无法进行分层划分。")

    missing_cells = int(X_numeric.isna().sum().sum())
    incomplete_rows = int(X_numeric.isna().any(axis=1).sum())

    print("\n数据概况")
    print(f"  样本数：{len(data)}")
    print(f"  特征数：{X_numeric.shape[1]}")
    print(f"  类别计数：{class_counts.to_dict()}")
    print(f"  含缺失值的病例数：{incomplete_rows}")
    print(f"  缺失单元格数：{missing_cells}")
    print(f"  其中由非数值内容产生：{nonnumeric_cells}")

    return X_numeric, y, label_encoder


def build_imputers() -> dict[str, object]:
    """返回需要比较的四种插补方法。"""
    return {
        "Mean": SimpleImputer(strategy="mean"),
        "Median": SimpleImputer(strategy="median"),
        "MostFrequent": SimpleImputer(strategy="most_frequent"),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "imputer",
                    KNNImputer(n_neighbors=5, weights="distance"),
                ),
            ]
        ),
    }


def incremental_tree_counts(total: int, batch_size: int) -> list[int]:
    """生成逐批增加的树数量，最后一个值一定等于 total。"""
    counts = list(range(batch_size, total + 1, batch_size))
    if not counts or counts[-1] != total:
        counts.append(total)
    return counts


def train_random_forest(
    X_train,
    y_train,
    sample_weight,
    method_name: str,
) -> RandomForestClassifier:
    """通过 warm_start 分批增加树，显示每种插补方法的训练进度。"""
    model = RandomForestClassifier(
        n_estimators=1,
        max_features="sqrt",
        min_samples_leaf=2,
        warm_start=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    previous = 0
    with tqdm(
        total=RF_TREES,
        desc=f"{method_name} + RF",
        unit="tree",
    ) as progress:
        for current in incremental_tree_counts(RF_TREES, RF_BATCH_SIZE):
            model.set_params(n_estimators=current)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            progress.update(current - previous)
            previous = current
    return model


def evaluate_model(
    method_name: str,
    model: RandomForestClassifier,
    X_test,
    y_test,
    n_classes: int,
) -> dict[str, float | str]:
    """计算与四模型实验一致的评价指标。"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)

    metrics: dict[str, float | str] = {
        "ImputationMethod": method_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "BalancedAccuracy": balanced_accuracy_score(y_test, predictions),
        "PrecisionMacro": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "RecallMacro": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "F1Macro": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }

    try:
        if n_classes == 2:
            metrics["ROC_AUC"] = roc_auc_score(y_test, probabilities[:, 1])
        else:
            metrics["ROC_AUC"] = roc_auc_score(
                y_test,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        metrics["ROC_AUC"] = np.nan

    return metrics


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y, label_encoder = load_raw_data(CSV_PATH, LABEL_COLUMN)

    # 先划分，再拟合插补器。所有插补方法共用完全相同的样本划分。
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    all_missing_train_columns = X_train_raw.columns[
        X_train_raw.isna().all(axis=0)
    ].tolist()
    if all_missing_train_columns:
        raise ValueError(
            "以下特征在训练集中全部缺失，无法完成插补："
            + ", ".join(all_missing_train_columns)
        )

    print(f"  训练/测试样本数：{len(X_train_raw)}/{len(X_test_raw)}")
    print(f"  训练集缺失单元格：{int(X_train_raw.isna().sum().sum())}")
    print(f"  测试集缺失单元格：{int(X_test_raw.isna().sum().sum())}")
    print(f"  类别编码顺序：{label_encoder.classes_.tolist()}")

    sample_weight = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )
    results = []

    print("\n开始插补实验……")
    for method_name, imputer in build_imputers().items():
        # fit_transform 只使用训练集；测试集只调用 transform。
        X_train = imputer.fit_transform(X_train_raw)
        X_test = imputer.transform(X_test_raw)

        if np.isnan(X_train).any() or np.isnan(X_test).any():
            raise ValueError(f"{method_name} 插补后仍存在缺失值。")

        model = train_random_forest(
            X_train,
            y_train,
            sample_weight,
            method_name,
        )
        results.append(
            evaluate_model(
                method_name,
                model,
                X_test,
                y_test,
                len(label_encoder.classes_),
            )
        )

    comparison = pd.DataFrame(results).sort_values(
        by="F1Macro",
        ascending=False,
    )
    numeric_columns = comparison.select_dtypes(include="number").columns
    comparison[numeric_columns] = comparison[numeric_columns].round(6)

    output_path = OUTPUT_DIR / "rf_imputation_comparison.csv"
    comparison.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\n插补方法对比（按宏平均 F1 从高到低排序）")
    print(
        comparison.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )
    best_method = comparison.iloc[0]["ImputationMethod"]
    print(f"\n本次固定划分下，F1Macro 最高的插补方法：{best_method}")
    print(f"汇总结果已保存到：{output_path.resolve()}")


if __name__ == "__main__":
    main()

"""
血液指标分类教学：LightGBM、Random Forest、XGBoost、GradientBoosting

安装依赖：
    python -m pip install pandas numpy scikit-learn lightgbm xgboost tqdm matplotlib

直接运行（脚本已将数据路径设为默认值）：
    python blood_classification_demo.py

指定数据和输出目录：
    python blood_classification_demo.py ^
        --data "C:\\Users\\booming\\Desktop\\新建文件夹\\demo.csv" ^
        --output "C:\\Users\\booming\\Desktop\\新建文件夹\\ml_outputs"

说明：
1. Diagnosis 是分类标签，其余列是特征。
2. 数据按 80%/20% 分为训练集和测试集，并保持类别比例。
3. 默认删除任何含缺失值、无穷值或非数值指标的病例；也可选择训练集中位数填补。
4. 四个模型训练时都会显示 tqdm 进度条。
5. 测试集只用于最终评价，不参与训练、调参或早停。
6. 最终只保存 model_comparison.csv 和四张单模型混淆矩阵 PNG。
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from tqdm.auto import tqdm


RANDOM_STATE = 42
TEST_SIZE = 0.20

RF_TREES = 300
RF_BATCH_SIZE = 10

GB_TREES = 200
GB_BATCH_SIZE = 5

LGBM_ROUNDS = 500
XGB_ROUNDS = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用四个树模型完成血液指标分类")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(r"C:\Users\booming\Desktop\新建文件夹\demo.csv"),
        help="输入 CSV 文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(r"C:\Users\booming\Desktop\新建文件夹\ml_outputs"),
        help="模型和评价结果的保存目录",
    )
    parser.add_argument(
        "--label",
        default="Diagnosis",
        help="标签列名，默认是 Diagnosis",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["drop", "median"],
        default="drop",
        help="缺失/非数值指标处理方式：drop=删除整行（默认），median=训练集中位数填补",
    )
    return parser.parse_args()


def load_and_split_data(
    data_path: Path,
    label_column: str,
    missing_policy: str,
):
    """读取数据、检查标签、转换数值特征，并完成分层划分。"""
    if not data_path.exists():
        raise FileNotFoundError(f"找不到数据文件：{data_path}")

    data = pd.read_csv(data_path, encoding="utf-8-sig")
    data.columns = data.columns.astype(str).str.strip()

    if label_column not in data.columns:
        raise ValueError(
            f"数据中没有标签列 {label_column!r}。实际列名：{data.columns.tolist()}"
        )
    if data[label_column].isna().any():
        missing_count = int(data[label_column].isna().sum())
        raise ValueError(f"标签列有 {missing_count} 个缺失值，请先处理标签缺失。")

    y_text = data[label_column].astype(str).str.strip()
    if (y_text == "").any():
        raise ValueError("标签列包含空字符串，请先处理标签缺失。")

    X_raw = data.drop(columns=[label_column]).copy()
    if X_raw.shape[1] == 0:
        raise ValueError("除标签列外没有可用特征。")

    original_sample_count = len(data)

    # 非数值内容和无穷值统一标记为 NaN。
    X_numeric = X_raw.apply(pd.to_numeric, errors="coerce")
    newly_missing = int((X_numeric.isna() & X_raw.notna()).sum().sum())
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)

    invalid_row_mask = X_numeric.isna().any(axis=1)
    invalid_row_count = int(invalid_row_mask.sum())
    if missing_policy == "drop" and invalid_row_count:
        X_numeric = X_numeric.loc[~invalid_row_mask].copy()
        y_text = y_text.loc[~invalid_row_mask].copy()

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    class_counts = pd.Series(y_text).value_counts()
    if len(label_encoder.classes_) < 2:
        raise ValueError("Diagnosis 只有一个类别，无法训练分类模型。")
    if class_counts.min() < 5:
        raise ValueError(
            "最少类别的样本数小于 5，无法稳定地进行三段分层划分。"
        )

    # 严格按 80%/20% 分层划分训练集和测试集。
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_numeric,
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
            "以下特征在训练集中全部缺失，无法进行中位数填补："
            + ", ".join(all_missing_train_columns)
        )

    imputer = SimpleImputer(strategy="median")
    feature_names = X_numeric.columns.tolist()
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw),
        columns=feature_names,
        index=X_train_raw.index,
    )
    X_test = pd.DataFrame(
        imputer.transform(X_test_raw),
        columns=feature_names,
        index=X_test_raw.index,
    )

    print("\n数据概况")
    print(f"  原始样本数：{original_sample_count}")
    print(f"  建模样本数：{len(X_numeric)}")
    print(f"  特征数：{len(feature_names)}")
    print(f"  类别：{label_encoder.classes_.tolist()}")
    print(f"  类别计数：{class_counts.to_dict()}")
    print(f"  训练/测试：{len(X_train)}/{len(X_test)}")
    if invalid_row_count:
        if missing_policy == "drop":
            print(
                f"  已删除 {invalid_row_count} 行含缺失、无穷值或非数值指标的病例"
                f"（其中 {newly_missing} 个单元格由非数值内容产生）。"
            )
        else:
            print(
                f"  有 {invalid_row_count} 行含缺失、无穷值或非数值指标；"
                f"将仅使用训练集的中位数填补（其中 {newly_missing} 个单元格由非数值内容产生）。"
            )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        label_encoder,
        imputer,
    )


def incremental_tree_counts(total: int, batch_size: int) -> list[int]:
    """生成逐批增加的树数量，最后一个值一定等于 total。"""
    counts = list(range(batch_size, total + 1, batch_size))
    if not counts or counts[-1] != total:
        counts.append(total)
    return counts


def train_random_forest(X_train, y_train, sample_weight):
    """通过 warm_start 分批增加树，为随机森林显示真实训练进度。"""
    model = RandomForestClassifier(
        n_estimators=1,
        max_features="sqrt",
        min_samples_leaf=2,
        warm_start=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    previous = 0
    with tqdm(total=RF_TREES, desc="Random Forest", unit="tree") as progress:
        for current in incremental_tree_counts(RF_TREES, RF_BATCH_SIZE):
            model.set_params(n_estimators=current)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            progress.update(current - previous)
            previous = current
    return model


def train_gradient_boosting(X_train, y_train, sample_weight):
    """通过 warm_start 分批增加树，为 GradientBoosting 显示训练进度。"""
    model = GradientBoostingClassifier(
        n_estimators=1,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        subsample=0.9,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    previous = 0
    with tqdm(total=GB_TREES, desc="GradientBoosting", unit="tree") as progress:
        for current in incremental_tree_counts(GB_TREES, GB_BATCH_SIZE):
            model.set_params(n_estimators=current)
            model.fit(X_train, y_train, sample_weight=sample_weight)
            progress.update(current - previous)
            previous = current
    return model


def make_lightgbm_progress_callback(progress: tqdm):
    """把 LightGBM 每一轮 boosting 更新到 tqdm。"""

    def callback(environment):
        progress.update(1)
        if environment.evaluation_result_list:
            dataset, metric, value, _ = environment.evaluation_result_list[-1][:4]
            progress.set_postfix_str(f"{dataset}-{metric}={value:.4f}")

    callback.order = 10
    callback.before_iteration = False
    return callback


def train_lightgbm(
    X_train,
    y_train,
    sample_weight,
    n_classes: int,
):
    """在完整训练集上训练 LightGBM，并显示每轮 boosting 的进度。"""
    objective = "binary" if n_classes == 2 else "multiclass"
    model = lgb.LGBMClassifier(
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        n_estimators=LGBM_ROUNDS,
        learning_rate=0.03,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    progress = tqdm(total=LGBM_ROUNDS, desc="LightGBM", unit="round")
    try:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weight,
            callbacks=[make_lightgbm_progress_callback(progress)],
        )
    finally:
        # 早停时把进度条总数改成实际完成轮数，让完成状态清晰可见。
        if progress.n < progress.total:
            progress.total = max(progress.n, 1)
            progress.refresh()
        progress.close()
    return model


class XGBoostProgressCallback(xgb.callback.TrainingCallback):
    """XGBoost 原生训练回调：每轮更新 tqdm。"""

    def __init__(self, progress: tqdm):
        self.progress = progress

    def after_iteration(self, model, epoch: int, evals_log) -> bool:
        self.progress.update(1)
        if "valid" in evals_log and evals_log["valid"]:
            metric_name = next(reversed(evals_log["valid"]))
            metric_history = evals_log["valid"][metric_name]
            value = metric_history[-1]
            if isinstance(value, tuple):
                value = value[0]
            self.progress.set_postfix_str(f"valid-{metric_name}={float(value):.4f}")
        return False


def train_xgboost(
    X_train,
    y_train,
    sample_weight,
    feature_names: list[str],
    n_classes: int,
):
    """使用 xgb.train 训练 XGBoost，便于稳定地接入逐轮进度回调。"""
    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        weight=sample_weight,
        feature_names=feature_names,
    )
    params = {
        "objective": "binary:logistic" if n_classes == 2 else "multi:softprob",
        "eval_metric": "logloss" if n_classes == 2 else "mlogloss",
        "eta": 0.03,
        "max_depth": 4,
        "min_child_weight": 2,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "lambda": 1.0,
        "alpha": 0.1,
        "tree_method": "hist",
        "seed": RANDOM_STATE,
        "nthread": -1,
    }
    if n_classes > 2:
        params["num_class"] = n_classes

    progress = tqdm(total=XGB_ROUNDS, desc="XGBoost", unit="round")
    try:
        booster = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=XGB_ROUNDS,
            verbose_eval=False,
            callbacks=[XGBoostProgressCallback(progress)],
        )
    finally:
        if progress.n < progress.total:
            progress.total = max(progress.n, 1)
            progress.refresh()
        progress.close()
    return booster


def predict_xgboost(
    booster,
    X_test,
    feature_names: list[str],
    n_classes: int,
):
    """使用全部训练轮数进行 XGBoost 预测。"""
    dtest = xgb.DMatrix(X_test, feature_names=feature_names)
    best_iteration = getattr(booster, "best_iteration", XGB_ROUNDS - 1)

    try:
        raw_prediction = booster.predict(
            dtest,
            iteration_range=(0, int(best_iteration) + 1),
        )
    except TypeError:
        # 兼容较旧版本 XGBoost。
        best_ntree_limit = getattr(booster, "best_ntree_limit", 0)
        raw_prediction = booster.predict(dtest, ntree_limit=best_ntree_limit)

    if n_classes == 2:
        positive_probability = np.asarray(raw_prediction)
        probabilities = np.column_stack(
            [1.0 - positive_probability, positive_probability]
        )
        predictions = (positive_probability >= 0.5).astype(int)
    else:
        probabilities = np.asarray(raw_prediction)
        predictions = probabilities.argmax(axis=1)
    return predictions, probabilities


def evaluate_predictions(
    model_name: str,
    y_true,
    y_pred,
    probabilities,
    class_names: list[str],
    output_dir: Path,
):
    """计算评价指标，并只保存该模型的混淆矩阵图片。"""
    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "PrecisionMacro": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "RecallMacro": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "F1Macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    try:
        if len(class_names) == 2:
            metrics["ROC_AUC"] = roc_auc_score(y_true, probabilities[:, 1])
        else:
            metrics["ROC_AUC"] = roc_auc_score(
                y_true,
                probabilities,
                multi_class="ovr",
                average="macro",
            )
    except ValueError:
        metrics["ROC_AUC"] = np.nan

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    safe_name = model_name.replace(" ", "_")
    matrix = confusion_matrix(y_true, y_pred)
    matrix_df = pd.DataFrame(
        matrix,
        index=[f"Actual_{name}" for name in class_names],
        columns=[f"Predicted_{name}" for name in class_names],
    )

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )
    axis.set_title(f"{model_name} Confusion Matrix")
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("Actual class")
    figure.tight_layout()
    figure.savefig(
        output_dir / f"confusion_matrix_{safe_name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"\n{model_name} 分类报告")
    print(report)
    print("混淆矩阵：")
    print(matrix_df)
    return metrics


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        label_encoder,
        _imputer,
    ) = load_and_split_data(args.data, args.label, args.missing_policy)

    class_names = [str(name) for name in label_encoder.classes_]
    n_classes = len(class_names)
    train_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    results = []

    print("\n开始训练四个模型……")

    # 1. Random Forest
    random_forest = train_random_forest(X_train, y_train, train_weights)
    rf_pred = random_forest.predict(X_test)
    rf_prob = random_forest.predict_proba(X_test)
    results.append(
        evaluate_predictions(
            "Random Forest",
            y_test,
            rf_pred,
            rf_prob,
            class_names,
            args.output,
        )
    )
    # 2. GradientBoosting
    gradient_boosting = train_gradient_boosting(
        X_train, y_train, train_weights
    )
    gb_pred = gradient_boosting.predict(X_test)
    gb_prob = gradient_boosting.predict_proba(X_test)
    results.append(
        evaluate_predictions(
            "GradientBoosting",
            y_test,
            gb_pred,
            gb_prob,
            class_names,
            args.output,
        )
    )
    # 3. LightGBM
    lightgbm_model = train_lightgbm(
        X_train,
        y_train,
        train_weights,
        n_classes,
    )
    lgbm_pred = lightgbm_model.predict(X_test).astype(int)
    lgbm_prob = lightgbm_model.predict_proba(X_test)
    results.append(
        evaluate_predictions(
            "LightGBM",
            y_test,
            lgbm_pred,
            lgbm_prob,
            class_names,
            args.output,
        )
    )
    # 4. XGBoost
    xgboost_model = train_xgboost(
        X_train,
        y_train,
        train_weights,
        feature_names,
        n_classes,
    )
    xgb_pred, xgb_prob = predict_xgboost(
        xgboost_model,
        X_test,
        feature_names,
        n_classes,
    )
    results.append(
        evaluate_predictions(
            "XGBoost",
            y_test,
            xgb_pred,
            xgb_prob,
            class_names,
            args.output,
        )
    )
    # 最终只保存四模型汇总结果 CSV；各模型混淆矩阵 PNG 已在评价时保存。
    results_df = pd.DataFrame(results).sort_values(
        by="F1Macro", ascending=False
    )
    results_df.to_csv(
        args.output / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n模型最终对比（按宏平均 F1 从高到低排序）")
    print(results_df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\n所有结果已保存到：{args.output.resolve()}")


if __name__ == "__main__":
    main()

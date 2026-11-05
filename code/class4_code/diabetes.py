
import sys; sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd                     # pandas：读表格、看表格
import matplotlib.pyplot as plt         # matplotlib：画图
from sklearn.model_selection import train_test_split          # 把数据切成训练集和测试集
from sklearn.linear_model import LogisticRegression           # 模型一：逻辑回归
from sklearn.ensemble import RandomForestClassifier           # 模型二：随机森林
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay   # 混淆矩阵：算数、画图

df = pd.read_csv('diabetes_clinic.csv')

print(df.head())

print('\n患病 / 未患病人数:\n', df['diabetes'].value_counts())

print('\n两组人各项指标的平均值:\n', df.groupby('diabetes').mean().round(1).T)


X = df.drop(columns=['diabetes'])
y = df['diabetes']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


models = {'Logistic regression': LogisticRegression(max_iter=1000),
          'Random forest': RandomForestClassifier(random_state=0)}

# 准备一张大图，横着放 3 个小图：两个混淆矩阵 + 一个特征重要性
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, model) in zip(axes, models.items()):
    model.fit(X_tr, y_tr)               # fit = 训练：让模型从训练集里学
    pred = model.predict(X_te)          # predict = 预测：对测试集里的每个人给出 0 或 1

    # 混淆矩阵把预测结果分成四类：
    #   tn 真阴性：没病，模型也说没病       fp 假阳性：没病，模型说有病
    #   fn 假阴性：有病，模型说没病（漏诊）  tp 真阳性：有病，模型也说有病
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()

    acc = (tp + tn) / len(y_te)         # 准确率：所有人里判对的比例
    sens = tp / (tp + fn)               # 敏感度：真正有病的人里，被找出来的比例
    print(f'\n{name}: 准确率 {acc:.0%}，敏感度 {sens:.0%}（{tp+fn} 个病人里找出 {tp} 个）')

    ConfusionMatrixDisplay.from_predictions(y_te, pred, display_labels=['healthy', 'diabetes'],
                                            ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'{name}\naccuracy {acc:.0%}, sensitivity {sens:.0%}')


imp = pd.Series(models['Random forest'].feature_importances_, index=X.columns).sort_values()
print('\n特征重要性:\n', imp[::-1].round(3))

imp.plot.barh(ax=axes[2], color='tab:orange')
axes[2].set_title('Which features does the forest rely on?')
axes[2].set_xlabel('importance')

fig.suptitle(f'Features used: {", ".join(X.columns)}', fontsize=9)
fig.tight_layout()
fig.savefig('results.png', dpi=120)
print('\n图已保存为 results.png')
plt.show()

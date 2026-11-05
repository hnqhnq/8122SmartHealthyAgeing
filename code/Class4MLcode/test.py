python -m pip install --upgrade pip
python -m pip install pandas numpy scikit-learn lightgbm xgboost tqdm joblib
python -m pip install matplotlib

import platform
import sys

import matplotlib
import numpy
import pandas
import sklearn


print("=" * 50)
print("机器学习课程环境检查")
print("=" * 50)
print("操作系统：", platform.platform())
print("Python 版本：", sys.version)
print("Python 路径：", sys.executable)
print("numpy 版本：", numpy.__version__)
print("pandas 版本：", pandas.__version__)
print("scikit-learn 版本：", sklearn.__version__)
print("matplotlib 版本：", matplotlib.__version__)
print("\n环境配置成功，可以开始机器学习实验。")



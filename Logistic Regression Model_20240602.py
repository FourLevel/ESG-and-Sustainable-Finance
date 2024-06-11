import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
from matplotlib.font_manager import fontManager
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 設置字體
for i in sorted(fontManager.get_font_names()):
    print(i)
matplotlib.rc('font', family='Microsoft JhengHei')  # 將繪圖字型改為微軟正黑體

# 讀取數據
data = pd.read_csv("D:\\ESG\全公司_補抓_20240602_selected.csv")

# 重新定義自變數 (X) 和應變數 (Y)
X = data.drop(columns=['代號', '名稱', '取樣年 T-1', 'TSE 產業別', 'TSE新產業_名稱', '是否違約', '會計師事務所', '是否投保董監責任險(Y/N)'])
Y = data['是否違約']

# 分離數值型變數和分類變數
numerical_features = ['總負債/總淨值', '稅前純益/實收資本', '營業利益率', '稅後淨利成長率', '存貨週轉率（次）', 'TESG分數', '環境構面分數', '社會構面分數', '公司治理構面分數',
                      '資產總額', 'ROA稅後息前', 'ROE稅後', '每股淨值',	'每股現金流量',	'營收成長率', '營業毛利成長率',	'營業利益/實收資本比', '折舊合計', 'ROA稅後息前折舊前']
categorical_features = ['是否為四大', '是否投保董監責任險']

# 標準化數值型變數
scaler = StandardScaler()
X_numerical_standardized = scaler.fit_transform(X[numerical_features])

# 將標準化後的數值型變數轉換為數據框
standardized_column_names = [f"{col}_標準化" for col in numerical_features]
X_numerical_standardized_df = pd.DataFrame(X_numerical_standardized, columns=standardized_column_names)


# 獲取分類變數
X_categorical_df = pd.get_dummies(X[categorical_features])

# 合併原始數據和標準化後數據
X_combined = pd.concat([X[numerical_features], X_numerical_standardized_df, X[categorical_features]], axis=1)

# 顯示合併後的數據框
X_combined.head()

# 為自變數 X 添加截距項
X_combined = sm.add_constant(X_combined)

# 設定不同模型的變數
variables_model_1 = ['TESG分數_標準化', '公司治理構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化',
                     '是否為四大', '是否投保董監責任險']
variables_model_2 = ['TESG分數_標準化', '環境構面分數_標準化', '社會構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化',
                     '是否為四大']
variables_model_3 = ['TESG分數_標準化', '環境構面分數_標準化', '社會構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化', '折舊合計_標準化',
                     '是否為四大']
numerical_variables_model_1 = ['TESG分數_標準化', '公司治理構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化']
numerical_variables_model_2 = ['TESG分數_標準化', '環境構面分數_標準化', '社會構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化']
numerical_variables_model_3 = ['TESG分數_標準化', '環境構面分數_標準化', '社會構面分數_標準化', '總負債/總淨值_標準化', 'ROA稅後息前_標準化', '折舊合計_標準化']


# 計算 VIF 值的函數
def calculate_vif(df, variables):
    vif = pd.DataFrame()
    vif["變數"] = variables
    vif["VIF"] = [variance_inflation_factor(df[variables].values, i) for i in range(len(variables))]
    return vif

# 訓練和評估模型
def train_and_evaluate(X, Y, threshold=0.5):
    X = sm.add_constant(X)
    model = sm.Logit(Y, X)
    result = model.fit(disp=0)  # 設置 disp=0 以抑制輸出
    Y_pred_prob = result.predict(X)
    Y_pred = Y_pred_prob >= threshold
    summary = result.summary()
    accuracy = accuracy_score(Y, Y_pred)
    report = classification_report(Y, Y_pred)
    conf_matrix = confusion_matrix(Y, Y_pred)
    return summary, accuracy, report, Y_pred_prob, conf_matrix

# 繪製ROC曲線
def plot_roc_curve(Y, Y_pred_prob, model_name):
    fpr, tpr, _ = roc_curve(Y, Y_pred_prob)
    roc_auc = roc_auc_score(Y, Y_pred_prob)
    
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUROC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title(f'ROC Curve and AUROC - {model_name}', fontsize=18)
    plt.legend(loc="lower right", fontsize=12)
    plt.show()

# 繪製Confusion Matrix
def plot_confusion_matrix(conf_matrix, model_name):
    plt.figure(figsize=(8, 6), dpi=200)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'], cbar=False, annot_kws={'size': 36})
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predicted label', fontsize=16)
    plt.title(f'Confusion Matrix - {model_name}', fontsize=18)
    plt.show()



# 評估 Model 1，ESG 係數為正
X_1 = X_combined[variables_model_1]
vif_model_1 = calculate_vif(X_combined, numerical_variables_model_1)
summary1, accuracy1, report1, Y_pred_prob1, conf_matrix1 = train_and_evaluate(X_1, Y, threshold=0.3)
print(vif_model_1, "\n")
print(summary1, "\n")
print("Model 1 Accuracy:", accuracy1, "\n")
print("Model 1 Classification Report:\n", report1)
plot_roc_curve(Y, Y_pred_prob1, "Model 1")
plot_confusion_matrix(conf_matrix1, "Model 1")

# 評估 Model 2，ESG 係數為負
X_2 = X_combined[variables_model_2]
vif_model_2 = calculate_vif(X_combined, numerical_variables_model_2)
summary2, accuracy2, report2, Y_pred_prob2, conf_matrix2 = train_and_evaluate(X_2, Y, threshold=0.3)
print(vif_model_2, "\n")
print(summary2, "\n")
print("Model 2 Accuracy:", accuracy2, "\n")
print("Model 2 Classification Report:\n", report2)
plot_roc_curve(Y, Y_pred_prob2, "Model 2")
plot_confusion_matrix(conf_matrix2, "Model 2")

# 評估 Model 3，ESG 係數為負
X_3 = X_combined[variables_model_3]
vif_model_3 = calculate_vif(X_combined, numerical_variables_model_3)
summary3, accuracy3, report3, Y_pred_prob3, conf_matrix3 = train_and_evaluate(X_3, Y, threshold=0.25)
print(vif_model_3, "\n")
print(summary3, "\n")
print("Model 3 Accuracy:", accuracy3, "\n")
print("Model 3 Classification Report:\n", report3)
plot_roc_curve(Y, Y_pred_prob3, "Model 3")
plot_confusion_matrix(conf_matrix3, "Model 3")
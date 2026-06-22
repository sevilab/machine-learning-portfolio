# Gerekli kütüphaneler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve



# Veriyi yükleme
file_path = 'C:/Users/HP/Desktop/4.0/E Commerce Dataset .xlsx'
excel_data = pd.ExcelFile(file_path)
data = excel_data.parse('E Comm')

# İlk 5 satıra göz atalım
print(data.head())

# Genel istatistikler
print(data.describe())
print(data.info())

# Eksik veri kontrolü
missing_data = data.isnull().sum()
print(missing_data)

# Eksik verileri doldurma
data['Tenure'].fillna(data['Tenure'].median(), inplace=True)
data['HourSpendOnApp'].fillna(data['HourSpendOnApp'].mean(), inplace=True)
data['WarehouseToHome'].fillna(data['WarehouseToHome'].median(), inplace=True)
data['OrderAmountHikeFromlastYear'].fillna(data['OrderAmountHikeFromlastYear'].median(), inplace=True)
data['CouponUsed'].fillna(data['CouponUsed'].median(), inplace=True)
data['OrderCount'].fillna(data['OrderCount'].median(), inplace=True)
data['DaySinceLastOrder'].fillna(data['DaySinceLastOrder'].median(),inplace=True)


# Eksik veri kontrolü
missing_data = data.isnull().sum()
print(missing_data)

# Cinsiyet ve müşteri kaybı (Churn) arasındaki ilişki
sns.countplot(data=data, x='Gender', hue='Churn')
plt.title("Gender vs Churn")
plt.show()

# Tercih edilen ödeme yöntemi ve müşteri kaybı
sns.countplot(data=data, x='PreferredPaymentMode', hue='Churn')
plt.title("Preferred Payment Mode vs Churn")
plt.xticks(rotation=45)
plt.show()

# Sayısal değişkenlerin dağılım grafiği
data[['Tenure', 'HourSpendOnApp', 'OrderCount', 'OrderAmountHikeFromlastYear']].hist(bins=15, figsize=(10, 8))
plt.show()

# Kategorik verileri etiketleme
label_encoder = LabelEncoder()
categorical_cols = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 'MaritalStatus', 'PreferedOrderCat']
for col in categorical_cols:
    data[col] = label_encoder.fit_transform(data[col])
    
# Korelasyon matrisi
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Hedef ve özellikler
X = data.drop(columns=['CustomerID', 'Churn'])
y = data['Churn']

# Veriyi ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Veriyi bölme
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)



# Model oluşturma ve eğitim
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Özellik Önem Düzeyi Görselleştirme
feature_importances = pd.Series(model.feature_importances_, index=data.drop(columns=['CustomerID', 'Churn']).columns)
plt.figure(figsize=(10, 6))
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Rapor ve metrikler
print(classification_report(y_test, y_pred))

# ROC AUC Skoru
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"ROC AUC Score: {roc_auc}")

# ROC Eğrisi
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, color='orange', label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



# Modelin önemli metrikleri
from sklearn.metrics import f1_score, precision_score, recall_score

f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"F1 Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")



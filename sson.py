
"""
SINIFLANDIRMA ÇALIŞMASI
Meme Kanseri Teşhisi — Breast Cancer Wisconsin (Diagnostic)
"""
#Kütüphanelerin yüklenmesi
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# 1. VERİ YÜKLEME
#wdbc.data dosyası bir CSV dosyasıdır ancak içinde başlıklar (sütun isimleri) yazmıyor
#Bu yüzden sütun isimleri elle liste şeklinde oluşturuldu
sutun_isimleri = [
    'id', 'teshis',
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dim_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se',
    'smoothness_se', 'compactness_se', 'concavity_se',
    'concave_points_se', 'symmetry_se', 'fractal_dim_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst',
    'smoothness_worst', 'compactness_worst', 'concavity_worst',
    'concave_points_worst', 'symmetry_worst', 'fractal_dim_worst'
]

df = pd.read_csv('C:/Users/HP/Desktop/YL/veribilimi/proje/wdbc.data',
                 header=None, names=sutun_isimleri)
#header=None: Dosyada başlık satırı olmadığını bildirir
df = df.drop(columns=['id'])
#id sütunu hastaların kimlik numarasıdır kanser olup olmadığını etkilemez

print("=" * 50)
print("VERİ SETİ GENEL BİLGİLER")
print("=" * 50)
print(f"Satır / Sütun : {df.shape[0]} / {df.shape[1]}")
print(f"Eksik değer   : {df.isnull().sum().sum()}")
print(f"\nSınıf dağılımı:\n{df['teshis'].value_counts()}")

# 2. KEŞİFSEL VERİ ANALİZİ (EDA)

# Sınıf dağılımının incelenmesi
plt.figure(figsize=(6, 4))
df['teshis'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Tümör Sınıf Dağılımı')
plt.xlabel('Teşhis (B=İyi Huylu, M=Kötü Huylu)')
plt.ylabel('Hasta Sayısı')
plt.xticks(rotation=0)
plt.tight_layout(); plt.show()

# Korelasyon matrisi (mean özellikler)
mean_kolonlar = [col for col in df.columns if 'mean' in col]

plt.figure(figsize=(12, 8))
sns.heatmap(df[mean_kolonlar].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', linewidths=0.5)
plt.title('Özellikler Arası Korelasyon (Mean Değerler)')
plt.tight_layout(); plt.show()

# Kutu grafikleri (B vs M)
onemli_ozellikler = [
    'radius_mean', 'texture_mean', 'area_mean',
    'concavity_mean', 'symmetry_mean', 'compactness_mean'
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, ozellik in zip(axes.flatten(), onemli_ozellikler):
    df.boxplot(column=ozellik, by='teshis', ax=ax,
               boxprops=dict(color='steelblue'),
               medianprops=dict(color='red'))
    ax.set_title(ozellik)
    ax.set_xlabel('Teşhis')
plt.suptitle('B vs M — Özellik Karşılaştırması')
plt.tight_layout(); plt.show()

# 3. ÖN İŞLEME & TRAIN/TEST SPLIT

X = df.drop(columns=['teshis'])#tahmin edeceğimiz değişken
y = LabelEncoder().fit_transform(df['teshis'])   # B=0, M=1

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #%80 eğitim %20 test  

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

print(f"\nEğitim: {X_train.shape[0]} örnek  |  Test: {X_test.shape[0]} örnek")

# 4. MODEL EĞİTİMİ & DEĞERLENDİRME

modeller = {
    'Logistic Regression': LogisticRegression(max_iter=10000, random_state=42),
    'KNN'                : KNeighborsClassifier(n_neighbors=5),
    'Random Forest'      : RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM'                : SVC(kernel='rbf', random_state=42)
}

for isim, model in modeller.items():
    model.fit(X_train, y_train)

# --- Classification report ---
for isim, model in modeller.items():
    y_pred = model.predict(X_test)
    print(f"\n{'='*50}")
    print(f"  {isim}  —  Accuracy: %{accuracy_score(y_test, y_pred)*100:.2f}")
    print('='*50)
    print(classification_report(y_test, y_pred,
                                target_names=['Benign (B)', 'Malignant (M)']))

# --- Confusion matrix ---
fig, axes = plt.subplots(1, 4, figsize=(20, 4))
for ax, (isim, model) in zip(axes, modeller.items()):
    y_pred = model.predict(X_test)
    sns.heatmap(confusion_matrix(y_test, y_pred),
                annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f'{isim}\nAcc: %{accuracy_score(y_test, y_pred)*100:.1f}')
    ax.set_xlabel('Tahmin'); ax.set_ylabel('Gerçek')
plt.suptitle('Confusion Matrix Karşılaştırması', fontsize=14)
plt.tight_layout(); plt.show()

# --- Accuracy karşılaştırma ---
isimler     = list(modeller.keys())
accuracyler = [accuracy_score(y_test, m.predict(X_test)) * 100
               for m in modeller.values()]

plt.figure(figsize=(8, 5))
bars = plt.bar(isimler, accuracyler,
               color=['steelblue', 'tomato', 'seagreen', 'orange'])
plt.ylim(90, 100)
plt.ylabel('Doğruluk (%)')
plt.title('Model Karşılaştırması — Accuracy')
for bar, acc in zip(bars, accuracyler):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'%{acc:.2f}', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout(); plt.show()

# 5. CROSS-VALIDATION

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

print("\n" + "=" * 50)
print("CROSS-VALIDATION SONUÇLARI (10-Fold)")
print("=" * 50)

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

cv_sonuclar = {}
for isim, model in modeller.items():
    # Scaler + model birlikte pipeline'a alınıyor (veri sızıntısı önlenir)
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    cv_sonuclar[isim] = scores
    print(f"\n{isim}")
    print(f"  Fold Skorları : {[f'%{s*100:.1f}' for s in scores]}")
    print(f"  Ortalama      : %{scores.mean()*100:.2f}")
    print(f"  Std Sapma     : ±%{scores.std()*100:.2f}")

# --- CV Görselleştirme ---
plt.figure(figsize=(10, 5))
plt.boxplot(
    [cv_sonuclar[isim] * 100 for isim in cv_sonuclar],
    labels=list(cv_sonuclar.keys()),
    patch_artist=True,
    boxprops=dict(facecolor='steelblue', color='navy'),
    medianprops=dict(color='red', linewidth=2)
)
plt.ylabel('Accuracy (%)')
plt.title('Cross-Validation Sonuçları (10-Fold)')
plt.ylim(88, 100)
plt.tight_layout()
plt.show()
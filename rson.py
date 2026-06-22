
"""
REGRESYON ÇALIŞMASI
Perakende Sektöründe Günlük Gelir Tahmini
Veri seti: Online Retail
"""

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

# 1. VERİ YÜKLEME

df = pd.read_excel('C:/Users/HP/Desktop/YL/veribilimi/proje/Online Retail.xlsx')

print("=" * 50)
print("VERİ SETİ GENEL BİLGİLERİ")
print("=" * 50)
print(f"Satır / Sütun : {df.shape[0]:,} / {df.shape[1]}")
print(f"\nEksik değerler:\n{df.isnull().sum()}")
print(f"\nİstatistiksel özet:\n{df.describe().round(2)}")

# 2. VERİ TEMİZLEME

onceki = df.shape[0]

df = df[df['Quantity'] > 0]                                      # iadeler
df = df[df['UnitPrice'] > 0]                                     # geçersiz fiyat
df = df.dropna(subset=['CustomerID'])                            # eksik müşteri
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]        # iptal faturalar

print(f"\nTemizleme : {onceki:,} → {df.shape[0]:,} satır  "
      f"(kaldırılan: {onceki - df.shape[0]:,})")

# Türetilen sütunlar
df['TotalRevenue'] = df['Quantity'] * df['UnitPrice']
df['Month']        = df['InvoiceDate'].dt.month
df['DayOfWeek']    = df['InvoiceDate'].dt.dayofweek
df['Hour']         = df['InvoiceDate'].dt.hour
df['IsWeekend']    = (df['DayOfWeek'] >= 5).astype(int)
df['Date']         = df['InvoiceDate'].dt.date

# 3. GÜNLÜK FEATURE ENGINEERING

daily = df.groupby('Date').agg(
    NumTransactions = ('InvoiceNo',    'nunique'),
    NumItems        = ('Quantity',     'sum'),
    TotalRevenue    = ('TotalRevenue', 'sum'),
    NumCustomers    = ('CustomerID',   'nunique'),
    NumProducts     = ('StockCode',    'nunique'),
    AvgUnitPrice    = ('UnitPrice',    'mean'),
    AvgQuantity     = ('Quantity',     'mean'),
    Month           = ('Month',        'first'),
    DayOfWeek       = ('DayOfWeek',    'first'),
    IsWeekend       = ('IsWeekend',    'first'),
).reset_index()

daily['RevenuePerCustomer']    = daily['TotalRevenue'] / daily['NumCustomers']
daily['RevenuePerTransaction'] = daily['TotalRevenue'] / daily['NumTransactions']
daily['ItemsPerTransaction']   = daily['NumItems']     / daily['NumTransactions']

print(f"\nGünlük veri : {daily.shape[0]} gün, {daily.shape[1]} özellik")

# 4. KEŞİFSEL VERİ ANALİZİ (EDA)

# --- Dağılım grafikleri ---
kolonlar = ['TotalRevenue', 'NumTransactions', 'NumItems',
            'NumCustomers', 'NumProducts', 'AvgUnitPrice']

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Günlük Satış Değişkenlerinin Dağılımı', fontsize=14, fontweight='bold')
for ax, col in zip(axes.flatten(), kolonlar):
    ax.hist(daily[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_title(col); ax.set_xlabel('Değer'); ax.set_ylabel('Frekans')
plt.tight_layout(); plt.savefig('dagilim.png', dpi=150); plt.show()

# --- Korelasyon matrisi ---
corr_cols = ['TotalRevenue', 'NumTransactions', 'NumItems', 'NumCustomers',
             'NumProducts', 'AvgUnitPrice', 'RevenuePerCustomer',
             'RevenuePerTransaction', 'ItemsPerTransaction', 'Month', 'DayOfWeek']
corr = daily[corr_cols].corr()

plt.figure(figsize=(12, 9))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            mask=np.triu(np.ones_like(corr, dtype=bool)),
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Korelasyon Matrisi', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.savefig('korelasyon.png', dpi=150); plt.show()

print("\nTotalRevenue ile korelasyonlar (sıralı):")
print(corr['TotalRevenue'].drop('TotalRevenue').sort_values(ascending=False))

# --- Scatter (TotalRevenue vs önemli değişkenler) ---
pairs = [('NumTransactions', 'steelblue'),
         ('NumCustomers',    'darkorange'),
         ('NumItems',        'seagreen')]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('TotalRevenue vs Bağımsız Değişkenler', fontsize=14, fontweight='bold')
for ax, (col, renk) in zip(axes, pairs):
    ax.scatter(daily[col], daily['TotalRevenue'],
               alpha=0.5, color=renk, edgecolors='white', linewidth=0.3)
    z = np.polyfit(daily[col], daily['TotalRevenue'], 1)
    x_line = np.linspace(daily[col].min(), daily[col].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=2)
    ax.set_xlabel(col); ax.set_ylabel('TotalRevenue')
    ax.set_title(f'r = {daily[col].corr(daily["TotalRevenue"]):.2f}')
plt.tight_layout(); plt.savefig('scatter.png', dpi=150); plt.show()

# --- Zamansal trend ---
plt.figure(figsize=(14, 5))
tarih = pd.to_datetime(daily['Date'])
plt.plot(tarih, daily['TotalRevenue'], color='steelblue', linewidth=1.2, alpha=0.8)
plt.fill_between(tarih, daily['TotalRevenue'], alpha=0.15, color='steelblue')
plt.title('Günlük Toplam Gelir Trendi', fontsize=14, fontweight='bold')
plt.xlabel('Tarih'); plt.ylabel('Toplam Gelir (£)')
plt.tight_layout(); plt.savefig('trend.png', dpi=150); plt.show()

# 5. VERİ HAZIRLAMA

# --- Aykırı değer temizleme (IQR × 3) ---
Q1, Q3 = daily['TotalRevenue'].quantile(0.25), daily['TotalRevenue'].quantile(0.75)
IQR = Q3 - Q1
daily_clean = daily[
    (daily['TotalRevenue'] >= Q1 - 3*IQR) &
    (daily['TotalRevenue'] <= Q3 + 3*IQR)
].copy()

print(f"\nAykırı değer temizleme : {len(daily)} → {len(daily_clean)} gün  "
      f"(kaldırılan: {len(daily) - len(daily_clean)})")

# --- Log dönüşümü ---
daily_clean['log_Revenue']            = np.log1p(daily_clean['TotalRevenue'])
daily_clean['log_NumItems']           = np.log1p(daily_clean['NumItems'])
daily_clean['log_RevenuePerCustomer'] = np.log1p(daily_clean['RevenuePerCustomer'])

# --- Feature seçimi (yüksek korelasyonlu değişkenler çıkarıldı) ---
# NumCustomers çıkarıldı  → NumTransactions ile r=0.99
# RevenuePerTransaction çıkarıldı → RevenuePerCustomer ile r=0.99
features = [
    'NumTransactions',         # r=0.60
    'log_NumItems',            # r=0.94
    'NumProducts',             # r=0.52
    'log_RevenuePerCustomer',  # r=0.77
    'ItemsPerTransaction',     # r=0.64
    'Month',
    'DayOfWeek',
    'IsWeekend',
]

X = daily_clean[features]
y = daily_clean['log_Revenue']

# --- Train / Test Split (%80 / %20) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")

# 6. MODEL EĞİTİMİ

modeller = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)'    : Ridge(alpha=1.0),
    'Lasso (α=0.01)'   : Lasso(alpha=0.01),
    'Random Forest'    : RandomForestRegressor(n_estimators=200, max_depth=10,
                                               random_state=42, n_jobs=-1),
}

sonuclar = {}
y_test_gercek = np.expm1(y_test)

for isim, model in modeller.items():
    # Random Forest ölçekleme gerektirmez
    if isim == 'Random Forest':
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    else:
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_test_sc)

    y_pred_gercek = np.expm1(y_pred)
    mae  = mean_absolute_error(y_test_gercek, y_pred_gercek)
    rmse = np.sqrt(mean_squared_error(y_test_gercek, y_pred_gercek))
    r2   = r2_score(y_test_gercek, y_pred_gercek)

    sonuclar[isim] = {'model': model, 'y_pred': y_pred_gercek,
                      'MAE': mae, 'RMSE': rmse, 'R2': r2}

    print(f"\n{'='*40}")
    print(f"  {isim}")
    print(f"  R²: {r2:.4f}  |  MAE: £{mae:,.0f}  |  RMSE: £{rmse:,.0f}")

# 7. GÖRSELLEŞTİRME & KARŞILAŞTIRMA

# --- Model karşılaştırma çubuk grafik ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Model Performans Karşılaştırması', fontsize=14, fontweight='bold')
renkler = ['steelblue', 'darkorange', 'seagreen', 'crimson']

for ax, (metrik, baslik) in zip(axes, [
    ('R2',   'R² (yüksek = iyi)'),
    ('MAE',  'MAE - £ (düşük = iyi)'),
    ('RMSE', 'RMSE - £ (düşük = iyi)')
]):
    degerler = [sonuclar[m][metrik] for m in sonuclar]
    isimler  = list(sonuclar.keys())
    bars = ax.bar(isimler, degerler, color=renkler, edgecolor='white', linewidth=0.8)
    ax.set_title(baslik, fontsize=11)
    ax.set_xticklabels(isimler, rotation=15, ha='right', fontsize=9)
    for bar, val in zip(bars, degerler):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
                f'{val:.4f}' if metrik == 'R2' else f'£{val:,.0f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
plt.tight_layout(); plt.savefig('model_karsilastirma.png', dpi=150); plt.show()

# --- Gerçek vs Tahmin ---
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Gerçek vs Tahmin Değerleri', fontsize=14, fontweight='bold')
for ax, (isim, veri) in zip(axes.flatten(), sonuclar.items()):
    ax.scatter(y_test_gercek, veri['y_pred'],
               alpha=0.6, color='steelblue', edgecolors='white', linewidth=0.3, s=40)
    mn, mx = min(y_test_gercek.min(), veri['y_pred'].min()), \
             max(y_test_gercek.max(), veri['y_pred'].max())
    ax.plot([mn, mx], [mn, mx], 'r--', linewidth=2, label='Mükemmel Tahmin')
    ax.set_xlabel('Gerçek (£)'); ax.set_ylabel('Tahmin (£)')
    ax.set_title(f"{isim}\nR²={veri['R2']:.4f} | MAE=£{veri['MAE']:,.0f}")
    ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig('gercek_vs_tahmin.png', dpi=150); plt.show()

# --- Lasso artık analizi ---
lasso_pred = sonuclar['Lasso (α=0.01)']['y_pred']
residuals  = y_test_gercek.values - lasso_pred

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Lasso — Artık Analizi', fontsize=13, fontweight='bold')
axes[0].hist(residuals, bins=25, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(0, color='red', linestyle='--', linewidth=2)
axes[0].set_title('Artık Dağılımı'); axes[0].set_xlabel('Artık (£)'); axes[0].set_ylabel('Frekans')
axes[1].scatter(lasso_pred, residuals, alpha=0.6, color='darkorange',
                edgecolors='white', linewidth=0.3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title('Artık vs Tahmin'); axes[1].set_xlabel('Tahmin (£)'); axes[1].set_ylabel('Artık (£)')
plt.tight_layout(); plt.savefig('artik_analizi.png', dpi=150); plt.show()

# --- Lasso feature önemi ---
lasso_model = sonuclar['Lasso (α=0.01)']['model']
katsayilar  = pd.Series(np.abs(lasso_model.coef_), index=features).sort_values()

plt.figure(figsize=(9, 5))
renkler_f = ['crimson' if v == katsayilar.max() else 'steelblue' for v in katsayilar]
katsayilar.plot(kind='barh', color=renkler_f, edgecolor='white')
plt.title('Lasso — Feature Önemi (|Katsayı|)', fontsize=13, fontweight='bold')
plt.xlabel('Mutlak Katsayı Değeri')
plt.tight_layout(); plt.savefig('feature_importance.png', dpi=150); plt.show()

# --- Özet tablo ---
print("\n" + "=" * 55)
print("MODEL PERFORMANS ÖZETİ")
print("=" * 55)
print(f"{'Model':<25} {'R²':>8} {'MAE (£)':>12} {'RMSE (£)':>12}")
print("-" * 55)
for isim, v in sonuclar.items():
    print(f"{isim:<25} {v['R2']:>8.4f} {v['MAE']:>12,.0f} {v['RMSE']:>12,.0f}")

# 8. SENARYO ANALİZİ

print("\nReferans istatistikler (temizlenmiş veri):")
ref_cols = ['NumTransactions', 'NumItems', 'NumProducts',
            'RevenuePerCustomer', 'ItemsPerTransaction', 'Month', 'DayOfWeek', 'IsWeekend']
print(daily_clean[ref_cols].mean().round(2))

senaryolar = {
    'Düşük Trafik (Pazartesi)': {
        'NumTransactions': 40, 'log_NumItems': np.log1p(10000),
        'NumProducts': 600,    'log_RevenuePerCustomer': np.log1p(350),
        'ItemsPerTransaction': 220, 'Month': 6, 'DayOfWeek': 0, 'IsWeekend': 0,
    },
    'Ortalama Gün (Çarşamba)': {
        'NumTransactions': 65, 'log_NumItems': np.log1p(18000),
        'NumProducts': 900,    'log_RevenuePerCustomer': np.log1p(500),
        'ItemsPerTransaction': 280, 'Month': 9, 'DayOfWeek': 2, 'IsWeekend': 0,
    },
    'Yoğun Gün - Kasım (Cuma)': {
        'NumTransactions': 100, 'log_NumItems': np.log1p(35000),
        'NumProducts': 1200,    'log_RevenuePerCustomer': np.log1p(750),
        'ItemsPerTransaction': 350, 'Month': 11, 'DayOfWeek': 4, 'IsWeekend': 0,
    },
}
print("\n" + "=" * 55)
print("SENARYO TAHMİN SONUÇLARI")
print("=" * 55)
tahmin_sonuclari = []
for senaryo_adi, degerler in senaryolar.items():
    X_yeni    = pd.DataFrame([degerler])[features]
    log_tahmin    = lasso_model.predict(scaler.transform(X_yeni))[0]
    gercek_tahmin = np.expm1(log_tahmin)
    tahmin_sonuclari.append({'Senaryo': senaryo_adi, 'Tahmin (£)': gercek_tahmin})

    print(f"\n  {senaryo_adi}")
    print(f"  İşlem Sayısı      : {degerler['NumTransactions']}")
    print(f"  Satılan Ürün      : {np.expm1(degerler['log_NumItems']):,.0f}")
    print(f"  Müşteri Başı Gel. : £{np.expm1(degerler['log_RevenuePerCustomer']):,.0f}")
    print(f"  Tahmini Gelir     : £{gercek_tahmin:,.0f}")
# --- Senaryo karşılaştırma grafiği ---
df_tahmin = pd.DataFrame(tahmin_sonuclari)
plt.figure(figsize=(10, 5))
bars = plt.bar(df_tahmin['Senaryo'], df_tahmin['Tahmin (£)'],
               color=['#4e9af1', '#f4a236', '#e05c5c'],
               edgecolor='white', linewidth=0.8, width=0.5)
for bar, val in zip(bars, df_tahmin['Tahmin (£)']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'£{val:,.0f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
plt.title('Lasso — Senaryo Bazlı Günlük Gelir Tahmini', fontsize=13, fontweight='bold')
plt.ylabel('Tahmini Günlük Gelir (£)')
plt.tight_layout(); plt.savefig('senaryo_tahmini.png', dpi=150); plt.show()

# 9. MODEL KAYDETME

joblib.dump(lasso_model, 'lasso_model.pkl')
joblib.dump(scaler,      'scaler.pkl')

print("\nModel ve scaler kaydedildi → lasso_model.pkl | scaler.pkl")
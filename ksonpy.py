# -*- coding: utf-8 -*-
"""
KÜMELEME ÇALIŞMASI — Müşteri Segmentasyonu
Veri seti: Wholesale customers
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
sns.set_style("whitegrid")

COLORS = ['steelblue', 'coral', 'mediumseagreen']
FEATURES = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
SEGMENT_ISIMLERI = {0: 'Horeca Odaklı', 1: 'Büyük Ölçekli Karma', 2: 'Perakende Odaklı'}

# 1. VERİ YÜKLEME
df = pd.read_csv("C:/Users/HP/Desktop/YL/veribilimi/proje/Wholesale customers data.csv")

print("=" * 55)
print("VERİ SETİ GENEL BİLGİLER")
print("=" * 55)
print(f"Satır / Sütun : {df.shape[0]} / {df.shape[1]}")
print(f"Eksik değer  :\n{df.isnull().sum()}")
print(f"\nİstatistikler:\n{df.describe().round(0)}")

# Kategori etiketleri
df['Channel_Label'] = df['Channel'].map({1: 'Horeca', 2: 'Retail'})
df['Region_Label']  = df['Region'].map({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})

# 2. KEŞİFSEL VERİ ANALİZİ (EDA)

# --- Ham dağılımlar ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(FEATURES):
    axes.flatten()[i].hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.7)
    axes.flatten()[i].set_title(f'{col} — Ham Dağılım')
plt.suptitle('Ham Dağılımlar', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()

# --- Aykırı değer raporu (IQR) ---
print("\n" + "=" * 45)
print("AYKIRI DEĞER RAPORU (IQR)")
print("=" * 45)
for col in FEATURES:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    n = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)].shape[0]
    print(f"  {col:<22}: {n:3d} aykırı değer")

# --- Korelasyon matrisi ---
plt.figure(figsize=(8, 6))
mask = np.triu(np.ones((len(FEATURES), len(FEATURES)), dtype=bool))
sns.heatmap(df[FEATURES].corr(), annot=True, fmt='.2f', cmap='coolwarm',
            mask=mask, linewidths=0.5, vmin=-1, vmax=1, square=True)
plt.title('Korelasyon Matrisi', fontsize=13, fontweight='bold')
plt.tight_layout(); plt.show()

# --- Kanala göre ortalama harcamalar ---
print("\n" + "=" * 45)
print("KANAL'A GÖRE ORTALAMA HARCAMALAR")
print("=" * 45)
print(df.groupby('Channel_Label')[FEATURES].mean().round(0))

# 3. VERİ ÖN İŞLEME
# Log dönüşümü
df_log = np.log1p(df[FEATURES].copy())

# Log sonrası dağılımlar
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, col in enumerate(FEATURES):
    axes.flatten()[i].hist(df_log[col], bins=30, color='mediumseagreen', edgecolor='white', alpha=0.7)
    axes.flatten()[i].set_title(f'{col} — Log Sonrası')
plt.suptitle('Log Dönüşümü Sonrası Dağılımlar', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout(); plt.show()

# Winsorization (IQR kırpma)
for col in FEATURES:
    Q1, Q3 = df_log[col].quantile(0.25), df_log[col].quantile(0.75)
    IQR = Q3 - Q1
    df_log[col] = df_log[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Standartlaştırma
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(df_log)

# 4. OPTİMAL KÜME SAYISI
k_range  = range(2, 11)
inertia, sil_scores, db_scores = [], [], []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))
    db_scores.append(davies_bouldin_score(X_scaled, labels))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(k_range, inertia,    'o-', color='steelblue'); axes[0].set_title('Elbow (Inertia)')
axes[1].plot(k_range, sil_scores, 's-', color='coral');     axes[1].set_title('Silhouette Skoru')
axes[2].plot(k_range, db_scores,  '^-', color='mediumseagreen'); axes[2].set_title('Davies-Bouldin')
for ax in axes:
    ax.set_xlabel('Küme Sayısı (k)'); ax.grid(True, alpha=0.4); ax.set_xticks(k_range)
plt.suptitle('Küme Sayısı Değerlendirme Metrikleri', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# Sayısal özet
print("\n" + "=" * 55)
print(f"{'k':>4} {'Inertia':>12} {'Silhouette':>12} {'Davies-Bouldin':>16}")
print("-" * 55)
for i, k in enumerate(k_range):
    print(f"{k:>4} {inertia[i]:>12.1f} {sil_scores[i]:>12.4f} {db_scores[i]:>16.4f}")
print(f"\n  → En iyi Silhouette : k = {list(k_range)[sil_scores.index(max(sil_scores))]}")
print(f"  → En iyi Davies-Bouldin : k = {list(k_range)[db_scores.index(min(db_scores))]}")

# 5. K-MEANS FİNAL MODEL (k=3)

kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n" + "=" * 50)
print("K-MEANS FİNAL — k = 3")
print("=" * 50)
print(df['Cluster'].value_counts().sort_index())
print((df['Cluster'].value_counts(normalize=True).sort_index() * 100).round(1))

# --- Küme profilleri ---
cluster_profile = df.groupby('Cluster')[FEATURES].mean().round(0)
norm_profile    = cluster_profile.div(cluster_profile.max(axis=0), axis=1)

# Isı haritaları
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
sns.heatmap(cluster_profile,      annot=True, fmt='.0f',  cmap='YlOrRd', ax=axes[0],
            linewidths=0.5, cbar_kws={"shrink": 0.8})
axes[0].set_title('Ham Ortalamalar')
sns.heatmap(norm_profile,         annot=True, fmt='.2f',  cmap='YlOrRd', ax=axes[1],
            linewidths=0.5, vmin=0, vmax=1, cbar_kws={"shrink": 0.8})
axes[1].set_title('Normalize (0–1)')
plt.suptitle('Küme Profilleri', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()

# --- Radar grafiği ---
angles = np.linspace(0, 2*np.pi, len(FEATURES), endpoint=False).tolist() + [0]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for i in range(3):
    vals = norm_profile.iloc[i].tolist() + [norm_profile.iloc[i].tolist()[0]]
    ax.plot(angles, vals, 'o-', linewidth=2, color=COLORS[i], label=f'Küme {i}')
    ax.fill(angles, vals, alpha=0.1, color=COLORS[i])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(FEATURES, size=10)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Radar Grafiği', fontsize=13, fontweight='bold', pad=20)
plt.tight_layout(); plt.show()

# --- Kanal & Bölge dağılımı ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
pd.crosstab(df['Cluster'], df['Channel_Label'], normalize='index').mul(100).plot(
    kind='bar', ax=axes[0], color=['steelblue', 'coral'], edgecolor='white')
axes[0].set_title('Kanala Göre Dağılım (%)'); axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
pd.crosstab(df['Cluster'], df['Region_Label'],  normalize='index').mul(100).plot(
    kind='bar', ax=axes[1], edgecolor='white')
axes[1].set_title('Bölgeye Göre Dağılım (%)'); axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
plt.tight_layout(); plt.show()

# --- Yatay bar grafikleri ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for cid in range(3):
    means = df[df['Cluster'] == cid][FEATURES].mean()
    bars  = axes[cid].barh(FEATURES, means, color=COLORS[cid], alpha=0.8, edgecolor='white')
    for bar, val in zip(bars, means):
        axes[cid].text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                       f'{val:,.0f}', va='center', fontsize=9)
    axes[cid].set_title(f'Küme {cid} — {SEGMENT_ISIMLERI[cid]}', fontsize=11, fontweight='bold')
    axes[cid].set_xlim(0, means.max() * 1.3)
    axes[cid].grid(True, axis='x', alpha=0.3)
plt.suptitle('Küme Başına Harcama Profilleri', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout(); plt.show()

# 6. PCA — 2D GÖRSELLEŞTİRME + BİPLOT

pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
pca_df['Cluster'] = df['Cluster'].values

print("\n" + "=" * 45)
print(f"PC1: %{pca.explained_variance_ratio_[0]*100:.1f}  |  "
      f"PC2: %{pca.explained_variance_ratio_[1]*100:.1f}  |  "
      f"Toplam: %{sum(pca.explained_variance_ratio_)*100:.1f}")

loadings = pca.components_.T * 3.5

plt.figure(figsize=(11, 7))
for cid in range(3):
    m = pca_df['Cluster'] == cid
    plt.scatter(pca_df.loc[m, 'PC1'], pca_df.loc[m, 'PC2'],
                c=COLORS[cid], label=f'Küme {cid} — {SEGMENT_ISIMLERI[cid]}',
                alpha=0.7, s=55, edgecolors='white', linewidths=0.4)

centers_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers_pca[:, 0], centers_pca[:, 1],
            c='black', marker='X', s=200, zorder=5, label='Merkez')

for i, feat in enumerate(FEATURES):
    plt.arrow(0, 0, loadings[i, 0], loadings[i, 1],
              head_width=0.08, head_length=0.05, fc='darkred', ec='darkred', lw=1.5)
    plt.text(loadings[i, 0]*1.18, loadings[i, 1]*1.18, feat,
             fontsize=10, fontweight='bold', color='darkred', ha='center')

plt.title('PCA Biplot — Müşteri Kümeleri', fontsize=14, fontweight='bold')
plt.xlabel(f'PC1 (%{pca.explained_variance_ratio_[0]*100:.1f})')
plt.ylabel(f'PC2 (%{pca.explained_variance_ratio_[1]*100:.1f})')
plt.legend(fontsize=9); plt.grid(True, alpha=0.3)
plt.tight_layout(); plt.show()

# 7. ALTERNATİF KÜMELEME — Hierarchical (karşılaştırma)

agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster_Agg'] = agg.fit_predict(X_scaled)
sil_agg = silhouette_score(X_scaled, df['Cluster_Agg'])
db_agg  = davies_bouldin_score(X_scaled, df['Cluster_Agg'])

sil_km = silhouette_score(X_scaled, df['Cluster'])
db_km  = davies_bouldin_score(X_scaled, df['Cluster'])

print("\n" + "=" * 58)
print("KÜMELEME YÖNTEMLERİ KARŞILAŞTIRMASI")
print("=" * 58)
print(f"{'Yöntem':<25} {'k':>6} {'Silhouette':>12} {'Davies-Bouldin':>12}")
print("-" * 58)
print(f"{'K-Means':<25} {3:>6} {sil_km:>12.4f} {db_km:>12.4f}")
print(f"{'Hierarchical (Ward)':<25} {3:>6} {sil_agg:>12.4f} {db_agg:>12.4f}")

# 8. SONUÇ RAPORU
toplam = df[FEATURES].sum().sum()

print("\n" + "=" * 62)
print("FİNAL KÜME PROFİLLERİ")
print("=" * 62)
for cid in range(3):
    sub = df[df['Cluster'] == cid]
    cluster_toplam = sub[FEATURES].sum().sum()
    print(f"\n  Küme {cid} — {SEGMENT_ISIMLERI[cid]}")
    print(f"  Müşteri    : {len(sub)} ({len(sub)/len(df)*100:.1f}%)")
    print(f"  Kanal      : Horeca={( sub['Channel']==1).sum()} | Retail={(sub['Channel']==2).sum()}")
    print(f"  Harcama    : {cluster_toplam:,.0f}  (%{cluster_toplam/toplam*100:.1f} toplam)")
    print(f"  En yüksek  : {cluster_profile.loc[cid].idxmax()}")

print("\n" + "=" * 62)
print("MODEL PERFORMANS ÖZETİ")
print("=" * 62)
print(f"  Yöntem           : K-Means  (k=3, k-means++)")
print(f"  Ön işleme        : log1p → Winsorization → StandardScaler")
print(f"  Silhouette       : {sil_km:.4f}  (1'e yakın = iyi)")
print(f"  Davies-Bouldin   : {db_km:.4f}  (0'a yakın = iyi)")
print(f"  PCA varyans      : %{sum(pca.explained_variance_ratio_)*100:.1f}  (2 bileşen)")

print("\n" + "=" * 62)
print("İŞ ÖNERİLERİ")
print("=" * 62)
oneriler = {
    0: ["Taze ürün tedarik sıklığını artır.",
        "Dondurulmuş ürün paket kampanyaları düzenle.",
        "Büyük siparişlere özel lojistik destek sun."],
    1: ["VIP müşteri programına al (yüksek harcama potansiyeli).",
        "Toplu sipariş indirimleri ve uzun vadeli sözleşmeler öner.",
        "Özel hesap yöneticisi ata."],
    2: ["Grocery & Detergents_Paper için sadakat programı kur.",
        "Taze ürün tüketimini artırmak için promosyon hazırla.",
        "Otomatik sipariş yenileme sistemi sun."]
}
for cid, maddeler in oneriler.items():
    print(f"\n  Küme {cid} — {SEGMENT_ISIMLERI[cid]}")
    for i, m in enumerate(maddeler, 1):
        print(f"    {i}. {m}")
print("\n" + "=" * 62)
print("  KÜMELEME ÇALIŞMASI TAMAMLANDI")
print("=" * 62)
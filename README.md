# machine-learning-portfolio
An end-to-end Machine Learning portfolio featuring Customer Segmentation (K-Means), Daily Revenue Forecasting (Linear/Ridge/Lasso/RF), and Cancer Diagnosis (Pipeline-backed 10-Fold Cross-Validation).
# Veri Bilimi ve Makine Öğrenmesi Projeleri Portfolyosu 📊🤖

Bu depo; Python veri bilimi ekosistemi kullanılarak geliştirilmiş, makine öğrenmesinin üç ana temel direğini (Kümeleme, Regresyon ve Sınıflandırma) kapsayan uçtan uca üç farklı projenin kaynak kodlarını barındırmaktadır. Projelerde veri ön işleme, aykırı değer yönetimi, veri sızıntısının (Data Leakage) engellenmesi ve ileri düzey model değerlendirme metrikleri titizlikle uygulanmıştır.

---

## 📁 Portfolyo İçeriği ve Teknik Detaylar

### 1. K-Means ile Müşteri Segmentasyonu (Gözetimsiz Öğrenme — `ksonpy.py`)
* **Veri Seti:** Wholesale Customers Data (Toptan Satış Müşteri Verileri)
* **İş Problemi:** Ticari müşterilerin harcama alışkanlıklarına göre kümelenerek optimize edilmiş pazarlama ve lojistik stratejilerinin (Horeca, Perakende vb.) belirlenmesi.
* **Uygulanan Veri Ön İşleme:** * Varyansı dengelemek ve sağa çarpıklığı gidermek amacıyla veriye **Log Dönüşümü (`log1p`)** uygulanmıştır.
  * Aykırı değerlerin (Outliers) bozucu etkisini önlemek için **Winsorization (Sınırlandırma)** tekniği devreye alınmıştır.
  * Boyut indirgeme ve 2D görselleştirme için **PCA (Temel Bileşenler Analizi)** kullanılmıştır.
* **Model ve Performans:** `KMeans` (k=3, k-means++) mimarisi kurulmuş; model başarısı **Silhouette Score** ve **Davies-Bouldin Index** metrikleri ile doğrulanmıştır.

### 2. İleri Düzey Regresyon ile Günlük Gelir Tahmini (Gözetimli Öğrenme — `rson.py`)
* **Veri Seti:** Online Retail (E-Ticaret İngiltere Perakende Satış Verileri)
* **İş Problemi:** Geçmiş sipariş, işlem hacmi ve müşteri başına gelir verilerinden yola çıkarak gelecekteki günlük toplam gelirleri tahmin etmek.
* **Uygulanan Veri Ön İşleme:**
  * İptal edilen siparişler (C ile başlayan faturalar), eksik veriler ve mantıksal hatalar barındıran negatif fiyat/miktar satırları temizlenmiştir.
  * Regresyon modellerinin doğrusal varsayımlarını korumak için hedef değişkene logaritmik dönüşüm uygulanmıştır.
* **Model ve Performans:** `Linear Regression`, `Ridge`, `Lasso` ve `RandomForestRegressor` modelleri eğitilmiştir. Aşırı öğrenmeyi (Overfitting) engellemek amacıyla L1 ve L2 düzenlileştirmeleri optimize edilerek **MAE**, **MSE** ve **$R^2$** skorları üzerinden kıyaslama yapılmıştır.

### 3. Hatasız Cross-Validation ile Meme Kanseri Teşhisi (Sınıflandırma — `sson.py`)
* **Veri Seti:** Breast Cancer Wisconsin (Diagnostic)
* **İş Problemi:** Tümör özelliklerine ait biyolojik metrikleri inceleyerek tümörün iyi huylu (Benign) veya kötü huylu (Malignant) olduğunu yüksek doğrulukla sınıflandırmak.
* **Mühendislik İmzası (Veri Sızıntısının Önlenmesi):**
  * Sıklıkla yapılan en büyük hata; cross-validation yapmadan önce tüm veriyi ölçeklendirmektir (StandardScaler). Bu durum test verisinin bilgisinin eğitim verisine sızmasına (Data Leakage) neden olur.
  * Bu projede, veri sızıntısını kesin olarak engellemek amacıyla `StandardScaler` ve Sınıflandırıcılar **Scikit-learn Pipeline** yapısı altında birleştirilmiştir.
* **Model ve Performans:** `Logistic Regression`, `KNN`, `Random Forest` ve `SVM` modelleri **Stratified 10-Fold Cross-Validation** ile eğitilmiştir. Modeller sadece doğruluğa (Accuracy) göre değil; **Confusion Matrix**, **Precision**, **Recall** ve **F1-Score** metrikleriyle derinlemesine analiz edilmiştir.

---

## 🛠️ Teknolojik Altyapı ve Bağımlılıklar

* **Programlama Dili:** Python 3.10+
* **Veri Manipülasyonu & Analiz:** Pandas, NumPy
* **Makine Öğrenmesi Çerçevesi:** Scikit-learn (Sklearn)
* **Görselleştirme:** Matplotlib, Seaborn

---

## 🚀 Projelerin Çalıştırılması

1. Bu depoyu yerel bilgisayarınıza klonlayın:
```bash
git clone [https://github.com/kullaniciadi/machine-learning-data-science-portfolio.git](https://github.com/kullaniciadi/machine-learning-data-science-portfolio.git)
cd machine-learning-data-science-portfolio

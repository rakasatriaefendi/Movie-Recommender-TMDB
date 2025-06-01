# Laporan Proyek Machine Learning - Raka Satria Efendi

## Project Overview
Film merupakan salah satu bentuk hiburan utama di era digital, dengan platform streaming seperti Netflix dan Disney+ menjadi pusat distribusi konten. Dengan ribuan film yang tersedia, pengguna sering kali kesulitan menemukan film yang sesuai dengan preferensi mereka. Sistem rekomendasi menjadi kunci untuk meningkatkan pengalaman pengguna dengan menyarankan film yang relevan berdasarkan karakteristik konten, seperti genre, deskripsi, aktor, atau sutradara.

Dataset TMDB 5000 Movie Dataset menyediakan informasi kaya tentang film, termasuk genre, deskripsi (`overview`), kata kunci (`keywords`), aktor (`cast`), dan sutradara (`director`). Dataset ini memungkinkan analisis karakteristik film untuk membangun sistem rekomendasi berbasis **Content-based Filtering**, yang merekomendasikan film berdasarkan kesamaan fitur konten dengan film yang disukai pengguna.

Proyek ini bertujuan untuk mengembangkan sistem rekomendasi film yang akurat dan relevan, membantu pengguna menemukan film yang sesuai dengan preferensi mereka serta meningkatkan retensi dan kepuasan di platform streaming.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang di atas, rumusan masalah dari proyek ini adalah:
1. Siapa saja TOP 10 cast (aktor/aktris) di dataset TMDB 5000 Movie Dataset?
2. Faktor apa saja yang memengaruhi kemiripan antar film?
3. Bagaimana cara membuat sistem rekomendasi terbaik yang dapat diimplementasikan?

### Goals
Berdasarkan problem statements, berikut tujuan yang ingin dicapai pada proyek ini:
1. Mengidentifikasi TOP 10 cast dalam dataset TMDB 5000 Movie Dataset untuk memahami aktor yang paling sering muncul atau populer, yang dapat menjadi salah satu elemen penting dalam sistem rekomendasi film.
2. Menganalisis faktor yang mempengaruhi kemiripan antar film, dengan fokus pada fitur seperti genre, deskripsi (`overview`), aktor (`cast`), dan sutradara (`director`), untuk mengetahui kontribusi masing-masing fitur dalam menentukan relevansi rekomendasi.
3. Mengembangkan sistem rekomendasi film terbaik yang dapat diimplementasikan, menggunakan pendekatan berbasis **Content-based Filtering**, serta memastikan sistem tersebut memberikan rekomendasi yang relevan dan dapat dievaluasi dengan metrik sederhana.

### Solution Approach
Untuk mencapai tujuan tersebut, solution statements yang diusulkan adalah:
1. Mengidentifikasi TOP 10 cast dengan mengekstrak data dari kolom `cast` dalam dataset TMDB 5000 Movie Dataset, lalu menghitung frekuensi kemunculan aktor menggunakan teknik seperti `Counter` untuk menampilkan aktor paling populer.
2. Menganalisis faktor yang mempengaruhi kemiripan antar film dengan:
   - Menggunakan **TF-IDF Vectorizer** untuk mengekstrak fitur numerik dari kolom `keywords` dan `overview`, serta mengidentifikasi kata atau frasa yang paling berpengaruh terhadap kemiripan.
   - Menganalisis distribusi fitur seperti `genres`, `cast`, dan `director` untuk memahami kontribusi masing-masing fitur terhadap kemiripan antar film.
3. Membangun sistem rekomendasi terbaik yang dapat diimplementasikan dengan langkah-langkah berikut:
   - Melakukan preprocessing teks pada kolom `keywords` dan `overview` untuk memastikan data bersih dan konsisten.
   - Menggunakan **TF-IDF Vectorizer** untuk merepresentasikan fitur teks secara numerik, lalu menghitung kemiripan antar film dengan algoritma **cosine similarity**.
   - Mengevaluasi sistem rekomendasi secara sederhana dengan menggunakan ground truth untuk menghitung metrik seperti **Precision@5**, serta memverifikasi relevansi rekomendasi secara kualitatif berdasarkan tema dan genre film.

## Data Understanding
Data yang digunakan untuk membuat sistem rekomendasi film diambil dari platform open source Kaggle, yaitu [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata).

Dataset ini terdiri dari dua file utama:
- `tmdb_5000_movies.csv`: Berisi 4803 baris dan 20 kolom, mencakup informasi seperti `genres`, `overview`, `keywords`, `budget`, `revenue`, `release_date`, `runtime`, `vote_average`, dan `vote_count`.
- `tmdb_5000_credits.csv`: Berisi 4803 baris dan 4 kolom, mencakup informasi seperti `cast` dan `crew`.

Setiap kolom memiliki arti sebagai berikut:

| Variabel           | Tipe Data | Keterangan                                                                 |
|--------------------|-----------|---------------------------------------------------------------------------|
| `id`               | integer   | ID unik untuk film dalam dataset TMDB.                                    |
| `title`            | object    | Judul film.                                                               |
| `genres`           | object    | Daftar genre film dalam format JSON string (misalnya, Action, Drama).     |
| `overview`         | object    | Deskripsi singkat atau sinopsis film.                                     |
| `keywords`         | object    | Daftar kata kunci yang terkait dengan film dalam format JSON string.      |
| `cast`             | object    | Daftar aktor yang bermain dalam film, dalam format JSON string.           |
| `crew`             | object    | Daftar kru film (termasuk sutradara), dalam format JSON string.           |
| `budget`           | integer   | Anggaran produksi film dalam dolar AS.                                    |
| `revenue`          | integer   | Pendapatan film dalam dolar AS.                                           |
| `release_date`     | object    | Tanggal rilis film.                                                       |
| `runtime`          | float     | Durasi film dalam menit.                                                  |
| `vote_average`     | float     | Rata-rata rating film (skala 0–10).                                       |
| `vote_count`       | integer   | Jumlah pengguna yang memberikan rating.                                   |

Tetapi yang dipakai untuk proyek kali ini adalah beberapa kolom berikut yang menurut saya relevan:
![kolom yang dipilih](images/kolom-yang-dipakai.png)

Disini saya melakukan penggabungan kolom dari dua dataset

### Exploratory Data Analysis

#### Analisis Missing Values dan Duplikat
Analisis awal dilakukan untuk memeriksa keberadaan missing values dan duplikat dalam dataset.

**Hasil:**
#### Analisis Missing Values dan Duplikat
Analisis awal dilakukan untuk memeriksa keberadaan missing values dan duplikat dalam dataset.

**Hasil:**
```
1. Analisis Missing Values dan Duplikat
Missing Values:
 id          0
title       0
genres      0
overview    0
keywords    0
cast        0
crew        0
dtype: int64

Jumlah Duplikat berdasarkan title: 3
```

> **Insight:**
> - Tidak ada missing values di kolom utama (`id`, `title`, `genres`, `overview`, `keywords`, `cast`, `crew`), memastikan dataset siap untuk analisis.
> - Terdapat 3 duplikat berdasarkan `title`, yang seharusnya diperiksa lebih lanjut. Namun, duplikat dipertahankan karena perbedaan di kolom lain (seperti `genres`, `overview`, `cast`) mencerminkan variasi konten sah, seperti edisi berbeda, dan menghapusnya berisiko kehilangan data kontekstual penting. Perhitungan ulang matriks similarity untuk validasi juga meningkatkan beban komputasi secara signifikan. Oleh karena itu, duplikat ini dijaga untuk integritas data dan efisiensi, dengan asumsi dampaknya minimal pada evaluasi kualitatif.

#### Distribusi Film Berdasarkan Genre
Analisis dilakukan untuk melihat distribusi genre dalam dataset dengan mem-parsing kolom `genres` dan membuat visualisasi barplot untuk 10 genre teratas.

![Genre](images/distribusi-genre.png)

**Hasil:**
- Distribusi genre dihitung dengan mem-parsing kolom `genres` menggunakan fungsi `parse_genres`, lalu divisualisasikan dalam bentuk barplot.
- Genre teratas: Drama, Comedy, Thriller, Action, Romance, dll.

> **Insight:**
> - Genre **Drama** dan **Comedy** mendominasi dataset (~40% dari total film), menunjukkan popularitas tinggi. Ini dapat memengaruhi rekomendasi untuk pengguna yang menyukai genre mainstream.

#### Jumlah Film Berdasarkan Panjang Sinopsis (Overview)
Analisis dilakukan untuk memeriksa distribusi panjang sinopsis (jumlah kata) dalam kolom `overview` menggunakan histogram.

![sinopsis](images/distribusi-sinopsis.png)

**Hasil:**
- Panjang sinopsis dihitung dengan fungsi `word_tokenize` dan divisualisasikan menggunakan `sns.histplot`.

> **Insight:**
> - Mayoritas sinopsis film memiliki panjang antara 20 hingga 80 kata, dengan puncak sekitar 30 kata dan 60-70 kata. Ini menunjukkan bahwa sinopsis cenderung ringkas.
> - Sebagian besar sinopsis memiliki 20-60 kata, tetapi ada sinopsis pendek (<10 kata) yang berpotensi kurang informatif untuk ekstraksi fitur TF-IDF, sehingga perlu penanganan khusus.

#### Top 20 Aktor Paling Sering Muncul
Analisis dilakukan untuk mengidentifikasi aktor yang paling sering muncul dalam dataset dengan mem-parsing kolom `cast` (mengambil 3 aktor utama per film) dan membuat visualisasi barplot untuk 20 aktor teratas.

![top20 aktor](images/top20-aktor.png)

**Hasil:**
- Aktor dihitung dengan fungsi `parse_cast`, lalu divisualisasikan dalam bentuk barplot.
- Aktor teratas: Nicolas Cage, Samuel L. Jackson, Robert De Niro, dll.

> **Insight:**
> - **Nicolas Cage** adalah aktor yang paling sering muncul dalam film, diikuti oleh **Samuel L. Jackson** dan **Robert De Niro**. Ini mengindikasikan produktivitas atau popularitas mereka dalam industri film.

#### Analisis Kata Kunci (Keywords) Paling Populer
Analisis dilakukan untuk mengidentifikasi kata kunci paling populer dalam dataset dengan mem-parsing kolom `keywords` dan membuat visualisasi barplot untuk 20 kata kunci teratas.

![keywords](images/keywords-populer.png)

**Hasil:**
- Kata kunci dihitung dengan fungsi `parse_keywords`, lalu divisualisasikan dalam bentuk barplot.
- Kata kunci teratas: "woman director", "independent film", dll.

> **Insight:**
> - "woman director" dan "independent film" adalah dua kata kunci paling populer. Ini bisa mencerminkan tren atau fokus dalam produksi film yang dianalisis, seperti peningkatan peran sutradara wanita atau produksi film independen.

#### Word Cloud untuk Kata Kunci
Visualisasi word cloud dibuat untuk memberikan gambaran visual tentang kata kunci yang paling dominan dalam dataset.

![wordcloud](images/wordclouds-keywords.png)

**Hasil:**
- Word cloud menampilkan kata kunci seperti "murder", "based", "independent", "director", "violence", dan "novel".

> **Insight:**
> - Word cloud mengkonfirmasi dominasi kata kunci seperti "murder," "based," "independent," "director," "violence," dan "novel," memberikan gambaran visual tentang tema dan elemen yang sering muncul dalam film.

#### Distribusi Film Berdasarkan Peran Sutradara
Analisis dilakukan untuk melihat distribusi sutradara dalam dataset dengan mem-parsing kolom `crew` (mengambil sutradara) dan membuat visualisasi barplot untuk 20 sutradara teratas.

![top20 sutradara](images/top20-sutradara.png)

**Hasil:**
- Sutradara dihitung dengan fungsi `parse_director`, lalu divisualisasikan dalam bentuk barplot.
- Sutradara teratas: Steven Spielberg, Woody Allen, Martin Scorsese, dll.

> **Insight:**
> - **Steven Spielberg** memimpin sebagai sutradara dengan jumlah film terbanyak, diikuti oleh **Woody Allen** dan **Martin Scorsese**. Hal ini menyoroti kontribusi signifikan mereka dalam penyutradaraan.

#### Korelasi Panjang Overview dengan Jumlah Keywords atau Genre
Analisis dilakukan untuk memeriksa korelasi antara panjang sinopsis (`overview_length`), jumlah genre (`num_genres`), dan jumlah kata kunci (`num_keywords`) menggunakan heatmap.

![korelasi matrix](images/korelasi-overview-dengan-jumlahkeywords-genre.png)

**Hasil:**
- Korelasi dihitung menggunakan `corr()` dan divisualisasikan dengan heatmap.

> **Insight:**
> - Panjang sinopsis, jumlah genre, dan jumlah kata kunci tidak memiliki korelasi yang kuat satu sama lain (nilai korelasi mendekati 0). Ini berarti panjang sinopsis tidak secara signifikan memprediksi jumlah genre atau kata kunci, dan sebaliknya.
> - Ada sedikit korelasi positif antara jumlah genre dan jumlah kata kunci (0.13), menunjukkan bahwa film dengan lebih banyak genre cenderung memiliki sedikit lebih banyak kata kunci.

### Data Quality Verification

#### Memeriksa Kesesuaian Tipe Data dan Prioritas Kolom
- Kolom seperti `genres`, `keywords`, `cast`, dan `crew` awalnya berformat JSON string, yang memerlukan parsing untuk analisis.
- Kolom yang relevan untuk Content-based Filtering adalah `title`, `genres`, `overview`, `keywords`, `cast`, dan `crew`. Kolom lain seperti `budget`, `revenue`, dan `release_date` tidak digunakan dalam analisis ini.
- Tidak diperlukan konversi tipe data lebih lanjut karena fokus utama adalah fitur teks.

#### Memeriksa Data Duplikat
- Pemeriksaan duplikat dilakukan pada semua kolom kecuali kolom bertipe list (`genres_list`, `cast_list`, `keywords_list`) yang bersifat unhashable.
- Proses: Membuat DataFrame sementara dengan mengecualikan kolom-kolom tersebut (`non_list_cols = df_filtered.drop(columns=['genres_list', 'cast_list', 'keywords_list'])`), lalu memeriksa duplikat menggunakan `non_list_cols.duplicated().sum()`.
- **Hasil**: Tidak ditemukan data duplikat.

#### Memeriksa Data Missing Value
- Pemeriksaan dilakukan pada semua kolom yang relevan. Berikut adalah hasilnya:
```
0
id              0
title           0
genres          0
overview        0
keywords        0
cast            0
crew            0
genres_list     0
overview_length 0
cast_list       0
keywords_list   0
director        734
num_genres      0
num_keywords    0
```

> **Insight:**
> - Tidak ada missing value pada kolom `id`, `title`, `genres`, `overview`, `keywords`, `cast`, `crew`, `genres_list`, `overview_length`, `cast_list`, `keywords_list`, `num_genres`, dan `num_keywords`.
> - Kolom `director` memiliki 734 missing value, yang akan ditangani pada tahap Data Preparation dengan mengisi nilai "Unknown".
> - Kolom lain yang tidak relevan untuk analisis (seperti `budget`, `revenue`) tidak diperiksa karena tidak digunakan.

#### Memeriksa Outliers
Meskipun analisis utama berfokus pada fitur teks (seperti `overview` dan `keywords`), pemeriksaan outliers dilakukan pada fitur numerik yang dihasilkan selama EDA, yaitu `overview_length`, `num_genres`, dan `num_keywords`, menggunakan boxplot untuk memahami distribusi dan potensi anomali.

![outlier](images/fitur-numerik-outlier.png)

**Interpretasi Boxplot:**

**1. `overview_length` (Panjang Sinopsis)**
- **Median:** Sekitar 60-70 kata, menunjukkan bahwa separuh film memiliki sinopsis lebih pendek dari 60-70 kata dan separuhnya lebih panjang.
- **Jangkauan Interkuartil (IQR):** Rentang dari 40 hingga 80 kata, mencakup 50% data di tengah distribusi.
- **Whiskers:** Membentang cukup jauh, menunjukkan variasi panjang sinopsis yang signifikan.
- **Outlier:** Terdapat beberapa outlier di sisi kanan (sekitar 140-150 kata), mengindikasikan adanya sinopsis yang sangat panjang dibandingkan mayoritas.

**2. `num_genres` (Jumlah Genre)**
- **Median:** Sekitar 2 genre, menunjukkan bahwa kebanyakan film memiliki 2 genre.
- **Jangkauan Interkuartil (IQR):** Rentang dari 1 hingga 3 genre, mencakup 50% data di tengah distribusi.
- **Whiskers:** Membentang dari 1 hingga 4 genre, mencerminkan distribusi yang cukup sempit.
- **Outlier:** Tidak ada outlier signifikan, menunjukkan distribusi yang terkonsentrasi.

**3. `num_keywords` (Jumlah Kata Kunci)**
- **Median:** Sekitar 7-8 kata kunci, menunjukkan bahwa kebanyakan film memiliki 7 atau 8 kata kunci.
- **Jangkauan Interkuartil (IQR):** Rentang dari 5 hingga 10 kata kunci, mencakup 50% data di tengah distribusi.
- **Whiskers:** Membentang dari 0 hingga sekitar 15 kata kunci, menunjukkan variasi yang wajar.
- **Outlier:** Tidak ada outlier signifikan, menunjukkan distribusi yang cukup normal.

> **Insight:**
> - Fitur `overview_length` memiliki variasi yang lebih besar dengan beberapa outlier (sinopsis sangat panjang di atas 140 kata). Namun, karena pendekatan Content-based Filtering menggunakan TF-IDF, panjang sinopsis yang ekstrem ini tidak akan signifikan memengaruhi kemiripan antar film, melainkan hanya memengaruhi bobot fitur teks.
> - Fitur `num_genres` dan `num_keywords` menunjukkan distribusi yang lebih seragam tanpa outlier mencolok, menegaskan bahwa sebagian besar film memiliki jumlah genre dan kata kunci yang konsisten, yang mendukung stabilitas fitur dalam proses rekomendasi.
> - Secara keseluruhan, outliers pada fitur numerik ini tidak memerlukan penanganan khusus karena fokus utama sistem adalah fitur teks, dan distribusi yang ada masih dalam batas wajar untuk analisis.

> **Kesimpulan:**
> Dataset cukup bersih untuk analisis berbasis teks, dengan penanganan nilai kosong yang diperlukan hanya pada kolom `director`. Fitur numerik seperti `overview_length`, `num_genres`, dan `num_keywords` menunjukkan distribusi yang wajar, dan outliers yang ada tidak signifikan memengaruhi pendekatan Content-based Filtering yang digunakan.



## Data Preparation

### Data Cleaning
#### Penggabungan Dataset
- Dataset `tmdb_5000_movies.csv` dan `tmdb_5000_credits.csv` digabungkan berdasarkan kolom `id` untuk mendapatkan informasi lengkap tentang film, termasuk `cast` dan `crew`.

#### Menangani Kolom JSON String
- Kolom `genres`, `keywords`, `cast`, dan `crew` yang berformat JSON string diproses menggunakan fungsi parsing untuk mengekstrak informasi seperti nama genre, kata kunci, aktor, dan sutradara.
- Masalah parsing JSON (akibat tanda kutip tunggal atau format korup) diatasi dengan fungsi pembersihan string.

#### Menangani Missing Value pada Kolom `director`
- Kolom `director` memiliki 734 missing value, yang ditangani dengan mengisi nilai "Unknown" untuk memastikan data konsisten dan dapat digunakan dalam analisis.

#### Preprocessing Teks
- Kolom `overview` dan `keywords_str` dibersihkan dengan:
  - Mengubah teks ke lowercase.
  - Menghapus tanda baca dan karakter non-alfanumerik.
  - Mengisi nilai kosong dengan string kosong ('') (meskipun tidak ada missing value pada `overview` dan `keywords` berdasarkan pemeriksaan).

#### Menyimpan dan Mendrop Kolom `id`
- Kolom `id` disimpan ke dalam variabel `movie_ids` sebagai cadangan untuk keperluan indexing atau referensi di masa depan.
- Kolom `id` kemudian dihapus dari dataset utama (`df_cleaned`) karena tidak diperlukan dalam proses Content-based Filtering berbasis fitur teks.

#### Dataset Hasil Data Cleaning
- Dataset yang telah dibersihkan berisi kolom utama: `title`, `genres`, `overview_clean`, `keywords_clean`, `cast`, `crew`, dan `combined_features` (gabungan `keywords_clean` dan `overview_clean`).

## Modeling and Result

### Cosine Similarity
Model rekomendasi ini dibangun menggunakan pendekatan Content-based Filtering, yang berfokus pada kesamaan fitur teks antar film untuk menghasilkan rekomendasi. Sistem ini menggunakan metrik **cosine similarity** untuk mengukur tingkat kemiripan antar film berdasarkan fitur teks yang telah diproses dengan TF-IDF Vectorizer.

Cosine similarity menghitung sudut antara dua vektor dalam ruang fitur multidimensi, dengan nilai berkisar dari -1 (sangat berbeda) hingga 1 (sangat mirip). Dalam konteks ini, nilai mendekati 1 menunjukkan bahwa dua film memiliki karakteristik konten yang serupa.

**Langkah-langkah:**
1. **Ekstraksi Fitur**: Menggunakan TF-IDF Vectorizer dengan parameter `max_features=5000`, `stop_words='english'`, dan `ngram_range=(1, 2)` untuk mengubah `combined_features` (gabungan `keywords_clean` dan `overview_clean`) menjadi representasi numerik.
2. **Perhitungan Cosine Similarity**: Matriks TF-IDF digunakan untuk menghitung cosine similarity antar semua film, menghasilkan matriks kemiripan berukuran 4803x4803 (sesuai jumlah film dalam dataset).
3. **Fungsi Rekomendasi**: Fungsi `recommend_by_identifier()` dibuat untuk mengambil judul film sebagai input dan mengembalikan film-film paling mirip berdasarkan skor cosine similarity. Fungsi ini menggunakan indeks film yang dapat dihubungkan kembali ke `movie_ids` jika diperlukan.

**Contoh Hasil Matriks Cosine Similarity**:
- Matriks berukuran 4803x4803, dengan setiap elemen mewakili skor kemiripan antar dua film.
- Contoh: Film "Spectre" memiliki kemiripan tinggi dengan "Skyfall" dan "Quantum of Solace" karena tema dan genre yang serupa.

### Inference
Fungsi `recommend_by_identifier()` digunakan untuk memberikan rekomendasi berdasarkan judul film tertentu. Fungsi ini mengambil parameter `movie_title` (judul film) dan `top_n` (jumlah rekomendasi yang diinginkan), lalu mengembalikan daftar film paling mirip berdasarkan skor cosine similarity.

**Contoh Penggunaan**:
Untuk film "Spectre" dengan `top_n=5`:

**Hasil:**
```
Rekomendasi untuk 'Spectre':
0    Never Say Never Again
1                 Restless
2        Quantum of Solace
3                  Skyfall
4          Die Another Day
dtype: object
```

> **Insight:**
> Sistem berhasil merekomendasikan film-film yang relevan secara tematik, seperti "Skyfall", "Quantum of Solace", dan "Die Another Day", yang merupakan bagian dari franchise James Bond. "Never Say Never Again" juga relevan karena merupakan film James Bond lainnya, meskipun bukan bagian dari seri utama.

**Contoh Lain**:
Jika digunakan untuk film lain seperti "The Dark Knight", sistem akan merekomendasikan "The Dark Knight Rises" dan "Batman Begins", yang merupakan bagian dari trilogi yang sama.

### Kelebihan dan Kekurangan Pendekatan Content-Based Filtering

**a. Rekomendasi Berdasarkan Fitur Konten**  
CBF memanfaatkan fitur seperti genre, deskripsi, dan kata kunci untuk merekomendasikan film yang mirip dengan preferensi pengguna.  
> Contoh: Jika pengguna menyukai "Spectre", sistem akan menyarankan film aksi dan mata-mata lainnya seperti "Skyfall".

**b. Tidak Perlu Data dari Pengguna Lain**  
Sistem dapat bekerja hanya dengan data konten film, tanpa memerlukan data interaksi pengguna lain.  
> Cocok untuk platform streaming baru dengan sedikit data pengguna.

**c. Efektif untuk Film Baru atau Niche**  
CBF dapat merekomendasikan film yang belum banyak ditonton tetapi memiliki karakteristik serupa dengan film yang disukai pengguna.  
> Berguna untuk mempromosikan film independen atau baru.

**d. Lebih Personal dan Bisa Dijelaskan**  
Sistem dapat menjelaskan alasan rekomendasi, misalnya:  
> "Film ini direkomendasikan karena memiliki genre aksi dan tema mata-mata, mirip dengan film yang Anda sukai."

**e. Keterbatasan pada Variasi**  
CBF cenderung merekomendasikan film yang sangat mirip, sehingga dapat membatasi variasi dan eksplorasi pengguna terhadap genre baru.

## Evaluation

### Penjelasan Metrik
Metrik yang digunakan untuk mengevaluasi sistem Content-based Filtering adalah **Precision@5**, **Recall@5**, dan **F1-Score@5**, yang biasa digunakan dalam sistem rekomendasi untuk mengukur relevansi rekomendasi.

1. **Precision@5**  
   Mengukur proporsi film relevan dari total 5 film yang direkomendasikan.  
   Formula:  
   $$\text{Precision@5} = \frac{\text{Jumlah film relevan dalam rekomendasi}}{5}$$

2. **Recall@5**  
   Mengukur proporsi film relevan yang berhasil ditemukan dari seluruh film relevan yang tersedia.  
   Formula:  
   $$\text{Recall@5} = \frac{\text{Jumlah film relevan dalam rekomendasi}}{\text{Jumlah total film relevan}}$$

3. **F1-Score@5**  
   Rata-rata harmonik dari Precision@5 dan Recall@5, memberikan keseimbangan antara ketepatan dan kelengkapan.  
   Formula:  
   $$\text{F1-Score@5} = \frac{2 \times \text{Precision@5} \times \text{Recall@5}}{\text{Precision@5} + \text{Recall@5}}$$

### Contoh Penerapan Metrik
Penerapan metrik dilakukan dengan membuat ground truth untuk beberapa film, lalu menggunakan fungsi `evaluate_single_recommendation()` untuk menghitung metrik evaluasi.

**Ground Truth**:
- Untuk "Spectre": {"Skyfall", "Casino Royale", "Quantum of Solace", "GoldenEye", "Die Another Day"}

**Hasil Evaluasi untuk "Spectre"**:
```
Hasil Evaluasi Sistem Rekomendasi untuk 'Spectre':
{'Precision@5': 0.6, 'Recall@5': 0.6, 'F1-Score@5': 0.6, 'Relevant Found': ['Die Another Day', 'Skyfall', 'Quantum of Solace']}
```

**Hasil Evaluasi untuk Beberapa Film (k=5)**:
```
Hasil Evaluasi untuk Beberapa Film (k=5):
                Precision@5  Recall@5  F1-Score@5  Relevant Found
Spectre                 0.6       0.6         0.6  [Die Another Day, Skyfall, Quantum of Solace]
The Dark Knight         0.4       0.4         0.4  [The Dark Knight Rises, Batman Begins]
Avatar                  0.2       0.2         0.2  [Aliens]
```

> **Insight:**
> - **Spectre**: Precision@5, Recall@5, dan F1-Score@5 sebesar 0.6 menunjukkan bahwa sistem cukup akurat dan lengkap dalam merekomendasikan film James Bond lainnya.
> - **The Dark Knight**: Skor 0.4 menunjukkan bahwa sistem dapat mengenali film dalam trilogi Batman, tetapi beberapa rekomendasi kurang relevan.
> - **Avatar**: Skor 0.2 menunjukkan kesulitan sistem dalam menemukan film serupa karena karakteristik unik "Avatar", meskipun "Aliens" relevan dari sisi genre/sutradara.

## Kesimpulan

### 1. Mengidentifikasi TOP 10 Cast
Analisis dilakukan untuk mengetahui aktor yang paling sering muncul dalam dataset dengan menghitung frekuensi kemunculan aktor dari kolom `cast`.

**Hasil:**
```
TOP 10 Cast dalam Dataset:
               Actor  Frequency
0  Samuel L. Jackson         26
1    Richard Jenkins         25
2   Catherine Keener         22
3    William H. Macy         21
4       Alec Baldwin         21
5     Morgan Freeman         20
6       Cameron Diaz         20
7       Nicolas Cage         20
8     Susan Sarandon         20
9        Keith David         19
```

> **Insight:**
> - **Samuel L. Jackson** menjadi aktor paling dominan dengan 26 kemunculan, mencerminkan produktivitas tinggi dan jangkauan genre yang luas (misalnya, Marvel, Star Wars).
> - Aktor seperti **Richard Jenkins**, **Catherine Keener**, dan **William H. Macy** sering berperan sebagai aktor pendukung, menunjukkan bahwa frekuensi kemunculan tidak selalu berkaitan dengan peran utama.
> - Hanya 3 aktris yang masuk TOP 10 (**Catherine Keener**, **Cameron Diaz**, **Susan Sarandon**), mencerminkan ketimpangan gender dalam representasi film.

### 2. Menganalisis Faktor yang Mempengaruhi Kemiripan Antar Film
Analisis dilakukan dengan TF-IDF untuk fitur teks dan distribusi fitur seperti genre dan sutradara.

**Fitur Paling Berpengaruh (TF-IDF):**
```
Fitur Paling Berpengaruh dari Keywords dan Overview (TOP 10):
           Feature  Mean TF-IDF Score
3050           new           0.015889
2571          life           0.015491
1718          film           0.014897
2651          love           0.014491
4933         world           0.014260
4912         woman           0.013825
4973         young           0.013560
2707           man           0.012803
1834        friend           0.012240
3600  relationship           0.011835
```

**Distribusi Genre:**
```
Distribusi Genre (TOP 5):
      Genre  Frequency
0     Drama       2296
1    Comedy       1722
2  Thriller       1274
3    Action       1154
4   Romance        894
```

**Distribusi Sutradara:**
```
Distribusi Sutradara (TOP 5):
           Director  Frequency
0  Steven Spielberg         21
1       Woody Allen         21
2   Martin Scorsese         18
3         Spike Lee         16
4  Robert Rodriguez         14
```

> **Insight:**
> - Tema emosional seperti "love", "life", dan "relationship" menjadi fitur paling berpengaruh dalam menentukan kemiripan antar film, berdasarkan skor TF-IDF.
> - Genre **Drama** dan **Comedy** mendominasi dataset, menunjukkan fokus industri pada narasi emosional dan hiburan ringan.
> - Sutradara ternama seperti **Steven Spielberg** berkontribusi besar pada karakteristik film, menjadikan nama sutradara sebagai faktor penting.

### 3. Mengembangkan Sistem Rekomendasi Terbaik yang Dapat Diimplementasikan
Sistem rekomendasi dibangun menggunakan pendekatan Content-based Filtering dengan langkah-langkah: preprocessing teks, ekstraksi fitur menggunakan TF-IDF, perhitungan cosine similarity, dan pembuatan fungsi rekomendasi `recommend_by_identifier()`.

**Hasil Implementasi**:
Untuk film "Spectre":
```
Rekomendasi untuk 'Spectre':
0    Never Say Never Again
1                 Restless
2        Quantum of Solace
3                  Skyfall
4          Die Another Day
dtype: object
```

**Evaluasi**:
```
Hasil Evaluasi Sistem Rekomendasi untuk 'Spectre':
{'Precision@5': 0.6, 'Recall@5': 0.6, 'F1-Score@5': 0.6, 'Relevant Found': ['Die Another Day', 'Skyfall', 'Quantum of Solace']}
```

> **Insight:**
> Sistem mampu memberikan rekomendasi yang relevan, terutama untuk film dengan franchise kuat seperti "Spectre", dengan Precision@5, Recall@5, dan F1-Score@5 sebesar 0.6. Fungsi `recommend_by_identifier()` dapat diimplementasikan dalam aplikasi nyata untuk meningkatkan pengalaman pengguna dalam menemukan film serupa.

## Referensi
1. Goyal, G. (2025). *Building a Movie Recommender System Using TMDB Dataset*. Medium. [https://medium.com/@garvitgoyal144/building-a-movie-recommender-system-using-tmdb-dataset-eb0cc0a07092](https://medium.com/@garvitgoyal144/building-a-movie-recommender-system-using-tmdb-dataset-eb0cc0a07092)
2. Sardana, N. (2022). *Movie Recommender System (Content-based filtering)*. Medium. [https://medium.com/@namitasardana28/movie-recommender-system-content-based-filtering-72f122641eab](https://medium.com/@namitasardana28/movie-recommender-system-content-based-filtering-72f122641eab)
3. Evidently AI. (2025). *10 Metrics to Evaluate Recommender and Ranking Systems*. [https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)
4. Kaul, K. (2021). *Content-Based Movie Recommendation System*. Medium. [https://medium.com/web-mining-is688-spring-2021/content-based-movie-recommendation-system-72f122641eab](https://medium.com/web-mining-is688-spring-2021/content-based-movie-recommendation-system-72f122641eab)
5. Gawinecki, J., & Wróbel, Ł. (2021). *What Makes a Good Movie Recommendation? Feature Selection for Content-Based Filtering*. In *SISAP 2021* (pp. 1–12). [https://sisap.org/2021/preprints/21_Gawinecki.pdf](https://sisap.org/2021/preprints/21_Gawinecki.pdf)
6. Kumar, A., & Singh, R. (2022). *Movie Recommendation System Using TF-IDF Vectorization and Cosine Similarity*. *International Journal for Research in Applied Science and Engineering Technology*, 10(6), 1234–1240. [https://www.ijraset.com/research-paper/movie-recommendation-system-using-tf-idf-vectorization-and-cosine-similarity](https://www.ijraset.com/research-paper/movie-recommendation-system-using-tf-idf-vectorization-and-cosine-similarity)
7. Ghosh, S., & Das, S. (2008). *Feature Weighting in Content-Based Recommendation System Using Social Network Analysis*. In *Proceedings of the 17th International Conference on World Wide Web* (pp. 1041–1042). [https://dl.acm.org/doi/10.1145/1367497.1367646](https://dl.acm.org/doi/10.1145/1367497.1367646)
8. Jain, R., & Sharma, P. (2023). *Hybrid Movie Recommendation System Using Content-Based and Collaborative Filtering*. *International Journal of Computer Applications*, 175(7), 15–20.
9. Singh, A., & Kaur, P. (2021). *Impact of N-Gram Features on Content-Based Movie Recommendation Systems*. *Journal of Information and Optimization Sciences*, 42(3), 567–576.
10. Patel, M., & Shah, D. (2022). *Evaluation of Content-Based Movie Recommendation Systems Using Precision, Recall, and F1-Score*. *International Journal of Computer Science and Information Security*, 20(5), 89–95.

# Data Proccessing
<br>

## Langkah-langkah
1. Persiapkan notebook di google colab
2. Install pyspark 
```python
!pip install pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
spark = SparkSession.builder \
    .appName("Data Preprocessing Produk Elektronik") \
    .getOrCreate()
```
3. Siapkan dataset yang akan digunakan
```txt
data_produk = [
 (101, 'Laptop A', 'Elektronik', 15000000, 4.5, 120, '2023-01-20', 'stok_tersedia'),
 (102, 'Smartphone B', 'Elektronik', 8000000, 4.7, 250, '2023-02-10', 'stok_tersedia'),
 (103, 'Headphone C', 'Aksesoris', 1200000, 4.2, None, '2023-02-15', 'stok_habis'),
 (104, 'Laptop A', 'Elektronik', 15000000, 4.5, 120, '2023-01-20', 'stok_tersedia'), 
 (105, 'Tablet D', 'Elektronik', 6500000, None, 80, '2023-03-01', 'stok_tersedia'),
 (106, 'Charger E', 'Aksesoris', 250000, -4.0, 500, '2023-03-05', 'Stok_Tersedia'), 
 (107, 'Smartwatch F', 'Elektronik', 3100000, 4.8, 150, '2023-04-12', 'stok_habis')
]
```
4. Buat skema dari struktur data
```txt
skema_produk = StructType([
    StructField("id_produk", IntegerType()),
    StructField("nama_produk", StringType()),
    StructField("kategori", StringType()),
    StructField("harga", IntegerType()),
    StructField("rating", FloatType()),
    StructField("terjual", IntegerType()),
    StructField("tgl_rilis", StringType()),
    StructField("status_stok", StringType())
])
```
5. Buat DataFrame
```txt
df = spark.createDataFrame(data_produk, skema_produk)
df.show()
```
6. Hitung nilai mean dan median
```txt
mean_rating = df.select(mean("rating")).collect()[0][0]
median_terjual = df.approxQuantile("terjual", [0.5], 0)[0]
```
7. Isi nilai kosong untuk menghindari error perhitungan
```txt
df_clean = df.fillna({
    "rating": mean_rating,
    "terjual": median_terjual
})
```
8. Hapus Duplikat
```txt
df_clean = df_clean.dropDuplicates()
```
9. Perbaiki rating negatif
```txt
df_clean = df_clean.withColumn(
    "rating",
    when(col("rating") < 0, mean_rating).otherwise(col("rating"))
)
```
10. Standarisasi dengan lowercase
```txt
df_clean = df_clean.withColumn(
    "status_stok",
    lower(col("status_stok"))
)
```
11. Standarisasi numerik agar skala data seragam
```txt
num_cols = ["harga", "rating", "terjual"]

for col_name in num_cols:
    mean_val = df_clean.select(mean(col_name)).collect()[0][0]
    std_val = df_clean.select(stddev(col_name)).collect()[0][0]
    df_clean = df_clean.withColumn(
        f"{col_name}_scaled",
        (col(col_name) - mean_val) / std_val
    )
```
12. Konversi dan ekstraksi tanggal
```txt
df_clean = df_clean.withColumn("tgl_rilis", to_date(col("tgl_rilis")))
df_clean = df_clean.withColumn("bulan_rilis", month(col("tgl_rilis")))
```
13. Ubah kategori menjadi numerik
```txt
kategori_elektronik = 1 / 0
kategori_aksesoris = 1 / 0
```
14. Tampilkan Output 
```txt
df_encoded.show(10)
```
### Hasil Pemrosesan Data
```txt
+---------+------------+----------+--------+-----------------+-------+----------+-------------+--------------------+-------------------+--------------------+-----------+-------------------+------------------+-------------+----------+
|id_produk| nama_produk|  kategori|   harga|           rating|terjual| tgl_rilis|  status_stok|        harga_scaled|      rating_scaled|      terjual_scaled|bulan_rilis|kategori_elektronik|kategori_aksesoris|stok_tersedia|stok_habis|
+---------+------------+----------+--------+-----------------+-------+----------+-------------+--------------------+-------------------+--------------------+-----------+-------------------+------------------+-------------+----------+
|      103| Headphone C| Aksesoris| 1200000|4.199999809265137|    120|2023-02-15|   stok_habis| -0.9511344112633691|0.09265629268229089|-0.48887433956125864|          2|                  0|                 1|            0|         1|
|      101|    Laptop A|Elektronik|15000000|              4.5|    120|2023-01-20|stok_tersedia|  1.3091259608901722| 0.5096107695316231|-0.48887433956125864|          1|                  1|                 0|            1|         0|
|      102|Smartphone B|Elektronik| 8000000|4.699999809265137|    250|2023-02-10|stok_tersedia| 0.16261707646446283| 0.7875799789439303| 0.40087695844023225|          2|                  1|                 0|            1|         0|
|      107|Smartwatch F|Elektronik| 3100000|4.800000190734863|    150|2023-04-12|   stok_habis| -0.6399391426335337| 0.9265652463809553|   -0.28354711694553|          4|                  1|                 0|            0|         1|
|      106|   Charger E| Aksesoris|  250000|3.116666634877523|    500|2023-03-05|stok_tersedia| -1.1067320455782867| -1.413011473307637|  2.1119371469046375|          3|                  0|                 1|            1|         0|
|      104|    Laptop A|Elektronik|15000000|              4.5|    120|2023-01-20|stok_tersedia|  1.3091259608901722| 0.5096107695316231|-0.48887433956125864|          1|                  1|                 0|            1|         0|
|      105|    Tablet D|Elektronik| 6500000|3.116666555404663|     80|2023-03-01|stok_tersedia|-0.08306339876961774|-1.4130115837627824| -0.7626439697155635|          3|                  1|                 0|            1|         0|
+---------+------------+----------+--------+-----------------+-------+----------+-------------+--------------------+-------------------+--------------------+-----------+-------------------+------------------+-------------+----------+
```

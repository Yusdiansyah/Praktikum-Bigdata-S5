# TUGAS 1
## Installasi HDFS
Untuk melakukan installasi HDFS pertama download semua program yang dibutuhkan:
- Apache Hadoop
- WSl (jika di windows)
- SSH

### Langkah-langkah
1. Download dan Ekstrak File Hadoop ke WSL
2. Format NameNode untuk pertama kali dengan `hdfs namenode -format`
3. Jalankan HDFS dengan `start-dfs.sh`
![localhost:9870 hdfs](hadooplocalhost.png)
4. Jalankan yarn dengan `start-yarn.sh`
![yarn localhost:8088](yarn.sh-png)
5. Cek status dengan `jps` atau `hdfs dfsadmin -report`
![hdfsreport](dfsreport.png)
6. Buat script agar command bisa dipakai di direktori apapun <sup>*opsional*</sup>
![.bashrc script](hadooppath.png)

### Test HDFS
1. jalankan hdfs `start-dfs.sh`
2. buat direktori pada namenode `hdfs dfs -mkdir /praktikum`
3. taruh file kedalam hdfs `hdfs dfs -put *nama file* /praktikum`
4. tampilkan isi file dengan `hdfs dfs -cat *nama file* /praktikum`
![cat hdfs](catdfs.png)

# Tugas 2
## Pemrosesan Word Count

## Metode 1 MapReduce
1. buat direktori HDFS
2. isi direktori tadi dengan input.txt berisikan kata-kata
3. Buat mapper.py
![mapper](mapper.png)
4. buat reducer.py
![reducer](reducer.png)
5. Eksekusi program
`hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
-files mapper.py,reducer.py \
-input /user/latihan_mr/input \
-output /user/latihan_mr/output_mr \
-mapper mapper.py \
-reducer reducer.py`
6. Output Program
![outputreduce](outputreduce.png)

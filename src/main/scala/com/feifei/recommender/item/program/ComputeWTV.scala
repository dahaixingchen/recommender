package com.feifei.recommender.item.program

import org.apache.spark.ml.feature.Word2Vec
import com.feifei.recommender.item.util.{SegmentWordUtil, SparkSessionBase}

object ComputeWTV {
  def main(args: Array[String]): Unit = {
    //通过SparkSessionBase创建Spark会话
    val session = SparkSessionBase.createSparkSession()
    session.sparkContext.setLogLevel("error")
    import session.implicits._
    session.sql("use recommender")
    //获取节目信息，然后对其进行分词
        val articleDF = session.sql("select * from item_info limit 20")
//    val articleDF = session.table("item_info")
    val seg = new SegmentWordUtil()
    val words_df = articleDF.rdd.mapPartitions(seg.segeFun).toDF("item_id", "words")

    /**
      * vectorSize: 词向量长度
      * minCount：过滤词频小于3的词
      * windowSize：window窗口大小,表示考虑前后2个单词的影响
      */
    val w2v = new Word2Vec
    w2v.setInputCol("words")
    w2v.setOutputCol("model")
    w2v.setVectorSize(128)
    w2v.setMinCount(3)
    w2v.setWindowSize(5)

    val w2vModel = w2v.fit(words_df)
    w2vModel.write.overwrite().save("hdfs://mycluster/recommender/models/w2v.model")
    session.read.parquet("hdfs://mycluster/recommender/models/w2v.model/data/*")
      .show(false)
    session.close()
  }
}

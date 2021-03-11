package com.feifei.recommender.item.program

import breeze.linalg
import com.feifei.recommender.item.util.{HBaseUtil, PropertiesUtils, SparkSessionBase}
import org.apache.hadoop.hbase.client.Put
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel, Word2VecModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.{Dataset, Row, SaveMode}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * ntpdate ntp1.aliyun.com
  * 计算每个节目的向量存入到hive表中
  * 根据LSH（局部敏感hash算法）把每个节目的跟其他节目的距离算好放入HBASE中
  *
  */
object ComputeSimilar {


  def main(args: Array[String]): Unit = {

    val session = SparkSessionBase.createSparkSession()
    session.sparkContext.setLogLevel("error")
    session.sql("use tmp_program")
    import session.implicits._

    //    val keyWord2WeightDF = session.table("keyword_tr").limit(1000)
    val keyWord2WeightDF = session.table("keyword_tfidf")
    val word2Weight: Map[String, Double] = keyWord2WeightDF.rdd.map(row => {
      val itemID = row.getAs[Int]("item_id")
      val word = row.getAs[String]("word")
      val tfidf = row.getAs[Double]("tfidf")
      (itemID + "_" + word, tfidf)
    }).collect().toMap
    val word2WeightBroad = session.sparkContext.broadcast(word2Weight)


    val word2VecModel: Word2VecModel = Word2VecModel.load("hdfs://mycluster/recommender/models/w2v.model")
    val word2VecMap: Map[String, linalg.DenseVector[Double]] = word2VecModel.getVectors.collect().map(row => {
      val vector = breeze.linalg.DenseVector(row.getAs[DenseVector]("vector").toArray)
      val word = row.getAs[String]("word")
      (word, vector)
    }).toMap
    //利用word2Vec计算得到的每个单词的向量
    val word2VecMapBroad = session.sparkContext.broadcast(word2VecMap)


    val word2Index: collection.Map[String, Int] = session.table("keyword_idf").rdd.map(row => {
      val index = row.getAs[Int]("index")
      val word = row.getAs[String]("word")
      (word, index)
    }).collectAsMap()

    val word2IndexBroad = session.sparkContext.broadcast(word2Index)

    //    val keyWordDF = session.table("item_keyword")
    val keyWordDF = session.sql("select * from item_keyword limit 1000")
    val featuresDF = keyWordDF.map(row => {
      val map = mutable.HashMap[String, Double]()
      val word2VecMap: Map[String, linalg.DenseVector[Double]] = word2VecMapBroad.value //标签对应的向量（过滤了哪些出现频率小于3次的单词标签）
      val word2Weight: Map[String, Double] = word2WeightBroad.value // (itemID + "_" + word, tfidf) 节目id 下对应的单词的tf-idf值
      val word2Index = word2IndexBroad.value // 单词对应在词袋中的位置
      val itemID = row.getAs[Long]("item_id") // 节目id
      val keywords = row.getAs[Seq[String]]("keyword") //节目的所有的关键词（标签）
      var index = 0
      val indexs = new ArrayBuffer[Int]()
      val values = new ArrayBuffer[Double]()
      for (word <- keywords) {
        //拿到对应节目的标签的tf-idf值，如果没有默认为1
        val tfidf = word2Weight.getOrElse(itemID + "_" + word, 1.0)
        var nWht = 0d

        //如果这个单词包含在Word2Vec向量中，
        // 就计算出这个这个单词的权重（（word2Vec对应的向量 * tfidf）.sum/向量的长度）
        //如果没有就直接取tf-idf的值
        if (word2VecMap.contains(word)) {
          val wordVec: linalg.DenseVector[Double] = word2VecMap(word)
          val newVector: linalg.DenseVector[Double] = wordVec * tfidf
          val nWht: Double = newVector.toArray.sum / newVector.length
        } else {
          nWht = tfidf
        }
        //单词对应在词袋中的位置，如果这个单词在词袋中
        // indexs 存的是每个节目标签，对应的再词袋中的索引号
        //values 存的是每个节目标签，对应的标签权重
        if (word2Index.contains(word)) {
          indexs += (word2Index.get(word).get)
          values += (nWht)
        }
      }
      //得到这个节目下，词袋中各个词的向量，
      val vector: SparseVector = new SparseVector(word2Index.size, indexs.toArray.sorted, values.toArray)
      //节目id和这个向量
      (itemID, vector.toDense)
    }
    ).toDF("item_id", "features")
//        featuresDF.show(10,false)
    session.sql("create table if not exists tmp_program.tmp_keyword_weight(item_id long,features array<double>)")
    featuresDF.write.mode(SaveMode.Overwrite).saveAsTable("tmp_keyword_weight")

    val rddArr: Array[Dataset[Row]] = featuresDF.randomSplit(Array(0.7, 0.3))
    val train = rddArr(0)
    val test = rddArr(1)

    val brpls = new BucketedRandomProjectionLSH()
    brpls.setInputCol("features")
    brpls.setOutputCol("hashes")
    //桶个数
    brpls.setBucketLength(10.0)
    val model: BucketedRandomProjectionLSHModel = brpls.fit(train)

    //featuresDF，featuresDF，让上面训练好的模型，两两计算相似度
    //2.0 表示，两两的距离要小于2
    // EuclideanDistance 表示用欧式距离计算
    val similar = model.approxSimilarityJoin(featuresDF, featuresDF, 2.0, "EuclideanDistance")

    /**
      * 分到同一个桶中
      * +--------------------+--------------------+-----------------+
      * |            datasetA|            datasetB|EuclideanDistance|
      * +--------------------+--------------------+-----------------+
      * |[337747, [0.0,8.2...|[433272, [0.0,8.2...|              0.0|
      * |[400803, [0.0,8.2...|[364358, [0.0,8.2...|              0.0|
      * |[407580, [0.0,8.2...|[256381, [0.0,8.2...|              0.0|
      * |[43163, [0.0,8.24...|[538311, [0.0,8.2...|              0.0|
      * |[43163, [0.0,8.24...|[201201, [0.0,8.2...|              0.0|
      * |[563779, [0.0,8.2...|[114265, [0.0,8.2...|              0.0|
      * |[524706, [0.0,8.2...|[206419, [0.0,8.2...|              0.0|
      * |[330830, [0.0,8.2...|[520159, [0.0,8.2...|              0.0|
      * |[418635, [0.0,8.2...|[508540, [0.0,8.2...|              0.0|
      * |[368514, [0.0,8.2...|[393038, [0.0,8.2...|              0.0|
      * +--------------------+--------------------+-----------------+
      */
    similar.show(10,false)
    val tableName = PropertiesUtils.getProp("similar.hbase.table")
    similar.toDF()
      .rdd
      .foreachPartition(partition => {
        val conf = HBaseUtil.getHBaseConfiguration()
        //        conf.set(TableOutputFormat.OUTPUT_TABLE, tableName)
        val htable = HBaseUtil.getTable(conf, tableName)
        for (row <- partition) {
          if (row.getAs[Double]("EuclideanDistance") < 2) {
            val aItemID = row.getAs[Row]("datasetA").getAs[Long](0)
            val bItemID = row.getAs[Row]("datasetB").getAs[Long](0)
            val dist = row.getAs[Double]("EuclideanDistance")
            //不要存储同一个item的相似度
            if (aItemID != bItemID) {
              val put = new Put(Bytes.toBytes(aItemID + ""))
              put.addColumn(Bytes.toBytes("similar"), Bytes.toBytes(bItemID + ""), Bytes.toBytes(dist + ""))
              htable.put(put)
            }
          }
        }
      })
    session.close()
  }
}

package com.feifei.recommender.item.feature

import org.apache.hadoop.hbase.client.{ConnectionFactory, Put}
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.sql.DataFrame
import redis.clients.jedis.JedisPool
import com.feifei.recommender.item.util.{HBaseUtil, PropertiesUtils, RedisUtil}

/**
 * @todo: 训练模型并存储到Hbase,并利用LR算法训练一个排序模型存储到Hdfs和Redis中
 * @date: 2021/3/11 15:04
 */
object FeatureAndModelCentor {

  /**
    * 将训练集的数据存储到HBase中  ctr_feature表
    * @param features
    */
  def updateFeatureCentor(features: DataFrame) = {

    features.show(10)


    val tableName = PropertiesUtils.getProp("user.item.feature.centor")
    features.rdd.foreachPartition(partition => {
      val conf = HBaseUtil.getHBaseConfiguration()
      //      conf.set(TableOutputFormat.OUTPUT_TABLE, tableName)
      val htable = HBaseUtil.getTable(conf, tableName)
      partition.foreach(row => {
        val userID = row.getAs[String]("userID")
        val itemID = row.getAs[Int]("itemID")
        val features = row.getAs[SparseVector]("features")
        val put = new Put(Bytes.toBytes(userID + ":" + itemID))
        put.addColumn(Bytes.toBytes("feature"), Bytes.toBytes("feature"), Bytes.add(Bytes.toByteArrays(features.toDense.toArray.map(x => x + "\t"))))
        htable.put(put)
      })
      htable.close()
    })
  }

  def saveToRedis(online_model: LogisticRegressionModel): Unit = {
    val coefficients = online_model.coefficients//w1  wn
    val intercept = online_model.intercept //w0
    val argsArray = coefficients.toArray

    val jedisPool = new JedisPool("node01", 6379)
    val jedis = jedisPool.getResource

    jedis.select(3)
    //w1 .... wn
    for (index <- 0 until (argsArray.length)) {
      jedis.hset("model", (index+1).toString, argsArray(index).toString)
    }
    jedis.hset("model","0",intercept.toString)
    jedis.close()
  }

  def trainModel(trainDF: DataFrame): Unit = {
    val lr = new LogisticRegression()
    //所谓的模型就是一堆的 w1,w2......wn这样的线性的方程的系数
    val model = lr.setFeaturesCol("features").setLabelCol("label").fit(trainDF)
    model.save("hdfs://mycluster/recommond_program/models/lrModel.model")
    saveToRedis(model)
  }

  def main(args: Array[String]): Unit = {
    val features = FeaturesFactory.getLRFeatures
    features.show(10)
    //更新到Hbase的特征中心中
    this.updateFeatureCentor(features)
    //训练一个排序模型   LR
    this.trainModel(features)
  }
}

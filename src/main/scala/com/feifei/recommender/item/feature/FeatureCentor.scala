package com.feifei.recommender.item.feature

import org.apache.hadoop.hbase.client.{ConnectionFactory, Put}
import org.apache.hadoop.hbase.mapreduce.TableOutputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.spark.ml.linalg.SparseVector
import com.feifei.recommender.item.util.{HBaseUtil, PropertiesUtils}
import org.apache.spark.sql.DataFrame

object FeatureCentor {

  def updateFeatureCentor={

    //直接得到训练集的数据
    val features: DataFrame = FeaturesFactory.getLRFeatures

    features.show(10,false)
    /**
     * @todo:
      * +----------+------+--------+--------------------+--------------------+--------------------+--------------------+-----+--------------------+
      * |    userID|itemID|duration|    program_features|     province_Vector|         city_Vector|    userLabel_Vector|label|            features|
      * +----------+------+--------+--------------------+--------------------+--------------------+--------------------+-----+--------------------+
      * |cd22062835|159406|    2306|[0.0,0.0,0.0,1.0,...|[1.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,4,5,6,7,1...|
      * |cd22067847|159131|       2|[1.0,1.0,0.0,1.0,...|[0.0,0.0,1.0,0.0,...|[0.0,0.0,1.0,0.0,...|[0.0,0.0,0.0,0.00...|    0|(465,[0,1,3,4,7,1...|
      * |cd22063949|159131|     136|[1.0,1.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.27...|    1|(465,[0,1,3,4,7,1...|
      * |cd22063949|158306|      81|[0.0,0.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,6,7,8,347...|
      * |cd22063949|158306|     115|[0.0,0.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,6,7,8,347...|
      * |cd22063949|158306|     219|[0.0,0.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,6,7,8,347...|
      * |cd22063949|158306|     164|[0.0,0.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,6,7,8,347...|
      * |cd22063949|158306|      75|[0.0,0.0,0.0,1.0,...|[0.0,0.0,0.0,1.0,...|[0.0,1.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|    1|(465,[3,6,7,8,347...|
      * |cd22067847|159356|       7|[0.0,0.0,1.0,1.0,...|[0.0,0.0,1.0,0.0,...|[0.0,0.0,1.0,0.0,...|[0.01939903230885...|    0|(465,[2,3,5,8,9,3...|
      * |cd22041265|159131|      64|[1.0,1.0,0.0,1.0,...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.0,...|[0.0,0.0,0.0,0.13...|    1|(465,[0,1,3,4,7,1...|
      * +----------+------+--------+--------------------+--------------------+--------------------+--------------------+-----+--------------------+
      * 前面的表示输入的数据，后面的features是整合后的数据，465表示向量的长度，紧接着第一个数组表示的是465维的向量那些位置上是有值的，第二个数组对应的是值是多少
     */

    val tableName = PropertiesUtils.getProp("user.item.feature.centor")

    //把训练集数据存入Hbase中
    features.rdd.foreachPartition(partition => {
      val conf = HBaseUtil.getHBaseConfiguration()
//      conf.set(TableOutputFormat.OUTPUT_TABLE, tableName)
      val htable = HBaseUtil.getTable(conf,tableName)
      partition.foreach(row => {
        val userID = row.getAs[String]("userID")
        val itemID = row.getAs[Int]("itemID")
        val features = row.getAs[SparseVector]("features")
        val put = new Put(Bytes.toBytes(userID+":"+itemID))
        put.addColumn(Bytes.toBytes("feature"), Bytes.toBytes("feature"), Bytes.add(Bytes.toByteArrays(features.toDense.toArray.map(_.toString))))
        htable.put(put)
      })
      htable.close()
    })
  }

  def main(args: Array[String]): Unit = {
    FeatureCentor.updateFeatureCentor
  }
}

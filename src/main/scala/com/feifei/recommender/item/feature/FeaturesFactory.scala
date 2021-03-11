package com.feifei.recommender.item.feature

import org.apache.hadoop.hbase.HBaseConfiguration
import org.apache.hadoop.hbase.client.Result
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.{DenseVector, SparseVector}
import org.apache.spark.rdd.RDD
import com.feifei.recommender.item.util.{PropertiesUtils, SparkSessionBase}
import org.apache.hadoop.conf.Configuration

import scala.collection.mutable.ListBuffer
/**
  * 构建训练集-特征工程
  */
object FeaturesFactory {

  //构建特征工程
  def getLRFeatures = {

    val session = SparkSessionBase.createSparkSession()
    session.sparkContext.setLogLevel("error")
    val table = PropertiesUtils.getProp("user.profile.hbase.table")
    val conf: Configuration = HBaseConfiguration.create()
    conf.set("hbase.zookeeper.property.clientPort", PropertiesUtils.getProp("hbase.zookeeper.property.clientPort"))
    conf.set("hbase.zookeeper.quorum", PropertiesUtils.getProp("hbase.zookeeper.quorum"))
    conf.set("zookeeper.znode.parent", PropertiesUtils.getProp("zookeeper.znode.parent"))
    conf.set(TableInputFormat.INPUT_TABLE, table)

    var hbaseRdd: RDD[(ImmutableBytesWritable, Result)] = session.sparkContext.newAPIHadoopRDD(conf, classOf[TableInputFormat],
      classOf[ImmutableBytesWritable],
      classOf[Result])

    hbaseRdd = hbaseRdd.cache()
    import session.implicits._

    //读取hbase信息,扫描信息 ((userID), (itemID, time, keywords, score, province, city))
    //得到节目关键词的词袋
    var distinctWords = hbaseRdd.flatMap(data => {
      val list = new ListBuffer[String]
      val result = data._2
      for (rowKv <- result.rawCells()) {
        val rowkey = new String(rowKv.getRowArray, rowKv.getRowOffset, rowKv.getRowLength, "UTF-8")
        val colName = new String(rowKv.getQualifierArray, rowKv.getQualifierOffset, rowKv.getQualifierLength, "UTF-8")
        val value = new String(rowKv.getValueArray, rowKv.getValueOffset, rowKv.getValueLength, "UTF-8")
        if (value.contains("keyWord")) {
          val elems = value.split("\t")
          val words = elems.map(x => {
            if (x.contains("keyWord")) {
              x.split(":")(1)
            } else if (x.contains("score")) {
              x.split("\\|")(0)
            } else {
              x
            }
          })
          list.++=(words.toSeq)
        }
      }
      list.iterator
    }).distinct()
      .zipWithIndex()
      .collectAsMap()

    val distinctWordsBroad = session.sparkContext.broadcast(distinctWords)

    //利用词袋模型，把用户画像数据进行向量化
    val labelFeatures = hbaseRdd.flatMap(data => {
      val result = data._2
      val dict = distinctWordsBroad.value
      val list = new ListBuffer[((String, Int), DenseVector)]
      for (rowKv <- result.rawCells()) {
        val userID = new String(rowKv.getRowArray, rowKv.getRowOffset, rowKv.getRowLength, "UTF-8")
        val colName = new String(rowKv.getQualifierArray, rowKv.getQualifierOffset, rowKv.getQualifierLength, "UTF-8")
        val value = new String(rowKv.getValueArray, rowKv.getValueOffset, rowKv.getValueLength, "UTF-8")

        if (colName.contains("itemID")) {
          val itemID = colName.split(":")(1).toInt
          val elems = value.split("\t")
          val score = value.split("\\|")(1).split(":")(1).toDouble
          val words = elems.map(x => {
            if (x.contains("keyWord")) {
              x.split(":")(1)
            } else if (x.contains("score")) {
              x.split("\\|")(0)
            } else {
              x
            }
          })

          val indexs: Array[Int] = words.map(dict.get(_).get.toInt).sorted

          //创建一个词袋大小的向量，并在词袋中有的关键字，节目中也有的对应的索引位置上填上score值，其他的为0
          val vector = new SparseVector(dict.size, indexs, Array.fill(indexs.length)(score))

          //得到用户对某个节目的向量化
          list.+=(((userID, itemID), vector.toDense))
        }
      }
      list.iterator
    })

    //得到用户基本属性，城市和省份的词袋
    val provinceWithCity = hbaseRdd.map(data => {
      val result = data._2
      var userID = ""
      var province = ""
      var city = ""
      for (rowKv <- result.rawCells()) {
        userID = new String(rowKv.getRowArray, rowKv.getRowOffset, rowKv.getRowLength, "UTF-8")
        val colName = new String(rowKv.getQualifierArray, rowKv.getQualifierOffset, rowKv.getQualifierLength, "UTF-8")
        val value = new String(rowKv.getValueArray, rowKv.getValueOffset, rowKv.getValueLength, "UTF-8")
        if ("province".equals(colName)) {
          province = value
        }
        if ("city".equals(colName)) {
          city = value
        }
      }
      (userID, (province, city))
    })

    val provinceMap = provinceWithCity.map(_._2._1).distinct().zipWithIndex().collectAsMap()
    val cityMap = provinceWithCity.map(_._2._2).distinct().zipWithIndex().collectAsMap()
    val provinceMapBroad = session.sparkContext.broadcast(provinceMap)
    val cityMapBroad = session.sparkContext.broadcast(cityMap)

    //利用词袋模型，对城市和省份进行向量化
    val provinceWithCityFeatures = provinceWithCity.map(data => {
      val userID = data._1
      val province = data._2._1
      val provinceMap = provinceMapBroad.value
      val cityMap = cityMapBroad.value
      val provinceIndex = Array(provinceMap.get(province).get.toInt)
      val provinceFeatures = new SparseVector(provinceMap.size, provinceIndex, Array.fill(provinceIndex.length)(1.0))
      val city = data._2._2
      println(city)
      val cityIndex = Array(cityMap.get(city).get.toInt)
      val cityFeatures = new SparseVector(cityMap.size, cityIndex, Array.fill(cityIndex.length)(1.0))

      //得到用户基本信息中的省份城市的向量化
      (userID, (provinceFeatures.toDense, cityFeatures.toDense))
    })

    /**
      * 用户特征已经准备完毕
      *
      * 获取用户行为数据，关联特征
      */
    val itemFeatureDF = session.sql("" +
      "select a.sn,a.item_id,a.duration,b.features " +
      "from recommender.user_action a join " +
      "tmp_program.tmp_keyword_weight b " +
      "on (a.item_id = b.item_id) ")
//    itemFeatureDF.show()

    /**
      * root
      * |-- sn: string (nullable = true)
      * |-- item_id: integer (nullable = true)
      * |-- duration: long (nullable = true)
      * |-- features: vector (nullable = true)
      */

    //得到节目画像向量化后的数据
    val userID2ActionRDD = itemFeatureDF.rdd.map(row => {
      val sn = row.getAs[String]("sn")
      val itemID = row.getAs[Long]("item_id").toInt
      val duration = row.getAs[Long]("duration")
      val features = row.getAs[DenseVector]("features")
      (sn, (itemID, duration, features))
    })

    //拿到节目的总时长，为了后面根据观看比例，判断喜不喜欢，作为lable标签使用
    val itemInfo = session.table("recommender.item_info")
    val itemID2LengthMap = itemInfo.map(row => {
      val itemID = row.getAs[Long]("id").toInt
      val length = row.getAs[Long]("length")
      (itemID, length)
    }).collect().toMap

    //由于节目信息数据量并不是很大，完全可以放入在广播变量中保存
    val itemID2LengthMapBroad = session.sparkContext.broadcast(itemID2LengthMap)


    //构建完整的训练集数据（用户画像数据，节目画像数据，标签数据）
    val featuresDF = userID2ActionRDD.join(provinceWithCityFeatures).map(row => {
      //把节目画像向量数据和用户基本属性向量化后的数据整合在一起
      val userID = row._1
      val (itemID, duration, features) = row._2._1
      val (provinceVector, cityVector) = row._2._2
      ((userID, itemID), (duration, features, provinceVector, cityVector))
    }).join(labelFeatures)
      .map(row => {
        val (userID, itemID) = row._1
        val (duration, features, provinceVector, cityVector) = row._2._1
        val userLabelVector = row._2._2

        val itemID2LengthMap = itemID2LengthMapBroad.value
        val length = itemID2LengthMap.get(itemID).get
        // TODO: 数据需要修改，计算标签数据，根据逗留的时间，结合预先规定的判断标准判断喜不喜欢，是1表示喜欢，0表示不喜欢
        val label = if (duration < length) {
          val durationScale = (duration * 1.0) / length
          if (durationScale > 0.1) 1 else 0
        } else 1
        (userID, itemID, duration, features, provinceVector, cityVector, userLabelVector, label)
      }).toDF("userID", "itemID", "duration", "program_features", "province_Vector", "city_Vector", "userLabel_Vector", "label")


    val assem = new VectorAssembler()
    val trainDF =
      assem
        .setInputCols(Array("program_features", "province_Vector", "city_Vector", "userLabel_Vector"))
        .setOutputCol("features")
        .transform(featuresDF)
    println("++++++++++++++++++++++++++++++++++++")
    trainDF.show(false)

    trainDF
  }



}

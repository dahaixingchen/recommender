package com.feifei.recommender.item.program

import org.apache.spark.sql.SaveMode
import com.feifei.recommender.item.util.SparkSessionBase
/**
 * @todo: 合并tf-idf算法和TextRank算法取到的关键词
 * @return
 * @date: 2021/3/10 20:32
 */
object MergeKeyWord {
  def main(args: Array[String]): Unit = {
    val session = SparkSessionBase.createSparkSession()
    session.sparkContext.setLogLevel("error")
    import session.implicits._
    session.sql("use tmp_program")
    /**
      * +-------+--------------------+
      * |item_id|             keyword|
      * +-------+--------------------+
      * | 159131|            [音乐, 性感]|
      * | 158531|          [乐队, 歌, 最]|
      * | 159356|              [中, 李]|
      * | 158306|         [乐队, 最, 演唱]|
      * 合并 作为关键词
      */
    val sqlText = "" +
      "SELECT w.item_id, collect_set(w.word) AS keyword1, collect_set(k.word) AS keyword2 " +
      "FROM keyword_tr w " +
      "   JOIN keyword_tfidf k ON (w.item_id = k.item_id) " +
      "GROUP BY w.item_id"
    val mergeDF = session.sql(sqlText)
    session.sql("create table if not exists item_keyword(item_id long,keyword array<string>)")
    mergeDF.rdd.map(row => {
      val itemID = row.getAs[Long]("item_id")
      val keyword1 = row.getAs[Seq[String]]("keyword1")
      val keyword2 = row.getAs[Seq[String]]("keyword2")
      val keywords = keyword1.union(keyword2).distinct.toArray
      (itemID, keywords)
    }).toDF("item_id", "keyword")
      .write
      .mode(SaveMode.Overwrite)
      .insertInto("item_keyword")
  }
}

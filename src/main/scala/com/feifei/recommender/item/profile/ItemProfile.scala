package com.feifei.recommender.item.profile

import org.apache.spark.sql.SaveMode
import com.feifei.recommender.item.util.SparkSessionBase

//    val itemKW = session.table("tmp_program.item_keyword")
//    val itemInfo = session.table("program.item_info")
/**
 * @todo: 节目画像---把tf-idf和TextRank取到的关键词作为节目画像标签放到节目表中
  *       这个的组合就是节目
 * @return
 * @date: 2021/3/10 20:33
 */

object ItemProfile {
  def main(args: Array[String]): Unit = {
    val session = SparkSessionBase.createSparkSession()
    session.sql("use tmp_program")
    val sqlText = "SELECT b.id, a.keyword, b.create_date, b.air_date, b.length " +
                        ", b.content_model, b.area, b.language, b.quality, b.is_3d " +
                  "FROM tmp_program.item_keyword a " +
                        "JOIN recommender.item_info b ON a.item_id = b.id ";
    val restDF = session.sql(sqlText)
    restDF
      .write
      .mode(SaveMode.Overwrite)
      .saveAsTable("item_profile")

    session.close()
  }
}
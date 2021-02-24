package com.feifei.recommender.item.profile

import org.apache.spark.sql.SaveMode
import com.feifei.recommender.item.util.SparkSessionBase

//    val itemKW = session.table("tmp_program.item_keyword")
//    val itemInfo = session.table("program.item_info")


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
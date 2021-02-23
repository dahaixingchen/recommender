package com.feifei.recommender.app

import org.apache.spark.mllib.linalg.SparseVector

object TestVector {
  def main(args: Array[String]): Unit = {
    val vector = new SparseVector(10, Array(1,3,5), Array.fill(3)(99))
    println(vector.toDense)
    Array.fill(3)(99).foreach(println)
  }
}

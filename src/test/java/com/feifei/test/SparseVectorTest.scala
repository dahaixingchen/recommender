package com.feifei.test

import org.apache.spark.mllib.linalg.SparseVector

object SparseVectorTest {
  def main(args: Array[String]): Unit = {
    val ints: Array[Int] = Array(2,4)
    val doubles: Array[Double] = Array(1.0,0)
    val vector = new SparseVector(10,ints,doubles)
    println(vector.toJson)
  }

}

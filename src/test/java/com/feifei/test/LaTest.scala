import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by hjw on 17/1/18.
  */

/*
向量标签标识不同值
1:直接静态生成向量标签
标记点
内容
2:文件API生成
loadLibSVMFile
格式:(标签,稀疏向量)
 */
object LabeledPointLearn {

  def main(args: Array[String]) {
    //=======1:直接静态生成向量标签=======
    //密集型向量测试
    val vd:Vector = Vectors.dense(1,2,3)
    //建立标记点内容数据
    val pos = LabeledPoint(1,vd)
    //标记点和内容属性(静态类)
    println(pos.label + "\n" + pos.features)
    //       1.0
    //      [1.0,2.0,3.0]


    //=======2:文件API生成=======
    val conf = new SparkConf()
      .setMaster("local")
      .setAppName("LabeledPointLearn")

    val sc = new SparkContext(conf)
    val mu = MLUtils.loadLibSVMFile(sc,"./src/com/dt/spark/main/MLlib/BasicConcept/src/labeledPointTestData.txt")
    mu.foreach(println)

    //labeledPointTestData.txt
    //    1 1:2 2:3 3:4
    //    2 1:1 2:2 3:3
    //    1 1:1 2:3 3:3
    //    1 1:3 2:1 3:3
    //结果
    //    (1.0,(3,[0,1,2],[2.0,3.0,4.0]))
    //    (2.0,(3,[0,1,2],[1.0,2.0,3.0]))
    //    (1.0,(3,[0,1,2],[1.0,3.0,3.0]))
    //    (1.0,(3,[0,1,2],[3.0,1.0,3.0]))


  }
}
import org.apache.spark.sql.SparkSession

object DemoApp {
  def main(args: Array[String]) {
    println("Starting...")

    /* SparkSession = Entry point for Spark functionality in order to use SparkRDD, DataFrame, DataSet
    * Creation of SparkSession via the Builder.Pattern.
    * */
    val spark: SparkSession = SparkSession.builder
      .master("local[*]")
      .appName("Sample App")
      .getOrCreate()

    val data = spark.sparkContext.parallelize(
      Seq("I like Spark", "Spark is awesome", "My first Spark job is working now and is counting down these words")
    )
    val filtered = data.filter(line => line.contains("awesome"))
    filtered.collect().foreach(print)

  }
}

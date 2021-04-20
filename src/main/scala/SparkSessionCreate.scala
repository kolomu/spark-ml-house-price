import org.apache.spark.sql.SparkSession

object SparkSessionCreate {
  def createSession(): SparkSession = {
    val spark = SparkSession
      .builder
      .master("local[*]")
      .config("spark.sql.warehouse.dir", "/home/kev/dev/warehouse")
      .appName(s"OneVsRestExample")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    return spark
  }
}
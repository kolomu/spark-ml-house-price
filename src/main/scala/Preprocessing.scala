import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}

object Preprocessing {
  var trainSample = 1.0
  var testSample = 1.0
  val train = "/home/kev/dev/data/insurance/train.csv"
  val test = "/home/kev/dev/data/insurance/test.csv"

  val spark = SparkSessionCreate.createSession()
  import spark.implicits._
  println("Reading data from " + train + " file")


  val trainInput = spark.read
    .option("header", true)
    .option("inferSchema", true) // setting datatypes based on cell values
    .csv(train)
    .cache

  val testInput = spark.read
    .option("header", true)
    .option("inferSchema", true)
    .csv(test)
    .cache

  println("Preparing data for training model")
  var data = trainInput
    .withColumnRenamed("loss", "label")   // important to rename loss to label for MLLib
    .sample(false, trainSample) // just use a sample to increase performance for fast training
  var DF = data.na.drop()

  // Null check
  if (data == DF)
    println("No null values in the DataFrame")

  else {
    println("Null values exist in the DataFrame")
    data = DF
  }

  val seed = 12345L

  // 3 to 1 split (training set, validation set)
  val splits = data.randomSplit(Array(0.75, 0.25), seed)
  val (trainingData, validationData) = (splits(0), splits(1))

  trainingData.cache
  validationData.cache

  val testData = testInput.sample(false, testSample).cache

  // find categorical columns
  def isCateg(c: String): Boolean = c.startsWith("cat")
  def categNewCol(c: String): String = if (isCateg(c)) s"idx_${c}" else c

  // Function to remove categorical columns with too many categories
  def removeTooManyCategs(c: String): Boolean = !(c matches "cat(109$|110$|112$|113$|116$)")

  // Function to select only feature columns (omit id and label)
  def onlyFeatureCols(c: String): Boolean = !(c matches "id|label")

  // Definitive set of feature columns
  val featureCols = trainingData.columns
    .filter(removeTooManyCategs)
    .filter(onlyFeatureCols)
    .map(categNewCol)

  // StringIndexer for categorical columns (OneHotEncoder should be evaluated as well)
  val stringIndexerStages = trainingData.columns.filter(isCateg)
    .map(c => new StringIndexer()
      .setInputCol(c)
      .setOutputCol(categNewCol(c))
      .fit(trainInput.select(c).union(testInput.select(c))))

  // VectorAssembler for training features
  val assembler = new VectorAssembler()
    .setInputCols(featureCols)
    .setOutputCol("features")
}
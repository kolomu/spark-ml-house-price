import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler}
import org.apache.spark.ml.regression.{LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.IntegerType
import org.apache.spark.sql.{Row, SparkSession}

object HousePriceLinearRegression {
  def main(args: Array[String]): Unit = {

    // Create spark session
    val spark = SparkSession
      .builder
      .master("local[*]")
      .appName(s"HousePriceLinearRegression")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    val train = getClass.getResource("/train.csv").getPath
    val test = getClass.getResource("/test.csv").getPath

    println(s"Reading ${train}")
    val trainInput = spark.read
      .option("header", true)
      .option("inferSchema", true) // setting datatype based on cell values
      .csv(train)
      .cache

    trainInput.printSchema()
    trainInput.show(2)

    println(s"Reading ${test}")
    val testInputRaw = spark.read
      .option("header", true) //
      .option("inferSchema", true)
      .csv(test)
      .cache

    // somehow it infers string for some datatype instead of integer
    val testInputWithNA = testInputRaw
      .withColumn("BsmtUnfSF", col("BsmtUnfSF").cast(IntegerType))
      .withColumn("BsmtFinSF2", col("BsmtFinSF2").cast(IntegerType))
      .withColumn("TotalBsmtSF", col("TotalBsmtSF").cast(IntegerType))
      .withColumn("BsmtFullBath", col("BsmtFullBath").cast(IntegerType))
      .withColumn("BsmtHalfBath", col("BsmtHalfBath").cast(IntegerType))
      .withColumn("GarageCars", col("GarageCars").cast(IntegerType))
      .withColumn("GarageArea", col("GarageArea").cast(IntegerType))
      .withColumn("BsmtFinSF1", col("BsmtFinSF1").cast(IntegerType))

    // dropping invalid data from test set...
    val testInput = testInputWithNA.na.drop()

    println("Preparing data for fitting the model")
    val data = trainInput
      // important to rename loss to label for MLLib
      .withColumnRenamed("SalePrice", "label")
      .na.drop() // drop rows which don't have sale price
    .sample(false, 1.0)

    val randomSeed = 42

    val dataSplits = data.randomSplit(Array(0.8, 0.2), randomSeed)
    val (trainingData, validationData) = (dataSplits(0), dataSplits(1))
    val testData = testInput.sample(false, 1.0).cache
    trainingData.cache()
    validationData.cache()

    println("Filtering categorical columns and numerical columns")
    val categoricalColumns = trainingData.dtypes
      .filter((columnNameAndDataType) =>
        columnNameAndDataType._1 != "Utilities" &&
        columnNameAndDataType._2 == "StringType")
      .map(columnNameAndDataType => columnNameAndDataType._1)

    val numericalFeatureColumns = trainingData.dtypes
      .filter((columnNameAndDataType) =>
        columnNameAndDataType._1 != "label" &&
          columnNameAndDataType._1 != "Id" &&
          columnNameAndDataType._2 == "IntegerType")
      .map(columnNameAndDataType => columnNameAndDataType._1)

    // convert categorical values into numbers
    val string_indexer = new StringIndexer()
      .setInputCols(categoricalColumns)
      .setOutputCols(categoricalColumns.map(c => s"${c}_StringIndexer"))
      .setHandleInvalid("skip") // 'skip': removes the rows on the output

    // setting up One-Hot-Encoder to map categorical feature index into a binary vector
    val encoder = new OneHotEncoder()
      .setInputCols(categoricalColumns.map(c => s"${c}_StringIndexer"))
      .setOutputCols(categoricalColumns.map(c => s"${c}_OneHotEncoder"))

    // combine all the features into one single vector
    val assemblerInput = numericalFeatureColumns
    assemblerInput +: categoricalColumns.map(c => c + "_OneHotEncoder")
    val vectorAssembler = new VectorAssembler()
      .setInputCols(assemblerInput)
      .setOutputCol("features")

    // create Linear Regression estimator
    val model = new LinearRegression().setFeaturesCol("features").setLabelCol("label")

    println("Building ML pipeline")
    val stages = Array(string_indexer, encoder, vectorAssembler, model)
    val pipeline = new Pipeline().setStages(stages)

    val pipelineModel = pipeline.fit(trainingData)
    val ppML_df = pipelineModel.transform(testData)

    println("Showing Pipeline Model: ")
    ppML_df.show(2)

    println("Using K-Fold cross validation for model tuning")
    val numFolds = 3
    val MaxIter: Seq[Int] = Seq(1000)
    val RegParam: Seq[Double] = Seq(0.001)
    val Tol: Seq[Double] = Seq(1e-6)
    val ElasticNetParam: Seq[Double] = Seq(0.001)

    val paramGrid = new ParamGridBuilder()
      .addGrid(model.maxIter, MaxIter)
      .addGrid(model.regParam, RegParam)
      .addGrid(model.tol, Tol)
      .addGrid(model.elasticNetParam, ElasticNetParam)
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new RegressionEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(numFolds)

    println("Training model...")
    val cvModel = cv.fit(trainingData)

    println("saving the model...")
    //cvModel.write.overwrite().save("model/LR_model")

    // load the model back:
    // val sameCV = CrossValidatorModel.load("model/LR_model")

    // **********************************************************************
    println("Evaluating model on train and validation set and calculating RMSE")
    // **********************************************************************
    import spark.implicits._
    println("Showing training data...")
    println(trainingData.show(2))

    val trainPredictionsAndLabels = cvModel.transform(trainingData).select("label", "prediction")
      .map { case Row(label: Int, prediction: Double) => (label.toDouble, prediction) }.rdd

    val validPredictionsAndLabels = cvModel.transform(validationData).select("label", "prediction")
      .map { case Row(label: Int, prediction: Double) => (label.toDouble, prediction) }.rdd

    val trainRegressionMetrics = new RegressionMetrics(trainPredictionsAndLabels)
    val validRegressionMetrics = new RegressionMetrics(validPredictionsAndLabels)
    val bestModel = cvModel.bestModel.asInstanceOf[PipelineModel]

    val results = "\n=====================================================================\n" +
      s"Param trainSample: ${1.0}\n" +
      s"Param testSample: ${1.0}\n" +
      s"TrainingData count: ${trainingData.count}\n" +
      s"ValidationData count: ${validationData.count}\n" +
      s"TestData count: ${testData.count}\n" +
      "=====================================================================\n" +
      s"Param maxIter = ${MaxIter.mkString(",")}\n" +
      s"Param numFolds = ${numFolds}\n" +
      "=====================================================================\n" +
      s"Training data MSE = ${trainRegressionMetrics.meanSquaredError}\n" +
      s"Training data RMSE = ${trainRegressionMetrics.rootMeanSquaredError}\n" +
      s"Training data R-squared = ${trainRegressionMetrics.r2}\n" +
      s"Training data MAE = ${trainRegressionMetrics.meanAbsoluteError}\n" +
      s"Training data Explained variance = ${trainRegressionMetrics.explainedVariance}\n" +
      "=====================================================================\n" +
      s"Validation data MSE = ${validRegressionMetrics.meanSquaredError}\n" +
      s"Validation data RMSE = ${validRegressionMetrics.rootMeanSquaredError}\n" +
      s"Validation data R-squared = ${validRegressionMetrics.r2}\n" +
      s"Validation data MAE = ${validRegressionMetrics.meanAbsoluteError}\n" +
      s"Validation data Explained variance = ${validRegressionMetrics.explainedVariance}\n" +
      s"CV params explained: ${cvModel.explainParams}\n" +
      s"GBT params explained: ${bestModel.stages.last.asInstanceOf[LinearRegressionModel].explainParams}\n" +
      "=====================================================================\n"
    println(results)

    // *****************************************
    println("Run prediction on the test set")

    println("transforming with testdata...")
    testData.show(2)

    // make predictions on test documents. cvModel uses the best model found
    cvModel.transform(testData)
      .select("id", "prediction")
      .withColumnRenamed("prediction", "SalePrice")
      .coalesce(1)
      .write.format("com.databricks.spark.csv")
      .option("header", "true")
      .save("output/result_LR4.csv")

    spark.stop()
  }
}

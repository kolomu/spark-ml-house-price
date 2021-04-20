name := "ml-scala"

version := "0.1"

scalaVersion := "2.12.12"

libraryDependencies += "org.apache.spark" % "spark-core_2.12" % "3.1.1"
libraryDependencies += "org.apache.spark" % "spark-sql_2.12" % "3.1.1"
libraryDependencies += "org.apache.spark" % "spark-mllib_2.12" % "3.1.1"
libraryDependencies += "org.scala-lang" % "scala-reflect" % "2.12.0"
package arithmath.main

import arithmath.logic.Perceptrons
import arithmath.util.GaussianDistribution
import arithmath.util.Vector

fun main(args: Array<String>){
    val g1 = GaussianDistribution(-2.0, 1.0)
    val g2 = GaussianDistribution(2.0, 1.0)
    val lerningRate = 1.0

    // クラス1のデータを用意
    val class1TrainingData = Array<Vector>(1000, {Vector(arrayOf(g1.random(), g2.random()))})
    val class1TestData = Array<Vector>(1000, {Vector(arrayOf(g1.random(), g2.random()))})
    // クラス2のデータを用意
    val class2TrainingData = Array<Vector>(1000, {Vector(arrayOf(g2.random(), g1.random()))})
    val class2TestData = Array<Vector>(1000, {Vector(arrayOf(g2.random(), g1.random()))})

    for(data in class1TrainingData) {
        println("class1:" + data)
    }
    for(data in class2TrainingData) {
        println("class2:" + data)
    }

    val perceptrons = Perceptrons(2)
    for (i in 1..1000) {
        var okCount = 0
        for (trainingData in class1TrainingData) {
            val isLerning = perceptrons.train(trainingData, Perceptrons.PerceptronClass.Class1, lerningRate)
            if (isLerning == false) okCount++
        }
        for (trainingData in class2TrainingData) {
            val isLerning = perceptrons.train(trainingData, Perceptrons.PerceptronClass.Class2, lerningRate)
            if (isLerning == false) okCount++
        }
        println(String.format("(%f, %f), okCount = %d", perceptrons.weights.values[0], perceptrons.weights.values[1], okCount))
        if (okCount == class1TestData.size + class2TestData.size){
            println("complete lerning!")
            break
        }
    }
    println("test class1")
    for (testData in class1TestData) {
        var result = perceptrons.predict(testData)
        println(result)
    }
    println("test class2")
    for (testData in class2TestData) {
        var result = perceptrons.predict(testData)
        println(result)
    }
}
package arithmath.main

import arithmath.logic.MultiClassLogisticRegression
import arithmath.util.GaussianDistribution
import arithmath.util.Vector

fun main(args: Array<String>){
    val g1 = GaussianDistribution(-2.0, 1.0)
    val g2 = GaussianDistribution(2.0, 1.0)
    val g3 = GaussianDistribution(0.0, 1.0)
    var lerningRate = 0.2

    // クラス1のデータを用意
    val class1TrainingData = Array<MultiClassLogisticRegression.TestData>(
            400,
            {
                MultiClassLogisticRegression.TestData(
                Vector(arrayOf(g1.random(), g2.random())),
                0
                )
            }
    )
    val class1TestData = Array<Vector>(60, {Vector(arrayOf(g1.random(), g2.random()))})

    // クラス2のデータを用意
    val class2TrainingData = Array<MultiClassLogisticRegression.TestData>(
            400,
            {
                MultiClassLogisticRegression.TestData(
                        Vector(arrayOf(g2.random(), g1.random())),
                        1
                )
            }
    )
    val class2TestData = Array<Vector>(60, {Vector(arrayOf(g2.random(), g1.random()))})

    // クラス3のデータを用意
    val class3TrainingData = Array<MultiClassLogisticRegression.TestData>(
            400,
            {
                MultiClassLogisticRegression.TestData(
                        Vector(arrayOf(g3.random(), g3.random())),
                        2
                )
            }
    )
    val class3TestData = Array<Vector>(60, {Vector(arrayOf(g3.random(), g3.random()))})

//    // クラス4のデータを用意
//    val class4TrainingData = Array<LogisticRegression.TestData>(
//            400,
//            {
//                LogisticRegression.TestData(
//                        Vector(arrayOf(g2.random(), g2.random())),
//                        LogisticRegression.Result(false, false, false, true)
//                )
//            }
//    )
//    val class4TestData = Array<Vector>(60, {Vector(arrayOf(g2.random(), g2.random()))})

    // トレーニングデータを全部ごちゃ混ぜにしてシャッフルする
    val allTrainingData = (class1TrainingData + class2TrainingData + class3TrainingData).toMutableList().shuffled().toTypedArray()
    //val allTrainingData = (class1TrainingData + class2TrainingData + class3TrainingData + class4TrainingData).toMutableList().shuffled().toTypedArray()

    val minibatchDataSize = 50 // 1個のminibatchのサイズ
    val minibatchNum = 240 // minibatchの数
    //val minibatchNum = 320 // minibatchの数
    val minibatchList = Array<Array<MultiClassLogisticRegression.TestData>>(minibatchNum) {
        allTrainingData.sliceArray(it..it + minibatchDataSize - 1)
    }

    val logistic = MultiClassLogisticRegression(2, 3)

    // 2000エポックトレーニングを行う
    for (i in 0..2000) {
        for (minibatch in minibatchList) {
            logistic.train(minibatch, lerningRate)
        }
        lerningRate *= 0.95 // 徐々に減少させる
    }
    println("w1 = " + logistic.weights)
    println("b1 = " + logistic.biases)

    println("test class1")
    for (testData in class1TestData) {
        var result = logistic.predict(testData)
        println(result)
    }
    println("test class2")
    for (testData in class2TestData) {
        var result = logistic.predict(testData)
        println(result)
    }
    println("test class3")
    for (testData in class3TestData) {
        var result = logistic.predict(testData)
        println(result)
    }
//    println("test class4")
//    for (testData in class4TestData) {
//        var result = logistic.predict(testData)
//        println(result)
//    }
}
package arithmath.logic

import arithmath.util.Vector

class Perceptrons(
        val nuronNumber: Int
) {
    val weights: Vector
    init {
        // "val weights = Vector(Array(nuronNumber, {0.0}))" だとクラッシュするので暫定対応
        val  initData = Array(nuronNumber, {0.0})
        weights = Vector(initData)
    }

    enum class PerceptronClass(val rawValue: Int) {
        Class1(1),
        Class2(-1),
    }

    /**
     * 学習を行う
     */
    fun train(trainingData: Vector, trainingDataClass: PerceptronClass, lerningRate: Double): Boolean {
        // 既存の重みを使って、トレーニングデータを正しく分類できるか確認する。
        // 正しく分類できたら学習の必要がないので、学習をスキップする
        val preCheckResult = predict(trainingData)
        if (preCheckResult == trainingDataClass) {
            return false
        }

        // 正しく分類できなかったら学習を行う(weightsの更新)
        for (i in 0..weights.values.size - 1) {
            weights.values[i] += lerningRate * trainingData.values[i] * trainingDataClass.rawValue
        }
        return true
    }

    /**
     * 学習を元に分類を行う
     */
    fun predict (testData: Vector): PerceptronClass {
        var preactivation = 0.0
        for (i in 0..weights.values.size - 1) {
            preactivation += weights.values[i] * testData.values[i]
        }
        return if (preactivation >= 0) PerceptronClass.Class1 else PerceptronClass.Class2
    }

}
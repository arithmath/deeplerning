package arithmath.logic

import arithmath.util.Vector

/**
 * 4クラスまでに対応した多クラスロジスティック回帰
 * dimenstion: データの次元
 */
class LogisticRegression(
        val dimension: Int
) {
    // 重み(クラスごとに用意)
    var weights1: Vector
    var weights2: Vector
    var weights3: Vector
    var weights4: Vector
    init {
        val initData = Array<Double>(dimension, {0.0})
        weights1 = Vector(initData)
        weights2 = Vector(initData)
        weights3 = Vector(initData)
        weights4 = Vector(initData)
    }

    // b(クラスごとに用意)
    var b1 = 0.0
    var b2 = 0.0
    var b3 = 0.0
    var b4 = 0.0


    /**
     * テストデータの構造
     * value: テストデータの値。3次元なら(0.1, 0.2, 0.3)のような値
     * tag:   テストデータのタグ。4クラスでテストデータがクラス2に該当するなら(0, 1, 0, 0)のような値
     */
    data class TestData(
            val value: Vector,
            val result: Result
    )

    data class Result (
            val isClass1: Boolean,
            val isClass2: Boolean,
            val isClass3: Boolean,
            val isClass4: Boolean
    )

    /**
     * testData: テストデータ。3次元なら{(0.1, 0,2, 0.3), (1.1, 1.2, 1.3), (2.1, 2.2, 2.3), (2.2, 2.3, 2.4)...}のようなテストデータのリスト
     * tags:     正解データ。4クラスなら  {(1, 0, 0, 0),  (0, 1, 0, 0),    (0, 0, 1, 0),    (0, 0, 1, 0), ...}のようなデータ。
     */
    fun train(testDataList: Array<TestData>, lerningRate: Double) {
        val initData = Array<Double>(dimension, {0.0})
        var gradientWeights1 = Vector(initData)
        var gradientWeights2 = Vector(initData)
        var gradientWeights3 = Vector(initData)
        var gradientWeights4 = Vector(initData)
        var gradientB1 = 0.0
        var gradientB2 = 0.0
        var gradientB3 = 0.0
        var gradientB4 = 0.0

        // ∂E/∂w_jと∂E/∂b_jを計算する
        // ∂E/∂w_j = - Σ(n: 1 -> N) (t_nj - y_nj)x_n
        // ∂E/∂b_j = - Σ(n: 1 -> N) (t_nj - y_nj)

        // Σ(n: 1 -> N) (t_nj - y_nj)を計算する
        for (testData in testDataList) {
            val (y1, y2, y3, y4) = output(testData.value)

            // - (t_nj - y_nj)
            val dy1 = y1 - if (testData.result.isClass1) 1.0 else 0.0
            val dy2 = y2 - if (testData.result.isClass2) 1.0 else 0.0
            val dy3 = y3 - if (testData.result.isClass3) 1.0 else 0.0
            val dy4 = y4 - if (testData.result.isClass4) 1.0 else 0.0

            // dy1, dy2, dy3, dy4のリストはディープラーニングでも使うので、
            // 本当は返せるようにした方が望ましい
            gradientWeights1 = gradientWeights1.plus(testData.value.multiple(dy1))
            gradientWeights2 = gradientWeights2.plus(testData.value.multiple(dy2))
            gradientWeights3 = gradientWeights3.plus(testData.value.multiple(dy3))
            gradientWeights4 = gradientWeights4.plus(testData.value.multiple(dy4))

            gradientB1 += dy1
            gradientB2 += dy2
            gradientB3 += dy3
            gradientB4 += dy4
        }

        // wおよびbの更新
        // ※ なぜtestDataList.size(ミニバッチ内の要素数)で割る必要があるのかは不明...。
        //   ミニバッチ数で割るなら、多少納得感はあるが...。
        weights1 = weights1.minus(gradientWeights1.multiple((lerningRate/testDataList.size)))
        weights2 = weights2.minus(gradientWeights2.multiple((lerningRate/testDataList.size)))
        weights3 = weights3.minus(gradientWeights3.multiple((lerningRate/testDataList.size)))
        weights4 = weights4.minus(gradientWeights4.multiple((lerningRate/testDataList.size)))
        b1 = b1 - (lerningRate/testDataList.size) * gradientB1
        b2 = b2 - (lerningRate/testDataList.size) * gradientB2
        b3 = b3 - (lerningRate/testDataList.size) * gradientB3
        b4 = b4 - (lerningRate/testDataList.size) * gradientB4
    }

    fun predict(input: Vector): Result {
        val (softmax1, softmax2, softmax3, softmax4) = output(input)
        val maxOfSoftmax = Math.max(Math.max(Math.max(softmax1, softmax2), softmax3), softmax4)
        when {
            softmax1 == maxOfSoftmax -> {
                return Result(true, false, false, false)
            }
            softmax2 == maxOfSoftmax -> {
                return Result(false,true,false,false)
            }
            softmax3 == maxOfSoftmax -> {
                return Result(false, false, true,false)
            }
            else -> {
                return Result(false, false, false, true)
            }
        }
    }

    data class Softmax (
            val data1: Double,
            val data2: Double,
            val data3: Double,
            val data4: Double
    )

    fun output(input: Vector): Softmax {
        // wx + bを計算し、softmax関数にかける
        val preactivation1 = weights1.innerProduct(input) + b1
        val preactivation2 = weights2.innerProduct(input) + b2
        val preactivation3 = weights3.innerProduct(input) + b3
        val preactivation4 = weights4.innerProduct(input) + b4

        return softmax(preactivation1, preactivation2, preactivation3, preactivation4)
    }

    // exp(w1・x + b1) / (exp(w1・x + b1) + exp(w2・x + b2) + exp(w3・x + b3) + exp(w4・x + b4))
    // exp(w2・x + b2) / (exp(w1・x + b1) + exp(w2・x + b2) + exp(w3・x + b3) + exp(w4・x + b4))
    // exp(w3・x + b3) / (exp(w1・x + b1) + exp(w2・x + b2) + exp(w3・x + b3) + exp(w4・x + b4))
    // exp(w4・x + b4) / (exp(w1・x + b1) + exp(w2・x + b2) + exp(w3・x + b3) + exp(w4・x + b4))
    // を計算する
    fun softmax(v1: Double, v2: Double, v3: Double, v4: Double): Softmax {
        // exp(v_n) / {exp(v_1) + exp(v_2) + exp(v_3) + exp(v_4)} をそのまま計算するとオーバーフローする可能性があるので、
        // v_1, v_2, v_3, v4の中のmax値を探し、
        // exp(v_n - v_max) / {exp(v_1 - v_max) + exp(v_2 - v_max) + exp(v_3 - v_max) + exp(v_4 - v_max)}
        // を代わりに計算する
        // ※ 分子・分母共にexp(v_max)を因数分解してあげると、同じ値である事がわかる

        val vmax = Math.max(Math.max(Math.max(v1, v2), v3), v4)
        val exp1 = Math.exp(v1 - vmax)
        val exp2 = Math.exp(v2 - vmax)
        val exp3 = Math.exp(v3 - vmax)
        val exp4 = Math.exp(v4 - vmax)
        val expSum = exp1 + exp2 + exp3 + exp4

        return Softmax(
                exp1 / expSum,
                exp2 / expSum,
                exp3 / expSum,
                exp4 / expSum
        )
    }
}
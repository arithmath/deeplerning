package arithmath.logic

import arithmath.util.Vector
import arithmath.util.zeroVector

/**
 * 多クラスロジスティック回帰
 * dimenstion: データの次元
 * classNum: クラス数
 */
class MultiClassLogisticRegression(
        val dimension: Int, // データの次元
        val classNum: Int   // クラス数
) {
    // 重み
    // weights[0]がクラス0の重み、weights[1]がクラス1の重み、...
    val weights = Array<Vector>(classNum) {zeroVector(dimension)}

    // バイアス
    // biases[0]がクラス0のバイアス、biases[1]がクラス1のバイアス、...
    val biases = Array<Double>(classNum) {0.0}


    /**
     * テストデータの構造
     * value: テストデータの値。3次元なら(0.1, 0.2, 0.3)のような値
     * result: どのクラスに属するか。0ならクラス0、1ならクラス1、...
     */
    data class TestData(
            val value: Vector,
            val result: Int
    )

    /**
     * ラーニングを行う
     */
    fun train(testDataList: Array<TestData>, lerningRate: Double): Array<Array<Double>> {
        val gradientWeights = Array<Vector>(classNum) { zeroVector(dimension) }
        val gradientBiases = Array<Double>(classNum) {0.0}
        // ディープラーニングで使用するdyのリスト
        val dyList: MutableList<Array<Double>> = mutableListOf()

        // ∂E/∂w_cと∂E/∂b_cを計算する(cはクラス)
        // ∂E/∂w_c = - Σ(n: 1 -> N) (t_nc - y_nc)x_n
        // ∂E/∂b_c = - Σ(n: 1 -> N) (t_nc - y_nc)
        for (testData in testDataList) {
            // トレーニングデータに対してクラス0, クラス1, クラス2, ...である確率(?)がいくらか求める
            val y = output(testData.value)

            // 「計算した確率」と「期待値(100%カ0%か)」の差である
            // - (t_nj - y_nj)
            // を算出する
            val dy = Array<Double>(classNum) {
                y[it] - if (testData.result == it) 1.0 else 0.0
            }

            // 返り値で返す情報を保存しておく
            dyList.add(dy)

            // 最後に勾配の計算
            for (c in 0..classNum - 1) {
                gradientWeights[c] = gradientWeights[c].plus(testData.value.multiple(dy[c]))
                gradientBiases[c] = gradientBiases[c] + dy[c]
            }

        }

        // wおよびbの更新
        // ※ なぜtestDataList.size(ミニバッチ内の要素数)で割る必要があるのかは後で調査
        for (c in 0..classNum - 1) {
            weights[c] = weights[c].minus(gradientWeights[c].multiple(lerningRate/testDataList.size))
            biases[c] = biases[c] - (lerningRate/testDataList.size) * gradientBiases[c]
        }

        return dyList.toTypedArray()
    }

    /**
     * クラスの判定を行う。
     * クラス0なら0を、クラス1なら1、クラス2なら2を返す
     */
    fun predict(input: Vector): Int {
        val p = output(input)
        val pmax = p.max()
        for (c in 0..p.size - 1) {
            if (p[c] == pmax) {
                return c
            }
        }
        throw Exception()
    }

    /**
     * クラス0, クラス1, クラス2, ...である確率(?)を求める
     */
    fun output(input: Vector): Array<Double> {
        // w_0・x + b_0,
        // w_1・x + b_1,
        // w_2・x + b_2, ...を計算する(x = input)
        val preactivations = Array<Double>(classNum) {
            weights[it].innerProduct(input) + biases[it]
        }

        // (y_0, y_1, y_2, ...) = softmax(w_0・x + b_0, w_1・x + b_1, w_2・x + b_2, ...)
        // を算出する
        return softmax(preactivations)
    }

    /**
     * y_0 = exp(x_0) / (exp(x_0) + exp(x_1) + ...)
     * y_1 = exp(x_1) / (exp(x_0) + exp(x_1) + ...)
     * y_2 = exp(x_2) / (exp(x_0) + exp(x_1) + ...)
     * を算出し、yのリストを返す
     */
    private fun softmax(x: Array<Double>): Array<Double> {
        // y_n = exp(x_n) / (exp(x_0) + exp(x_1) + ...)
        // をそのまま計算するとexpの計算でオーバーフローする可能性があるので、
        // x_nの中の最大値max(x)を探し、
        // y_n = exp(x_n - max(x)) / (exp(x_0 - max(x)) + exp(x_1 - max(x)) + ...)
        // を計算する。
        // ※ exp(-max(x))で因数分解して約分すると、最初の式と後の式が同じである事がわかる
        val xmax = x.max()

        if (xmax == null) {
            return arrayOf()
        }

        // 分子のexp(x_0 - max(x)), exp(x_1 - max(x)), exp(x_2 - max(x)), ...を計算
        val expList = x.map { Math.exp(it - xmax) }

        // 分母の(exp(x_0 - max(x)) + exp(x_1 - max(x)) + ...)の計算
        val expSum = expList.sum()

        return expList.map { it / expSum }.toTypedArray()
    }
}
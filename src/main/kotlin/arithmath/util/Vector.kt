package arithmath.util

class Vector(
        val values: Array<Double>
) {
    val dimension = values.size

    override fun toString(): String {
        var result = "("
        for (value in values) {
            result += value
            result += ", "
        }
        return result + ")"
    }

    /**
     * 和を返す
     */
    fun plus(other: Vector): Vector {
        if (values.size != other.values.size) {
            throw IllegalArgumentException()
        }
        val result = Vector(values.clone())
        for (i in 0..result.dimension - 1) {
            result.values[i] += other.values[i]
        }
        return result
    }

    /**
     * 差を返す
     */
    fun minus(other: Vector): Vector {
        if (values.size != other.values.size) {
            throw IllegalArgumentException()
        }
        val result = Vector(values.clone())
        for (i in 0..result.dimension - 1) {
            result.values[i] -= other.values[i]
        }
        return result
    }

    /**
     * 定数倍する
     */
    fun multiple(k: Double): Vector {
        val result = Vector(values.clone())
        for (i in 0..result.dimension - 1) {
            result.values[i] *= k
        }
        return result
    }

    /**
     * 内積の計算を行う
     */
    fun innerProduct(other: Vector): Double {
        if (values.size != other.values.size) {
            throw IllegalArgumentException()
        }
        var sum = 0.0
        for (i in 0..dimension - 1) {
            sum = values[i] * other.values[i]
        }
        return sum
    }
}

fun zeroVector(dimension: Int): Vector {
    val values = Array<Double>(dimension) {0.0}
    return Vector(values)
}

fun vectorOf(vararg values: Double): Vector {
    return Vector(values.toTypedArray())
}
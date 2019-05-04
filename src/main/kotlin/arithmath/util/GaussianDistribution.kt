package arithmath.util

import java.util.*

class GaussianDistribution(
        val mean: Double,
        val num: Double
) {
    private val random = Random()

    fun random(): Double {
        var r = 0.0;
        while (r == 0.0) {
            r = random.nextDouble();
        }

        val c = Math.sqrt(-2.0 * Math.log(r));

        val temp = if (random.nextDouble() < 5.0) c * Math.sin (2.0 * Math.PI * random.nextDouble()) else c * Math.cos(2.0 * Math.PI * random.nextDouble())
        return temp * num + mean
    }
}
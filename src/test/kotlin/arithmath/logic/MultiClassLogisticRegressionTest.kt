package arithmath.logic

import org.junit.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals

class MultiClassLogisticRegressionTest {
    @Test
    fun testConstruct() {
        val obj = MultiClassLogisticRegression(3, 4)
        assertEquals(3, obj.dimension)
        assertEquals(4, obj.classNum)
    }
}
package com.metriccapture.app.guidance

import org.junit.Assert.assertEquals
import org.junit.Test

class OrbitControllerTest {
    private val upper = RoundDefinition(CaptureRound.UPPER, 55.0, 45.0, 300.0, 0.0, 20.0)
    private val lower = RoundDefinition(CaptureRound.LOWER, 25.0, 25.0, 300.0, 10.0, 20.0)

    @Test
    fun `two staggered passes produce 36 views at combined 10 degree coverage`() {
        assertEquals(18, upper.targetCount)
        assertEquals(18, lower.targetCount)
        val combined = (0 until upper.targetCount).map(upper::targetAzimuth) +
            (0 until lower.targetCount).map(lower::targetAzimuth)
        assertEquals((0 until 36).map { it * 10.0 }, combined.sorted())
    }

    @Test
    fun `saved capture count restores the exact next scheduled target`() {
        val upperResume = OrbitController().apply { start(listOf(upper, lower), 7) }
        assertEquals(CaptureRound.UPPER, upperResume.currentTarget()?.round)
        assertEquals(7, upperResume.currentTarget()?.index)

        val lowerResume = OrbitController().apply { start(listOf(upper, lower), 18) }
        assertEquals(CaptureRound.LOWER, lowerResume.currentTarget()?.round)
        assertEquals(0, lowerResume.currentTarget()?.index)

        val complete = OrbitController().apply { start(listOf(upper, lower), 36) }
        assertEquals(null, complete.currentTarget())
        assertEquals(true, complete.state is OrbitState.Complete)
    }
}

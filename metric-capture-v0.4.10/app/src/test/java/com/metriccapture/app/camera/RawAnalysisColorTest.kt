package com.metriccapture.app.camera

import org.junit.Assert.assertArrayEquals
import org.junit.Test

class RawAnalysisColorTest {
    @Test
    fun `neutral point produces analysis-only RGB balance gains`() {
        val gains = neutralAnalysisGains(doubleArrayOf(0.5, 1.0, 2.0))

        assertArrayEquals(doubleArrayOf(2.0, 1.0, 0.5), gains, 0.0)
    }

    @Test
    fun `missing or invalid neutral point falls back to unity`() {
        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            neutralAnalysisGains(null),
            0.0,
        )
        assertArrayEquals(
            doubleArrayOf(1.0, 1.0, 1.0),
            neutralAnalysisGains(doubleArrayOf(0.0, 1.0, 1.0)),
            0.0,
        )
    }
}

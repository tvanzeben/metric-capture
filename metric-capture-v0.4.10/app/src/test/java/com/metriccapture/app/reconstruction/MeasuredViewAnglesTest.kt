package com.metriccapture.app.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Test

class MeasuredViewAnglesTest {
    @Test
    fun `measured ordering estimates orbit center instead of assuming mat origin`() {
        val angles = measuredViewAngles(
            listOf(
                doubleArrayOf(110.0, 20.0, 100.0),
                doubleArrayOf(10.0, 120.0, 100.0),
                doubleArrayOf(-90.0, 20.0, 100.0),
                doubleArrayOf(10.0, -80.0, 100.0),
            )
        )

        listOf(0.0, 90.0, 180.0, 270.0).forEachIndexed { index, expected ->
            assertEquals(expected, angles[index].azimuthDegrees, 1e-9)
        }
        angles.forEach { assertEquals(45.0, it.elevationDegrees, 1e-9) }
    }
}

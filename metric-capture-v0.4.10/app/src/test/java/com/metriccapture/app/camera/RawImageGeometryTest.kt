package com.metriccapture.app.camera

import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

class RawImageGeometryTest {
    @Test
    fun `full pixel-array RAW retains inactive-margin offset`() {
        val geometry = resolveRawImageGeometry(
            rawWidth = 4100,
            rawHeight = 3100,
            pixelArrayWidth = 4100,
            pixelArrayHeight = 3100,
            activeLeft = 20,
            activeTop = 30,
            activeWidth = 4060,
            activeHeight = 3040,
            preCorrectionIntrinsics = listOf(3000f, 3000f, 2030f, 1520f),
        )

        assertEquals(2050.0, geometry.intrinsics[2], 0.0)
        assertEquals(1550.0, geometry.intrinsics[3], 0.0)
    }

    @Test
    fun `RAW validation coordinates map directly back to RAW pixels`() {
        val geometry = RawImageGeometry(
            4000,
            3000,
            doubleArrayOf(3000.0, 3000.0, 2000.0, 1500.0),
        )
        val scaled = geometry.scaledTo(1000, 750)
        val raw = geometry.validationToRaw(500.0, 375.0, 1000, 750)

        assertEquals(750.0, scaled[0], 0.0)
        assertEquals(500.0, scaled[2], 0.0)
        assertEquals(2000.0, raw.first, 0.0)
        assertEquals(1500.0, raw.second, 0.0)
    }

    @Test
    fun `bounded validation retains RAW orientation and even dimensions`() {
        val size = boundedRawValidationDimensions(6000, 4000)

        assertEquals(1200 to 800, size)
    }

    @Test
    fun `ambiguous scaled RAW geometry is rejected`() {
        assertThrows(IllegalStateException::class.java) {
            resolveRawImageGeometry(
                rawWidth = 2000,
                rawHeight = 1500,
                pixelArrayWidth = 4000,
                pixelArrayHeight = 3000,
                activeLeft = 20,
                activeTop = 20,
                activeWidth = 3960,
                activeHeight = 2970,
                preCorrectionIntrinsics = listOf(3000f, 3000f, 1980f, 1485f),
            )
        }
    }
}

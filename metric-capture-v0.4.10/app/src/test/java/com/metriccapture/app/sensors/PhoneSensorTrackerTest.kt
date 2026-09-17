package com.metriccapture.app.sensors

import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.math.sqrt

class PhoneSensorTrackerTest {
    @Test
    fun `slerp interpolates attitude on the exposure timestamp`() {
        val halfway = slerpQuaternion(
            doubleArrayOf(0.0, 0.0, 0.0, 1.0),
            doubleArrayOf(0.0, 0.0, 1.0, 0.0),
            0.5,
        )

        assertEquals(0.0, halfway[0], 1e-9)
        assertEquals(0.0, halfway[1], 1e-9)
        assertEquals(1.0 / sqrt(2.0), halfway[2], 1e-9)
        assertEquals(1.0 / sqrt(2.0), halfway[3], 1e-9)
    }

    @Test
    fun `angular interpolation follows the short path across zero`() {
        val first = sample(1_000L, 350.0)
        val second = sample(3_000L, 10.0)

        val synchronized = interpolatePhoneSensorSamples(first, second, 2_000L, 0.5)

        assertEquals(2_000L, synchronized.exposureTimestampNs)
        assertEquals(1_000L, synchronized.beforeTimestampNs)
        assertEquals(3_000L, synchronized.afterTimestampNs)
        assertEquals(0.0, synchronized.azimuthDegrees, 1e-9)
    }

    private fun sample(timestamp: Long, azimuth: Double) = PhoneSensorSample(
        timestampNs = timestamp,
        azimuthDegrees = azimuth,
        pitchDegrees = 0.0,
        rollDegrees = 0.0,
        quaternionX = 0.0,
        quaternionY = 0.0,
        quaternionZ = 0.0,
        quaternionW = 1.0,
    )
}

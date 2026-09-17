package com.metriccapture.app.geometry

import org.junit.Assert.assertEquals
import org.junit.Test

class GeometryTest {
    @Test
    fun `camera position is expressed relative to artifact center`() {
        val pose = artifactRelativePose(
            cameraCenter = Vec3(100.0, 0.0, 100.0),
            artifactCenter = Vec3(0.0, 0.0, 0.0),
        )

        assertEquals(0.0, pose.azimuthDegrees, 1e-9)
        assertEquals(45.0, pose.elevationDegrees, 1e-9)
        assertEquals(kotlin.math.sqrt(20_000.0), pose.radiusMm, 1e-9)
    }

    @Test
    fun `angular distance wraps across zero`() {
        assertEquals(2.0, angularDistanceDegrees(359.0, 1.0), 1e-9)
    }
}

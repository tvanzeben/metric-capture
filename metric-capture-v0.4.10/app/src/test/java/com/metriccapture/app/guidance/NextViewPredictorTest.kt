package com.metriccapture.app.guidance

import com.metriccapture.app.geometry.Vec3
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class NextViewPredictorTest {
    @Test
    fun `target camera is placed on requested artifact-centered orbit`() {
        val predictor = NextViewPredictor(1000.0, 1000.0, 640.0, 360.0)
        val predicted = predictor.predict(
            TargetView(CaptureRound.UPPER, 0, 0.0, 30.0, 30.0, 300.0),
            Vec3(0.0, 0.0, 20.0),
            null,
        )

        assertEquals(259.8076, predicted.cameraCenterMm.x, 1e-3)
        assertEquals(0.0, predicted.cameraCenterMm.y, 1e-9)
        assertEquals(170.0, predicted.cameraCenterMm.z, 1e-9)
        assertTrue(predicted.fiducialPixels.values.all { it.depth > 0.0 })
    }
}

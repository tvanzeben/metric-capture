package com.metriccapture.app.vision

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class CornerRefinementBoundsTest {
    @Test
    fun rejectsPointsOutsideImageAndRefinementMargin() {
        assertFalse(refinementWindowFits(1280, 960, -1.0, 100.0, 3))
        assertFalse(refinementWindowFits(1280, 960, 2.0, 100.0, 3))
        assertFalse(refinementWindowFits(1280, 960, 1278.0, 100.0, 3))
        assertFalse(refinementWindowFits(1280, 960, 100.0, 958.0, 3))
    }

    @Test
    fun acceptsPointWhoseCompleteRefinementWindowFits() {
        assertTrue(refinementWindowFits(1280, 960, 640.0, 480.0, 3))
        assertTrue(refinementWindowFits(1280, 960, 4.0, 4.0, 3))
    }

    @Test
    fun rejectsNonFiniteCoordinates() {
        assertFalse(refinementWindowFits(1280, 960, Double.NaN, 100.0, 3))
        assertFalse(refinementWindowFits(1280, 960, 100.0, Double.POSITIVE_INFINITY, 3))
    }
}

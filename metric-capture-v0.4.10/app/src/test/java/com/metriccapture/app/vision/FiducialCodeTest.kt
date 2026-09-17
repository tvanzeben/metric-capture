package com.metriccapture.app.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class FiducialCodeTest {
    @Test
    fun `mat model preserves the printed cyan yellow and monochrome codes`() {
        assertEquals(
            listOf(
                QuadrantTone.CYAN, QuadrantTone.BLACK,
                QuadrantTone.CYAN, QuadrantTone.WHITE,
            ),
            MatModel.expectedCode.getValue(FiducialId.A).quadrants,
        )
        assertEquals(
            listOf(
                QuadrantTone.YELLOW, QuadrantTone.WHITE,
                QuadrantTone.YELLOW, QuadrantTone.BLACK,
            ),
            MatModel.expectedCode.getValue(FiducialId.B).quadrants,
        )
        assertEquals(
            listOf(
                QuadrantTone.CYAN, QuadrantTone.WHITE,
                QuadrantTone.CYAN, QuadrantTone.BLACK,
            ),
            MatModel.expectedCode.getValue(FiducialId.G).quadrants,
        )
        assertEquals(
            listOf(
                QuadrantTone.YELLOW, QuadrantTone.BLACK,
                QuadrantTone.YELLOW, QuadrantTone.WHITE,
            ),
            MatModel.expectedCode.getValue(FiducialId.H).quadrants,
        )
        assertEquals(
            listOf(
                QuadrantTone.BLACK, QuadrantTone.WHITE,
                QuadrantTone.BLACK, QuadrantTone.WHITE,
            ),
            MatModel.expectedCode.getValue(FiducialId.C).quadrants,
        )
        assertEquals(
            MatModel.expectedCode.getValue(FiducialId.C),
            MatModel.expectedCode.getValue(FiducialId.D),
        )
        assertEquals(
            listOf(
                QuadrantTone.WHITE, QuadrantTone.BLACK,
                QuadrantTone.WHITE, QuadrantTone.BLACK,
            ),
            MatModel.expectedCode.getValue(FiducialId.E).quadrants,
        )
        assertEquals(
            MatModel.expectedCode.getValue(FiducialId.E),
            MatModel.expectedCode.getValue(FiducialId.F),
        )
        assertEquals(FiducialId.values().toSet(), MatModel.expectedCode.keys)
    }

    @Test
    fun `code comparison tolerates camera quarter turns`() {
        val expected = MatModel.expectedCode.getValue(FiducialId.A)
        val rotated = FiducialCode(
            expected.quadrants.drop(1) + expected.quadrants.take(1)
        )

        assertEquals(1.0, cyclicCodeMatch(rotated, expected), 1e-12)
        assertTrue(
            cyclicCodeMatch(
                MatModel.expectedCode.getValue(FiducialId.B),
                expected,
            ) < 1.0
        )
    }

    @Test
    fun `one shared quarter turn preserves spatial code distinctions`() {
        val a = MatModel.expectedCode.getValue(FiducialId.A)
        val g = MatModel.expectedCode.getValue(FiducialId.G)

        assertEquals(1.0, codeMatch(a, a, 0), 1e-12)
        assertTrue(codeMatch(a, g, 0) < 1.0)
        assertEquals(1.0, codeMatch(a, g, 2), 1e-12)
    }
}

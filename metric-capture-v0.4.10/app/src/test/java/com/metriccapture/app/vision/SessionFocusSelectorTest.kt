package com.metriccapture.app.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SessionFocusSelectorTest {
    @Test
    fun candidateSetIsBoundedAroundMeteredFocus() {
        assertEquals(listOf(1.5f, 2.0f, 2.5f), SessionFocusSelector.candidateDiopters(2f, 8f))
        assertEquals(listOf(0f), SessionFocusSelector.candidateDiopters(0f, 0f))
    }

    @Test
    fun selectsCandidateWithBestWorstCaseRingEdges() {
        val blurred = measurement(1.5f, median = 3.0, p90 = 5.0)
        val covered = measurement(2.0f, median = 1.2, p90 = 1.8)
        val nearOnly = measurement(2.5f, median = 0.9, p90 = 3.1)

        val calibration = SessionFocusSelector.select(
            listOf(blurred, covered, nearOnly),
            meteredFocusDiopters = 2.0f,
        )!!

        assertEquals(2.0f, calibration.focusDistanceDiopters)
        assertEquals(3, calibration.candidateCount)
        assertTrue(calibration.accepts(covered.analysis))
        assertFalse(calibration.accepts(blurred.analysis))
    }

    @Test
    fun rejectsCandidateSetWithoutDistributedDepthEvidence() {
        val insufficient = measurement(2f, median = 1.0, p90 = 1.2, samples = 8)
        assertEquals(null, SessionFocusSelector.select(listOf(insufficient), 2f))
    }

    @Test
    fun rejectsEdgesThatDoNotFollowEqualGraySteps() {
        val falseEdges = measurement(
            2f,
            median = 1.0,
            p90 = 1.2,
            equalStepConsistency = 0.2,
        )
        assertEquals(null, SessionFocusSelector.select(listOf(falseEdges), 2f))
    }

    private fun measurement(
        focus: Float,
        median: Double,
        p90: Double,
        samples: Int = 30,
        equalStepConsistency: Double = 1.0,
    ): FocusCandidateMeasurement {
        val values = List(samples) { index ->
            GrayStepFocusSample(
                boundaryRadiusMm = 50.0 + index % 4 * 5.0,
                xMm = index.toDouble(),
                yMm = 0.0,
                cameraDepthMm = 400.0 + index,
                edgeWidthCanonicalPx = if (index < samples * 9 / 10) median else p90,
                measuredStepContrast = 12.0,
            )
        }
        return FocusCandidateMeasurement(
            focus,
            GrayStepFocusAnalysis(
                detectedBoundaryCount = 4,
                samples = values,
                medianEdgeWidthCanonicalPx = median,
                percentile90EdgeWidthCanonicalPx = p90,
                nearSampleDepthMm = 400.0,
                farSampleDepthMm = 430.0,
                medianStepContrast = 12.0,
                equalStepConsistency = equalStepConsistency,
            ),
        )
    }
}

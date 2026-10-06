package com.metriccapture.app.vision

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class PlanarCorrespondenceTest {
    private val modelX = MatModel.pointsMm.mapValues { it.value.x }
    private val modelY = MatModel.pointsMm.mapValues { it.value.y }

    private data class Detection(val x: Double, val y: Double, val cls: TargetColorClass)

    private fun run(detections: List<Detection>, maxResults: Int = 4) = PlanarCorrespondence.candidates(
        xs = DoubleArray(detections.size) { detections[it].x },
        ys = DoubleArray(detections.size) { detections[it].y },
        classes = detections.map { it.cls },
        quality = DoubleArray(detections.size) { 1.0 },
        modelXmm = modelX,
        modelYmm = modelY,
        maxResults = maxResults,
    )

    private val mono = TargetColorClass.MONO
    private val cyan = TargetColorClass.CYAN
    private val yellow = TargetColorClass.YELLOW

    /** Device log 2026-10-01 22:27:32: all eight targets decoded cleanly, but the old solver's best RMSE was 76.5 px. */
    private val fullFrame = listOf(
        Detection(354.0, 823.0, mono),   // D
        Detection(47.0, 614.0, mono),    // C
        Detection(393.0, 440.0, mono),   // F
        Detection(433.0, 654.0, mono),   // E
        Detection(178.0, 480.0, cyan),   // G
        Detection(217.0, 694.0, yellow), // H
        Detection(257.0, 309.0, yellow), // B
        Detection(564.0, 519.0, cyan),   // A
    )
    private val fullFrameExpected = listOf(
        FiducialId.D, FiducialId.C, FiducialId.F, FiducialId.E,
        FiducialId.G, FiducialId.H, FiducialId.B, FiducialId.A,
    )

    @Test
    fun `real eight-target frame resolves to the geometrically correct assignment`() {
        val best = run(fullFrame).first()
        assertEquals(fullFrameExpected, best.ids)
        assertTrue("homography RMS ${best.homographyRmsPx}", best.homographyRmsPx < 2.0)
    }

    @Test
    fun `detection order does not affect the result`() {
        val order = listOf(7, 2, 5, 0, 6, 3, 1, 4)
        val shuffled = order.map { fullFrame[it] }
        val best = run(shuffled).first()
        assertEquals(order.map { fullFrameExpected[it] }, best.ids)
    }

    @Test
    fun `winding check leaves only the correct full assignment`() {
        val fullAssignments = run(fullFrame, maxResults = 1000).filter { c -> c.ids.all { it != null } }
        assertEquals(1, fullAssignments.size)
        assertEquals(fullFrameExpected, fullAssignments.single().ids)
    }

    @Test
    fun `homography fit alone separates a swapped assignment by a wide margin`() {
        val xs = DoubleArray(8) { fullFrame[it].x }
        val ys = DoubleArray(8) { fullFrame[it].y }
        fun rms(ids: List<FiducialId>) = PlanarCorrespondence.homographyRms(
            DoubleArray(8) { modelX.getValue(ids[it]) }, DoubleArray(8) { modelY.getValue(ids[it]) }, xs, ys,
        )!!
        val correct = rms(fullFrameExpected)
        // Swap E and F, the kind of guess the previous code-based bucketing could make.
        val swapped = rms(fullFrameExpected.map {
            when (it) { FiducialId.E -> FiducialId.F; FiducialId.F -> FiducialId.E; else -> it }
        })
        assertTrue("correct=$correct swapped=$swapped", correct < 2.0 && swapped > 50.0)
    }

    @Test
    fun `a spurious extra blob is left unassigned`() {
        val withNoise = fullFrame + Detection(600.0, 900.0, mono)
        val best = run(withNoise).first()
        assertEquals(fullFrameExpected + listOf(null), best.ids)
    }

    @Test
    fun `mirrored assignments are rejected`() {
        val mirrored = fullFrame.map { it.copy(x = 1000.0 - it.x) }
        val best = run(mirrored).firstOrNull()
        // A mirrored image of the mat cannot be produced by a camera above it.
        assertTrue(best == null || best.ids != fullFrameExpected)
    }

    @Test
    fun `color class comes from the decoded code`() {
        assertEquals(cyan, colorClassOf(MatModel.expectedCode.getValue(FiducialId.G)))
        assertEquals(yellow, colorClassOf(MatModel.expectedCode.getValue(FiducialId.B)))
        assertEquals(mono, colorClassOf(MatModel.expectedCode.getValue(FiducialId.E)))
    }
}

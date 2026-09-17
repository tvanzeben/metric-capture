package com.metriccapture.app.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

class ScheduleMatchPlannerTest {
    private val frames = (0 until 18).map { frame(0, it, it * 20.0) } +
        (0 until 18).map { frame(1, it, 10.0 + it * 20.0) }

    @Test
    fun `lower frame uses all four predetermined bracketing upper candidates`() {
        val candidates = ScheduleMatchPlanner().incrementalCandidates(frames)
            .filter { it.secondFrameId == "lower-0" && it.crossCamera }

        assertEquals(4, candidates.size)
        assertEquals(
            setOf("upper-0", "upper-1", "upper-2", "upper-17"),
            candidates.map(ScheduledMatchCandidate::firstFrameId).toSet(),
        )
        assertEquals(listOf(10.0, 10.0, 30.0, 30.0),
            candidates.map(ScheduledMatchCandidate::scheduledSeparationDegrees).sorted())
    }

    @Test
    fun `planner ignores measured azimuth and uses schedule`() {
        val candidates = ScheduleMatchPlanner().incrementalCandidates(
            frames.map { it.copy(azimuthDegrees = 123.0) }
        )

        assertTrue(candidates.any {
            it.firstFrameId == "upper-0" && it.secondFrameId == "lower-0"
        })
        assertTrue(candidates.any {
            it.firstFrameId == "upper-1" && it.secondFrameId == "lower-0"
        })
    }

    @Test
    fun `post pass adds wider accumulated-model candidates`() {
        val planner = ScheduleMatchPlanner()
        val initial = planner.incrementalCandidates(frames)
        val post = planner.postPassCandidates(frames, initial.map { it.id }.toSet())

        assertTrue(post.any { it.crossCamera && it.scheduledSeparationDegrees == 50.0 })
        assertEquals(6, planner.minimumUsefulCrossCameraEdges(frames))
    }

    private fun frame(camera: Int, target: Int, scheduled: Double) = ReconstructionFrame(
        frameId = if (camera == 0) "upper-$target" else "lower-$target",
        workingImage = File("unused-$camera-$target.tif"),
        imageWidth = 1280,
        imageHeight = 960,
        foregroundMask = null,
        cameraIndex = camera,
        targetIndex = target,
        scheduledAzimuthDegrees = scheduled,
        azimuthDegrees = 123.0,
        elevationDegrees = 0.0,
        cameraFx = 1.0,
        cameraFy = 1.0,
        cameraCx = 0.0,
        cameraCy = 0.0,
        distortion = DoubleArray(5),
        rotationVector = DoubleArray(3),
        translationVector = doubleArrayOf(0.0, 0.0, 1.0),
    )
}

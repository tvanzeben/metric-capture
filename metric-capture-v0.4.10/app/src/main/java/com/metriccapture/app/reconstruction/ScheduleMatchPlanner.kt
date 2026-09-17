package com.metriccapture.app.reconstruction

import com.metriccapture.app.geometry.angularDistanceDegrees

enum class MatchPhase { INCREMENTAL, POST_PASS }

data class ScheduledMatchCandidate(
    val firstFrameId: String,
    val secondFrameId: String,
    val phase: MatchPhase,
    val crossCamera: Boolean,
    val scheduledSeparationDegrees: Double,
) {
    val id: String get() = listOf(firstFrameId, secondFrameId).sorted().joinToString("__")
}

/**
 * Builds the sparse match graph from the capture schedule. Measured pose angles
 * remain reconstruction data; they do not decide which images are compared.
 */
class ScheduleMatchPlanner(
    private val sameCameraWindowDegrees: Double = 40.0,
    private val incrementalCrossCameraWindowDegrees: Double = 30.0,
    private val postPassCrossCameraWindowDegrees: Double = 70.0,
    private val maximumIncrementalSameCameraConnections: Int = 4,
    private val maximumIncrementalCrossCameraConnections: Int = 4,
) {
    fun incrementalCandidates(frames: List<ReconstructionFrame>): List<ScheduledMatchCandidate> {
        val ordered = frames.sortedWith(
            compareBy(ReconstructionFrame::cameraIndex).thenBy(ReconstructionFrame::targetIndex)
        )
        val available = mutableListOf<ReconstructionFrame>()
        val planned = linkedMapOf<String, ScheduledMatchCandidate>()
        for (current in ordered) {
            available.filter { it.cameraIndex == current.cameraIndex }
                .map { it to separation(it, current) }
                .filter { it.second <= sameCameraWindowDegrees }
                .sortedWith(compareBy<Pair<ReconstructionFrame, Double>> { it.second }
                    .thenBy { it.first.targetIndex })
                .take(maximumIncrementalSameCameraConnections)
                .forEach { (previous, distance) ->
                    candidate(previous, current, MatchPhase.INCREMENTAL, distance)
                        .let { planned.putIfAbsent(it.id, it) }
                }
            if (current.cameraIndex > 0) {
                available.filter { it.cameraIndex < current.cameraIndex }
                    .map { it to separation(it, current) }
                    .filter { it.second <= incrementalCrossCameraWindowDegrees }
                    .sortedWith(compareBy<Pair<ReconstructionFrame, Double>> { it.second }
                        .thenBy { it.first.targetIndex })
                    .take(maximumIncrementalCrossCameraConnections)
                    .forEach { (previous, distance) ->
                        candidate(previous, current, MatchPhase.INCREMENTAL, distance)
                            .let { planned.putIfAbsent(it.id, it) }
                    }
            }
            available += current
        }
        return planned.values.toList()
    }

    /**
     * Once every image exists, add untried schedule-supported connections. The
     * wider cross-camera window supplies fallback connective tissue without
     * requiring a partner for every frame.
     */
    fun postPassCandidates(
        frames: List<ReconstructionFrame>,
        triedCandidateIds: Set<String>,
    ): List<ScheduledMatchCandidate> {
        val ordered = frames.sortedWith(
            compareBy(ReconstructionFrame::cameraIndex).thenBy(ReconstructionFrame::targetIndex)
        )
        return buildList {
            for (firstIndex in ordered.indices) {
                for (secondIndex in firstIndex + 1 until ordered.size) {
                    val first = ordered[firstIndex]
                    val second = ordered[secondIndex]
                    val crossCamera = first.cameraIndex != second.cameraIndex
                    val distance = separation(first, second)
                    val withinWindow = if (crossCamera) {
                        distance <= postPassCrossCameraWindowDegrees
                    } else {
                        distance <= sameCameraWindowDegrees
                    }
                    if (!withinWindow) continue
                    val candidate = candidate(first, second, MatchPhase.POST_PASS, distance)
                    if (candidate.id !in triedCandidateIds) add(candidate)
                }
            }
        }.sortedWith(
            compareBy<ScheduledMatchCandidate> { if (it.crossCamera) 0 else 1 }
                .thenBy(ScheduledMatchCandidate::scheduledSeparationDegrees)
                .thenBy(ScheduledMatchCandidate::firstFrameId)
                .thenBy(ScheduledMatchCandidate::secondFrameId)
        )
    }

    fun minimumUsefulCrossCameraEdges(frames: List<ReconstructionFrame>): Int {
        val smallestCamera = frames.groupingBy(ReconstructionFrame::cameraIndex)
            .eachCount().values.minOrNull() ?: 0
        return maxOf(4, smallestCamera / 3)
    }

    private fun separation(first: ReconstructionFrame, second: ReconstructionFrame): Double =
        angularDistanceDegrees(
            first.scheduledAzimuthDegrees,
            second.scheduledAzimuthDegrees,
        )

    private fun candidate(
        first: ReconstructionFrame,
        second: ReconstructionFrame,
        phase: MatchPhase,
        distance: Double,
    ) = ScheduledMatchCandidate(
        firstFrameId = first.frameId,
        secondFrameId = second.frameId,
        phase = phase,
        crossCamera = first.cameraIndex != second.cameraIndex,
        scheduledSeparationDegrees = distance,
    )
}

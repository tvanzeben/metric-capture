package com.metriccapture.app.reconstruction

import com.metriccapture.app.compute.ComputePermission
import com.metriccapture.app.compute.ReconstructionStage
import com.metriccapture.app.compute.ResourceGovernor
import com.metriccapture.app.compute.ResourceSnapshot
import com.metriccapture.app.compute.StageEstimate
import java.io.File

sealed interface ReconstructionOutcome {
    data class Complete(
        val plyFile: File,
        val pointCount: Int,
        val sparseEdgeCount: Int,
        val densePairCount: Int,
    ) : ReconstructionOutcome

    data class Paused(
        val stage: ReconstructionStage,
        val permission: ComputePermission.Pause,
    ) : ReconstructionOutcome

    data class Failed(val stage: ReconstructionStage, val message: String) : ReconstructionOutcome
}

interface ReconstructionProgress {
    fun onStage(stage: ReconstructionStage, completed: Int, total: Int)
}

/** Operational local pipeline. Every expensive stage passes through the governor. */
class OnDeviceReconstructionEngine(
    private val resourceGovernor: ResourceGovernor,
    private val resourceSnapshot: () -> ResourceSnapshot,
    private val progress: ReconstructionProgress,
) {
    fun reconstruct(
        frames: List<ReconstructionFrame>,
        candidateEnvelope: Aabb,
        outputPly: File,
        checkpointDirectory: File = File(
            outputPly.absoluteFile.parentFile ?: error("PLY output has no parent directory"),
            "checkpoint",
        ),
    ): ReconstructionOutcome {
        if (frames.size < 36) {
            return ReconstructionOutcome.Failed(
                ReconstructionStage.SPARSE_FEATURES,
                "All 36 validated frames across both 18-frame passes are required.",
            )
        }
        val cameras = frames.map(ReconstructionFrame::cameraIndex).toSet()
        if (cameras != setOf(0, 1)) {
            return ReconstructionOutcome.Failed(
                ReconstructionStage.SPARSE_FEATURES,
                "Both photogrammetric cameras are required.",
            )
        }
        if (frames.groupingBy(ReconstructionFrame::cameraIndex).eachCount().values.any { it < 18 }) {
            return ReconstructionOutcome.Failed(
                ReconstructionStage.SPARSE_FEATURES,
                "Each photogrammetric camera must contain 18 validated frames.",
            )
        }

        val checkpointStore = ReconstructionCheckpointStore(checkpointDirectory)
        var checkpoint = checkpointStore.load()
        var sparseEdgeCount = checkpoint.sparseEdgeCount
        var reconstructionFrames = applyRefinedPoses(frames, checkpoint.refinedCameraPoses)
        val fusion: VoxelFusion
        if (checkpoint.sparseComplete && checkpointStore.voxelFile.exists()) {
            fusion = VoxelFusion.load(checkpointStore.voxelFile)
        } else {
            val sparsePermission = permission(
                ReconstructionStage.SPARSE_FEATURES,
                StageEstimate(600_000_000, 250_000_000, true),
            )
            if (sparsePermission is ComputePermission.Pause) {
                return ReconstructionOutcome.Paused(ReconstructionStage.SPARSE_FEATURES, sparsePermission)
            }
            val bundlePermission = permission(
                ReconstructionStage.BUNDLE_ADJUSTMENT,
                StageEstimate(450_000_000, 80_000_000, true),
            )
            if (bundlePermission is ComputePermission.Pause) {
                return ReconstructionOutcome.Paused(
                    ReconstructionStage.BUNDLE_ADJUSTMENT,
                    bundlePermission,
                )
            }
            progress.onStage(ReconstructionStage.SPARSE_FEATURES, 0, 1)
            progress.onStage(ReconstructionStage.BUNDLE_ADJUSTMENT, 0, 1)
            val sparse = try {
                SparseReconstructor().reconstruct(frames)
            } catch (exception: Exception) {
                return ReconstructionOutcome.Failed(
                    ReconstructionStage.SPARSE_FEATURES,
                    exception.message ?: exception.javaClass.simpleName,
                )
            }
            if (sparse.points.size < 100) {
                return ReconstructionOutcome.Failed(
                    ReconstructionStage.TRIANGULATION,
                    "Sparse reconstruction produced insufficient metric structure.",
                )
            }
            val minimumCrossCameraEdges = ScheduleMatchPlanner().minimumUsefulCrossCameraEdges(frames)
            if (sparse.crossCameraEdgeCount < minimumCrossCameraEdges ||
                sparse.crossCameraTrackCount < 20
            ) {
                return ReconstructionOutcome.Failed(
                    ReconstructionStage.TRIANGULATION,
                    "The two photogrammetric cameras did not produce enough shared multi-view structure.",
                )
            }
            sparseEdgeCount = sparse.matchedEdges.size
            reconstructionFrames = sparse.refinedFrames
            fusion = VoxelFusion().apply { add(sparse.points) }
            checkpoint = checkpoint.copy(
                sparseComplete = true,
                sparseEdgeCount = sparseEdgeCount,
                refinedCameraPoses = reconstructionFrames.associate { frame ->
                    frame.frameId to RefinedCameraPose(
                        frame.rotationVector.copyOf(),
                        frame.translationVector.copyOf(),
                    )
                },
            )
            checkpointStore.save(checkpoint, fusion)
            progress.onStage(ReconstructionStage.SPARSE_FEATURES, 1, 1)
            progress.onStage(ReconstructionStage.BUNDLE_ADJUSTMENT, 1, 1)
        }

        val densePairs = adjacentPairs(reconstructionFrames)
        val stereo = ConstrainedStereoReconstructor()
        val completedIds = checkpoint.completedDensePairIds.toMutableSet()
        var completedDensePairs = completedIds.size
        progress.onStage(ReconstructionStage.DENSE_DEPTH, 0, densePairs.size)
        for ((first, second) in densePairs) {
            val pairId = "${first.frameId}__${second.frameId}"
            if (pairId in completedIds) continue
            val densePermission = permission(
                ReconstructionStage.DENSE_DEPTH,
                StageEstimate(900_000_000, 500_000_000, true),
            )
            if (densePermission is ComputePermission.Pause) {
                return ReconstructionOutcome.Paused(ReconstructionStage.DENSE_DEPTH, densePermission)
            }
            val longEdge = when (densePermission) {
                is ComputePermission.Run -> densePermission.geometryLongEdgePixels
                is ComputePermission.Throttle -> densePermission.geometryLongEdgePixels
                is ComputePermission.Pause -> error("handled above")
            }
            try {
                stereo.reconstructPair(first, second, candidateEnvelope, longEdge)?.let {
                    fusion.add(it.points)
                    completedIds += pairId
                    completedDensePairs++
                    checkpoint = checkpoint.copy(completedDensePairIds = completedIds.toSet())
                    checkpointStore.save(checkpoint, fusion)
                }
            } catch (exception: Exception) {
                // A single weak baseline does not invalidate the complete orbit.
            }
            progress.onStage(ReconstructionStage.DENSE_DEPTH, completedDensePairs, densePairs.size)
        }
        if (completedDensePairs < maxOf(4, densePairs.size / 4)) {
            return ReconstructionOutcome.Failed(
                ReconstructionStage.DENSE_DEPTH,
                "Too few adjacent view pairs produced valid dense depth.",
            )
        }

        val points = fusion.fusedPoints(minimumObservations = 2)
        if (points.size < 1_000) {
            return ReconstructionOutcome.Failed(
                ReconstructionStage.DEPTH_FUSION,
                "Fused reconstruction contains too few supported metric points.",
            )
        }
        val exportPermission = permission(
            ReconstructionStage.PLY_EXPORT,
            StageEstimate(200_000_000, points.size * 19L + 1_000_000, false),
        )
        if (exportPermission is ComputePermission.Pause) {
            return ReconstructionOutcome.Paused(ReconstructionStage.PLY_EXPORT, exportPermission)
        }
        progress.onStage(ReconstructionStage.PLY_EXPORT, 0, 1)
        PlyWriter.writeBinary(points, outputPly)
        checkpointStore.clearAfterSuccess()
        progress.onStage(ReconstructionStage.PLY_EXPORT, 1, 1)
        return ReconstructionOutcome.Complete(
            outputPly,
            points.size,
            sparseEdgeCount,
            completedDensePairs,
        )
    }

    private fun permission(stage: ReconstructionStage, estimate: StageEstimate): ComputePermission =
        resourceGovernor.permission(resourceSnapshot(), stage, estimate)

    private fun adjacentPairs(frames: List<ReconstructionFrame>): List<Pair<ReconstructionFrame, ReconstructionFrame>> =
        frames.groupBy(ReconstructionFrame::cameraIndex).values.flatMap { camera ->
            val sorted = camera.sortedBy(ReconstructionFrame::targetIndex)
            sorted.indices.map { index -> sorted[index] to sorted[(index + 1) % sorted.size] }
        }

    private fun applyRefinedPoses(
        frames: List<ReconstructionFrame>,
        poses: Map<String, RefinedCameraPose>,
    ): List<ReconstructionFrame> = frames.map { frame ->
        poses[frame.frameId]?.let { pose ->
            frame.copy(
                rotationVector = pose.rotationVector.copyOf(),
                translationVector = pose.translationVector.copyOf(),
            )
        } ?: frame
    }
}

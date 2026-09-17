package com.metriccapture.app.reconstruction

import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.vision.MatPoseResult
import com.metriccapture.app.vision.NegativeSpaceExtractor
import com.metriccapture.app.vision.NegativeSpaceResult
import org.opencv.core.Mat

data class IncrementalGeometryResult(
    val negativeSpace: NegativeSpaceResult,
    val candidateEnvelope: Aabb?,
    val estimatedArtifactCenterMm: Vec3?,
    val remainingLeafCount: Int,
) : AutoCloseable {
    override fun close() = negativeSpace.close()
}

/** Connects canonical mat subtraction directly to visual-hull space carving. */
class IncrementalGeometryEngine(
    private val negativeSpaceExtractor: NegativeSpaceExtractor,
    reconstructionBounds: Aabb,
    maximumOctreeDepth: Int = 7,
    private val distortion: DoubleArray = DoubleArray(5),
) {
    private val hull = VisualHullOctree(reconstructionBounds, maximumOctreeDepth)

    fun integrate(
        imageGray: Mat,
        pose: MatPoseResult,
        carveHull: Boolean = true,
    ): IncrementalGeometryResult? {
        val negativeSpace = negativeSpaceExtractor.extract(imageGray, pose.fiducials) ?: return null
        if (carveHull) {
            val imageMask = negativeSpace.imageForeground.toBinaryMask()
            hull.integrate(
                imageMask,
                CameraProjection(
                    fx = pose.cameraFx,
                    fy = pose.cameraFy,
                    cx = pose.cameraCx,
                    cy = pose.cameraCy,
                    rotationVector = pose.rotationVector,
                    translationVector = pose.translationVector,
                    distortion = distortion,
                ),
            )
        }
        val envelope = hull.candidateEnvelope()
        return IncrementalGeometryResult(
            negativeSpace = negativeSpace,
            candidateEnvelope = envelope,
            estimatedArtifactCenterMm = envelope?.center,
            remainingLeafCount = hull.candidateLeafBounds().size,
        )
    }

    fun depthSearchIntervals(cameraOrigin: Vec3, rayDirection: Vec3): List<DepthInterval> =
        hull.depthIntervals(cameraOrigin, rayDirection)
}

private fun Mat.toBinaryMask(): BinaryMask {
    require(channels() == 1)
    val bytes = ByteArray(rows() * cols())
    get(0, 0, bytes)
    return BinaryMask(cols(), rows(), bytes)
}

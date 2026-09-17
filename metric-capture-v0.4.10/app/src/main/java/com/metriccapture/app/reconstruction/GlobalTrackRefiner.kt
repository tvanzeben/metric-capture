package com.metriccapture.app.reconstruction

import org.opencv.geometry.Geometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import kotlin.math.sqrt

internal data class TrackObservation(
    val frameId: String,
    val featureIndex: Int,
    val pixel: Point,
    val undistortedPixel: Point,
    val red: Int,
    val green: Int,
    val blue: Int,
)

internal data class MultiViewFeatureTrack(val observations: List<TrackObservation>)

internal data class GlobalTrackRefinementResult(
    val points: List<MetricPoint>,
    val refinedFrames: Map<String, ReconstructionFrame>,
    val acceptedTrackCount: Int,
    val crossCameraAcceptedTrackCount: Int,
    val reprojectionRmsePx: Double,
)

/**
 * Alternates multi-view DLT over every feature track with all-camera LM pose
 * updates. Each pose is clamped to its exact mat-PnP prior so the global pass
 * cannot change metric scale, axes, or the Z=0 datum.
 */
internal class GlobalTrackRefiner(
    private val iterations: Int = 3,
    private val maximumRotationCorrectionRadians: Double = 0.012,
    private val maximumTranslationCorrectionMm: Double = 1.0,
    private val maximumTrackRmsePx: Double = 3.0,
) {
    fun refine(
        originalFrames: List<ReconstructionFrame>,
        tracks: List<MultiViewFeatureTrack>,
    ): GlobalTrackRefinementResult {
        require(originalFrames.size >= 2)
        val priors = originalFrames.associateBy(ReconstructionFrame::frameId)
        var frames = priors
        repeat(iterations) {
            val estimates = triangulateTracks(tracks, frames, filter = false)
            if (estimates.isEmpty()) return@repeat
            frames = frames.mapValues { (frameId, frame) ->
                refineFramePose(frame, priors.getValue(frameId), estimates) ?: frame
            }
        }
        val accepted = triangulateTracks(tracks, frames, filter = true)
        val points = accepted.map { estimate ->
            val observations = estimate.track.observations
            MetricPoint(
                xMm = estimate.point[0],
                yMm = estimate.point[1],
                zMm = estimate.point[2],
                red = observations.map(TrackObservation::red).average().toInt().coerceIn(0, 255),
                green = observations.map(TrackObservation::green).average().toInt().coerceIn(0, 255),
                blue = observations.map(TrackObservation::blue).average().toInt().coerceIn(0, 255),
                confidence = (
                    observations.size.coerceAtMost(6) / 6.0 / (1.0 + estimate.rmse)
                ).coerceIn(0.0, 1.0),
            )
        }
        return GlobalTrackRefinementResult(
            points = points,
            refinedFrames = frames,
            acceptedTrackCount = accepted.size,
            crossCameraAcceptedTrackCount = accepted.count { estimate ->
                estimate.track.observations.map {
                    frames.getValue(it.frameId).cameraIndex
                }.toSet().size > 1
            },
            reprojectionRmsePx = if (accepted.isEmpty()) Double.POSITIVE_INFINITY else
                sqrt(accepted.map { it.rmse * it.rmse }.average()),
        )
    }

    private data class TrackEstimate(
        val track: MultiViewFeatureTrack,
        val point: DoubleArray,
        val rmse: Double,
    )

    private data class CameraGeometry(
        val rotation: Array<DoubleArray>,
        val projection: Array<DoubleArray>,
    )

    private fun triangulateTracks(
        tracks: List<MultiViewFeatureTrack>,
        frames: Map<String, ReconstructionFrame>,
        filter: Boolean,
    ): List<TrackEstimate> {
        val geometry = frames.mapValues { (_, frame) ->
            val rotation = rodrigues(frame.rotationVector)
            CameraGeometry(rotation, projection(frame, rotation))
        }
        return tracks.mapNotNull { track ->
            val observations = track.observations.filter { it.frameId in frames }
                .distinctBy(TrackObservation::frameId)
            if (observations.size < 2) return@mapNotNull null
            val point = triangulate(observations, geometry) ?: return@mapNotNull null
            if (point.any { !it.isFinite() } || point[2] < -1.0) return@mapNotNull null
            val errors = observations.map { observation ->
                reprojectionError(
                    frames.getValue(observation.frameId),
                    geometry.getValue(observation.frameId).rotation,
                    point,
                    observation.undistortedPixel,
                )
            }
            val finite = errors.filter(Double::isFinite)
            if (finite.size < 2) return@mapNotNull null
            val positiveDepthFraction = observations.count {
                positiveDepth(
                    frames.getValue(it.frameId),
                    geometry.getValue(it.frameId).rotation,
                    point,
                )
            }.toDouble() / observations.size
            val rmse = sqrt(finite.sumOf { it * it } / finite.size)
            if (filter && (positiveDepthFraction < 0.8 || rmse > maximumTrackRmsePx)) {
                return@mapNotNull null
            }
            TrackEstimate(track.copy(observations = observations), point, rmse)
        }
    }

    private fun refineFramePose(
        current: ReconstructionFrame,
        prior: ReconstructionFrame,
        estimates: List<TrackEstimate>,
    ): ReconstructionFrame? {
        val visible = estimates.mapNotNull { estimate ->
            estimate.track.observations.firstOrNull { it.frameId == current.frameId }
                ?.let { estimate.point to it.pixel }
        }
        if (visible.size < 12) return null
        val objectPoints = MatOfPoint3f(*visible.map {
            Point3(it.first[0], it.first[1], it.first[2])
        }.toTypedArray())
        val imagePoints = MatOfPoint2f(*visible.map { it.second }.toTypedArray())
        val rvec = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, *current.rotationVector) }
        val tvec = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, *current.translationVector) }
        val camera = cameraMatrix(current)
        val distortion = MatOfDouble(*current.distortion)
        return try {
            Geometry.solvePnPRefineLM(objectPoints, imagePoints, camera, distortion, rvec, tvec)
            current.copy(
                rotationVector = clamp(
                    DoubleArray(3) { rvec[it, 0][0] },
                    prior.rotationVector,
                    maximumRotationCorrectionRadians,
                ),
                translationVector = clamp(
                    DoubleArray(3) { tvec[it, 0][0] },
                    prior.translationVector,
                    maximumTranslationCorrectionMm,
                ),
            )
        } catch (_: Exception) {
            null
        } finally {
            objectPoints.release(); imagePoints.release(); rvec.release(); tvec.release()
            camera.release(); distortion.release()
        }
    }

    private fun triangulate(
        observations: List<TrackObservation>,
        geometry: Map<String, CameraGeometry>,
    ): DoubleArray? {
        val system = Mat(observations.size * 2, 4, CvType.CV_64F)
        val w = Mat(); val u = Mat(); val vt = Mat()
        return try {
            observations.forEachIndexed { index, observation ->
                val projection = geometry.getValue(observation.frameId).projection
                val x = observation.undistortedPixel.x
                val y = observation.undistortedPixel.y
                for (column in 0..3) {
                    system.put(index * 2, column, x * projection[2][column] - projection[0][column])
                    system.put(index * 2 + 1, column, y * projection[2][column] - projection[1][column])
                }
            }
            Core.SVDecomp(system, w, u, vt)
            if (vt.rows() < 4 || vt.cols() < 4) return null
            val homogeneous = DoubleArray(4) { column -> vt[3, column][0] }
            if (kotlin.math.abs(homogeneous[3]) < 1e-12) return null
            DoubleArray(3) { homogeneous[it] / homogeneous[3] }
        } finally {
            system.release(); w.release(); u.release(); vt.release()
        }
    }

    private fun projection(
        frame: ReconstructionFrame,
        rotation: Array<DoubleArray>,
    ): Array<DoubleArray> {
        val output = Array(3) { DoubleArray(4) }
        for (column in 0..2) {
            output[0][column] = frame.cameraFx * rotation[0][column] +
                frame.cameraCx * rotation[2][column]
            output[1][column] = frame.cameraFy * rotation[1][column] +
                frame.cameraCy * rotation[2][column]
            output[2][column] = rotation[2][column]
        }
        output[0][3] = frame.cameraFx * frame.translationVector[0] +
            frame.cameraCx * frame.translationVector[2]
        output[1][3] = frame.cameraFy * frame.translationVector[1] +
            frame.cameraCy * frame.translationVector[2]
        output[2][3] = frame.translationVector[2]
        return output
    }

    private fun reprojectionError(
        frame: ReconstructionFrame,
        rotation: Array<DoubleArray>,
        point: DoubleArray,
        observed: Point,
    ): Double {
        val x = (0..2).sumOf { rotation[0][it] * point[it] } + frame.translationVector[0]
        val y = (0..2).sumOf { rotation[1][it] * point[it] } + frame.translationVector[1]
        val z = (0..2).sumOf { rotation[2][it] * point[it] } + frame.translationVector[2]
        if (z <= 1e-9) return Double.POSITIVE_INFINITY
        val dx = frame.cameraFx * x / z + frame.cameraCx - observed.x
        val dy = frame.cameraFy * y / z + frame.cameraCy - observed.y
        return sqrt(dx * dx + dy * dy)
    }

    private fun positiveDepth(
        frame: ReconstructionFrame,
        rotation: Array<DoubleArray>,
        point: DoubleArray,
    ): Boolean {
        return (0..2).sumOf { rotation[2][it] * point[it] } + frame.translationVector[2] > 0.0
    }

    private fun rodrigues(vector: DoubleArray): Array<DoubleArray> {
        val source = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, *vector) }
        val destination = Mat()
        return try {
            Geometry.Rodrigues(source, destination)
            Array(3) { row -> DoubleArray(3) { column -> destination[row, column][0] } }
        } finally {
            source.release(); destination.release()
        }
    }

    private fun cameraMatrix(frame: ReconstructionFrame): Mat =
        Mat.eye(3, 3, CvType.CV_64F).apply {
            put(0, 0, frame.cameraFx); put(1, 1, frame.cameraFy)
            put(0, 2, frame.cameraCx); put(1, 2, frame.cameraCy)
        }

    private fun clamp(value: DoubleArray, prior: DoubleArray, maximumNorm: Double): DoubleArray {
        val delta = DoubleArray(value.size) { value[it] - prior[it] }
        val norm = sqrt(delta.sumOf { it * it })
        if (norm <= maximumNorm || norm < 1e-12) return value
        return DoubleArray(value.size) { prior[it] + delta[it] * maximumNorm / norm }
    }
}

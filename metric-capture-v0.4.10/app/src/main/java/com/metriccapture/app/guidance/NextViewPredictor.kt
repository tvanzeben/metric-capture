package com.metriccapture.app.guidance

import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.geometry.degreesToRadians
import com.metriccapture.app.reconstruction.Aabb
import com.metriccapture.app.reconstruction.CameraProjection
import com.metriccapture.app.reconstruction.ProjectedPoint
import com.metriccapture.app.vision.FiducialId
import com.metriccapture.app.vision.MatModel
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

data class PredictedView(
    val target: TargetView,
    val cameraCenterMm: Vec3,
    val rotationVector: DoubleArray,
    val translationVector: DoubleArray,
    val fiducialPixels: Map<FiducialId, PredictedPoint>,
    val artifactBoundsPixels: PredictedBounds?,
)

data class PredictedPoint(val x: Double, val y: Double, val depth: Double)
data class PredictedBounds(val left: Double, val top: Double, val right: Double, val bottom: Double)

/** Produces the measurable geometry that the next live frame must satisfy. */
class NextViewPredictor(
    private val fx: Double,
    private val fy: Double,
    private val cx: Double,
    private val cy: Double,
    private val distortion: DoubleArray = DoubleArray(5),
) {
    fun predict(
        target: TargetView,
        artifactCenterMm: Vec3,
        candidateEnvelope: Aabb?,
    ): PredictedView {
        val azimuth = degreesToRadians(target.azimuthDegrees)
        val positionAngle = degreesToRadians(target.positionAngleDegrees)
        val horizontalRadius = target.radiusMm * cos(positionAngle)
        val cameraCenter = Vec3(
            artifactCenterMm.x + horizontalRadius * cos(azimuth),
            artifactCenterMm.y + horizontalRadius * sin(azimuth),
            artifactCenterMm.z + target.radiusMm * sin(positionAngle),
        )
        // Tilt is independently guided. It is not inferred by forcing the optical
        // axis through the artifact center; the whole mat is the framing invariant.
        val tilt = degreesToRadians(target.tiltDegrees)
        val forward = Vec3(
            -cos(tilt) * cos(azimuth),
            -cos(tilt) * sin(azimuth),
            -sin(tilt),
        )
        val rotation = viewRotation(forward)
        val translation = doubleArrayOf(
            -dot(rotation[0], cameraCenter),
            -dot(rotation[1], cameraCenter),
            -dot(rotation[2], cameraCenter),
        )
        val rotationVector = matrixToRodrigues(rotation)
        val camera = CameraProjection(
            fx, fy, cx, cy, rotationVector, translation, distortion
        )
        val fiducials = MatModel.pointsMm.mapValues { (_, point) ->
            camera.project(Vec3(point.x, point.y, point.z))?.let {
                PredictedPoint(it.x, it.y, it.depth)
            } ?: PredictedPoint(Double.NaN, Double.NaN, Double.NaN)
        }
        val artifactBounds = candidateEnvelope?.corners()
            ?.mapNotNull(camera::project)
            ?.takeIf { it.isNotEmpty() }
            ?.let {
                PredictedBounds(
                    left = it.minOf(ProjectedPoint::x),
                    top = it.minOf(ProjectedPoint::y),
                    right = it.maxOf(ProjectedPoint::x),
                    bottom = it.maxOf(ProjectedPoint::y),
                )
            }
        return PredictedView(
            target,
            cameraCenter,
            rotationVector,
            translation,
            fiducials,
            artifactBounds,
        )
    }

    private fun viewRotation(requestedForward: Vec3): Array<DoubleArray> {
        val forward = normalize(requestedForward)
        val worldUp = if (kotlin.math.abs(forward.z) > 0.98) {
            Vec3(0.0, 1.0, 0.0)
        } else {
            Vec3(0.0, 0.0, 1.0)
        }
        val right = normalize(cross(forward, worldUp))
        val down = normalize(cross(forward, right))
        return arrayOf(
            doubleArrayOf(right.x, right.y, right.z),
            doubleArrayOf(down.x, down.y, down.z),
            doubleArrayOf(forward.x, forward.y, forward.z),
        )
    }

    private fun matrixToRodrigues(r: Array<DoubleArray>): DoubleArray {
        val trace = r[0][0] + r[1][1] + r[2][2]
        val angle = kotlin.math.acos(((trace - 1.0) * 0.5).coerceIn(-1.0, 1.0))
        if (angle < 1e-10) return doubleArrayOf(0.0, 0.0, 0.0)
        val denominator = 2.0 * sin(angle)
        return doubleArrayOf(
            (r[2][1] - r[1][2]) / denominator * angle,
            (r[0][2] - r[2][0]) / denominator * angle,
            (r[1][0] - r[0][1]) / denominator * angle,
        )
    }

    private fun normalize(v: Vec3): Vec3 {
        val length = sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
        require(length > 1e-12)
        return Vec3(v.x / length, v.y / length, v.z / length)
    }

    private fun cross(a: Vec3, b: Vec3) = Vec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x,
    )

    private fun dot(row: DoubleArray, v: Vec3): Double =
        row[0] * v.x + row[1] * v.y + row[2] * v.z
}

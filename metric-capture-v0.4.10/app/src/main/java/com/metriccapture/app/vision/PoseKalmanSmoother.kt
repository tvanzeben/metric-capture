package com.metriccapture.app.vision

import com.metriccapture.app.geometry.Vec3
import org.opencv.geometry.Geometry
import org.opencv.core.CvType
import org.opencv.core.Mat

/** Six independent constant-velocity Kalman filters for Rodrigues pose coordinates. */
class PoseKalmanSmoother {
    private val rotation = Array(3) { AxisFilter(processAccelerationVariance = 0.08) }
    private val translation = Array(3) { AxisFilter(processAccelerationVariance = 900.0) }
    private var lastTimestampNs: Long? = null

    fun filter(measurement: MatPoseResult): MatPoseResult {
        val previous = lastTimestampNs
        lastTimestampNs = measurement.timestampNs
        val dt = if (previous == null) 1.0 / 30.0 else
            ((measurement.timestampNs - previous) / 1_000_000_000.0).coerceIn(1.0 / 120.0, 0.20)
        val pixelVariance = measurement.reprojectionRmsePx * measurement.reprojectionRmsePx
        val rvec = DoubleArray(3) { axis ->
            rotation[axis].update(measurement.rotationVector[axis], dt, 0.000025 + pixelVariance * 0.00001)
        }
        val tvec = DoubleArray(3) { axis ->
            translation[axis].update(measurement.translationVector[axis], dt, 0.25 + pixelVariance * 0.35)
        }
        return measurement.copy(
            rotationVector = rvec,
            translationVector = tvec,
            cameraCenterMm = cameraCenter(rvec, tvec),
        )
    }

    fun reset() {
        rotation.forEach(AxisFilter::reset)
        translation.forEach(AxisFilter::reset)
        lastTimestampNs = null
    }

    private fun cameraCenter(rotationVector: DoubleArray, translationVector: DoubleArray): Vec3 {
        val rvec = Mat(3, 1, CvType.CV_64F).apply { put(0, 0, *rotationVector) }
        val rotation = Mat(); Geometry.Rodrigues(rvec, rotation); rvec.release()
        val result = Vec3(
            -(rotation[0, 0][0] * translationVector[0] + rotation[1, 0][0] * translationVector[1] +
                rotation[2, 0][0] * translationVector[2]),
            -(rotation[0, 1][0] * translationVector[0] + rotation[1, 1][0] * translationVector[1] +
                rotation[2, 1][0] * translationVector[2]),
            -(rotation[0, 2][0] * translationVector[0] + rotation[1, 2][0] * translationVector[1] +
                rotation[2, 2][0] * translationVector[2]),
        )
        rotation.release()
        return result
    }

    private class AxisFilter(private val processAccelerationVariance: Double) {
        private var initialized = false
        private var position = 0.0
        private var velocity = 0.0
        private var p00 = 1.0
        private var p01 = 0.0
        private var p10 = 0.0
        private var p11 = 10.0

        fun update(measurement: Double, dt: Double, measurementVariance: Double): Double {
            if (!initialized) {
                position = measurement
                initialized = true
                return position
            }
            position += velocity * dt
            val q00 = processAccelerationVariance * dt * dt * dt * dt / 4.0
            val q01 = processAccelerationVariance * dt * dt * dt / 2.0
            val q11 = processAccelerationVariance * dt * dt
            val predicted00 = p00 + dt * (p10 + p01) + dt * dt * p11 + q00
            val predicted01 = p01 + dt * p11 + q01
            val predicted10 = p10 + dt * p11 + q01
            val predicted11 = p11 + q11
            val innovationVariance = predicted00 + measurementVariance
            val k0 = predicted00 / innovationVariance
            val k1 = predicted10 / innovationVariance
            val innovation = measurement - position
            position += k0 * innovation
            velocity += k1 * innovation
            p00 = (1.0 - k0) * predicted00
            p01 = (1.0 - k0) * predicted01
            p10 = predicted10 - k1 * predicted00
            p11 = predicted11 - k1 * predicted01
            return position
        }

        fun reset() {
            initialized = false; position = 0.0; velocity = 0.0
            p00 = 1.0; p01 = 0.0; p10 = 0.0; p11 = 10.0
        }
    }
}

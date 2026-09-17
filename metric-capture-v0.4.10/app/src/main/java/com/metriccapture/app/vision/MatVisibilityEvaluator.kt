package com.metriccapture.app.vision

import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.reconstruction.CameraProjection
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

data class MatContainment(
    val fullMatVisible: Boolean,
    val minimumFrameMarginPx: Double,
)

/** Projects the known circular mat perimeter; printed degree labels are never inspected. */
class MatVisibilityEvaluator(
    private val matRadiusMm: Double = 75.0,
    private val perimeterSamples: Int = 72,
    private val distortion: DoubleArray = DoubleArray(5),
) {
    init {
        require(matRadiusMm > 0.0)
        require(perimeterSamples >= 24)
    }

    fun evaluate(pose: MatPoseResult, width: Int, height: Int): MatContainment {
        val camera = CameraProjection(
            pose.cameraFx,
            pose.cameraFy,
            pose.cameraCx,
            pose.cameraCy,
            pose.rotationVector,
            pose.translationVector,
            distortion,
        )
        var minimumMargin = Double.POSITIVE_INFINITY
        repeat(perimeterSamples) { index ->
            val angle = 2.0 * Math.PI * index / perimeterSamples
            val projected = camera.project(
                Vec3(matRadiusMm * cos(angle), matRadiusMm * sin(angle), 0.0)
            ) ?: return MatContainment(false, Double.NEGATIVE_INFINITY)
            val margin = min(
                min(projected.x, width - 1.0 - projected.x),
                min(projected.y, height - 1.0 - projected.y),
            )
            minimumMargin = min(minimumMargin, margin)
        }
        return MatContainment(minimumMargin >= 0.0, minimumMargin)
    }
}

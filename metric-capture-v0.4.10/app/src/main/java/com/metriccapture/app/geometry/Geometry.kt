package com.metriccapture.app.geometry

import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.sqrt

data class Vec3(val x: Double, val y: Double, val z: Double) {
    operator fun minus(other: Vec3) = Vec3(x - other.x, y - other.y, z - other.z)
    val norm: Double get() = sqrt(x * x + y * y + z * z)
}

data class ArtifactRelativePose(
    val azimuthDegrees: Double,
    val elevationDegrees: Double,
    val radiusMm: Double,
)

fun artifactRelativePose(cameraCenter: Vec3, artifactCenter: Vec3): ArtifactRelativePose {
    val d = cameraCenter - artifactCenter
    return ArtifactRelativePose(
        azimuthDegrees = wrapDegrees(Math.toDegrees(atan2(d.y, d.x))),
        elevationDegrees = Math.toDegrees(atan2(d.z, sqrt(d.x * d.x + d.y * d.y))),
        radiusMm = d.norm,
    )
}

fun angularDistanceDegrees(a: Double, b: Double): Double =
    kotlin.math.abs(((a - b + 180.0) % 360.0 + 360.0) % 360.0 - 180.0)

fun wrapDegrees(value: Double): Double = ((value % 360.0) + 360.0) % 360.0

fun degreesToRadians(value: Double): Double = value * PI / 180.0

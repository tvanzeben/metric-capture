package com.metriccapture.app.sensors

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.view.Surface
import java.util.TreeMap
import kotlin.math.acos
import kotlin.math.sin
import kotlin.math.sqrt

/** Raw, timestamped device attitude. These values are never baseline-adjusted. */
data class PhoneSensorSample(
    val timestampNs: Long,
    val azimuthDegrees: Double,
    val pitchDegrees: Double,
    val rollDegrees: Double,
    val quaternionX: Double,
    val quaternionY: Double,
    val quaternionZ: Double,
    val quaternionW: Double,
)

data class UserRelativeSensorPose(
    val azimuthDeltaDegrees: Double,
    val tiltDeltaDegrees: Double,
    val rollDeltaDegrees: Double,
)

/** Phone attitude interpolated onto a Camera2 exposure timestamp. */
data class SynchronizedPhoneSensorSample(
    val exposureTimestampNs: Long,
    val beforeTimestampNs: Long,
    val afterTimestampNs: Long,
    val interpolationFraction: Double,
    val azimuthDegrees: Double,
    val pitchDegrees: Double,
    val rollDegrees: Double,
    val quaternionX: Double,
    val quaternionY: Double,
    val quaternionZ: Double,
    val quaternionW: Double,
)

/**
 * Owns the phone-attitude stream and keeps its two consumers separate:
 * raw samples are synchronized with captures; a persistent baseline-adjusted
 * view exists only for user guidance.
 */
class PhoneSensorTracker(context: Context) : SensorEventListener, AutoCloseable {
    private val appContext = context
    private val manager = context.getSystemService(SensorManager::class.java)
    private val rotationSensor = manager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)
        ?: manager.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)
    private val samples = TreeMap<Long, PhoneSensorSample>()

    @Volatile private var latestSample: PhoneSensorSample? = null
    @Volatile private var guidanceBaseline: PhoneSensorSample? = null

    val isAvailable: Boolean get() = rotationSensor != null

    fun start() {
        rotationSensor?.let {
            manager.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME)
        }
    }

    fun establishUserGuidanceBaseline(): Boolean {
        val sample = latestSample ?: return false
        guidanceBaseline = sample
        return true
    }

    fun clearUserGuidanceBaseline() {
        guidanceBaseline = null
    }

    fun latestRaw(): PhoneSensorSample? = latestSample

    @Synchronized
    fun sampleAt(
        timestampNs: Long,
        maximumDeltaNs: Long = 75_000_000L,
    ): SynchronizedPhoneSensorSample? {
        val before = samples.floorEntry(timestampNs)
        val after = samples.ceilingEntry(timestampNs)
        if (before != null && after != null &&
            timestampNs - before.key <= maximumDeltaNs &&
            after.key - timestampNs <= maximumDeltaNs
        ) {
            val span = after.key - before.key
            val fraction = if (span == 0L) 0.0 else
                (timestampNs - before.key).toDouble() / span.toDouble()
            return interpolatePhoneSensorSamples(
                before.value,
                after.value,
                timestampNs,
                fraction.coerceIn(0.0, 1.0),
            )
        }
        return null
    }

    fun userRelative(): UserRelativeSensorPose? {
        val baseline = guidanceBaseline ?: return null
        val current = latestSample ?: return null
        return UserRelativeSensorPose(
            azimuthDeltaDegrees = signedDelta(baseline.azimuthDegrees, current.azimuthDegrees),
            // Downward phone pitch is positive in the guidance convention.
            tiltDeltaDegrees = -signedDelta(baseline.pitchDegrees, current.pitchDegrees),
            rollDeltaDegrees = signedDelta(baseline.rollDegrees, current.rollDegrees),
        )
    }

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_ROTATION_VECTOR &&
            event.sensor.type != Sensor.TYPE_GAME_ROTATION_VECTOR
        ) return
        val rotation = FloatArray(9)
        val displayRotation = FloatArray(9)
        val orientation = FloatArray(3)
        SensorManager.getRotationMatrixFromVector(rotation, event.values)
        val (axisX, axisY) = when (appContext.display?.rotation ?: Surface.ROTATION_0) {
            Surface.ROTATION_90 -> SensorManager.AXIS_Y to SensorManager.AXIS_MINUS_X
            Surface.ROTATION_180 -> SensorManager.AXIS_MINUS_X to SensorManager.AXIS_MINUS_Y
            Surface.ROTATION_270 -> SensorManager.AXIS_MINUS_Y to SensorManager.AXIS_X
            else -> SensorManager.AXIS_X to SensorManager.AXIS_Y
        }
        SensorManager.remapCoordinateSystem(rotation, axisX, axisY, displayRotation)
        SensorManager.getOrientation(displayRotation, orientation)
        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)
        val sample = PhoneSensorSample(
            timestampNs = event.timestamp,
            azimuthDegrees = Math.toDegrees(orientation[0].toDouble()),
            pitchDegrees = Math.toDegrees(orientation[1].toDouble()),
            rollDegrees = Math.toDegrees(orientation[2].toDouble()),
            quaternionX = quaternion[1].toDouble(),
            quaternionY = quaternion[2].toDouble(),
            quaternionZ = quaternion[3].toDouble(),
            quaternionW = quaternion[0].toDouble(),
        )
        latestSample = sample
        synchronized(this) {
            samples[event.timestamp] = sample
            while (samples.size > 240) samples.pollFirstEntry()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    override fun close() {
        manager.unregisterListener(this)
        synchronized(this) { samples.clear() }
        latestSample = null
        guidanceBaseline = null
    }

    private fun signedDelta(from: Double, to: Double): Double =
        ((to - from + 540.0) % 360.0) - 180.0
}

internal fun interpolatePhoneSensorSamples(
    before: PhoneSensorSample,
    after: PhoneSensorSample,
    exposureTimestampNs: Long,
    fraction: Double,
): SynchronizedPhoneSensorSample {
    val t = fraction.coerceIn(0.0, 1.0)
    val quaternion = slerpQuaternion(
        doubleArrayOf(
            before.quaternionX, before.quaternionY,
            before.quaternionZ, before.quaternionW,
        ),
        doubleArrayOf(
            after.quaternionX, after.quaternionY,
            after.quaternionZ, after.quaternionW,
        ),
        t,
    )
    fun angle(from: Double, to: Double): Double {
        val interpolated = from + (((to - from + 540.0) % 360.0) - 180.0) * t
        return ((interpolated + 540.0) % 360.0) - 180.0
    }
    return SynchronizedPhoneSensorSample(
        exposureTimestampNs = exposureTimestampNs,
        beforeTimestampNs = before.timestampNs,
        afterTimestampNs = after.timestampNs,
        interpolationFraction = t,
        azimuthDegrees = angle(before.azimuthDegrees, after.azimuthDegrees),
        pitchDegrees = angle(before.pitchDegrees, after.pitchDegrees),
        rollDegrees = angle(before.rollDegrees, after.rollDegrees),
        quaternionX = quaternion[0],
        quaternionY = quaternion[1],
        quaternionZ = quaternion[2],
        quaternionW = quaternion[3],
    )
}

/** Quaternion order is x, y, z, w. */
internal fun slerpQuaternion(first: DoubleArray, second: DoubleArray, fraction: Double): DoubleArray {
    require(first.size == 4 && second.size == 4)
    val a = normalizedQuaternion(first)
    var b = normalizedQuaternion(second)
    var dot = a.indices.sumOf { a[it] * b[it] }
    if (dot < 0.0) {
        b = DoubleArray(4) { -b[it] }
        dot = -dot
    }
    val t = fraction.coerceIn(0.0, 1.0)
    if (dot > 0.9995) {
        return normalizedQuaternion(DoubleArray(4) { a[it] + t * (b[it] - a[it]) })
    }
    val angle = acos(dot.coerceIn(-1.0, 1.0))
    val denominator = sin(angle)
    val firstWeight = sin((1.0 - t) * angle) / denominator
    val secondWeight = sin(t * angle) / denominator
    return DoubleArray(4) { firstWeight * a[it] + secondWeight * b[it] }
}

private fun normalizedQuaternion(value: DoubleArray): DoubleArray {
    val length = sqrt(value.sumOf { it * it })
    require(length > 1e-12)
    return DoubleArray(4) { value[it] / length }
}

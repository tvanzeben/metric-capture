package com.metriccapture.app.reconstruction

import com.metriccapture.app.geometry.Vec3
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

data class CameraProjection(
    val fx: Double,
    val fy: Double,
    val cx: Double,
    val cy: Double,
    val rotationVector: DoubleArray,
    val translationVector: DoubleArray,
    /** OpenCV order: k1, k2, p1, p2, k3. */
    val distortion: DoubleArray = DoubleArray(5),
) {
    init {
        require(rotationVector.size == 3)
        require(translationVector.size == 3)
    }

    private val rotation: Array<DoubleArray> = rodrigues(rotationVector)

    fun project(point: Vec3): ProjectedPoint? {
        val x = rotation[0][0] * point.x + rotation[0][1] * point.y +
            rotation[0][2] * point.z + translationVector[0]
        val y = rotation[1][0] * point.x + rotation[1][1] * point.y +
            rotation[1][2] * point.z + translationVector[1]
        val z = rotation[2][0] * point.x + rotation[2][1] * point.y +
            rotation[2][2] * point.z + translationVector[2]
        if (z <= 1e-6) return null
        val xn = x / z
        val yn = y / z
        val k1 = distortion.getOrElse(0) { 0.0 }
        val k2 = distortion.getOrElse(1) { 0.0 }
        val p1 = distortion.getOrElse(2) { 0.0 }
        val p2 = distortion.getOrElse(3) { 0.0 }
        val k3 = distortion.getOrElse(4) { 0.0 }
        val r2 = xn * xn + yn * yn
        val radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        val xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        val yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return ProjectedPoint(fx * xd + cx, fy * yd + cy, z)
    }
}

data class ProjectedPoint(val x: Double, val y: Double, val depth: Double)

data class Aabb(val minimum: Vec3, val maximum: Vec3) {
    val center: Vec3 get() = Vec3(
        (minimum.x + maximum.x) * 0.5,
        (minimum.y + maximum.y) * 0.5,
        (minimum.z + maximum.z) * 0.5,
    )

    fun corners(): List<Vec3> = listOf(
        Vec3(minimum.x, minimum.y, minimum.z),
        Vec3(maximum.x, minimum.y, minimum.z),
        Vec3(minimum.x, maximum.y, minimum.z),
        Vec3(maximum.x, maximum.y, minimum.z),
        Vec3(minimum.x, minimum.y, maximum.z),
        Vec3(maximum.x, minimum.y, maximum.z),
        Vec3(minimum.x, maximum.y, maximum.z),
        Vec3(maximum.x, maximum.y, maximum.z),
    )

    fun split(): List<Aabb> {
        val c = center
        return buildList(8) {
            for (zHigh in listOf(false, true)) {
                for (yHigh in listOf(false, true)) {
                    for (xHigh in listOf(false, true)) {
                        add(
                            Aabb(
                                Vec3(
                                    if (xHigh) c.x else minimum.x,
                                    if (yHigh) c.y else minimum.y,
                                    if (zHigh) c.z else minimum.z,
                                ),
                                Vec3(
                                    if (xHigh) maximum.x else c.x,
                                    if (yHigh) maximum.y else c.y,
                                    if (zHigh) maximum.z else c.z,
                                ),
                            )
                        )
                    }
                }
            }
        }
    }

    fun rayInterval(origin: Vec3, direction: Vec3): ClosedFloatingPointRange<Double>? {
        var near = Double.NEGATIVE_INFINITY
        var far = Double.POSITIVE_INFINITY
        fun axis(originValue: Double, directionValue: Double, low: Double, high: Double): Boolean {
            if (kotlin.math.abs(directionValue) < 1e-12) return originValue in low..high
            var a = (low - originValue) / directionValue
            var b = (high - originValue) / directionValue
            if (a > b) a = b.also { b = a }
            near = max(near, a)
            far = min(far, b)
            return near <= far
        }
        if (!axis(origin.x, direction.x, minimum.x, maximum.x)) return null
        if (!axis(origin.y, direction.y, minimum.y, maximum.y)) return null
        if (!axis(origin.z, direction.z, minimum.z, maximum.z)) return null
        if (far < 0.0) return null
        return max(0.0, near)..far
    }
}

private enum class HullState { CANDIDATE, FREE }

private data class HullNode(
    val bounds: Aabb,
    val depth: Int,
    var state: HullState = HullState.CANDIDATE,
    var children: MutableList<HullNode>? = null,
)

data class DepthInterval(val nearMm: Double, val farMm: Double)

/**
 * Incremental visual hull. A node is removed as soon as any calibrated view
 * proves that its entire image projection lies in visible mat/background.
 */
class VisualHullOctree(
    bounds: Aabb,
    private val maximumDepth: Int = 7,
) {
    private val root = HullNode(bounds, 0)
    var integratedViewCount: Int = 0
        private set

    fun integrate(mask: BinaryMask, camera: CameraProjection) {
        val integral = BinaryIntegralImage(mask)
        carve(root, integral, camera)
        integratedViewCount++
    }

    fun candidateLeafBounds(): List<Aabb> = buildList { collectLeaves(root, this) }

    fun candidateEnvelope(): Aabb? {
        val leaves = candidateLeafBounds()
        if (leaves.isEmpty()) return null
        return Aabb(
            minimum = Vec3(
                leaves.minOf { it.minimum.x },
                leaves.minOf { it.minimum.y },
                leaves.minOf { it.minimum.z },
            ),
            maximum = Vec3(
                leaves.maxOf { it.maximum.x },
                leaves.maxOf { it.maximum.y },
                leaves.maxOf { it.maximum.z },
            ),
        )
    }

    fun depthIntervals(origin: Vec3, direction: Vec3): List<DepthInterval> {
        val intervals = candidateLeafBounds().mapNotNull { bounds ->
            bounds.rayInterval(origin, direction)?.let { DepthInterval(it.start, it.endInclusive) }
        }.sortedBy(DepthInterval::nearMm)
        if (intervals.isEmpty()) return emptyList()
        val merged = mutableListOf(intervals.first())
        for (next in intervals.drop(1)) {
            val last = merged.last()
            if (next.nearMm <= last.farMm + 1e-6) {
                merged[merged.lastIndex] = DepthInterval(last.nearMm, max(last.farMm, next.farMm))
            } else {
                merged += next
            }
        }
        return merged
    }

    private fun carve(
        node: HullNode,
        integral: BinaryIntegralImage,
        camera: CameraProjection,
    ) {
        if (node.state == HullState.FREE) return
        val projected = node.bounds.corners().mapNotNull(camera::project)
        if (projected.size != 8) return
        val left = floor(projected.minOf(ProjectedPoint::x)).toInt()
        val top = floor(projected.minOf(ProjectedPoint::y)).toInt()
        val right = ceil(projected.maxOf(ProjectedPoint::x)).toInt()
        val bottom = ceil(projected.maxOf(ProjectedPoint::y)).toInt()
        if (left < 0 || top < 0 || right >= integral.width || bottom >= integral.height) return
        val area = (right - left + 1) * (bottom - top + 1)
        val foreground = integral.sum(left, top, right, bottom)
        if (foreground == 0) {
            node.state = HullState.FREE
            node.children = null
            return
        }
        if (foreground == area || node.depth >= maximumDepth) return
        val children = node.children ?: node.bounds.split().mapTo(mutableListOf()) {
            HullNode(it, node.depth + 1)
        }.also { node.children = it }
        children.forEach { carve(it, integral, camera) }
    }

    private fun collectLeaves(node: HullNode, destination: MutableList<Aabb>) {
        if (node.state == HullState.FREE) return
        val children = node.children
        if (children == null) destination += node.bounds else children.forEach {
            collectLeaves(it, destination)
        }
    }
}

data class BinaryMask(val width: Int, val height: Int, val values: ByteArray) {
    init {
        require(values.size == width * height)
    }
}

private class BinaryIntegralImage(mask: BinaryMask) {
    val width = mask.width
    val height = mask.height
    private val stride = width + 1
    private val sums = IntArray((width + 1) * (height + 1))

    init {
        for (y in 0 until height) {
            var rowSum = 0
            for (x in 0 until width) {
                if (mask.values[y * width + x].toInt() != 0) rowSum++
                sums[(y + 1) * stride + x + 1] = sums[y * stride + x + 1] + rowSum
            }
        }
    }

    fun sum(left: Int, top: Int, right: Int, bottom: Int): Int {
        val x0 = left.coerceIn(0, width)
        val y0 = top.coerceIn(0, height)
        val x1 = (right + 1).coerceIn(0, width)
        val y1 = (bottom + 1).coerceIn(0, height)
        return sums[y1 * stride + x1] - sums[y0 * stride + x1] -
            sums[y1 * stride + x0] + sums[y0 * stride + x0]
    }
}

private fun rodrigues(vector: DoubleArray): Array<DoubleArray> {
    val theta = sqrt(vector.sumOf { it * it })
    if (theta < 1e-12) return arrayOf(
        doubleArrayOf(1.0, 0.0, 0.0),
        doubleArrayOf(0.0, 1.0, 0.0),
        doubleArrayOf(0.0, 0.0, 1.0),
    )
    val x = vector[0] / theta
    val y = vector[1] / theta
    val z = vector[2] / theta
    val c = cos(theta)
    val s = sin(theta)
    val oneMinusC = 1.0 - c
    return arrayOf(
        doubleArrayOf(c + x * x * oneMinusC, x * y * oneMinusC - z * s, x * z * oneMinusC + y * s),
        doubleArrayOf(y * x * oneMinusC + z * s, c + y * y * oneMinusC, y * z * oneMinusC - x * s),
        doubleArrayOf(z * x * oneMinusC - y * s, z * y * oneMinusC + x * s, c + z * z * oneMinusC),
    )
}

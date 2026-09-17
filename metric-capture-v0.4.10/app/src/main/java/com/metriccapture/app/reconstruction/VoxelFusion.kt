package com.metriccapture.app.reconstruction

import kotlin.math.floor
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.BufferedInputStream
import java.io.BufferedOutputStream
import java.io.FileInputStream
import java.io.FileOutputStream

private data class VoxelKey(val x: Int, val y: Int, val z: Int)

private data class Accumulator(
    var x: Double = 0.0,
    var y: Double = 0.0,
    var z: Double = 0.0,
    var red: Double = 0.0,
    var green: Double = 0.0,
    var blue: Double = 0.0,
    var weight: Double = 0.0,
    var observations: Int = 0,
)

/** Fixed-memory-density fusion: point count is bounded by occupied metric voxels. */
class VoxelFusion(private val voxelSizeMm: Double = 0.35) {
    private val voxels = HashMap<VoxelKey, Accumulator>()

    init {
        require(voxelSizeMm > 0.0)
    }

    fun add(points: Iterable<MetricPoint>) {
        points.forEach { point ->
            if (!point.xMm.isFinite() || !point.yMm.isFinite() || !point.zMm.isFinite()) return@forEach
            if (point.zMm < -0.5) return@forEach
            val key = key(point.xMm, point.yMm, point.zMm)
            val accumulator = voxels.getOrPut(key, ::Accumulator)
            val weight = point.confidence.coerceIn(0.01, 1.0)
            accumulator.x += point.xMm * weight
            accumulator.y += point.yMm * weight
            accumulator.z += point.zMm * weight
            accumulator.red += point.red * weight
            accumulator.green += point.green * weight
            accumulator.blue += point.blue * weight
            accumulator.weight += weight
            accumulator.observations++
        }
    }

    fun fusedPoints(minimumObservations: Int = 2): List<MetricPoint> {
        val retained = voxels.filterValues { it.observations >= minimumObservations }
        return retained.mapNotNull { (key, accumulator) ->
            val neighbors = neighborCount(key, retained)
            if (neighbors < 2 && accumulator.observations < minimumObservations + 2) return@mapNotNull null
            MetricPoint(
                xMm = accumulator.x / accumulator.weight,
                yMm = accumulator.y / accumulator.weight,
                zMm = maxOf(0.0, accumulator.z / accumulator.weight),
                red = (accumulator.red / accumulator.weight).toInt().coerceIn(0, 255),
                green = (accumulator.green / accumulator.weight).toInt().coerceIn(0, 255),
                blue = (accumulator.blue / accumulator.weight).toInt().coerceIn(0, 255),
                confidence = (accumulator.observations / 6.0).coerceAtMost(1.0),
            )
        }
    }

    val occupiedVoxelCount: Int get() = voxels.size

    fun save(file: File) {
        file.parentFile?.mkdirs()
        DataOutputStream(BufferedOutputStream(FileOutputStream(file))).use { output ->
            output.writeInt(0x4D435658) // MCVX
            output.writeInt(1)
            output.writeDouble(voxelSizeMm)
            output.writeInt(voxels.size)
            voxels.forEach { (key, value) ->
                output.writeInt(key.x); output.writeInt(key.y); output.writeInt(key.z)
                output.writeDouble(value.x); output.writeDouble(value.y); output.writeDouble(value.z)
                output.writeDouble(value.red); output.writeDouble(value.green); output.writeDouble(value.blue)
                output.writeDouble(value.weight); output.writeInt(value.observations)
            }
        }
    }

    companion object {
        fun load(file: File): VoxelFusion {
            DataInputStream(BufferedInputStream(FileInputStream(file))).use { input ->
                require(input.readInt() == 0x4D435658) { "Invalid voxel checkpoint" }
                require(input.readInt() == 1) { "Unsupported voxel checkpoint version" }
                val fusion = VoxelFusion(input.readDouble())
                repeat(input.readInt()) {
                    val key = VoxelKey(input.readInt(), input.readInt(), input.readInt())
                    fusion.voxels[key] = Accumulator(
                        x = input.readDouble(), y = input.readDouble(), z = input.readDouble(),
                        red = input.readDouble(), green = input.readDouble(), blue = input.readDouble(),
                        weight = input.readDouble(), observations = input.readInt(),
                    )
                }
                return fusion
            }
        }
    }

    private fun neighborCount(
        key: VoxelKey,
        retained: Map<VoxelKey, Accumulator>,
    ): Int {
        var count = 0
        for (dz in -1..1) for (dy in -1..1) for (dx in -1..1) {
            if (dx == 0 && dy == 0 && dz == 0) continue
            if (VoxelKey(key.x + dx, key.y + dy, key.z + dz) in retained) count++
        }
        return count
    }

    private fun key(x: Double, y: Double, z: Double) = VoxelKey(
        floor(x / voxelSizeMm).toInt(),
        floor(y / voxelSizeMm).toInt(),
        floor(z / voxelSizeMm).toInt(),
    )
}

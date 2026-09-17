package com.metriccapture.app.reconstruction

import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

object PlyWriter {
    fun writeBinary(points: List<MetricPoint>, destination: File) {
        destination.parentFile?.mkdirs()
        BufferedOutputStream(FileOutputStream(destination), 1 shl 20).use { output ->
            val header = buildString {
                appendLine("ply")
                appendLine("format binary_little_endian 1.0")
                appendLine("comment coordinates are millimeters in mat reference frame")
                appendLine("element vertex ${points.size}")
                appendLine("property float x")
                appendLine("property float y")
                appendLine("property float z")
                appendLine("property uchar red")
                appendLine("property uchar green")
                appendLine("property uchar blue")
                appendLine("property float confidence")
                appendLine("end_header")
            }
            output.write(header.toByteArray(Charsets.US_ASCII))
            val record = ByteBuffer.allocate(19).order(ByteOrder.LITTLE_ENDIAN)
            points.forEach { point ->
                record.clear()
                record.putFloat(point.xMm.toFloat())
                record.putFloat(point.yMm.toFloat())
                record.putFloat(point.zMm.toFloat())
                record.put(point.red.toByte())
                record.put(point.green.toByte())
                record.put(point.blue.toByte())
                record.putFloat(point.confidence.toFloat())
                output.write(record.array())
            }
        }
    }
}

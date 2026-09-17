package com.metriccapture.app.camera

import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.TotalCaptureResult
import android.media.Image
import android.util.Size
import com.metriccapture.app.vision.FramePlanes
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.pow
import org.opencv.core.CvType
import org.opencv.core.Mat

/** Files and interpretation values produced from one untouched RAW_SENSOR buffer. */
data class RawFrameArtifact(
    val rawMaster: File,
    val workingTiff: File,
    val width: Int,
    val height: Int,
    val rowStrideBytes: Int,
    val pixelStrideBytes: Int,
    val rawBufferBytes: Long,
    val colorFilterArrangement: Int,
    val blackLevels: DoubleArray,
    val whiteLevel: Int,
    /** Analysis-only neutral gains; never applied to the preserved RAW or linear TIFF. */
    val validationNeutralGains: DoubleArray,
    val validationFrame: FramePlanes,
)

/**
 * Copies the Camera2 RAW plane verbatim before deriving a deterministic linear TIFF.
 * The TIFF is an uncompressed 16-bit RGB file written one row at a time, so a
 * maximum-resolution frame never needs a second full-resolution RGB allocation.
 */
class RawFrameProcessor(
    private val characteristics: CameraCharacteristics,
    private val validationSize: Size,
) {
    /**
     * Bounded linear green-channel image for edge-spread measurement. It bypasses
     * neutral gains, color conversion, gamma, tone mapping, and sharpening.
     */
    fun linearFocusGray(image: Image, result: TotalCaptureResult): Mat {
        require(image.format == android.graphics.ImageFormat.RAW_SENSOR)
        val plane = image.planes.single()
        val cfa = colorFilterArrangement()
        val black = blackLevels(cfa, result)
        val white = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
            ?: characteristics[CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL]
            ?: 65_535
        val outputWidth = validationSize.width.coerceAtMost(image.width).let { it - (it and 1) }
        val outputHeight = validationSize.height.coerceAtMost(image.height).let { it - (it and 1) }
        val packed = ByteArray(outputWidth * outputHeight)
        val source = plane.buffer.duplicate().order(ByteOrder.nativeOrder())
        for (row in 0 until outputHeight) {
            val sourceY = (row.toLong() * image.height / outputHeight).toInt()
            for (column in 0 until outputWidth) {
                val sourceX = (column.toLong() * image.width / outputWidth).toInt()
                var sum = 0.0
                var count = 0
                val startX = ((sourceX / 2) * 2).coerceIn(0, image.width - 1)
                val startY = ((sourceY / 2) * 2).coerceIn(0, image.height - 1)
                for (dy in 0..1) for (dx in 0..1) {
                    val x = (startX + dx).coerceAtMost(image.width - 1)
                    val y = (startY + dy).coerceAtMost(image.height - 1)
                    if (cfaChannel(cfa, x, y) != 1) continue
                    val raw = source.getShort(
                        y * plane.rowStride + x * plane.pixelStride
                    ).toInt() and 0xffff
                    val level = black[(y and 1) * 2 + (x and 1)]
                    sum += ((raw - level) / (white - level).coerceAtLeast(1.0))
                        .coerceIn(0.0, 1.0)
                    count++
                }
                val value = if (count == 0) 0 else (sum / count * 255.0 + 0.5).toInt()
                packed[row * outputWidth + column] = value.coerceIn(0, 255).toByte()
            }
        }
        return Mat(outputHeight, outputWidth, CvType.CV_8UC1).apply { put(0, 0, packed) }
    }

    /** Builds only the bounded RAW-derived frame used for empty-mat calibration. */
    fun validationFrame(image: Image, result: TotalCaptureResult): FramePlanes {
        require(image.format == android.graphics.ImageFormat.RAW_SENSOR)
        val plane = image.planes.single()
        val cfa = colorFilterArrangement()
        val blackLevels = blackLevels(cfa, result)
        val whiteLevel = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
            ?: characteristics[CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL]
            ?: 65_535
        val neutralGains = validationNeutralGains(result)
        return developValidationFrame(
            source = plane.buffer,
            timestampNs = image.timestamp,
            width = image.width,
            height = image.height,
            rowStride = plane.rowStride,
            pixelStride = plane.pixelStride,
            cfa = cfa,
            blackLevels = blackLevels,
            whiteLevel = whiteLevel,
            neutralGains = neutralGains,
        )
    }

    fun write(
        image: Image,
        result: TotalCaptureResult,
        rawMaster: File,
        workingTiff: File,
    ): RawFrameArtifact {
        require(image.format == android.graphics.ImageFormat.RAW_SENSOR)
        val plane = image.planes.single()
        require(plane.pixelStride >= 2) { "RAW_SENSOR pixel stride is smaller than 16 bits" }
        val cfa = colorFilterArrangement()

        rawMaster.parentFile?.mkdirs()
        val rawBytes = copyVerbatimPlane(plane.buffer, rawMaster)
        val blackLevels = blackLevels(cfa, result)
        val whiteLevel = result.get(CaptureResult.SENSOR_DYNAMIC_WHITE_LEVEL)
            ?: characteristics[CameraCharacteristics.SENSOR_INFO_WHITE_LEVEL]
            ?: 65_535
        val neutralGains = validationNeutralGains(result)

        val validationFrame = writeLinearRgbTiff(
            source = plane.buffer,
            timestampNs = image.timestamp,
            width = image.width,
            height = image.height,
            rowStride = plane.rowStride,
            pixelStride = plane.pixelStride,
            cfa = cfa,
            blackLevels = blackLevels,
            whiteLevel = whiteLevel,
            neutralGains = neutralGains,
            destination = workingTiff,
        )
        return RawFrameArtifact(
            rawMaster,
            workingTiff,
            image.width,
            image.height,
            plane.rowStride,
            plane.pixelStride,
            rawBytes,
            cfa,
            blackLevels,
            whiteLevel,
            neutralGains,
            validationFrame,
        )
    }

    private fun colorFilterArrangement(): Int {
        val cfa = characteristics[CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT]
            ?: error("The camera did not report its RAW color-filter arrangement")
        require(cfa in setOf(
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB,
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG,
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG,
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR,
        )) { "Only Bayer RAW_SENSOR layouts are supported" }
        return cfa
    }

    private fun blackLevels(cfa: Int, result: TotalCaptureResult): DoubleArray {
        val fixedBlack = characteristics[CameraCharacteristics.SENSOR_BLACK_LEVEL_PATTERN]
        val dynamicBlack = result.get(CaptureResult.SENSOR_DYNAMIC_BLACK_LEVEL)
        return DoubleArray(4) { index ->
            val xParity = index % 2
            val yParity = index / 2
            val dynamicIndex = when (cfaChannel(cfa, xParity, yParity)) {
                0 -> 0
                2 -> 3
                else -> if (yParity == 0) 1 else 2
            }
            dynamicBlack?.getOrNull(dynamicIndex)?.toDouble()
                ?: fixedBlack?.getOffsetForIndex(xParity, yParity)?.toDouble()
                ?: 0.0
        }
    }

    /**
     * Converts the reported linear sensor neutral point into bounded RGB gains.
     * This only stabilizes yellow/cyan/black/white target classification in the
     * bounded validation frame. Scientific image products remain unbalanced.
     */
    private fun validationNeutralGains(result: TotalCaptureResult): DoubleArray {
        val neutral = result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)
            ?.map { it.numerator.toDouble() / it.denominator.toDouble() }
            ?.toDoubleArray()
        return neutralAnalysisGains(neutral)
    }

    private fun developValidationFrame(
        source: ByteBuffer,
        timestampNs: Long,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        cfa: Int,
        blackLevels: DoubleArray,
        whiteLevel: Int,
        neutralGains: DoubleArray,
    ): FramePlanes {
        val input = source.duplicate().order(ByteOrder.nativeOrder())
        var previous = normalizedRow(input, 0, width, rowStride, pixelStride, blackLevels, whiteLevel)
        var current = previous
        var next = normalizedRow(
            input, if (height > 1) 1 else 0, width, rowStride, pixelStride,
            blackLevels, whiteLevel,
        )
        val encoded = ByteArray(width * 6)
        val validationWidth = validationSize.width.coerceAtMost(width).let { it - (it and 1) }
        val validationHeight = validationSize.height.coerceAtMost(height).let { it - (it and 1) }
        val y = ByteArray(validationWidth * validationHeight)
        val u = ByteArray(validationWidth * validationHeight / 4)
        val v = ByteArray(validationWidth * validationHeight / 4)
        var validationRow = 0
        for (row in 0 until height) {
            var byteIndex = 0
            for (column in 0 until width) for (channel in 0..2) {
                val value = interpolate(
                    previous, current, next, column, row, width, height, cfa, channel,
                )
                encoded[byteIndex++] = (value and 0xff).toByte()
                encoded[byteIndex++] = ((value ushr 8) and 0xff).toByte()
            }
            while (validationRow < validationHeight &&
                validationRow.toLong() * height / validationHeight == row.toLong()
            ) {
                downsampleLinearRgbRow(
                    encoded, width, validationRow, validationWidth, y, u, v, neutralGains,
                )
                validationRow++
            }
            previous = current
            current = next
            next = normalizedRow(
                input, (row + 2).coerceAtMost(height - 1), width, rowStride,
                pixelStride, blackLevels, whiteLevel,
            )
        }
        return FramePlanes(
            validationWidth, validationHeight, timestampNs,
            y, validationWidth, 1,
            u, validationWidth / 2, 1,
            v, validationWidth / 2, 1,
        )
    }

    private fun copyVerbatimPlane(buffer: ByteBuffer, destination: File): Long {
        val source = buffer.duplicate().apply { rewind() }
        val byteCount = source.remaining().toLong()
        FileOutputStream(destination).channel.use { channel ->
            while (source.hasRemaining()) channel.write(source)
            channel.force(true)
        }
        return byteCount
    }

    private fun writeLinearRgbTiff(
        source: ByteBuffer,
        timestampNs: Long,
        width: Int,
        height: Int,
        rowStride: Int,
        pixelStride: Int,
        cfa: Int,
        blackLevels: DoubleArray,
        whiteLevel: Int,
        neutralGains: DoubleArray,
        destination: File,
    ): FramePlanes {
        destination.parentFile?.mkdirs()
        val input = source.duplicate().order(ByteOrder.nativeOrder())
        val pixelBytes = width.toLong() * height * 6L
        require(pixelBytes + 512L < 0xffff_ffffL) {
            "The linear TIFF exceeds the classic TIFF 4 GiB offset limit"
        }
        RandomAccessFile(destination, "rw").use { output ->
            output.setLength(0L)
            output.write('I'.code); output.write('I'.code)
            output.writeShortLe(42)
            output.writeIntLe(0L) // Patched with the IFD location after pixels are streamed.

            var previous = normalizedRow(input, 0, width, rowStride, pixelStride, blackLevels, whiteLevel)
            var current = previous
            var next = normalizedRow(
                input,
                if (height > 1) 1 else 0,
                width,
                rowStride,
                pixelStride,
                blackLevels,
                whiteLevel,
            )
            val encoded = ByteArray(width * 6)
            val previewWidth = validationSize.width.coerceAtMost(width).let { it - (it and 1) }
            val previewHeight = validationSize.height.coerceAtMost(height).let { it - (it and 1) }
            require(previewWidth >= 2 && previewHeight >= 2)
            val previewY = ByteArray(previewWidth * previewHeight)
            val previewU = ByteArray(previewWidth * previewHeight / 4)
            val previewV = ByteArray(previewWidth * previewHeight / 4)
            var nextPreviewRow = 0
            for (row in 0 until height) {
                var byteIndex = 0
                for (column in 0 until width) {
                    for (channel in 0..2) {
                        val value = interpolate(
                            previous,
                            current,
                            next,
                            column,
                            row,
                            width,
                            height,
                            cfa,
                            channel,
                        )
                        encoded[byteIndex++] = (value and 0xff).toByte()
                        encoded[byteIndex++] = ((value ushr 8) and 0xff).toByte()
                    }
                }
                while (
                    nextPreviewRow < previewHeight &&
                    nextPreviewRow.toLong() * height / previewHeight == row.toLong()
                ) {
                    downsampleLinearRgbRow(
                        encoded,
                        width,
                        nextPreviewRow,
                        previewWidth,
                        previewY,
                        previewU,
                        previewV,
                        neutralGains,
                    )
                    nextPreviewRow++
                }
                output.write(encoded)
                previous = current
                current = next
                next = normalizedRow(
                    input,
                    (row + 2).coerceAtMost(height - 1),
                    width,
                    rowStride,
                    pixelStride,
                    blackLevels,
                    whiteLevel,
                )
            }

            val ifdOffset = output.filePointer
            val entryCount = 11
            val extraOffset = ifdOffset + 2L + entryCount * 12L + 4L
            val bitsPerSampleOffset = extraOffset
            output.writeShortLe(entryCount)
            output.writeIfdLong(256, width.toLong())
            output.writeIfdLong(257, height.toLong())
            output.writeIfdOffset(258, 3, bitsPerSampleOffset)
            output.writeIfdShort(259, 1) // no compression
            output.writeIfdShort(262, 2) // RGB
            output.writeIfdLong(273, 8L) // one strip begins after the TIFF header
            output.writeIfdShort(274, 1) // top-left orientation
            output.writeIfdShort(277, 3)
            output.writeIfdLong(278, height.toLong())
            output.writeIfdLong(279, pixelBytes)
            output.writeIfdShort(284, 1) // chunky RGB
            output.writeIntLe(0L)
            repeat(3) { output.writeShortLe(16) }
            output.seek(4L)
            output.writeIntLe(ifdOffset)
            output.fd.sync()
            return FramePlanes(
                width = previewWidth,
                height = previewHeight,
                timestampNs = timestampNs,
                y = previewY,
                yRowStride = previewWidth,
                yPixelStride = 1,
                u = previewU,
                uRowStride = previewWidth / 2,
                uPixelStride = 1,
                v = previewV,
                vRowStride = previewWidth / 2,
                vPixelStride = 1,
            )
        }
    }

    private fun downsampleLinearRgbRow(
        fullRowRgb16Le: ByteArray,
        fullWidth: Int,
        previewRow: Int,
        previewWidth: Int,
        yPlane: ByteArray,
        uPlane: ByteArray,
        vPlane: ByteArray,
        neutralGains: DoubleArray,
    ) {
        for (previewColumn in 0 until previewWidth) {
            val sourceColumn = (previewColumn.toLong() * fullWidth / previewWidth).toInt()
            val source = sourceColumn * 6
            val red = display8(((((fullRowRgb16Le[source + 1].toInt() and 0xff) shl 8) or
                (fullRowRgb16Le[source].toInt() and 0xff)) * neutralGains[0]).toInt())
            val green = display8(((((fullRowRgb16Le[source + 3].toInt() and 0xff) shl 8) or
                (fullRowRgb16Le[source + 2].toInt() and 0xff)) * neutralGains[1]).toInt())
            val blue = display8(((((fullRowRgb16Le[source + 5].toInt() and 0xff) shl 8) or
                (fullRowRgb16Le[source + 4].toInt() and 0xff)) * neutralGains[2]).toInt())
            val y = ((66 * red + 129 * green + 25 * blue + 128) shr 8) + 16
            yPlane[previewRow * previewWidth + previewColumn] = y.coerceIn(0, 255).toByte()
            if ((previewRow and 1) == 0 && (previewColumn and 1) == 0) {
                val chromaIndex = previewRow / 2 * (previewWidth / 2) + previewColumn / 2
                val u = ((-38 * red - 74 * green + 112 * blue + 128) shr 8) + 128
                val v = ((112 * red - 94 * green - 18 * blue + 128) shr 8) + 128
                uPlane[chromaIndex] = u.coerceIn(0, 255).toByte()
                vPlane[chromaIndex] = v.coerceIn(0, 255).toByte()
            }
        }
    }

    private fun display8(linear16: Int): Int {
        val linear = linear16.coerceIn(0, 65_535) / 65_535.0
        val encoded = if (linear <= 0.0031308) 12.92 * linear
            else 1.055 * linear.pow(1.0 / 2.4) - 0.055
        return (encoded * 255.0 + 0.5).toInt().coerceIn(0, 255)
    }

    private fun normalizedRow(
        source: ByteBuffer,
        row: Int,
        width: Int,
        rowStride: Int,
        pixelStride: Int,
        blackLevels: DoubleArray,
        whiteLevel: Int,
    ): IntArray = IntArray(width) { column ->
        val raw = source.getShort(row * rowStride + column * pixelStride).toInt() and 0xffff
        val black = blackLevels[(row and 1) * 2 + (column and 1)]
        val span = (whiteLevel - black).coerceAtLeast(1.0)
        (((raw - black) / span) * 65_535.0).toInt().coerceIn(0, 65_535)
    }

    private fun interpolate(
        previous: IntArray,
        current: IntArray,
        next: IntArray,
        column: Int,
        absoluteRow: Int,
        width: Int,
        height: Int,
        cfa: Int,
        requestedChannel: Int,
    ): Int {
        val centerChannel = cfaChannel(cfa, column, absoluteRow)
        if (centerChannel == requestedChannel) return current[column]
        val left = adjacent(column, -1, width)
        val right = adjacent(column, 1, width)
        val upRow = if (absoluteRow > 0) previous else next
        val downRow = if (absoluteRow + 1 < height) next else previous
        return when {
            requestedChannel == 1 -> (
                current[left].toLong() + current[right] + upRow[column] + downRow[column]
            ).div(4).toInt()
            centerChannel == 1 -> {
                if (cfaChannel(cfa, left, absoluteRow) == requestedChannel) {
                    ((current[left].toLong() + current[right]) / 2L).toInt()
                } else {
                    ((upRow[column].toLong() + downRow[column]) / 2L).toInt()
                }
            }
            else -> (
                upRow[left].toLong() + upRow[right] + downRow[left] + downRow[right]
            ).div(4).toInt()
        }
    }

    private fun adjacent(value: Int, delta: Int, size: Int): Int {
        val candidate = value + delta
        return if (candidate in 0 until size) candidate else value - delta
    }

    /** 0=red, 1=green, 2=blue. */
    private fun cfaChannel(cfa: Int, x: Int, y: Int): Int {
        val index = (y and 1) * 2 + (x and 1)
        return when (cfa) {
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB -> when (index) {
                0 -> 0; 3 -> 2; else -> 1
            }
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG -> when (index) {
                1 -> 0; 2 -> 2; else -> 1
            }
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG -> when (index) {
                2 -> 0; 1 -> 2; else -> 1
            }
            CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR -> when (index) {
                3 -> 0; 0 -> 2; else -> 1
            }
            else -> error("Unsupported CFA $cfa")
        }
    }

    private fun RandomAccessFile.writeShortLe(value: Int) {
        write(value and 0xff)
        write((value ushr 8) and 0xff)
    }

    private fun RandomAccessFile.writeIntLe(value: Long) {
        write((value and 0xff).toInt())
        write(((value ushr 8) and 0xff).toInt())
        write(((value ushr 16) and 0xff).toInt())
        write(((value ushr 24) and 0xff).toInt())
    }

    private fun RandomAccessFile.writeIfdShort(tag: Int, value: Int) {
        writeShortLe(tag); writeShortLe(3); writeIntLe(1); writeShortLe(value); writeShortLe(0)
    }

    private fun RandomAccessFile.writeIfdLong(tag: Int, value: Long) {
        writeShortLe(tag); writeShortLe(4); writeIntLe(1); writeIntLe(value)
    }

    private fun RandomAccessFile.writeIfdOffset(tag: Int, count: Int, offset: Long) {
        writeShortLe(tag); writeShortLe(3); writeIntLe(count.toLong()); writeIntLe(offset)
    }
}

internal fun neutralAnalysisGains(neutral: DoubleArray?): DoubleArray {
    if (neutral == null || neutral.size < 3 ||
        neutral.take(3).any { !it.isFinite() || it <= 0.0 }
    ) return doubleArrayOf(1.0, 1.0, 1.0)
    val green = neutral[1]
    return doubleArrayOf(
        (green / neutral[0]).coerceIn(0.25, 4.0),
        1.0,
        (green / neutral[2]).coerceIn(0.25, 4.0),
    )
}

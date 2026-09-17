package com.metriccapture.app.camera

import android.graphics.Rect
import android.util.Size
import kotlin.math.ceil
import kotlin.math.abs

/** Explicit mapping from pre-correction sensor coordinates into one RAW buffer. */
data class RawImageGeometry(
    val imageWidth: Int,
    val imageHeight: Int,
    val intrinsics: DoubleArray,
) {
    init {
        require(imageWidth > 0 && imageHeight > 0)
        require(intrinsics.size >= 4)
    }

    fun scaledTo(output: Size): DoubleArray = scaledTo(output.width, output.height)

    fun scaledTo(outputWidth: Int, outputHeight: Int): DoubleArray {
        val sx = outputWidth.toDouble() / imageWidth
        val sy = outputHeight.toDouble() / imageHeight
        return doubleArrayOf(
            intrinsics[0] * sx,
            intrinsics[1] * sy,
            intrinsics[2] * sx,
            intrinsics[3] * sy,
        )
    }

    fun validationToRaw(x: Double, y: Double, validationSize: Size): Pair<Double, Double> =
        validationToRaw(x, y, validationSize.width, validationSize.height)

    fun validationToRaw(
        x: Double,
        y: Double,
        validationWidth: Int,
        validationHeight: Int,
    ): Pair<Double, Double> =
        x * imageWidth / validationWidth to y * imageHeight / validationHeight
}

/**
 * RAW may expose the full pixel array (including inactive margins) or only the
 * pre-correction active array. Both mappings are explicit; an ambiguous RAW
 * geometry is rejected instead of being treated as a cropped YUV stream.
 */
fun resolveRawImageGeometry(
    rawSize: Size,
    pixelArraySize: Size,
    preCorrectionActiveArray: Rect,
    preCorrectionIntrinsics: List<Float>,
): RawImageGeometry = resolveRawImageGeometry(
    rawSize.width,
    rawSize.height,
    pixelArraySize.width,
    pixelArraySize.height,
    preCorrectionActiveArray.left,
    preCorrectionActiveArray.top,
    preCorrectionActiveArray.width(),
    preCorrectionActiveArray.height(),
    preCorrectionIntrinsics,
)

fun resolveRawImageGeometry(
    rawWidth: Int,
    rawHeight: Int,
    pixelArrayWidth: Int,
    pixelArrayHeight: Int,
    activeLeft: Int,
    activeTop: Int,
    activeWidth: Int,
    activeHeight: Int,
    preCorrectionIntrinsics: List<Float>,
): RawImageGeometry {
    require(preCorrectionIntrinsics.size >= 4)
    val base = preCorrectionIntrinsics.map(Float::toDouble)
    val scaledPixelArrayMatch = sameAspect(rawWidth, rawHeight, pixelArrayWidth, pixelArrayHeight)
    val scaledActiveArrayMatch = sameAspect(rawWidth, rawHeight, activeWidth, activeHeight)
    val rawIsPixelArray = rawWidth == pixelArrayWidth && rawHeight == pixelArrayHeight
    val rawIsActiveArray = rawWidth == activeWidth && rawHeight == activeHeight
    if (!rawIsPixelArray && !rawIsActiveArray && scaledPixelArrayMatch &&
        scaledActiveArrayMatch &&
        (pixelArrayWidth != activeWidth || pixelArrayHeight != activeHeight)
    ) {
        error(
            "RAW dimensions ${rawWidth}x${rawHeight} have an ambiguous " +
                "mapping to the reported pixel and pre-correction active arrays"
        )
    }
    val intrinsics = when {
        rawIsPixelArray -> doubleArrayOf(
            base[0],
            base[1],
            base[2] + activeLeft,
            base[3] + activeTop,
        )
        rawIsActiveArray -> doubleArrayOf(base[0], base[1], base[2], base[3])
        scaledPixelArrayMatch -> {
            val sx = rawWidth.toDouble() / pixelArrayWidth
            val sy = rawHeight.toDouble() / pixelArrayHeight
            doubleArrayOf(
                base[0] * sx,
                base[1] * sy,
                (base[2] + activeLeft) * sx,
                (base[3] + activeTop) * sy,
            )
        }
        scaledActiveArrayMatch -> {
            val sx = rawWidth.toDouble() / activeWidth
            val sy = rawHeight.toDouble() / activeHeight
            doubleArrayOf(base[0] * sx, base[1] * sy, base[2] * sx, base[3] * sy)
        }
        else -> error(
            "RAW dimensions ${rawWidth}x${rawHeight} do not map to the " +
                "reported pixel or pre-correction active array"
        )
    }
    return RawImageGeometry(rawWidth, rawHeight, intrinsics)
}

fun boundedRawValidationSize(rawSize: Size, maximumLongEdge: Int = 1280): Size {
    val dimensions = boundedRawValidationDimensions(
        rawSize.width,
        rawSize.height,
        maximumLongEdge,
    )
    return Size(dimensions.first, dimensions.second)
}

fun boundedRawValidationDimensions(
    rawWidth: Int,
    rawHeight: Int,
    maximumLongEdge: Int = 1280,
): Pair<Int, Int> {
    require(maximumLongEdge >= 320)
    val divisor = ceil(maxOf(rawWidth, rawHeight).toDouble() / maximumLongEdge)
        .toInt().coerceAtLeast(1)
    fun even(value: Int) = (value.coerceAtLeast(2) and -2).coerceAtLeast(2)
    return even(rawWidth / divisor) to even(rawHeight / divisor)
}

private fun sameAspect(
    firstWidth: Int,
    firstHeight: Int,
    secondWidth: Int,
    secondHeight: Int,
): Boolean {
    val firstAspect = firstWidth.toDouble() / firstHeight
    val secondAspect = secondWidth.toDouble() / secondHeight
    return abs(firstAspect - secondAspect) <= 0.001
}

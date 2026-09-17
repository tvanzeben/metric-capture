package com.metriccapture.app.vision

import android.media.Image
import org.opencv.core.CvType
import org.opencv.core.Mat

/** Immutable copy so the Camera2 Image can be closed before OpenCV processing finishes. */
data class FramePlanes(
    val width: Int,
    val height: Int,
    val timestampNs: Long,
    val y: ByteArray,
    val yRowStride: Int,
    val yPixelStride: Int,
    val u: ByteArray,
    val uRowStride: Int,
    val uPixelStride: Int,
    val v: ByteArray,
    val vRowStride: Int,
    val vPixelStride: Int,
) {
    fun grayMat(): Mat {
        val packed = if (yRowStride == width && yPixelStride == 1 && y.size == width * height) {
            y
        } else {
            ByteArray(width * height).also { destination ->
                for (row in 0 until height) for (column in 0 until width) {
                    destination[row * width + column] =
                        y[row * yRowStride + column * yPixelStride]
                }
            }
        }
        return Mat(height, width, CvType.CV_8UC1).apply { put(0, 0, packed) }
    }

    companion object {
        fun copyOf(image: Image): FramePlanes {
            require(image.planes.size == 3)
            fun packedPlane(index: Int, planeWidth: Int, planeHeight: Int): ByteArray {
                val plane = image.planes[index]
                val source = plane.buffer.duplicate()
                val start = source.position()
                return ByteArray(planeWidth * planeHeight).also { destination ->
                    for (row in 0 until planeHeight) {
                        val sourceRow = start + row * plane.rowStride
                        val destinationRow = row * planeWidth
                        for (column in 0 until planeWidth) {
                            destination[destinationRow + column] = source.get(
                                sourceRow + column * plane.pixelStride
                            )
                        }
                    }
                }
            }
            val chromaWidth = (image.width + 1) / 2
            val chromaHeight = (image.height + 1) / 2
            return FramePlanes(
                width = image.width,
                height = image.height,
                timestampNs = image.timestamp,
                y = packedPlane(0, image.width, image.height),
                yRowStride = image.width,
                yPixelStride = 1,
                u = packedPlane(1, chromaWidth, chromaHeight),
                uRowStride = chromaWidth,
                uPixelStride = 1,
                v = packedPlane(2, chromaWidth, chromaHeight),
                vRowStride = chromaWidth,
                vPixelStride = 1,
            )
        }
    }
}

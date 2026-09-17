package com.metriccapture.app.reconstruction

import com.metriccapture.app.geometry.Vec3
import org.opencv.calib.Calib
import org.opencv.geometry.Geometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.stereo.Stereo
import org.opencv.stereo.StereoSGBM
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

data class DensePairResult(
    val firstFrameId: String,
    val secondFrameId: String,
    val minimumDisparity: Int,
    val disparityCount: Int,
    val points: List<MetricPoint>,
)

/**
 * Rectifies only neighboring guided views and derives the SGBM disparity range
 * from the current visual-hull envelope instead of searching arbitrary depth.
 */
class ConstrainedStereoReconstructor(
    private val blockSize: Int = 5,
    private val outputStride: Int = 2,
) {
    fun reconstructPair(
        first: ReconstructionFrame,
        second: ReconstructionFrame,
        candidateEnvelope: Aabb,
        geometryLongEdgePixels: Int = 2048,
    ): DensePairResult? {
        var firstColor = readBoundedWorkingImage(first)
        var secondColor = readBoundedWorkingImage(second)
        if (firstColor.empty() || secondColor.empty() || firstColor.size() != secondColor.size()) {
            firstColor.release(); secondColor.release(); return null
        }
        val firstDecodeScale = firstColor.cols().toDouble() / first.imageWidth
        val secondDecodeScale = secondColor.cols().toDouble() / second.imageWidth
        val inputLongEdge = max(firstColor.cols(), firstColor.rows())
        val scale = min(1.0, geometryLongEdgePixels.toDouble() / inputLongEdge)
        val firstFrame = scaledFrame(first, firstDecodeScale * scale)
        val secondFrame = scaledFrame(second, secondDecodeScale * scale)
        if (scale < 1.0) {
            val resizedFirst = Mat(); val resizedSecond = Mat()
            Imgproc.resize(firstColor, resizedFirst, Size(), scale, scale, Imgproc.INTER_AREA)
            Imgproc.resize(secondColor, resizedSecond, Size(), scale, scale, Imgproc.INTER_AREA)
            firstColor.release(); secondColor.release()
            firstColor = resizedFirst; secondColor = resizedSecond
        }
        val firstGray = Mat()
        val secondGray = Mat()
        Imgproc.cvtColor(firstColor, firstGray, Imgproc.COLOR_BGR2GRAY)
        Imgproc.cvtColor(secondColor, secondGray, Imgproc.COLOR_BGR2GRAY)
        val k1 = cameraMatrix(firstFrame)
        val k2 = cameraMatrix(secondFrame)
        val d1 = MatOfDouble(*firstFrame.distortion)
        val d2 = MatOfDouble(*secondFrame.distortion)
        val worldToFirst = rotation(firstFrame.rotationVector)
        val worldToSecond = rotation(secondFrame.rotationVector)
        val firstTranspose = worldToFirst.t()
        val firstToSecond = Mat()
        val emptyProduct = Mat()
        Core.gemm(worldToSecond, firstTranspose, 1.0, emptyProduct, 0.0, firstToSecond)
        val relativeTranslation = relativeTranslation(firstFrame, secondFrame, firstToSecond)
        val rectify1 = Mat()
        val rectify2 = Mat()
        val projection1 = Mat()
        val projection2 = Mat()
        val q = Mat()
        val roi1 = Rect()
        val roi2 = Rect()
        try {
            Stereo.stereoRectify(
                k1, d1, k2, d2, firstGray.size(),
                firstToSecond, relativeTranslation,
                rectify1, rectify2, projection1, projection2, q,
                Calib.CALIB_ZERO_DISPARITY,
                0.0,
                firstGray.size(),
                roi1,
                roi2,
            )
            val disparityRange = hullDisparityRange(
                candidateEnvelope, firstFrame, secondFrame, k1, d1, k2, d2,
                rectify1, rectify2, projection1, projection2,
            ) ?: return null
            val map11 = Mat(); val map12 = Mat(); val map21 = Mat(); val map22 = Mat()
            Imgproc.initUndistortRectifyMap(
                k1, d1, rectify1, projection1, firstGray.size(), CvType.CV_32FC1, map11, map12
            )
            Imgproc.initUndistortRectifyMap(
                k2, d2, rectify2, projection2, secondGray.size(), CvType.CV_32FC1, map21, map22
            )
            val rectified1 = Mat(); val rectified2 = Mat(); val rectifiedColor = Mat()
            Imgproc.remap(firstGray, rectified1, map11, map12, Imgproc.INTER_LINEAR)
            Imgproc.remap(secondGray, rectified2, map21, map22, Imgproc.INTER_LINEAR)
            Imgproc.remap(firstColor, rectifiedColor, map11, map12, Imgproc.INTER_LINEAR)
            val sourceMask = loadMask(first.foregroundMask, firstGray.size())
            val secondSourceMask = loadMask(second.foregroundMask, secondGray.size())
            val rectifiedMask = Mat()
            val secondRectifiedMask = Mat()
            Imgproc.remap(sourceMask, rectifiedMask, map11, map12, Imgproc.INTER_NEAREST)
            Imgproc.remap(secondSourceMask, secondRectifiedMask, map21, map22, Imgproc.INTER_NEAREST)
            Core.bitwise_and(rectifiedMask, secondRectifiedMask, rectifiedMask)
            val maskKernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
            Imgproc.erode(rectifiedMask, rectifiedMask, maskKernel)
            maskKernel.release()
            val disparity16 = Mat()
            val stereo = StereoSGBM.create(
                disparityRange.first,
                disparityRange.second,
                blockSize,
                8 * blockSize * blockSize,
                32 * blockSize * blockSize,
                1,
                31,
                10,
                100,
                2,
                StereoSGBM.MODE_SGBM_3WAY,
            )
            stereo.compute(rectified1, rectified2, disparity16)
            val disparity = Mat()
            disparity16.convertTo(disparity, CvType.CV_32F, 1.0 / 16.0)
            val points3d = Mat()
            Stereo.reprojectImageTo3D(disparity, points3d, q, true, CvType.CV_32F)
            val firstRectifiedToWorld = rectifiedCameraToWorld(firstFrame, rectify1)
            val points = extractPoints(
                points3d,
                disparity,
                rectifiedColor,
                rectifiedMask,
                firstRectifiedToWorld.first,
                firstRectifiedToWorld.second,
                candidateEnvelope,
                disparityRange.first,
            )
            listOf(map11, map12, map21, map22, rectified1, rectified2, rectifiedColor,
                sourceMask, secondSourceMask, rectifiedMask, secondRectifiedMask,
                disparity16, disparity, points3d, firstRectifiedToWorld.first).forEach(Mat::release)
            stereo.clear()
            return DensePairResult(
                first.frameId,
                second.frameId,
                disparityRange.first,
                disparityRange.second,
                points,
            )
        } finally {
            listOf(firstColor, secondColor, firstGray, secondGray, k1, k2, d1, d2,
                worldToFirst, worldToSecond, firstTranspose, emptyProduct, firstToSecond, relativeTranslation,
                rectify1, rectify2, projection1, projection2, q).forEach(Mat::release)
        }
    }

    private fun hullDisparityRange(
        envelope: Aabb,
        first: ReconstructionFrame,
        second: ReconstructionFrame,
        k1: Mat,
        d1: Mat,
        k2: Mat,
        d2: Mat,
        rectify1: Mat,
        rectify2: Mat,
        projection1: Mat,
        projection2: Mat,
    ): Pair<Int, Int>? {
        val image1 = envelope.corners().mapNotNull { project(first, it) }
        val image2 = envelope.corners().mapNotNull { project(second, it) }
        if (image1.size != 8 || image2.size != 8) return null
        val rectifiedPoints1 = MatOfPoint2f()
        val rectifiedPoints2 = MatOfPoint2f()
        val sourcePoints1 = MatOfPoint2f(*image1.toTypedArray())
        val sourcePoints2 = MatOfPoint2f(*image2.toTypedArray())
        Geometry.undistortPoints(sourcePoints1, rectifiedPoints1, k1, d1, rectify1, projection1)
        Geometry.undistortPoints(sourcePoints2, rectifiedPoints2, k2, d2, rectify2, projection2)
        val disparities = rectifiedPoints1.toArray().zip(rectifiedPoints2.toArray()).map { (a, b) -> a.x - b.x }
        sourcePoints1.release(); sourcePoints2.release()
        rectifiedPoints1.release(); rectifiedPoints2.release()
        val low = floor(disparities.minOrNull()!! - 8.0).toInt()
        val high = ceil(disparities.maxOrNull()!! + 8.0).toInt()
        val minimum = floor(low / 16.0).toInt() * 16
        val count = (ceil((high - minimum).coerceAtLeast(16) / 16.0).toInt() * 16)
            .coerceIn(16, 256)
        return minimum to count
    }

    private fun extractPoints(
        points3d: Mat,
        disparity: Mat,
        color: Mat,
        foregroundMask: Mat,
        rectifiedToWorldRotation: Mat,
        rectifiedToWorldTranslation: DoubleArray,
        envelope: Aabb,
        invalidDisparity: Int,
    ): List<MetricPoint> {
        val result = ArrayList<MetricPoint>()
        for (row in 0 until points3d.rows() step outputStride) {
            for (column in 0 until points3d.cols() step outputStride) {
                if (foregroundMask[row, column][0] == 0.0) continue
                val d = disparity[row, column][0]
                if (!d.isFinite() || d <= invalidDisparity.toDouble()) continue
                val p = points3d[row, column]
                if (p.size < 3 || p.any { !it.isFinite() } || kotlin.math.abs(p[2]) > 10_000.0) continue
                val world = DoubleArray(3) { axis ->
                    rectifiedToWorldRotation[axis, 0][0] * p[0] +
                        rectifiedToWorldRotation[axis, 1][0] * p[1] +
                        rectifiedToWorldRotation[axis, 2][0] * p[2] +
                        rectifiedToWorldTranslation[axis]
                }
                if (world[0] !in envelope.minimum.x..envelope.maximum.x ||
                    world[1] !in envelope.minimum.y..envelope.maximum.y ||
                    world[2] !in max(-0.5, envelope.minimum.z)..envelope.maximum.z
                ) continue
                val bgr = color[row, column]
                result += MetricPoint(
                    world[0], world[1], world[2],
                    bgr[2].toInt().coerceIn(0, 255),
                    bgr[1].toInt().coerceIn(0, 255),
                    bgr[0].toInt().coerceIn(0, 255),
                    confidence = 1.0,
                )
            }
        }
        return result
    }

    private fun rectifiedCameraToWorld(frame: ReconstructionFrame, rectify: Mat): Pair<Mat, DoubleArray> {
        val worldToCamera = rotation(frame.rotationVector)
        val combined = Mat()
        val empty = Mat()
        Core.gemm(rectify, worldToCamera, 1.0, empty, 0.0, combined)
        empty.release()
        val inverse = combined.t()
        val rectifiedTranslation = DoubleArray(3) { row ->
            (0..2).sumOf { column -> rectify[row, column][0] * frame.translationVector[column] }
        }
        val translation = DoubleArray(3) { axis ->
            -(inverse[axis, 0][0] * rectifiedTranslation[0] +
                inverse[axis, 1][0] * rectifiedTranslation[1] +
                inverse[axis, 2][0] * rectifiedTranslation[2])
        }
        worldToCamera.release(); combined.release()
        return inverse to translation
    }

    private fun relativeTranslation(
        first: ReconstructionFrame,
        second: ReconstructionFrame,
        firstToSecond: Mat,
    ): Mat {
        val result = Mat(3, 1, CvType.CV_64F)
        for (row in 0..2) {
            val transformedFirstT = (0..2).sumOf { column ->
                firstToSecond[row, column][0] * first.translationVector[column]
            }
            result.put(row, 0, second.translationVector[row] - transformedFirstT)
        }
        return result
    }

    private fun project(frame: ReconstructionFrame, point: Vec3): Point? {
        val projected = CameraProjection(
            frame.cameraFx,
            frame.cameraFy,
            frame.cameraCx,
            frame.cameraCy,
            frame.rotationVector,
            frame.translationVector,
            frame.distortion,
        ).project(point) ?: return null
        return Point(projected.x, projected.y)
    }

    private fun cameraMatrix(frame: ReconstructionFrame): Mat = Mat.eye(3, 3, CvType.CV_64F).apply {
        put(0, 0, frame.cameraFx); put(1, 1, frame.cameraFy)
        put(0, 2, frame.cameraCx); put(1, 2, frame.cameraCy)
    }

    private fun rotation(vector: DoubleArray): Mat {
        val source = Mat(3, 1, CvType.CV_64F)
        source.put(0, 0, *vector)
        return Mat().also { Geometry.Rodrigues(source, it); source.release() }
    }

    private fun loadMask(file: java.io.File?, size: Size): Mat {
        if (file == null || !file.exists()) return Mat.ones(size, CvType.CV_8UC1)
        val loaded = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        if (loaded.empty()) {
            loaded.release()
            return Mat.ones(size, CvType.CV_8UC1)
        }
        if (loaded.size() == size) return loaded
        val resized = Mat()
        Imgproc.resize(loaded, resized, size, 0.0, 0.0, Imgproc.INTER_NEAREST)
        loaded.release()
        return resized
    }

    private fun readBoundedWorkingImage(frame: ReconstructionFrame): Mat {
        val longEdge = max(frame.imageWidth, frame.imageHeight)
        val flag = when {
            longEdge > 16_000 -> Imgcodecs.IMREAD_REDUCED_COLOR_8
            longEdge > 8_000 -> Imgcodecs.IMREAD_REDUCED_COLOR_4
            longEdge > 4_000 -> Imgcodecs.IMREAD_REDUCED_COLOR_2
            else -> Imgcodecs.IMREAD_COLOR
        }
        return Imgcodecs.imread(frame.workingImage.absolutePath, flag)
    }

    private fun scaledFrame(frame: ReconstructionFrame, scale: Double): ReconstructionFrame =
        if (scale == 1.0) frame else frame.copy(
            cameraFx = frame.cameraFx * scale,
            cameraFy = frame.cameraFy * scale,
            cameraCx = frame.cameraCx * scale,
            cameraCy = frame.cameraCy * scale,
        )
}

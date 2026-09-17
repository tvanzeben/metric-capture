package com.metriccapture.app.vision

import org.opencv.geometry.Geometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

data class NegativeSpaceResult(
    val canonicalForeground: Mat,
    val canonicalVisibleMat: Mat,
    val imageForeground: Mat,
    val imageToCanonical: Mat,
    val canonicalToImage: Mat,
    val radiometricScale: Double,
    val radiometricOffset: Double,
    val radiometricMeanAbsoluteError: Double,
    val highlightClippedFraction: Double,
    val shadowClippedFraction: Double,
    val foregroundFraction: Double,
) : AutoCloseable {
    override fun close() {
        canonicalForeground.release()
        canonicalVisibleMat.release()
        imageForeground.release()
        imageToCanonical.release()
        canonicalToImage.release()
    }
}

/**
 * Rectifies the observed mat, solves a robust per-frame radiometric transform,
 * and returns the connected foreground component occupying the artifact field.
 */
class NegativeSpaceExtractor(
    private val canonicalTemplateGray: Mat,
    private val pixelsPerMm: Double,
    private val minimumXmm: Double,
    private val maximumYmm: Double,
    private val artifactFieldRadiusMm: Double,
    private val comparisonThreshold: Double = 18.0,
) {
    init {
        require(canonicalTemplateGray.type() == CvType.CV_8UC1)
        require(pixelsPerMm > 0.0)
        require(artifactFieldRadiusMm > 0.0)
    }

    fun extract(
        imageGray: Mat,
        fiducials: List<FiducialObservation>,
    ): NegativeSpaceResult? {
        if (fiducials.size < 4) return null
        val imagePoints = MatOfPoint2f(*fiducials.map(FiducialObservation::pixel).toTypedArray())
        val canonicalPoints = MatOfPoint2f(*fiducials.map {
            canonicalPoint(MatModel.pointsMm.getValue(it.id).x, MatModel.pointsMm.getValue(it.id).y)
        }.toTypedArray())
        val imageToCanonical = Geometry.findHomography(
            imagePoints,
            canonicalPoints,
            Geometry.RANSAC,
            2.0,
        )
        imagePoints.release()
        canonicalPoints.release()
        if (imageToCanonical.empty()) return null
        val canonicalToImage = imageToCanonical.inv()

        val rectified = Mat()
        Imgproc.warpPerspective(
            imageGray,
            rectified,
            imageToCanonical,
            canonicalTemplateGray.size(),
            Imgproc.INTER_LINEAR,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
        )
        val validImage = Mat.ones(imageGray.size(), CvType.CV_8UC1)
        val canonicalValid = Mat()
        Imgproc.warpPerspective(
            validImage,
            canonicalValid,
            imageToCanonical,
            canonicalTemplateGray.size(),
            Imgproc.INTER_NEAREST,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
        )
        validImage.release()

        val referenceMask = radiometricReferenceMask(canonicalValid)
        val referencePixelCount = Core.countNonZero(referenceMask).coerceAtLeast(1)
        val highlightMask = Mat()
        val shadowMask = Mat()
        Imgproc.threshold(rectified, highlightMask, 249.0, 255.0, Imgproc.THRESH_BINARY)
        Imgproc.threshold(rectified, shadowMask, 5.0, 255.0, Imgproc.THRESH_BINARY_INV)
        Core.bitwise_and(highlightMask, referenceMask, highlightMask)
        Core.bitwise_and(shadowMask, referenceMask, shadowMask)
        val highlightClippedFraction = Core.countNonZero(highlightMask).toDouble() / referencePixelCount
        val shadowClippedFraction = Core.countNonZero(shadowMask).toDouble() / referencePixelCount
        highlightMask.release()
        shadowMask.release()
        val radiometry = robustRadiometricFit(canonicalTemplateGray, rectified, referenceMask)
        val normalized = normalize(rectified, radiometry.first, radiometry.second)
        rectified.release()

        val difference = Mat()
        Core.absdiff(normalized, canonicalTemplateGray, difference)
        val radiometricMeanAbsoluteError = Core.mean(difference, referenceMask).`val`[0]
        referenceMask.release()
        normalized.release()
        val rawForeground = Mat()
        Imgproc.threshold(
            difference,
            rawForeground,
            comparisonThreshold,
            255.0,
            Imgproc.THRESH_BINARY,
        )
        difference.release()
        Core.bitwise_and(rawForeground, canonicalValid, rawForeground)
        val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.morphologyEx(rawForeground, rawForeground, Imgproc.MORPH_OPEN, kernel)
        Imgproc.morphologyEx(rawForeground, rawForeground, Imgproc.MORPH_CLOSE, kernel)
        kernel.release()

        val foreground = selectArtifactComponent(rawForeground)
        rawForeground.release()
        val visibleMat = Mat()
        Core.bitwise_not(foreground, visibleMat)
        Core.bitwise_and(visibleMat, canonicalValid, visibleMat)
        canonicalValid.release()

        val imageForeground = Mat.zeros(imageGray.size(), CvType.CV_8UC1)
        Imgproc.warpPerspective(
            foreground,
            imageForeground,
            canonicalToImage,
            imageGray.size(),
            Imgproc.INTER_NEAREST,
            Core.BORDER_CONSTANT,
            Scalar(0.0),
        )
        val fraction = Core.countNonZero(foreground).toDouble() /
            (foreground.rows().toDouble() * foreground.cols().toDouble())
        return NegativeSpaceResult(
            canonicalForeground = foreground,
            canonicalVisibleMat = visibleMat,
            imageForeground = imageForeground,
            imageToCanonical = imageToCanonical,
            canonicalToImage = canonicalToImage,
            radiometricScale = radiometry.first,
            radiometricOffset = radiometry.second,
            radiometricMeanAbsoluteError = radiometricMeanAbsoluteError,
            highlightClippedFraction = highlightClippedFraction,
            shadowClippedFraction = shadowClippedFraction,
            foregroundFraction = fraction,
        )
    }

    private fun canonicalPoint(xMm: Double, yMm: Double): Point = Point(
        (xMm - minimumXmm) * pixelsPerMm,
        (maximumYmm - yMm) * pixelsPerMm,
    )

    private fun radiometricReferenceMask(valid: Mat): Mat {
        val mask = Mat.zeros(valid.size(), CvType.CV_8UC1)
        val center = canonicalPoint(0.0, 0.0)
        val inner = (artifactFieldRadiusMm * pixelsPerMm).toInt()
        val maximumRadius = min(mask.cols(), mask.rows()) / 2 - 3
        Imgproc.circle(mask, center, maximumRadius, Scalar(255.0), -1)
        Imgproc.circle(mask, center, inner, Scalar(0.0), -1)
        Core.bitwise_and(mask, valid, mask)
        return mask
    }

    /** Returns observed = scale * template + offset using iterative residual rejection. */
    private fun robustRadiometricFit(
        template: Mat,
        observed: Mat,
        mask: Mat,
    ): Pair<Double, Double> {
        var scale = 1.0
        var offset = 0.0
        var limit = 40.0
        repeat(3) {
            var n = 0L
            var sx = 0.0
            var sy = 0.0
            var sxx = 0.0
            var sxy = 0.0
            for (row in 0 until template.rows() step 3) {
                for (column in 0 until template.cols() step 3) {
                    if (mask[row, column][0] == 0.0) continue
                    val x = template[row, column][0]
                    val y = observed[row, column][0]
                    if (abs(y - (scale * x + offset)) > limit) continue
                    n++
                    sx += x
                    sy += y
                    sxx += x * x
                    sxy += x * y
                }
            }
            val denominator = n * sxx - sx * sx
            if (n > 100 && abs(denominator) > 1e-9) {
                scale = ((n * sxy - sx * sy) / denominator).coerceIn(0.25, 4.0)
                offset = ((sy - scale * sx) / n).coerceIn(-128.0, 128.0)
            }
            limit = max(12.0, limit * 0.6)
        }
        return scale to offset
    }

    private fun normalize(observed: Mat, scale: Double, offset: Double): Mat {
        val normalized = Mat()
        observed.convertTo(normalized, CvType.CV_8UC1, 1.0 / scale, -offset / scale)
        return normalized
    }

    private fun selectArtifactComponent(binary: Mat): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()
        val count = Imgproc.connectedComponentsWithStats(binary, labels, stats, centroids)
        val expectedCenter = canonicalPoint(0.0, 0.0)
        var bestLabel = 0
        var bestScore = Double.NEGATIVE_INFINITY
        val acceptedLabels = mutableListOf<Int>()
        val fieldRadiusPixels = artifactFieldRadiusMm * pixelsPerMm * 1.10
        for (label in 1 until count) {
            val area = stats[label, Imgproc.CC_STAT_AREA][0]
            if (area < 25.0) continue
            val cx = centroids[label, 0][0]
            val cy = centroids[label, 1][0]
            val distance = sqrt((cx - expectedCenter.x) * (cx - expectedCenter.x) +
                (cy - expectedCenter.y) * (cy - expectedCenter.y))
            val score = area / (1.0 + distance)
            if (distance <= fieldRadiusPixels) acceptedLabels += label
            if (score > bestScore) {
                bestScore = score
                bestLabel = label
            }
        }
        val selected = Mat.zeros(binary.size(), CvType.CV_8UC1)
        val labelsToKeep = acceptedLabels.ifEmpty { listOfNotNull(bestLabel.takeIf { it > 0 }) }
        val component = Mat()
        for (label in labelsToKeep) {
            Core.compare(labels, Scalar(label.toDouble()), component, Core.CMP_EQ)
            Core.bitwise_or(selected, component, selected)
        }
        component.release()
        labels.release()
        stats.release()
        centroids.release()
        return selected
    }
}

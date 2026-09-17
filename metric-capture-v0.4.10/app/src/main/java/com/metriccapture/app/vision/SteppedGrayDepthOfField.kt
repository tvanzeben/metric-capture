package com.metriccapture.app.vision

import org.opencv.geometry.Geometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.cos
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin

data class GrayStepFocusSample(
    val boundaryRadiusMm: Double,
    val xMm: Double,
    val yMm: Double,
    val cameraDepthMm: Double,
    /** 10-90% edge-spread width after metric rectification. */
    val edgeWidthCanonicalPx: Double,
    val measuredStepContrast: Double,
)

data class GrayStepFocusAnalysis(
    val detectedBoundaryCount: Int,
    val samples: List<GrayStepFocusSample>,
    val medianEdgeWidthCanonicalPx: Double,
    val percentile90EdgeWidthCanonicalPx: Double,
    val nearSampleDepthMm: Double,
    val farSampleDepthMm: Double,
    val medianStepContrast: Double,
    /** 1.0 when the detected ring transitions have the same local contrast. */
    val equalStepConsistency: Double,
) {
    val sampleCount: Int get() = samples.size
    val sampledDepthSpanMm: Double get() = farSampleDepthMm - nearSampleDepthMm
    val hasSufficientEvidence: Boolean
        get() = detectedBoundaryCount >= 2 && sampleCount >= 20 &&
            sampledDepthSpanMm >= 20.0 && equalStepConsistency >= 0.45
}

data class FocusCandidateMeasurement(
    val focusDistanceDiopters: Float,
    val analysis: GrayStepFocusAnalysis,
)

/** Session-wide reference derived from exact RAW focus-bracket exposures. */
data class DepthOfFieldCalibration(
    val focusDistanceDiopters: Float,
    val candidateCount: Int,
    val detectedBoundaryCount: Int,
    val sampleCount: Int,
    val referenceMedianEdgeWidthCanonicalPx: Double,
    val referenceP90EdgeWidthCanonicalPx: Double,
    val maximumP90EdgeWidthCanonicalPx: Double,
    val nearReferenceDepthMm: Double,
    val farReferenceDepthMm: Double,
    val medianStepContrast: Double,
    val referenceEqualStepConsistency: Double,
) {
    fun score(analysis: GrayStepFocusAnalysis?): Double {
        if (analysis == null || analysis.detectedBoundaryCount < 2 || analysis.sampleCount < 12) {
            return 0.0
        }
        val widthScore = (
            maximumP90EdgeWidthCanonicalPx /
                analysis.percentile90EdgeWidthCanonicalPx.coerceAtLeast(0.05)
            ).coerceIn(0.0, 1.0)
        val sampleScore = (analysis.sampleCount.toDouble() /
            max(12, sampleCount / 2).toDouble()).coerceIn(0.0, 1.0)
        val boundaryScore = (analysis.detectedBoundaryCount.toDouble() /
            max(2, detectedBoundaryCount / 2).toDouble()).coerceIn(0.0, 1.0)
        val equalStepScore = (
            analysis.equalStepConsistency /
                (referenceEqualStepConsistency * 0.75).coerceAtLeast(0.45)
            ).coerceIn(0.0, 1.0)
        return min(widthScore, min(sampleScore, min(boundaryScore, equalStepScore)))
    }

    fun accepts(analysis: GrayStepFocusAnalysis?): Boolean = score(analysis) >= 0.80
}

object SessionFocusSelector {
    /** Three RAW probes are enough to test the metered focus and one neighbor on either side. */
    fun candidateDiopters(
        meteredFocusDiopters: Float,
        minimumFocusDistanceDiopters: Float,
    ): List<Float> {
        if (minimumFocusDistanceDiopters <= 0f) return listOf(0f)
        val center = meteredFocusDiopters.coerceIn(0f, minimumFocusDistanceDiopters)
        val offset = max(0.15f, center * 0.25f)
        return listOf(center - offset, center, center + offset)
            .map { it.coerceIn(0f, minimumFocusDistanceDiopters) }
            .distinctBy { (it * 10_000f).toInt() }
            .sorted()
    }

    fun select(
        measurements: List<FocusCandidateMeasurement>,
        meteredFocusDiopters: Float,
    ): DepthOfFieldCalibration? {
        val usable = measurements.filter { it.analysis.hasSufficientEvidence }
        if (usable.isEmpty()) return null
        val maximumSamples = usable.maxOf { it.analysis.sampleCount }.toDouble()
        val maximumBoundaries = usable.maxOf { it.analysis.detectedBoundaryCount }.toDouble()
        val selected = usable.minWithOrNull(
            compareBy<FocusCandidateMeasurement> {
                it.analysis.percentile90EdgeWidthCanonicalPx *
                    (maximumSamples / it.analysis.sampleCount) *
                    (maximumBoundaries / it.analysis.detectedBoundaryCount)
            }.thenBy {
                it.analysis.medianEdgeWidthCanonicalPx
            }.thenBy {
                abs(it.focusDistanceDiopters - meteredFocusDiopters)
            }
        ) ?: return null
        val analysis = selected.analysis
        return DepthOfFieldCalibration(
            focusDistanceDiopters = selected.focusDistanceDiopters,
            candidateCount = measurements.size,
            detectedBoundaryCount = analysis.detectedBoundaryCount,
            sampleCount = analysis.sampleCount,
            referenceMedianEdgeWidthCanonicalPx = analysis.medianEdgeWidthCanonicalPx,
            referenceP90EdgeWidthCanonicalPx = analysis.percentile90EdgeWidthCanonicalPx,
            maximumP90EdgeWidthCanonicalPx =
                analysis.percentile90EdgeWidthCanonicalPx * 1.40 + 0.25,
            nearReferenceDepthMm = analysis.nearSampleDepthMm,
            farReferenceDepthMm = analysis.farSampleDepthMm,
            medianStepContrast = analysis.medianStepContrast,
            referenceEqualStepConsistency = analysis.equalStepConsistency,
        )
    }
}

/**
 * Uses the mat's equal stepped-gray rings as a dense focus chart. Ring boundaries
 * are detected from the current registered frame, so the printed artwork remains
 * the authority and no hard-coded ring radii are required.
 */
class SteppedGrayDepthOfFieldEvaluator(
    private val pixelsPerMm: Double = 4.0,
    private val minimumXmm: Double = -75.0,
    private val maximumYmm: Double = 75.0,
    private val widthMm: Double = 150.0,
    private val heightMm: Double = 150.0,
    private val analysisInnerRadiusMm: Double = 45.0,
    private val analysisOuterRadiusMm: Double = 71.0,
) {
    private val canonicalSize = Size(widthMm * pixelsPerMm, heightMm * pixelsPerMm)
    private val center = Point(-minimumXmm * pixelsPerMm, maximumYmm * pixelsPerMm)

    fun evaluate(
        imageGray: Mat,
        pose: MatPoseResult,
        canonicalVisibleMat: Mat? = null,
    ): GrayStepFocusAnalysis? {
        if (pose.fiducials.size < 4 || imageGray.empty()) return null
        val imagePoints = MatOfPoint2f(*pose.fiducials.map(FiducialObservation::pixel).toTypedArray())
        val canonicalPoints = MatOfPoint2f(*pose.fiducials.map {
            val point = MatModel.pointsMm.getValue(it.id)
            canonicalPoint(point.x, point.y)
        }.toTypedArray())
        val homography = Geometry.findHomography(
            imagePoints, canonicalPoints, Geometry.RANSAC, 2.0,
        )
        imagePoints.release()
        canonicalPoints.release()
        if (homography.empty()) {
            homography.release()
            return null
        }

        val rectified = Mat()
        val sourceValid = Mat.ones(imageGray.size(), CvType.CV_8UC1)
        val valid = Mat()
        try {
            Imgproc.warpPerspective(
                imageGray, rectified, homography, canonicalSize,
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0),
            )
            Imgproc.warpPerspective(
                sourceValid, valid, homography, canonicalSize,
                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0),
            )
            if (canonicalVisibleMat != null &&
                canonicalVisibleMat.size() == valid.size() &&
                canonicalVisibleMat.type() == CvType.CV_8UC1
            ) {
                Core.bitwise_and(valid, canonicalVisibleMat, valid)
            }
            excludePrintedTargets(valid)
            val radii = detectStepRadii(rectified, valid)
            if (radii.size < 2) return null
            val samples = sampleBoundaries(rectified, valid, radii, pose)
            if (samples.isEmpty()) return null
            val widths = samples.map(GrayStepFocusSample::edgeWidthCanonicalPx).sorted()
            val contrasts = samples.map(GrayStepFocusSample::measuredStepContrast).sorted()
            val boundaryContrasts = samples.groupBy(GrayStepFocusSample::boundaryRadiusMm)
                .values.map { boundarySamples ->
                    percentile(
                        boundarySamples.map(GrayStepFocusSample::measuredStepContrast).sorted(),
                        0.50,
                    )
                }
            val equalStepConsistency = equalStepConsistency(boundaryContrasts)
            val depths = samples.map(GrayStepFocusSample::cameraDepthMm)
            return GrayStepFocusAnalysis(
                detectedBoundaryCount = radii.size,
                samples = samples,
                medianEdgeWidthCanonicalPx = percentile(widths, 0.50),
                percentile90EdgeWidthCanonicalPx = percentile(widths, 0.90),
                nearSampleDepthMm = depths.minOrNull() ?: 0.0,
                farSampleDepthMm = depths.maxOrNull() ?: 0.0,
                medianStepContrast = percentile(contrasts, 0.50),
                equalStepConsistency = equalStepConsistency,
            )
        } finally {
            homography.release()
            rectified.release()
            sourceValid.release()
            valid.release()
        }
    }

    private fun detectStepRadii(rectified: Mat, valid: Mat): List<Double> {
        val inner = ceil(analysisInnerRadiusMm * pixelsPerMm).toInt()
        val outer = floor(analysisOuterRadiusMm * pixelsPerMm).toInt()
        val profile = DoubleArray(outer + 1) { Double.NaN }
        for (radius in inner..outer) {
            var sum = 0.0
            var count = 0
            for (degrees in 0 until 360 step 3) {
                val angle = degrees * PI / 180.0
                val x = center.x + radius * cos(angle)
                val y = center.y - radius * sin(angle)
                val value = bilinear(rectified, x, y) ?: continue
                if (!maskContains(valid, x, y)) continue
                sum += value
                count++
            }
            if (count >= 48) profile[radius] = sum / count
        }
        data class Peak(val radius: Int, val strength: Double)
        val peaks = mutableListOf<Peak>()
        for (radius in inner + 3..outer - 3) {
            val left = profile[radius - 2]
            val right = profile[radius + 2]
            if (!left.isFinite() || !right.isFinite()) continue
            val strength = abs(right - left) / 4.0
            val prior = if (profile[radius - 3].isFinite())
                abs(profile[radius + 1] - profile[radius - 3]) / 4.0 else 0.0
            val next = if (profile[radius + 3].isFinite())
                abs(profile[radius + 3] - profile[radius - 1]) / 4.0 else 0.0
            if (strength >= 1.0 && strength >= prior && strength >= next) {
                peaks += Peak(radius, strength)
            }
        }
        val retained = mutableListOf<Peak>()
        for (peak in peaks.sortedByDescending(Peak::strength)) {
            if (retained.none { abs(it.radius - peak.radius) < 8 }) retained += peak
            if (retained.size >= 10) break
        }
        if (retained.size < 2) return emptyList()
        val strongest = retained.maxOf(Peak::strength)
        return retained.filter { it.strength >= max(1.0, strongest * 0.18) }
            .map { it.radius.toDouble() }
            .sorted()
    }

    private fun sampleBoundaries(
        rectified: Mat,
        valid: Mat,
        radiiPx: List<Double>,
        pose: MatPoseResult,
    ): List<GrayStepFocusSample> {
        val rotationVector = Mat(3, 1, CvType.CV_64F)
        rotationVector.put(0, 0, *pose.rotationVector)
        val rotation = Mat()
        Geometry.Rodrigues(rotationVector, rotation)
        rotationVector.release()
        return try {
            val rzX = rotation[2, 0][0]
            val rzY = rotation[2, 1][0]
            val tz = pose.translationVector[2]
            buildList {
                for (radiusPx in radiiPx) {
                    for (degrees in 0 until 360 step 12) {
                        val angle = degrees * PI / 180.0
                        val profile = mutableListOf<Pair<Double, Double>>()
                        for (offset in -8..8) {
                            val radius = radiusPx + offset
                            val x = center.x + radius * cos(angle)
                            val y = center.y - radius * sin(angle)
                            if (!maskContains(valid, x, y)) continue
                            val value = bilinear(rectified, x, y) ?: continue
                            profile += offset.toDouble() to value
                        }
                        val edge = edgeSpread(profile) ?: continue
                        val xMm = radiusPx / pixelsPerMm * cos(angle)
                        val yMm = radiusPx / pixelsPerMm * sin(angle)
                        add(
                            GrayStepFocusSample(
                                boundaryRadiusMm = radiusPx / pixelsPerMm,
                                xMm = xMm,
                                yMm = yMm,
                                cameraDepthMm = rzX * xMm + rzY * yMm + tz,
                                edgeWidthCanonicalPx = edge.first,
                                measuredStepContrast = edge.second,
                            )
                        )
                    }
                }
            }
        } finally {
            rotation.release()
        }
    }

    /** Returns 10-90 width and endpoint contrast. */
    private fun edgeSpread(profile: List<Pair<Double, Double>>): Pair<Double, Double>? {
        if (profile.size < 13) return null
        val sorted = profile.sortedBy { it.first }
        val lowPlateau = sorted.take(3).map { it.second }.average()
        val highPlateau = sorted.takeLast(3).map { it.second }.average()
        val contrast = abs(highPlateau - lowPlateau)
        if (contrast < 4.0) return null
        val oriented = sorted.map { (position, value) ->
            val normalized = if (highPlateau >= lowPlateau) {
                (value - lowPlateau) / contrast
            } else {
                (lowPlateau - value) / contrast
            }
            position to normalized.coerceIn(-0.25, 1.25)
        }
        val ten = crossing(oriented, 0.10) ?: return null
        val ninety = crossing(oriented, 0.90) ?: return null
        val width = ninety - ten
        if (width <= 0.0 || width > 16.0) return null
        return width to contrast
    }

    private fun crossing(profile: List<Pair<Double, Double>>, level: Double): Double? {
        for (index in 1 until profile.size) {
            val (x0, y0) = profile[index - 1]
            val (x1, y1) = profile[index]
            if (y0 <= level && y1 >= level && abs(y1 - y0) > 1e-9) {
                return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
            }
        }
        return null
    }

    private fun excludePrintedTargets(mask: Mat) {
        for ((id, point) in MatModel.pointsMm) {
            val radiusMm = if (id in setOf(
                    FiducialId.A, FiducialId.B, FiducialId.C, FiducialId.D
                )) 10.0 else 6.0
            Imgproc.circle(
                mask,
                canonicalPoint(point.x, point.y),
                (radiusMm * pixelsPerMm).toInt(),
                Scalar(0.0),
                -1,
            )
        }
    }

    private fun canonicalPoint(xMm: Double, yMm: Double) = Point(
        (xMm - minimumXmm) * pixelsPerMm,
        (maximumYmm - yMm) * pixelsPerMm,
    )

    private fun maskContains(mask: Mat, x: Double, y: Double): Boolean {
        val column = x.toInt()
        val row = y.toInt()
        return row in 0 until mask.rows() && column in 0 until mask.cols() &&
            mask[row, column][0] != 0.0
    }

    private fun bilinear(image: Mat, x: Double, y: Double): Double? {
        val x0 = floor(x).toInt()
        val y0 = floor(y).toInt()
        val x1 = x0 + 1
        val y1 = y0 + 1
        if (x0 < 0 || y0 < 0 || x1 >= image.cols() || y1 >= image.rows()) return null
        val dx = x - x0
        val dy = y - y0
        val top = image[y0, x0][0] * (1.0 - dx) + image[y0, x1][0] * dx
        val bottom = image[y1, x0][0] * (1.0 - dx) + image[y1, x1][0] * dx
        return top * (1.0 - dy) + bottom * dy
    }

    private fun percentile(sorted: List<Double>, fraction: Double): Double {
        if (sorted.isEmpty()) return Double.POSITIVE_INFINITY
        val position = (sorted.size - 1) * fraction.coerceIn(0.0, 1.0)
        val lower = floor(position).toInt()
        val upper = ceil(position).toInt()
        if (lower == upper) return sorted[lower]
        val weight = position - lower
        return sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }

    /**
     * The artwork specifies equal gray increments. The absolute printed contrast
     * may vary with exposure, but the median local contrast of each concentric
     * boundary should agree. A robust median absolute deviation makes this test
     * insensitive to a few occluded or contaminated angular samples.
     */
    private fun equalStepConsistency(boundaryContrasts: List<Double>): Double {
        if (boundaryContrasts.size < 2) return 0.0
        val sorted = boundaryContrasts.sorted()
        val median = percentile(sorted, 0.50)
        if (median <= 1e-6) return 0.0
        val relativeMad = percentile(
            boundaryContrasts.map { abs(it - median) }.sorted(),
            0.50,
        ) / median
        return (1.0 - relativeMad / 0.50).coerceIn(0.0, 1.0)
    }
}

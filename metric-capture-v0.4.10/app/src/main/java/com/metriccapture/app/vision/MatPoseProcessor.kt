package com.metriccapture.app.vision

import android.graphics.Rect
import android.os.SystemClock
import android.util.Size
import android.util.Log
import com.metriccapture.app.geometry.Vec3
import org.opencv.geometry.Geometry
import org.opencv.core.CvException
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfPoint2f
import org.opencv.core.MatOfPoint3f
import org.opencv.core.Point
import org.opencv.core.Point3
import org.opencv.core.RotatedRect
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import org.opencv.video.Video
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

data class TargetObservation(
    val center: Point,
    val radiusPx: Double,
    val code: FiducialCode,
    val codeConfidence: Double,
    val saddleScore: Double,
)

data class FiducialObservation(
    val id: FiducialId,
    val pixel: Point,
    val code: FiducialCode,
    val codeConfidence: Double,
    val codeQuarterTurn: Int,
    val saddleScore: Double,
)

data class MatPoseResult(
    val timestampNs: Long,
    val cameraCenterMm: Vec3,
    val rotationVector: DoubleArray,
    val translationVector: DoubleArray,
    val reprojectionRmsePx: Double,
    val fiducials: List<FiducialObservation>,
    val cameraFx: Double,
    val cameraFy: Double,
    val cameraCx: Double,
    val cameraCy: Double,
)

/**
 * Keeps the measurement that belongs to the image separate from the temporally
 * filtered copy used by the live interface. Reconstruction must consume [raw].
 */
data class MatPoseEstimate(
    val raw: MatPoseResult,
    val filtered: MatPoseResult,
)

private data class TargetCandidate(
    val center: Point,
    val semiAxisX: Double,
    val semiAxisY: Double,
    val ellipseAngleDegrees: Double,
) {
    val radiusPx: Double get() = sqrt(semiAxisX * semiAxisY)
    val minimumAxisPx: Double get() = min(semiAxisX, semiAxisY)

    fun scaled(factor: Double) = TargetCandidate(
        center = Point(center.x * factor, center.y * factor),
        semiAxisX = semiAxisX * factor,
        semiAxisY = semiAxisY * factor,
        ellipseAngleDegrees = ellipseAngleDegrees,
    )

    /** Maps a normalized point on the printed circular target into its image ellipse. */
    fun imagePoint(angleDegrees: Double, radiusFraction: Double): Point {
        val local = angleDegrees * PI / 180.0
        val ellipse = ellipseAngleDegrees * PI / 180.0
        val x = cos(local) * semiAxisX * radiusFraction
        val y = sin(local) * semiAxisY * radiusFraction
        return Point(
            center.x + cos(ellipse) * x - sin(ellipse) * y,
            center.y + sin(ellipse) * x + cos(ellipse) * y,
        )
    }
}

class MatPoseProcessor(
    sensorIntrinsics: List<Float>,
    distortion: List<Float>?,
    activeArray: Rect,
    outputSize: Size,
    outputIntrinsics: DoubleArray? = null,
) : AutoCloseable {
    private val cameraMatrix = outputIntrinsics?.let(::directCameraMatrix)
        ?: scaledCameraMatrix(sensorIntrinsics, activeArray, outputSize)
    private val distortionCoefficients = MatOfDouble().apply {
        val values = distortion?.map(Float::toDouble)?.toDoubleArray() ?: DoubleArray(5)
        fromArray(*values)
    }
    private var previousGray: Mat? = null
    private var previousFiducials: List<FiducialObservation> = emptyList()
    private var framesSinceFullDetection = Int.MAX_VALUE
    private var consecutiveTrackingFailures = 0
    private var lastDetectionDiagnosticNs = Long.MIN_VALUE
    private val poseSmoother = PoseKalmanSmoother()

    @Synchronized
    fun process(frame: FramePlanes): MatPoseEstimate? {
        val gray = frame.grayMat()
        var result: MatPoseResult? = null
        try {
            if (framesSinceFullDetection < 20) {
                result = trackPrevious(frame.timestampNs, gray)
                    ?.takeIf { it.reprojectionRmsePx <= 1.75 }
            }
            if (result == null) {
                result = detectAndSolve(frame.timestampNs, gray, frame)
                framesSinceFullDetection = 0
            } else {
                framesSinceFullDetection++
            }

            if (result != null) {
                val raw = result
                val filtered = poseSmoother.filter(raw)
                previousFiducials = raw.fiducials
                consecutiveTrackingFailures = 0
                previousGray?.release()
                previousGray = gray.clone()
                return MatPoseEstimate(raw, filtered)
            } else {
                consecutiveTrackingFailures++
                if (consecutiveTrackingFailures >= 3) {
                    previousFiducials = emptyList()
                    poseSmoother.reset()
                }
            }
            previousGray?.release()
            previousGray = gray.clone()
            return null
        } finally {
            gray.release()
        }
    }

    /**
     * Solves the pose of an exact still-derived validation frame independently of
     * preview tracking and smoothing. This call may run after newer preview frames, so it deliberately
     * does not mutate the optical-flow or Kalman state.
     */
    @Synchronized
    fun processExposure(frame: FramePlanes): MatPoseResult? {
        val gray = frame.grayMat()
        return try {
            detectAndSolve(frame.timestampNs, gray, frame)
        } finally {
            gray.release()
        }
    }

    private fun detectAndSolve(timestamp: Long, gray: Mat, frame: FramePlanes): MatPoseResult? {
        val targets = detectTargets(gray, frame)
        if (targets.size < 8) {
            logDetectionDiagnostic(
                "target decode ${targets.size}/8; frame=${gray.cols()}x${gray.rows()}"
            )
            return null
        }
        val pose = candidateAssignments(targets)
            .mapNotNull { solve(timestamp, it) }
            .filter { it.cameraCenterMm.z > 0.0 }
            .minByOrNull(MatPoseResult::reprojectionRmsePx)
            ?.takeIf { it.reprojectionRmsePx <= 3.0 }
        logDetectionDiagnostic(
            if (pose == null) {
                "decoded ${targets.size} targets; no valid PnP solution"
            } else {
                "pose accepted: ${pose.fiducials.size} targets; " +
                    "RMSE=%.2f px".format(pose.reprojectionRmsePx)
            }
        )
        return pose
    }

    private fun logDetectionDiagnostic(message: String) {
        val now = SystemClock.elapsedRealtimeNanos()
        if (lastDetectionDiagnosticNs != Long.MIN_VALUE &&
            now - lastDetectionDiagnosticNs < DIAGNOSTIC_INTERVAL_NS
        ) return
        lastDetectionDiagnosticNs = now
        Log.d(TAG, message)
    }

    private fun trackPrevious(timestamp: Long, gray: Mat): MatPoseResult? {
        val priorGray = previousGray ?: return null
        if (previousFiducials.size < 4) return null
        val previousPoints = MatOfPoint2f(
            *previousFiducials.map(FiducialObservation::pixel).toTypedArray()
        )
        val predictedPoints = MatOfPoint2f()
        val status = MatOfByte()
        val errors = MatOfFloat()
        return try {
            Video.calcOpticalFlowPyrLK(
                priorGray,
                gray,
                previousPoints,
                predictedPoints,
                status,
                errors,
                org.opencv.core.Size(21.0, 21.0),
                3,
                org.opencv.core.TermCriteria(
                    org.opencv.core.TermCriteria.EPS or org.opencv.core.TermCriteria.COUNT,
                    20,
                    0.01,
                ),
                0,
                1e-4,
            )
            val tracked = predictedPoints.toArray()
            val accepted = previousFiducials.indices.mapNotNull { index ->
                val valid = status[index, 0]?.get(0)?.toInt() == 1
                val error = errors[index, 0]?.get(0) ?: Double.POSITIVE_INFINITY
                if (!valid || error > 18.0) return@mapNotNull null
                val point = tracked[index]
                if (point.x < 0.0 || point.x >= gray.cols().toDouble() ||
                    point.y < 0.0 || point.y >= gray.rows().toDouble()
                ) return@mapNotNull null
                val refined = refineSaddle(gray, point, 10.0)
                    ?: return@mapNotNull null
                val score = saddleScore(
                    gray,
                    TargetCandidate(refined, 10.0, 10.0, 0.0),
                )
                if (score < 0.30) return@mapNotNull null
                previousFiducials[index].copy(pixel = refined, saddleScore = score)
            }
            if (accepted.size < 6) return null
            solve(timestamp, accepted)
        } finally {
            listOf(previousPoints, predictedPoints, status, errors).forEach(Mat::release)
        }
    }

    private fun detectTargets(gray: Mat, frame: FramePlanes): List<TargetObservation> {
        val detectionScale = min(
            1.0,
            1280.0 / max(gray.cols(), gray.rows()).toDouble(),
        )
        val detectionGray = Mat()
        if (detectionScale < 1.0) {
            Imgproc.resize(
                gray,
                detectionGray,
                org.opencv.core.Size(),
                detectionScale,
                detectionScale,
                Imgproc.INTER_AREA,
            )
        } else {
            gray.copyTo(detectionGray)
        }
        val blurred = Mat()
        Imgproc.GaussianBlur(detectionGray, blurred, org.opencv.core.Size(5.0, 5.0), 1.2)
        val candidates = ellipseCandidates(blurred).toMutableList()
        val circles = Mat()
        Imgproc.HoughCircles(
            blurred,
            circles,
            Imgproc.HOUGH_GRADIENT,
            1.2,
            min(blurred.cols(), blurred.rows()) / 12.0,
            120.0,
            22.0,
            max(5, min(blurred.cols(), blurred.rows()) / 80),
            max(12, min(blurred.cols(), blurred.rows()) / 10),
        )
        for (column in 0 until circles.cols()) {
            val circle = circles.get(0, column) ?: continue
            val fallback = TargetCandidate(
                center = Point(circle[0], circle[1]),
                semiAxisX = circle[2],
                semiAxisY = circle[2],
                ellipseAngleDegrees = 0.0,
            )
            if (candidates.none { distance(it.center, fallback.center) < it.radiusPx * 0.55 }) {
                candidates += fallback
            }
        }
        val result = mutableListOf<TargetObservation>()
        val fullResolutionCandidates = candidates.map { it.scaled(1.0 / detectionScale) }
        for (candidate in fullResolutionCandidates) {
            val score = saddleScore(gray, candidate)
            if (score < 0.45) continue
            val refined = refineSaddle(gray, candidate.center, candidate.minimumAxisPx)
                ?: continue
            val refinedCandidate = candidate.copy(center = refined)
            val decoded = decodeQuadrantCode(frame, refinedCandidate) ?: continue
            result += TargetObservation(
                center = refined,
                radiusPx = candidate.radiusPx,
                code = decoded.code,
                codeConfidence = decoded.confidence,
                saddleScore = score,
            )
        }
        detectionGray.release()
        blurred.release()
        circles.release()
        return result.sortedByDescending(TargetObservation::saddleScore).take(12)
    }

    /**
     * Finds printed circles after projective foreshortening. Hough circles remain
     * only as a near-fronto-parallel fallback; ellipse contours drive oblique views.
     */
    private fun ellipseCandidates(gray: Mat): List<TargetCandidate> {
        val edges = Mat()
        val hierarchy = Mat()
        val contours = mutableListOf<org.opencv.core.MatOfPoint>()
        val minimumAxis = min(gray.cols(), gray.rows()) / 100.0
        val maximumAxis = min(gray.cols(), gray.rows()) / 7.0
        return try {
            Imgproc.Canny(gray, edges, 35.0, 110.0, 3, true)
            Imgproc.findContours(
                edges,
                contours,
                hierarchy,
                Imgproc.RETR_LIST,
                Imgproc.CHAIN_APPROX_NONE,
            )
            val ranked = contours.mapNotNull { contour ->
                val points = contour.toArray()
                if (points.size < 20) return@mapNotNull null
                val point2f = MatOfPoint2f(*points)
                val ellipse: RotatedRect = try {
                    Geometry.fitEllipse(point2f)
                } catch (_: Exception) {
                    point2f.release()
                    return@mapNotNull null
                }
                point2f.release()
                val semiX = ellipse.size.width / 2.0
                val semiY = ellipse.size.height / 2.0
                val small = min(semiX, semiY)
                val large = max(semiX, semiY)
                if (small < minimumAxis || large > maximumAxis || large / small > 3.8) {
                    return@mapNotNull null
                }
                if (ellipse.center.x < large ||
                    ellipse.center.x >= gray.cols().toDouble() - large ||
                    ellipse.center.y < large ||
                    ellipse.center.y >= gray.rows().toDouble() - large
                ) return@mapNotNull null
                val ellipseArea = PI * semiX * semiY
                val enclosedArea = abs(Geometry.contourArea(contour))
                val fillAgreement = (enclosedArea / ellipseArea).coerceAtMost(1.0)
                if (fillAgreement < 0.45) return@mapNotNull null
                TargetCandidate(
                    ellipse.center,
                    semiX,
                    semiY,
                    ellipse.angle,
                ) to fillAgreement
            }.sortedByDescending { it.second }

            val distinct = mutableListOf<TargetCandidate>()
            for ((candidate, _) in ranked) {
                if (distinct.none {
                        distance(it.center, candidate.center) <
                            min(it.radiusPx, candidate.radiusPx) * 0.60
                    }
                ) distinct += candidate
                if (distinct.size >= 16) break
            }
            distinct
        } finally {
            contours.forEach(Mat::release)
            edges.release(); hierarchy.release()
        }
    }

    private fun saddleScore(gray: Mat, candidate: TargetCandidate): Double {
        val patch = max(2, (candidate.minimumAxisPx * 0.20).toInt())
        var best = 0.0
        for (axisRotationDegrees in 0 until 90 step 5) {
            val means = listOf(-135.0, -45.0, 45.0, 135.0).map { quadrantAngle ->
                val sample = candidate.imagePoint(quadrantAngle + axisRotationDegrees, 0.35)
                meanPatch(
                    gray,
                    sample.x.toInt(),
                    sample.y.toInt(),
                    patch,
                )
            }
            // Every printed target has at least one repeated opposite pair;
            // the other pair may intentionally be black/white on color codes.
            val repeatedOppositeAgreement = 1.0 - min(
                1.0,
                min(abs(means[0] - means[2]), abs(means[1] - means[3])) / 90.0,
            )
            val adjacentContrast = min(
                1.0,
                (abs(means[0] - means[1]) + abs(means[1] - means[2]) +
                    abs(means[2] - means[3]) + abs(means[3] - means[0])) / 360.0,
            )
            best = max(best, repeatedOppositeAgreement * adjacentContrast)
        }
        return best
    }

    private fun refineSaddle(gray: Mat, initial: Point, radius: Double): Point? {
        if (!initial.x.isFinite() || !initial.y.isFinite()) return null
        val windowPixels = max(2, (radius * 0.25).toInt())
        if (!refinementWindowFits(
                gray.cols(), gray.rows(), initial.x, initial.y, windowPixels
            )
        ) return null
        val points = MatOfPoint2f(initial)
        return try {
            val window = windowPixels.toDouble()
            Imgproc.cornerSubPix(
                gray,
                points,
                org.opencv.core.Size(window, window),
                org.opencv.core.Size(-1.0, -1.0),
                org.opencv.core.TermCriteria(
                    org.opencv.core.TermCriteria.EPS or org.opencv.core.TermCriteria.MAX_ITER,
                    30,
                    0.01,
                ),
            )
            points.toArray().firstOrNull()?.takeIf { refined ->
                refined.x.isFinite() && refined.y.isFinite() &&
                    refined.x >= 0.0 && refined.x < gray.cols() &&
                    refined.y >= 0.0 && refined.y < gray.rows()
            }
        } catch (_: CvException) {
            null
        } finally {
            points.release()
        }
    }

    private data class Rgb(val red: Double, val green: Double, val blue: Double) {
        val luma: Double get() = 0.299 * red + 0.587 * green + 0.114 * blue
    }

    private data class DecodedCode(
        val code: FiducialCode,
        val confidence: Double,
    )

    /**
     * Decodes the four spatially distinct quadrants. The cross orientation is
     * searched over one quarter-turn. Cyclic matching is used here only to
     * reject non-code candidates; assignment later requires one quarter-turn
     * shared by the complete A-H layout.
     */
    private fun decodeQuadrantCode(
        frame: FramePlanes,
        candidate: TargetCandidate,
    ): DecodedCode? {
        var best: DecodedCode? = null
        for (axisRotationDegrees in 0 until 90 step 5) {
            val means = listOf(-135.0, -45.0, 45.0, 135.0).map { quadrantAngle ->
                meanQuadrantRgb(frame, candidate, quadrantAngle + axisRotationDegrees)
                    ?: return@map null
            }
            if (means.any { it == null }) continue
            val decoded = classifyQuadrants(means.filterNotNull()) ?: continue
            val templateMatch = MatModel.expectedCode.values.maxOf {
                cyclicCodeMatch(decoded.code, it)
            }
            val candidate = decoded.copy(confidence = decoded.confidence * templateMatch)
            if (best == null || candidate.confidence > best.confidence) best = candidate
        }
        return best?.takeIf { it.confidence >= 0.60 }
    }

    private fun candidateAssignments(targets: List<TargetObservation>): List<List<FiducialObservation>> {
        val center = Point(targets.map { it.center.x }.average(), targets.map { it.center.y }.average())
        val byRadius = targets.sortedBy { distance(it.center, center) }
        val inner = byRadius.take(4)
        val outer = byRadius.takeLast(4)

        // Ring position and printed target size are both part of the symbol system.
        // Keep this check deliberately loose because perspective changes local scale.
        if (outer.map { it.radiusPx }.average() < inner.map { it.radiusPx }.average() * 1.20) {
            return emptyList()
        }

        // All crosses were printed in one mat coordinate system. Try one shared
        // quarter-turn for the entire layout; do not rotate each code independently,
        // which would discard the black/white ordering that distinguishes the codes.
        return (0 until 4).flatMap { codeQuarterTurn ->
            candidateAssignmentsForQuarterTurn(outer, inner, codeQuarterTurn)
        }
    }

    private fun candidateAssignmentsForQuarterTurn(
        outer: List<TargetObservation>,
        inner: List<TargetObservation>,
        codeQuarterTurn: Int,
    ): List<List<FiducialObservation>> {
        val (a, b) = bestDistinctAnchors(
            outer, FiducialId.A, FiducialId.B, codeQuarterTurn
        )
            ?: return emptyList()
        val (g, h) = bestDistinctAnchors(
            inner, FiducialId.G, FiducialId.H, codeQuarterTurn
        )
            ?: return emptyList()
        val outerMono = outer.filter { it !== a && it !== b }
        val innerMono = inner.filter { it !== g && it !== h }
        if (outerMono.size != 2 || innerMono.size != 2) return emptyList()
        if (outerMono.any { codeScore(it, FiducialId.C, codeQuarterTurn) < 0.60 } ||
            innerMono.any { codeScore(it, FiducialId.E, codeQuarterTurn) < 0.60 }
        ) return emptyList()

        val assignments = mutableListOf<List<FiducialObservation>>()
        for (swapOuter in listOf(false, true)) for (swapInner in listOf(false, true)) {
            val c = outerMono[if (swapOuter) 1 else 0]
            val d = outerMono[if (swapOuter) 0 else 1]
            val e = innerMono[if (swapInner) 1 else 0]
            val f = innerMono[if (swapInner) 0 else 1]
            assignments += listOf(
                a.asFiducial(FiducialId.A, codeQuarterTurn),
                b.asFiducial(FiducialId.B, codeQuarterTurn),
                c.asFiducial(FiducialId.C, codeQuarterTurn),
                d.asFiducial(FiducialId.D, codeQuarterTurn),
                e.asFiducial(FiducialId.E, codeQuarterTurn),
                f.asFiducial(FiducialId.F, codeQuarterTurn),
                g.asFiducial(FiducialId.G, codeQuarterTurn),
                h.asFiducial(FiducialId.H, codeQuarterTurn),
            )
        }
        return assignments
    }

    private fun solve(timestamp: Long, observations: List<FiducialObservation>): MatPoseResult? {
        val objectPoints = MatOfPoint3f(*observations.map { MatModel.pointsMm.getValue(it.id) }.toTypedArray())
        val imagePoints = MatOfPoint2f(*observations.map(FiducialObservation::pixel).toTypedArray())
        val rvec = Mat()
        val tvec = Mat()
        val projected = MatOfPoint2f()
        val rotation = Mat()
        val rt = Mat()
        val center = Mat()
        val empty = Mat()
        return try {
            val solved = Geometry.solvePnP(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distortionCoefficients,
                rvec,
                tvec,
                false,
                Geometry.SOLVEPNP_IPPE,
            )
            if (!solved) return null
            Geometry.solvePnPRefineLM(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distortionCoefficients,
                rvec,
                tvec,
            )
            Geometry.projectPoints(
                objectPoints, rvec, tvec, cameraMatrix, distortionCoefficients, projected
            )
            val actual = imagePoints.toArray()
            val predicted = projected.toArray()
            val rmse = sqrt(actual.indices.sumOf { index ->
                val dx = actual[index].x - predicted[index].x
                val dy = actual[index].y - predicted[index].y
                dx * dx + dy * dy
            } / actual.size)
            Geometry.Rodrigues(rvec, rotation)
            org.opencv.core.Core.transpose(rotation, rt)
            org.opencv.core.Core.gemm(rt, tvec, -1.0, empty, 0.0, center)
            MatPoseResult(
                timestampNs = timestamp,
                cameraCenterMm = Vec3(center[0, 0][0], center[1, 0][0], center[2, 0][0]),
                rotationVector = DoubleArray(3) { rvec[it, 0][0] },
                translationVector = DoubleArray(3) { tvec[it, 0][0] },
                reprojectionRmsePx = rmse,
                fiducials = observations,
                cameraFx = cameraMatrix[0, 0][0],
                cameraFy = cameraMatrix[1, 1][0],
                cameraCx = cameraMatrix[0, 2][0],
                cameraCy = cameraMatrix[1, 2][0],
            )
        } finally {
            listOf(objectPoints, imagePoints, rvec, tvec, projected, rotation, rt, center, empty)
                .forEach(Mat::release)
        }
    }

    override fun close() {
        previousGray?.release()
        previousGray = null
        previousFiducials = emptyList()
        cameraMatrix.release()
        distortionCoefficients.release()
    }

    private fun meanPatch(gray: Mat, x: Int, y: Int, radius: Int): Double {
        val left = max(0, x - radius)
        val top = max(0, y - radius)
        val right = min(gray.cols(), x + radius + 1)
        val bottom = min(gray.rows(), y + radius + 1)
        if (left >= right || top >= bottom) return 127.0
        val roi = gray.submat(top, bottom, left, right)
        return org.opencv.core.Core.mean(roi).`val`[0].also { roi.release() }
    }

    private fun meanQuadrantRgb(
        frame: FramePlanes,
        candidate: TargetCandidate,
        angleDegrees: Double,
    ): Rgb? {
        var red = 0.0
        var green = 0.0
        var blue = 0.0
        var count = 0
        for (radiusFraction in listOf(0.30, 0.42, 0.54, 0.64)) {
            for (angleOffset in listOf(-20.0, -10.0, 0.0, 10.0, 20.0)) {
                val sample = candidate.imagePoint(angleDegrees + angleOffset, radiusFraction)
                val x = sample.x.toInt()
                val y = sample.y.toInt()
                if (x !in 0 until frame.width || y !in 0 until frame.height) continue
                val value = rgbAt(frame, x, y)
                red += value.red
                green += value.green
                blue += value.blue
                count++
            }
        }
        if (count < 8) return null
        return Rgb(red / count, green / count, blue / count)
    }

    private fun rgbAt(frame: FramePlanes, x: Int, y: Int): Rgb {
        val yy = sample(frame.y, frame.yRowStride, frame.yPixelStride, x, y)
        val chromaX = x / 2
        val chromaY = y / 2
        val uu = sample(
            frame.u, frame.uRowStride, frame.uPixelStride, chromaX, chromaY
        ) - 128
        val vv = sample(
            frame.v, frame.vRowStride, frame.vPixelStride, chromaX, chromaY
        ) - 128
        return Rgb(
            red = (yy + 1.402 * vv).coerceIn(0.0, 255.0),
            green = (yy - 0.344136 * uu - 0.714136 * vv).coerceIn(0.0, 255.0),
            blue = (yy + 1.772 * uu).coerceIn(0.0, 255.0),
        )
    }

    private fun classifyQuadrants(means: List<Rgb>): DecodedCode? {
        if (means.size != 4) return null
        val tones = MutableList(4) { QuadrantTone.UNKNOWN }
        means.forEachIndexed { index, rgb ->
            tones[index] = when {
                rgb.red - rgb.blue >= 32.0 && rgb.green - rgb.blue >= 24.0 &&
                    rgb.red >= 85.0 && rgb.green >= 75.0 -> QuadrantTone.YELLOW
                rgb.green - rgb.red >= 24.0 && rgb.blue - rgb.red >= 24.0 &&
                    rgb.green >= 80.0 && rgb.blue >= 80.0 -> QuadrantTone.CYAN
                else -> QuadrantTone.UNKNOWN
            }
        }
        val colored = tones.indices.filter { tones[it] != QuadrantTone.UNKNOWN }
        if (colored.size != 0 && colored.size != 2) return null
        if (colored.size == 2 && tones[colored[0]] != tones[colored[1]]) return null
        val achromatic = tones.indices.filter { tones[it] == QuadrantTone.UNKNOWN }
            .sortedBy { means[it].luma }
        val split = achromatic.size / 2
        achromatic.forEachIndexed { order, index ->
            tones[index] = if (order < split) QuadrantTone.BLACK else QuadrantTone.WHITE
        }
        val darkLuma = achromatic.take(split).map { means[it].luma }.average()
        val lightLuma = achromatic.drop(split).map { means[it].luma }.average()
        val luminanceConfidence = ((lightLuma - darkLuma) / 90.0).coerceIn(0.0, 1.0)
        if (luminanceConfidence < 0.28) return null
        val chromaConfidence = if (colored.isEmpty()) 1.0 else colored.map { index ->
            val rgb = means[index]
            when (tones[index]) {
                QuadrantTone.YELLOW -> min(rgb.red - rgb.blue, rgb.green - rgb.blue)
                QuadrantTone.CYAN -> min(rgb.green - rgb.red, rgb.blue - rgb.red)
                else -> 0.0
            }
        }.average().div(80.0).coerceIn(0.0, 1.0)
        return DecodedCode(
            code = FiducialCode(tones),
            confidence = (0.60 * luminanceConfidence + 0.40 * chromaConfidence)
                .coerceIn(0.0, 1.0),
        )
    }

    private fun bestDistinctAnchors(
        targets: List<TargetObservation>,
        firstId: FiducialId,
        secondId: FiducialId,
        codeQuarterTurn: Int,
    ): Pair<TargetObservation, TargetObservation>? {
        var best: Pair<TargetObservation, TargetObservation>? = null
        var bestScore = Double.NEGATIVE_INFINITY
        for (first in targets) for (second in targets) {
            if (first === second) continue
            val firstScore = codeScore(first, firstId, codeQuarterTurn)
            val secondScore = codeScore(second, secondId, codeQuarterTurn)
            if (firstScore < 0.60 || secondScore < 0.60) continue
            val score = firstScore + secondScore
            if (score > bestScore) {
                bestScore = score
                best = first to second
            }
        }
        return best
    }

    private fun codeScore(
        target: TargetObservation,
        id: FiducialId,
        codeQuarterTurn: Int,
    ): Double = target.codeConfidence * codeMatch(
        target.code,
        MatModel.expectedCode.getValue(id),
        codeQuarterTurn,
    )

    private fun sample(bytes: ByteArray, rowStride: Int, pixelStride: Int, x: Int, y: Int): Int {
        val index = y * rowStride + x * pixelStride
        return if (index in bytes.indices) bytes[index].toInt() and 0xff else 128
    }

    private fun distance(a: Point, b: Point): Double =
        sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y))

    private fun TargetObservation.asFiducial(id: FiducialId, codeQuarterTurn: Int) =
        FiducialObservation(
            id, center, code, codeConfidence, codeQuarterTurn, saddleScore
        )

    private fun scaledCameraMatrix(
        intrinsics: List<Float>,
        activeArray: Rect,
        output: Size,
    ): Mat {
        require(intrinsics.size >= 4)
        val sensorWidth = activeArray.width().toDouble()
        val sensorHeight = activeArray.height().toDouble()
        val scale = max(output.width / sensorWidth, output.height / sensorHeight)
        val cropWidth = output.width / scale
        val cropHeight = output.height / scale
        // Intrinsic calibration is local to the pre-correction active-array
        // coordinate system; activeArray.left/top belong to the full pixel array.
        val cropLeft = (sensorWidth - cropWidth) / 2.0
        val cropTop = (sensorHeight - cropHeight) / 2.0
        val fx = intrinsics[0] * scale
        val fy = intrinsics[1] * scale
        val cx = (intrinsics[2] - cropLeft) * scale
        val cy = (intrinsics[3] - cropTop) * scale
        return Mat.eye(3, 3, CvType.CV_64F).apply {
            put(0, 0, fx)
            put(1, 1, fy)
            put(0, 2, cx)
            put(1, 2, cy)
        }
    }

    private fun directCameraMatrix(intrinsics: DoubleArray): Mat {
        require(intrinsics.size >= 4)
        return Mat.eye(3, 3, CvType.CV_64F).apply {
            put(0, 0, intrinsics[0])
            put(1, 1, intrinsics[1])
            put(0, 2, intrinsics[2])
            put(1, 2, intrinsics[3])
        }
    }

    private companion object {
        const val TAG = "MetricVision"
        const val DIAGNOSTIC_INTERVAL_NS = 1_000_000_000L
    }
}

internal fun refinementWindowFits(
    imageWidth: Int,
    imageHeight: Int,
    x: Double,
    y: Double,
    windowPixels: Int,
): Boolean {
    if (!x.isFinite() || !y.isFinite() || windowPixels < 1) return false
    val margin = windowPixels + 1.0
    return x >= margin && x < imageWidth - margin &&
        y >= margin && y < imageHeight - margin
}

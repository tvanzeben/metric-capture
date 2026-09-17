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

data class TemplateCalibrationProgress(
    val acceptedFrames: Int,
    val requiredFrames: Int,
    val completedTemplate: Mat?,
)

/** Builds the canonical empty-mat radiometric reference from several registered views. */
class MatTemplateCalibrator(
    val pixelsPerMm: Double = 4.0,
    val minimumXmm: Double = -75.0,
    val maximumYmm: Double = 75.0,
    private val widthMm: Double = 150.0,
    private val heightMm: Double = 150.0,
    private val requiredFrames: Int = 8,
) : AutoCloseable {
    val canonicalSize = Size(widthMm * pixelsPerMm, heightMm * pixelsPerMm)
    private val sum = Mat.zeros(canonicalSize, CvType.CV_32FC1)
    private val count = Mat.zeros(canonicalSize, CvType.CV_32FC1)
    private var accepted = 0

    fun ingest(frame: FramePlanes, pose: MatPoseResult): TemplateCalibrationProgress {
        if (accepted >= requiredFrames) {
            return TemplateCalibrationProgress(accepted, requiredFrames, finish())
        }
        val imagePoints = MatOfPoint2f(*pose.fiducials.map(FiducialObservation::pixel).toTypedArray())
        val canonicalPoints = MatOfPoint2f(*pose.fiducials.map {
            val model = MatModel.pointsMm.getValue(it.id)
            canonicalPoint(model.x, model.y)
        }.toTypedArray())
        val homography = Geometry.findHomography(imagePoints, canonicalPoints, Geometry.RANSAC, 2.0)
        imagePoints.release(); canonicalPoints.release()
        if (homography.empty()) {
            homography.release()
            return TemplateCalibrationProgress(accepted, requiredFrames, null)
        }
        val gray = frame.grayMat()
        val rectified = Mat()
        val sourceValid = Mat.ones(gray.size(), CvType.CV_8UC1)
        val valid = Mat()
        val validFloat = Mat()
        try {
            Imgproc.warpPerspective(
                gray, rectified, homography, canonicalSize,
                Imgproc.INTER_LINEAR, Core.BORDER_CONSTANT, Scalar(0.0),
            )
            Imgproc.warpPerspective(
                sourceValid, valid, homography, canonicalSize,
                Imgproc.INTER_NEAREST, Core.BORDER_CONSTANT, Scalar(0.0),
            )
            valid.convertTo(validFloat, CvType.CV_32FC1, 1.0 / 255.0)
            Imgproc.accumulate(rectified, sum, valid)
            Core.add(count, validFloat, count)
            accepted++
        } finally {
            gray.release(); rectified.release(); sourceValid.release(); valid.release()
            validFloat.release(); homography.release()
        }
        return TemplateCalibrationProgress(
            accepted,
            requiredFrames,
            if (accepted >= requiredFrames) finish() else null,
        )
    }

    private fun finish(): Mat {
        val safeCount = count.clone()
        val emptyPixels = Mat()
        val mean = Mat()
        val output = Mat()
        Core.compare(safeCount, Scalar(0.0), emptyPixels, Core.CMP_EQ)
        safeCount.setTo(Scalar(1.0), emptyPixels)
        Core.divide(sum, safeCount, mean)
        mean.convertTo(output, CvType.CV_8UC1)
        safeCount.release(); emptyPixels.release(); mean.release()
        return output
    }

    private fun canonicalPoint(xMm: Double, yMm: Double) = Point(
        (xMm - minimumXmm) * pixelsPerMm,
        (maximumYmm - yMm) * pixelsPerMm,
    )

    override fun close() {
        sum.release()
        count.release()
    }
}

package com.metriccapture.app.guidance

import com.metriccapture.app.geometry.ArtifactRelativePose
import com.metriccapture.app.geometry.angularDistanceDegrees
import com.metriccapture.app.geometry.wrapDegrees
import com.metriccapture.app.sensors.UserRelativeSensorPose

enum class AcquisitionMode { CAMERA_ORBIT, TURNTABLE }

enum class CaptureRound { UPPER, LOWER }

/** Planned capture band. Targets guide coverage; measured poses are stored separately. */
data class RoundDefinition(
    val round: CaptureRound,
    val targetPositionAngleDegrees: Double,
    val targetTiltDegrees: Double,
    val targetRadiusMm: Double,
    val startAzimuthDegrees: Double,
    val azimuthStepDegrees: Double = 20.0,
) {
    init {
        require(azimuthStepDegrees > 0.0 && azimuthStepDegrees <= 90.0)
        require(targetRadiusMm > 0.0)
        val exactCount = 360.0 / azimuthStepDegrees
        require(kotlin.math.abs(exactCount - exactCount.toInt()) < 1e-9) {
            "The azimuth step must divide 360 degrees exactly"
        }
    }

    val targetCount: Int get() = (360.0 / azimuthStepDegrees).toInt()

    fun targetAzimuth(index: Int): Double =
        wrapDegrees(startAzimuthDegrees + index * azimuthStepDegrees)
}

data class GuidanceTolerances(
    val azimuthDegrees: Double = 4.0,
    val positionAngleDegrees: Double = 6.0,
    val tiltDegrees: Double = 7.0,
    val rollDegrees: Double = 6.0,
    val minimumStableFrames: Int = 4,
)

/**
 * Evidence used only to decide whether the next requested view is ready.
 * [guidancePose] and [userSensorPose] are baseline-relative UI quantities.
 * The unmodified metric pose remains in VisionFrameAnalysis and capture metadata.
 */
data class LiveFrameEvidence(
    val guidancePose: ArtifactRelativePose,
    val userSensorPose: UserRelativeSensorPose,
    val fiducialReprojectionRmsePx: Double,
    val predictedFiducialRmsePx: Double,
    val overlapFraction: Double,
    val fullMatVisible: Boolean,
    val minimumMatFrameMarginPx: Double,
    val sharpnessScore: Double,
    val depthOfFieldScore: Double,
    val radiometricScore: Double,
    val highlightClippedFraction: Double,
    val artifactCenterConfidence: Double,
)

enum class GateFailure {
    WRONG_AZIMUTH,
    WRONG_POSITION_ANGLE,
    WRONG_TILT,
    WRONG_ROLL,
    POSE_UNCERTAIN,
    PREDICTION_MISMATCH,
    OVERLAP_TOO_LOW,
    MAT_CLIPPED,
    IMAGE_BLURRED,
    DEPTH_OF_FIELD_INVALID,
    RADIOMETRY_INVALID,
    HIGHLIGHTS_CLIPPED,
    ARTIFACT_CENTER_UNCERTAIN,
    NOT_TEMPORALLY_STABLE,
}

data class GateThresholds(
    val maximumPoseRmsePx: Double = 1.5,
    val maximumPredictionRmsePx: Double = 10.0,
    val minimumOverlapFraction: Double = 0.60,
    val minimumMatFrameMarginPx: Double = 12.0,
    val minimumSharpnessScore: Double,
    val minimumDepthOfFieldScore: Double = 0.80,
    val minimumRadiometricScore: Double = 0.75,
    val maximumHighlightClippedFraction: Double = 0.001,
    val minimumArtifactCenterConfidence: Double = 0.75,
)

data class TargetView(
    val round: CaptureRound,
    val index: Int,
    /** Person-relative/turntable-relative coverage angle. */
    val azimuthDegrees: Double,
    val positionAngleDegrees: Double,
    val tiltDegrees: Double,
    val radiusMm: Double,
)

/** Actual values paired with an accepted image; planned target values never replace these. */
data class AcceptedFrameMeasurement(
    val matAzimuthDegrees: Double,
    val matElevationDegrees: Double,
    val matRadiusMm: Double,
    val guidanceAzimuthDegrees: Double,
    val guidancePositionAngleDegrees: Double,
    val guidanceTiltDegrees: Double,
    val guidanceRollDegrees: Double,
    val minimumMatFrameMarginPx: Double,
)

sealed interface CaptureDecision {
    data class Accept(
        val target: TargetView,
        val measurement: AcceptedFrameMeasurement,
    ) : CaptureDecision

    data class Reject(
        val target: TargetView,
        val failures: Set<GateFailure>,
        val correction: GuidanceCorrection,
    ) : CaptureDecision
}

data class GuidanceCorrection(
    val azimuthDeltaDegrees: Double,
    val positionAngleDeltaDegrees: Double,
    val tiltDeltaDegrees: Double,
    val rollDeltaDegrees: Double,
    val additionalMatMarginPx: Double,
)

class NextFrameGate(
    private val tolerances: GuidanceTolerances,
    private val thresholds: GateThresholds,
) {
    private var consecutivePassingFrames = 0

    fun evaluate(
        target: TargetView,
        frame: LiveFrameEvidence,
        measuredMatPose: ArtifactRelativePose,
    ): CaptureDecision {
        val failures = linkedSetOf<GateFailure>()
        val azimuthError = angularDistanceDegrees(
            frame.guidancePose.azimuthDegrees,
            target.azimuthDegrees,
        )
        if (azimuthError > tolerances.azimuthDegrees) failures += GateFailure.WRONG_AZIMUTH
        if (kotlin.math.abs(
                frame.guidancePose.elevationDegrees - target.positionAngleDegrees
            ) > tolerances.positionAngleDegrees
        ) failures += GateFailure.WRONG_POSITION_ANGLE
        if (kotlin.math.abs(
                frame.userSensorPose.tiltDeltaDegrees - target.tiltDegrees
            ) > tolerances.tiltDegrees
        ) failures += GateFailure.WRONG_TILT
        if (kotlin.math.abs(frame.userSensorPose.rollDeltaDegrees) > tolerances.rollDegrees) {
            failures += GateFailure.WRONG_ROLL
        }
        if (frame.fiducialReprojectionRmsePx > thresholds.maximumPoseRmsePx) {
            failures += GateFailure.POSE_UNCERTAIN
        }
        if (frame.predictedFiducialRmsePx > thresholds.maximumPredictionRmsePx) {
            failures += GateFailure.PREDICTION_MISMATCH
        }
        if (frame.overlapFraction < thresholds.minimumOverlapFraction) {
            failures += GateFailure.OVERLAP_TOO_LOW
        }
        if (!frame.fullMatVisible ||
            frame.minimumMatFrameMarginPx < thresholds.minimumMatFrameMarginPx
        ) failures += GateFailure.MAT_CLIPPED
        if (frame.sharpnessScore < thresholds.minimumSharpnessScore) {
            failures += GateFailure.IMAGE_BLURRED
        }
        if (frame.depthOfFieldScore < thresholds.minimumDepthOfFieldScore) {
            failures += GateFailure.DEPTH_OF_FIELD_INVALID
        }
        if (frame.radiometricScore < thresholds.minimumRadiometricScore) {
            failures += GateFailure.RADIOMETRY_INVALID
        }
        if (frame.highlightClippedFraction > thresholds.maximumHighlightClippedFraction) {
            failures += GateFailure.HIGHLIGHTS_CLIPPED
        }
        if (frame.artifactCenterConfidence < thresholds.minimumArtifactCenterConfidence) {
            failures += GateFailure.ARTIFACT_CENTER_UNCERTAIN
        }

        if (failures.isEmpty()) consecutivePassingFrames++ else consecutivePassingFrames = 0
        if (failures.isEmpty() && consecutivePassingFrames < tolerances.minimumStableFrames) {
            failures += GateFailure.NOT_TEMPORALLY_STABLE
        }

        val correction = GuidanceCorrection(
            azimuthDeltaDegrees = signedAngularDelta(
                frame.guidancePose.azimuthDegrees,
                target.azimuthDegrees,
            ),
            positionAngleDeltaDegrees =
                target.positionAngleDegrees - frame.guidancePose.elevationDegrees,
            tiltDeltaDegrees = target.tiltDegrees - frame.userSensorPose.tiltDeltaDegrees,
            rollDeltaDegrees = -frame.userSensorPose.rollDeltaDegrees,
            additionalMatMarginPx =
                (thresholds.minimumMatFrameMarginPx - frame.minimumMatFrameMarginPx)
                    .coerceAtLeast(0.0),
        )

        return if (failures.isEmpty()) {
            consecutivePassingFrames = 0
            CaptureDecision.Accept(
                target,
                AcceptedFrameMeasurement(
                    matAzimuthDegrees = measuredMatPose.azimuthDegrees,
                    matElevationDegrees = measuredMatPose.elevationDegrees,
                    matRadiusMm = measuredMatPose.radiusMm,
                    guidanceAzimuthDegrees = frame.guidancePose.azimuthDegrees,
                    guidancePositionAngleDegrees = frame.guidancePose.elevationDegrees,
                    guidanceTiltDegrees = frame.userSensorPose.tiltDeltaDegrees,
                    guidanceRollDegrees = frame.userSensorPose.rollDeltaDegrees,
                    minimumMatFrameMarginPx = frame.minimumMatFrameMarginPx,
                ),
            )
        } else {
            CaptureDecision.Reject(target, failures, correction)
        }
    }

    fun resetStability() {
        consecutivePassingFrames = 0
    }
}

internal fun signedAngularDelta(from: Double, to: Double): Double =
    ((to - from + 540.0) % 360.0) - 180.0

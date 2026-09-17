package com.metriccapture.app.guidance

import com.metriccapture.app.geometry.ArtifactRelativePose
import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.geometry.artifactRelativePose
import com.metriccapture.app.geometry.wrapDegrees
import com.metriccapture.app.reconstruction.Aabb
import com.metriccapture.app.sensors.UserRelativeSensorPose
import com.metriccapture.app.vision.MatPoseResult
import com.metriccapture.app.vision.VisionFrameAnalysis
import kotlin.math.sqrt

data class GuidedCaptureStatus(
    val state: OrbitState,
    val mode: AcquisitionMode,
    val target: TargetView?,
    val decision: CaptureDecision?,
    val capturedFrames: Int,
    val totalFrames: Int,
    val captureInFlight: Boolean,
)

/** Turns live metric evidence and the persistent user-relative sensor frame into capture actions. */
class GuidedCaptureCoordinator(
    private val triggerCapture: () -> Boolean,
    private val userSensorPose: () -> UserRelativeSensorPose?,
    private val statusListener: (GuidedCaptureStatus) -> Unit,
    private val distortion: DoubleArray = DoubleArray(5),
    private val upperPositionAngleDegrees: Double = 55.0,
    private val upperTiltDegrees: Double = 45.0,
    private val lowerPositionAngleDegrees: Double = 25.0,
    private val lowerTiltDegrees: Double = 25.0,
    private val azimuthStepDegrees: Double = 20.0,
    private val lowerStartAzimuthDegrees: Double = 10.0,
) {
    private val orbit = OrbitController()
    private val gate = NextFrameGate(
        GuidanceTolerances(minimumStableFrames = 4),
        GateThresholds(minimumSharpnessScore = 80.0),
    )
    private var artifactCenter: Vec3? = null
    private var guidanceArtifactCenter: Vec3? = null
    private var candidateEnvelope: Aabb? = null
    private var guidanceSceneBaseline: ArtifactRelativePose? = null
    private var captureInFlight = false
    private var capturedFrames = 0
    private var lastDecision: CaptureDecision? = null
    private var targetBeingCaptured: TargetView? = null
    private var measurementBeingCaptured: AcceptedFrameMeasurement? = null
    private var requestedStart = false
    private var restoredCaptureCount = 0

    var acquisitionMode: AcquisitionMode = AcquisitionMode.CAMERA_ORBIT
        private set

    /** Called once after the user is at printed 0°, vertically centered, and the full mat is visible. */
    fun startAtUserGuidanceZero(mode: AcquisitionMode, committedCaptureCount: Int = 0) {
        require(committedCaptureCount in 0..36)
        acquisitionMode = mode
        restoredCaptureCount = committedCaptureCount
        capturedFrames = committedCaptureCount
        requestedStart = true
    }

    fun onAnalysis(analysis: VisionFrameAnalysis) {
        if (guidanceArtifactCenter == null) {
            val measuredCenter = analysis.artifactCenterMm
            if (measuredCenter != null) {
                artifactCenter = artifactCenter?.let { prior ->
                    Vec3(
                        prior.x * 0.85 + measuredCenter.x * 0.15,
                        prior.y * 0.85 + measuredCenter.y * 0.15,
                        prior.z * 0.85 + measuredCenter.z * 0.15,
                    )
                } ?: measuredCenter
            }
        }
        analysis.candidateEnvelope?.let { candidateEnvelope = it }
        if (guidanceArtifactCenter == null) {
            candidateEnvelope?.center?.let { hullCenter ->
                artifactCenter = artifactCenter?.copy(z = hullCenter.z)
            }
        }

        val center = guidanceArtifactCenter ?: artifactCenter
        val measuredPose = center?.let { artifactRelativePose(analysis.pose.cameraCenterMm, it) }
        if (requestedStart && center != null && measuredPose != null &&
            userSensorPose() != null && orbit.state is OrbitState.AwaitingUserGuidanceZero
        ) {
            guidanceArtifactCenter = center
            guidanceSceneBaseline = measuredPose
            val workingRadius = measuredPose.radiusMm
            orbit.start(
                listOf(
                    RoundDefinition(
                        CaptureRound.UPPER,
                        upperPositionAngleDegrees,
                        upperTiltDegrees,
                        workingRadius,
                        startAzimuthDegrees = 0.0,
                        azimuthStepDegrees = azimuthStepDegrees,
                    ),
                    RoundDefinition(
                        CaptureRound.LOWER,
                        lowerPositionAngleDegrees,
                        lowerTiltDegrees,
                        workingRadius,
                        startAzimuthDegrees = lowerStartAzimuthDegrees,
                        azimuthStepDegrees = azimuthStepDegrees,
                    ),
                ),
                completedTargets = restoredCaptureCount,
            )
        }

        val target = orbit.currentTarget()
        val baseline = guidanceSceneBaseline
        val sensorPose = userSensorPose()
        if (target == null || center == null || measuredPose == null || baseline == null ||
            sensorPose == null || captureInFlight
        ) {
            publish(target)
            return
        }
        if (candidateEnvelope == null) candidateEnvelope = envelopeFromCenter(center)

        val guidancePose = ArtifactRelativePose(
            azimuthDegrees = wrapDegrees(
                signedAngularDelta(baseline.azimuthDegrees, measuredPose.azimuthDegrees)
            ),
            elevationDegrees = measuredPose.elevationDegrees - baseline.elevationDegrees,
            radiusMm = measuredPose.radiusMm,
        )
        val modelTarget = target.copy(
            azimuthDegrees = wrapDegrees(baseline.azimuthDegrees + target.azimuthDegrees),
            positionAngleDegrees = baseline.elevationDegrees + target.positionAngleDegrees,
            radiusMm = measuredPose.radiusMm,
        )
        val predictor = NextViewPredictor(
            analysis.pose.cameraFx,
            analysis.pose.cameraFy,
            analysis.pose.cameraCx,
            analysis.pose.cameraCy,
            distortion,
        )
        val predicted = predictor.predict(modelTarget, center, candidateEnvelope)
        val evidence = LiveFrameEvidence(
            guidancePose = guidancePose,
            userSensorPose = sensorPose,
            fiducialReprojectionRmsePx = analysis.pose.reprojectionRmsePx,
            predictedFiducialRmsePx = predictedRmse(analysis.pose, predicted),
            overlapFraction = estimatedOverlap(guidancePose, target),
            fullMatVisible = analysis.fullMatVisible,
            minimumMatFrameMarginPx = analysis.minimumMatFrameMarginPx,
            sharpnessScore = analysis.sharpnessVariance,
            depthOfFieldScore = analysis.depthOfFieldScore,
            radiometricScore = analysis.radiometricScore,
            highlightClippedFraction = analysis.highlightClippedFraction,
            artifactCenterConfidence = analysis.artifactCenterConfidence,
        )
        lastDecision = gate.evaluate(target, evidence, measuredPose)
        val accepted = lastDecision as? CaptureDecision.Accept
        if (accepted != null) {
            if (triggerCapture()) {
                captureInFlight = true
                targetBeingCaptured = target
                measurementBeingCaptured = accepted.measurement
            } else {
                gate.resetStability()
                lastDecision = null
            }
        }
        publish(target)
    }

    fun onCaptureSaved() {
        if (!captureInFlight) return
        captureInFlight = false
        capturedFrames++
        orbit.acceptCurrentTarget()
        gate.resetStability()
        if (orbit.state is OrbitState.RoundReview) orbit.continueAfterRoundReview()
        lastDecision = null
        targetBeingCaptured = null
        measurementBeingCaptured = null
        publish(orbit.currentTarget())
    }

    fun onCaptureFailed() {
        captureInFlight = false
        targetBeingCaptured = null
        measurementBeingCaptured = null
        gate.resetStability()
        publish(orbit.currentTarget())
    }

    val isComplete: Boolean get() = orbit.state is OrbitState.Complete
    val envelope: Aabb? get() = candidateEnvelope
    val pendingCaptureTarget: TargetView? get() = targetBeingCaptured
    val pendingCaptureMeasurement: AcceptedFrameMeasurement? get() = measurementBeingCaptured

    private fun publish(target: TargetView?) {
        statusListener(
            GuidedCaptureStatus(
                orbit.state,
                acquisitionMode,
                target,
                lastDecision,
                capturedFrames,
                orbit.totalTargets.takeIf { it > 0 } ?: 36,
                captureInFlight,
            )
        )
    }

    private fun predictedRmse(pose: MatPoseResult, predicted: PredictedView): Double {
        val errors = pose.fiducials.mapNotNull { observed ->
            val expected = predicted.fiducialPixels[observed.id] ?: return@mapNotNull null
            if (!expected.x.isFinite() || !expected.y.isFinite()) return@mapNotNull null
            val dx = observed.pixel.x - expected.x
            val dy = observed.pixel.y - expected.y
            dx * dx + dy * dy
        }
        return if (errors.isEmpty()) Double.POSITIVE_INFINITY else sqrt(errors.average())
    }

    private fun estimatedOverlap(pose: ArtifactRelativePose, target: TargetView): Double {
        val azimuthError = com.metriccapture.app.geometry.angularDistanceDegrees(
            pose.azimuthDegrees,
            target.azimuthDegrees,
        )
        val elevationError = kotlin.math.abs(
            pose.elevationDegrees - target.positionAngleDegrees
        )
        return (1.0 - azimuthError / 90.0 - elevationError / 90.0).coerceIn(0.0, 1.0)
    }

    private fun envelopeFromCenter(center: Vec3) = Aabb(
        Vec3(center.x - 55.0, center.y - 55.0, 0.0),
        Vec3(center.x + 55.0, center.y + 55.0, 100.0),
    )
}

package com.metriccapture.app.guidance

import com.metriccapture.app.geometry.ArtifactRelativePose
import com.metriccapture.app.sensors.UserRelativeSensorPose
import org.junit.Assert.assertTrue
import org.junit.Test

class NextFrameGateTest {
    private val target = TargetView(CaptureRound.UPPER, 0, 0.0, 55.0, 45.0, 300.0)
    private val measured = ArtifactRelativePose(18.0, 54.5, 335.0)
    private val goodFrame = LiveFrameEvidence(
        guidancePose = ArtifactRelativePose(0.5, 55.5, 335.0),
        userSensorPose = UserRelativeSensorPose(0.5, 45.5, 0.5),
        fiducialReprojectionRmsePx = 0.5,
        predictedFiducialRmsePx = 0.8,
        overlapFraction = 0.8,
        fullMatVisible = true,
        minimumMatFrameMarginPx = 20.0,
        sharpnessScore = 150.0,
        depthOfFieldScore = 1.0,
        radiometricScore = 0.95,
        highlightClippedFraction = 0.0,
        artifactCenterConfidence = 0.95,
    )

    @Test
    fun `capture waits for temporal stability and keeps measured pose`() {
        val gate = NextFrameGate(
            GuidanceTolerances(minimumStableFrames = 3),
            GateThresholds(minimumSharpnessScore = 100.0),
        )

        assertTrue(gate.evaluate(target, goodFrame, measured) is CaptureDecision.Reject)
        assertTrue(gate.evaluate(target, goodFrame, measured) is CaptureDecision.Reject)
        val accepted = gate.evaluate(target, goodFrame, measured)
        assertTrue(accepted is CaptureDecision.Accept)
        assertTrue((accepted as CaptureDecision.Accept).measurement.matAzimuthDegrees == 18.0)
    }

    @Test
    fun `clipped mat prevents capture`() {
        val gate = NextFrameGate(
            GuidanceTolerances(minimumStableFrames = 1),
            GateThresholds(minimumSharpnessScore = 100.0),
        )
        val result = gate.evaluate(
            target,
            goodFrame.copy(fullMatVisible = false, minimumMatFrameMarginPx = -3.0),
            measured,
        )

        assertTrue(result is CaptureDecision.Reject)
        assertTrue((result as CaptureDecision.Reject).failures.contains(GateFailure.MAT_CLIPPED))
    }

    @Test
    fun `blur prevents capture`() {
        val gate = NextFrameGate(
            GuidanceTolerances(minimumStableFrames = 1),
            GateThresholds(minimumSharpnessScore = 100.0),
        )
        val result = gate.evaluate(target, goodFrame.copy(sharpnessScore = 20.0), measured)

        assertTrue(result is CaptureDecision.Reject)
        assertTrue((result as CaptureDecision.Reject).failures.contains(GateFailure.IMAGE_BLURRED))
    }

    @Test
    fun `stepped gray depth failure prevents capture`() {
        val gate = NextFrameGate(
            GuidanceTolerances(minimumStableFrames = 1),
            GateThresholds(minimumSharpnessScore = 100.0),
        )
        val result = gate.evaluate(target, goodFrame.copy(depthOfFieldScore = 0.4), measured)

        assertTrue(result is CaptureDecision.Reject)
        assertTrue(
            (result as CaptureDecision.Reject).failures
                .contains(GateFailure.DEPTH_OF_FIELD_INVALID)
        )
    }
}

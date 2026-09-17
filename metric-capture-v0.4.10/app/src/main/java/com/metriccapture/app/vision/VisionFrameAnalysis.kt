package com.metriccapture.app.vision

import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.reconstruction.Aabb

data class VisionFrameAnalysis(
    val timestampNs: Long,
    val pose: MatPoseResult,
    val artifactCenterMm: Vec3?,
    val artifactVerticalCenterPx: Double?,
    val principalPointVerticalPx: Double,
    val displayFrameHeightPx: Int,
    val foregroundFraction: Double,
    val visibleMatFraction: Double,
    val fullMatVisible: Boolean,
    val minimumMatFrameMarginPx: Double,
    val sharpnessVariance: Double,
    val steppedGrayFocus: GrayStepFocusAnalysis?,
    val depthOfFieldScore: Double,
    val radiometricScore: Double,
    val highlightClippedFraction: Double,
    val artifactCenterConfidence: Double,
    val templateReady: Boolean,
    val candidateEnvelope: Aabb?,
)

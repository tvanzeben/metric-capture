package com.metriccapture.app.reconstruction

import java.io.File

data class ReconstructionFrame(
    val frameId: String,
    val workingImage: File,
    val imageWidth: Int,
    val imageHeight: Int,
    val foregroundMask: File?,
    /** Photogrammetric camera/view-set identity: 0 upper, 1 lower. */
    val cameraIndex: Int,
    val targetIndex: Int,
    /** Capture schedule angle; used only to plan the match graph. */
    val scheduledAzimuthDegrees: Double,
    /** Measured pose angle; retained for diagnostics, never used to discover sparse edges. */
    val azimuthDegrees: Double,
    val elevationDegrees: Double,
    val cameraFx: Double,
    val cameraFy: Double,
    val cameraCx: Double,
    val cameraCy: Double,
    val distortion: DoubleArray,
    val rotationVector: DoubleArray,
    val translationVector: DoubleArray,
)

data class MetricPoint(
    val xMm: Double,
    val yMm: Double,
    val zMm: Double,
    val red: Int,
    val green: Int,
    val blue: Int,
    val confidence: Double,
)

data class SparseMatchEdgeResult(
    val firstFrameId: String,
    val secondFrameId: String,
    val phase: MatchPhase,
    val crossCamera: Boolean,
    val rawMatchCount: Int,
    val ratioMatchCount: Int,
    val geometricInlierCount: Int,
)

data class SparseReconstructionResult(
    val points: List<MetricPoint>,
    val matchedEdges: List<SparseMatchEdgeResult>,
    val featureTrackCount: Int,
    val crossCameraEdgeCount: Int,
    val crossCameraTrackCount: Int,
    val refinedFrames: List<ReconstructionFrame>,
)

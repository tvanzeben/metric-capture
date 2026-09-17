package com.metriccapture.app.camera

import android.util.Size
import android.graphics.Rect
import android.util.Range

data class CaptureProfile(
    val cameraId: String,
    /** Supported PRIVATE output used only for the continuous display surface. */
    val previewSize: Size,
    /** Efficient processed stream used only for prospective live guidance. */
    val yuvSize: Size,
    /** Maximum native RAW_SENSOR stream used as the preserved reconstruction master. */
    val rawSize: Size,
    val usesMaximumResolutionPixelMode: Boolean,
    val sensorOrientationDegrees: Int,
    val focalLengthsMm: List<Float>,
    val availableApertures: List<Float>,
    val minimumFocusDistanceDiopters: Float,
    val sensitivityRange: Range<Int>,
    val exposureTimeRangeNs: Range<Long>,
    val maximumFrameDurationNs: Long,
    val sensorTimestampUsesRealtimeClock: Boolean,
    val intrinsicCalibration: List<Float>?,
    val distortion: List<Float>?,
    val rawIntrinsicCalibration: List<Float>?,
    val rawDistortion: List<Float>?,
    /** Default-mode array used by the live YUV coordinate system. */
    val activeArray: Rect,
    /** Array used by the selected RAW pixel mode. */
    val rawActiveArray: Rect,
    /** Full pixel-array coordinate system used by the selected RAW pixel mode. */
    val rawPixelArraySize: Size,
    val supportsManualPostProcessing: Boolean,
    val supportsOisOff: Boolean,
    val supportsEdgeOff: Boolean,
    val supportsNoiseReductionOff: Boolean,
    val supportsTonemapGamma: Boolean,
    val processingControls: ProcessingControlSupport,
)

data class ProcessingControlSupport(
    val videoStabilizationOff: Boolean,
    val hotPixelOff: Boolean,
    val shadingOff: Boolean,
    val aberrationOff: Boolean,
    val distortionOff: Boolean,
    val blackLevelLock: Boolean,
    val neutralPostRawSensitivity: Boolean,
    val rotateAndCropNone: Boolean,
)

enum class QualificationFailure {
    NO_REAR_CAMERA,
    PREVIEW_OUTPUT_UNAVAILABLE,
    MANUAL_SENSOR_UNAVAILABLE,
    MANUAL_WHITE_BALANCE_UNAVAILABLE,
    FOCUS_LOCK_UNAVAILABLE,
    YUV_OUTPUT_UNAVAILABLE,
    RAW_OUTPUT_UNAVAILABLE,
    UNSUPPORTED_RAW_COLOR_FILTER,
    RAW_LENS_SHADING_IS_APPLIED,
    RAW_HOT_PIXEL_CORRECTION_CANNOT_BE_DISABLED,
    OIS_CANNOT_BE_DISABLED,
    CAMERA_ACCESS_FAILED,
    INTRINSICS_UNAVAILABLE,
    RAW_GEOMETRY_UNRESOLVED,
    SENSOR_TIMESTAMP_NOT_REALTIME,
}

enum class CaptureLimitation {
    MANUAL_POST_PROCESSING_UNAVAILABLE,
    OIS_CANNOT_BE_EXPLICITLY_DISABLED,
    EDGE_OFF_UNAVAILABLE,
    NOISE_REDUCTION_OFF_UNAVAILABLE,
    INTRINSICS_NOT_REPORTED,
    DISTORTION_NOT_REPORTED,
    LINEAR_TONEMAP_UNAVAILABLE,
    VIDEO_STABILIZATION_OFF_UNAVAILABLE,
    HOT_PIXEL_OFF_UNAVAILABLE,
    SHADING_OFF_UNAVAILABLE,
    ABERRATION_OFF_UNAVAILABLE,
    DISTORTION_OFF_UNAVAILABLE,
    BLACK_LEVEL_LOCK_UNAVAILABLE,
    POST_RAW_SENSITIVITY_NEUTRAL_UNAVAILABLE,
    ROTATE_AND_CROP_NONE_UNAVAILABLE,
}

sealed interface DeviceQualification {
    data class Supported(val profile: CaptureProfile) : DeviceQualification
    data class Degraded(
        val profile: CaptureProfile,
        val limitations: Set<CaptureLimitation>,
    ) : DeviceQualification

    data class Unsupported(
        val reasons: Set<QualificationFailure>,
    ) : DeviceQualification
}

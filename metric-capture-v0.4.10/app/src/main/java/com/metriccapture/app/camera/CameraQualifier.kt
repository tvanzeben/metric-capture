package com.metriccapture.app.camera

import android.content.Context
import android.graphics.ImageFormat
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.params.StreamConfigurationMap
import android.util.Size
import android.util.Range

class CameraQualifier(context: Context) {
    private val manager = context.getSystemService(CameraManager::class.java)

    fun qualify(): DeviceQualification {
        val candidates = try {
            manager.cameraIdList.mapNotNull(::inspectCamera)
        } catch (_: CameraAccessException) {
            return DeviceQualification.Unsupported(setOf(QualificationFailure.CAMERA_ACCESS_FAILED))
        }

        if (candidates.isEmpty()) {
            return DeviceQualification.Unsupported(setOf(QualificationFailure.NO_REAR_CAMERA))
        }

        val viable = candidates.filter { it.failures.isEmpty() }
        if (viable.isEmpty()) {
            return DeviceQualification.Unsupported(
                candidates.flatMapTo(linkedSetOf()) { it.failures }
            )
        }

        val selected = viable.maxWithOrNull(
            compareBy<InspectedCamera> { it.profile.rawSize.area }
                .thenBy { -it.limitations.size }
                .thenBy { it.profile.yuvSize.area }
        ) ?: return DeviceQualification.Unsupported(setOf(QualificationFailure.NO_REAR_CAMERA))

        return if (selected.limitations.isEmpty()) {
            DeviceQualification.Supported(selected.profile)
        } else {
            DeviceQualification.Degraded(selected.profile, selected.limitations)
        }
    }

    private fun inspectCamera(cameraId: String): InspectedCamera? {
        val c = manager.getCameraCharacteristics(cameraId)
        if (c[CameraCharacteristics.LENS_FACING] != CameraMetadata.LENS_FACING_BACK) return null

        val capabilities = c[CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES]
            ?.toSet().orEmpty()
        val streamMap = c[CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP]
        val failures = linkedSetOf<QualificationFailure>()

        if (CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR !in capabilities) {
            failures += QualificationFailure.MANUAL_SENSOR_UNAVAILABLE
        }

        val minimumFocusDistance = c[CameraCharacteristics.LENS_INFO_MINIMUM_FOCUS_DISTANCE] ?: 0f
        if (minimumFocusDistance <= 0f) failures += QualificationFailure.FOCUS_LOCK_UNAVAILABLE

        val yuvSizes = outputSizes(streamMap, ImageFormat.YUV_420_888)
        val previewSizes = outputSizes(streamMap, SurfaceTexture::class.java)
        val defaultRawSizes = outputSizes(streamMap, ImageFormat.RAW_SENSOR) +
            highResolutionOutputSizes(streamMap, ImageFormat.RAW_SENSOR)
        val maximumResolutionMap = c[
            CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP_MAXIMUM_RESOLUTION
        ]
        val maximumRawSizes = outputSizes(maximumResolutionMap, ImageFormat.RAW_SENSOR) +
            highResolutionOutputSizes(maximumResolutionMap, ImageFormat.RAW_SENSOR)
        if (yuvSizes.isEmpty()) failures += QualificationFailure.YUV_OUTPUT_UNAVAILABLE
        if (previewSizes.isEmpty()) failures += QualificationFailure.PREVIEW_OUTPUT_UNAVAILABLE
        val rawCapability = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_RAW in capabilities
        val defaultRaw = defaultRawSizes.maxByOrNull { it.area }
        val maximumRaw = maximumRawSizes.maxByOrNull { it.area }
        val usesMaximumResolutionPixelMode = maximumRaw != null &&
            maximumRaw.area > (defaultRaw?.area ?: 0L)
        val rawSize = if (usesMaximumResolutionPixelMode) maximumRaw else defaultRaw
        if (!rawCapability || rawSize == null) {
            failures += QualificationFailure.RAW_OUTPUT_UNAVAILABLE
        }
        val aspectReference = rawSize ?: yuvSizes.maxByOrNull { it.area } ?: Size(4, 3)
        val yuvSize = chooseTrackingSize(yuvSizes, aspectReference) ?: Size(1, 1)
        val previewSize = chooseTrackingSize(previewSizes, yuvSize) ?: Size(1, 1)

        val limitations = linkedSetOf<CaptureLimitation>()
        val manualPost = CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_POST_PROCESSING in capabilities
        if (!manualPost) {
            limitations += CaptureLimitation.MANUAL_POST_PROCESSING_UNAVAILABLE
        }

        val oisOff = CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF in
            (c[CameraCharacteristics.LENS_INFO_AVAILABLE_OPTICAL_STABILIZATION] ?: intArrayOf())
        if (!oisOff) {
            limitations += CaptureLimitation.OIS_CANNOT_BE_EXPLICITLY_DISABLED
            failures += QualificationFailure.OIS_CANNOT_BE_DISABLED
        }

        val edgeOff = CameraMetadata.EDGE_MODE_OFF in
            (c[CameraCharacteristics.EDGE_AVAILABLE_EDGE_MODES] ?: intArrayOf())
        if (!edgeOff) limitations += CaptureLimitation.EDGE_OFF_UNAVAILABLE

        val noiseOff = CameraMetadata.NOISE_REDUCTION_MODE_OFF in
            (c[CameraCharacteristics.NOISE_REDUCTION_AVAILABLE_NOISE_REDUCTION_MODES] ?: intArrayOf())
        if (!noiseOff) limitations += CaptureLimitation.NOISE_REDUCTION_OFF_UNAVAILABLE

        val tonemapGamma = CameraMetadata.TONEMAP_MODE_GAMMA_VALUE in
            (c[CameraCharacteristics.TONEMAP_AVAILABLE_TONE_MAP_MODES] ?: intArrayOf())
        if (!tonemapGamma) limitations += CaptureLimitation.LINEAR_TONEMAP_UNAVAILABLE

        val availableRequestKeys = c.availableCaptureRequestKeys.toSet()
        val processingControls = ProcessingControlSupport(
            videoStabilizationOff = CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF in
                (c[CameraCharacteristics.CONTROL_AVAILABLE_VIDEO_STABILIZATION_MODES] ?: intArrayOf()),
            hotPixelOff = CameraMetadata.HOT_PIXEL_MODE_OFF in
                (c[CameraCharacteristics.HOT_PIXEL_AVAILABLE_HOT_PIXEL_MODES] ?: intArrayOf()),
            shadingOff = CameraMetadata.SHADING_MODE_OFF in
                (c[CameraCharacteristics.SHADING_AVAILABLE_MODES] ?: intArrayOf()),
            aberrationOff = CameraMetadata.COLOR_CORRECTION_ABERRATION_MODE_OFF in
                (c[CameraCharacteristics.COLOR_CORRECTION_AVAILABLE_ABERRATION_MODES] ?: intArrayOf()),
            distortionOff = CameraMetadata.DISTORTION_CORRECTION_MODE_OFF in
                (c[CameraCharacteristics.DISTORTION_CORRECTION_AVAILABLE_MODES] ?: intArrayOf()),
            blackLevelLock = CaptureRequest.BLACK_LEVEL_LOCK in availableRequestKeys,
            neutralPostRawSensitivity = c[
                CameraCharacteristics.CONTROL_POST_RAW_SENSITIVITY_BOOST_RANGE
            ]?.contains(100) == true,
            rotateAndCropNone = CaptureRequest.SCALER_ROTATE_AND_CROP in availableRequestKeys,
        )
        if (!processingControls.videoStabilizationOff) {
            limitations += CaptureLimitation.VIDEO_STABILIZATION_OFF_UNAVAILABLE
        }
        if (!processingControls.hotPixelOff) {
            limitations += CaptureLimitation.HOT_PIXEL_OFF_UNAVAILABLE
            failures += QualificationFailure.RAW_HOT_PIXEL_CORRECTION_CANNOT_BE_DISABLED
        }
        if (!processingControls.shadingOff) limitations += CaptureLimitation.SHADING_OFF_UNAVAILABLE
        if (!processingControls.aberrationOff) limitations += CaptureLimitation.ABERRATION_OFF_UNAVAILABLE
        if (!processingControls.distortionOff) limitations += CaptureLimitation.DISTORTION_OFF_UNAVAILABLE
        if (!processingControls.blackLevelLock) limitations += CaptureLimitation.BLACK_LEVEL_LOCK_UNAVAILABLE
        if (!processingControls.neutralPostRawSensitivity) {
            limitations += CaptureLimitation.POST_RAW_SENSITIVITY_NEUTRAL_UNAVAILABLE
        }
        if (!processingControls.rotateAndCropNone) {
            limitations += CaptureLimitation.ROTATE_AND_CROP_NONE_UNAVAILABLE
        }
        val cfa = c[CameraCharacteristics.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT]
        if (cfa !in setOf(
                CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_RGGB,
                CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GRBG,
                CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_GBRG,
                CameraMetadata.SENSOR_INFO_COLOR_FILTER_ARRANGEMENT_BGGR,
            )
        ) failures += QualificationFailure.UNSUPPORTED_RAW_COLOR_FILTER
        if (c[CameraCharacteristics.SENSOR_INFO_LENS_SHADING_APPLIED] == true) {
            failures += QualificationFailure.RAW_LENS_SHADING_IS_APPLIED
        }

        val activeArray = c[CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE]
            ?: c[CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE]
            ?: android.graphics.Rect(0, 0, aspectReference.width, aspectReference.height)
        val rawActiveArray = if (usesMaximumResolutionPixelMode) {
            c[CameraCharacteristics.SENSOR_INFO_PRE_CORRECTION_ACTIVE_ARRAY_SIZE_MAXIMUM_RESOLUTION]
                ?: c[CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE_MAXIMUM_RESOLUTION]
                ?: activeArray
        } else activeArray
        val rawPixelArraySize = if (usesMaximumResolutionPixelMode) {
            c[CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE_MAXIMUM_RESOLUTION]
                ?: Size(rawActiveArray.right, rawActiveArray.bottom)
        } else {
            c[CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE]
                ?: Size(rawActiveArray.right, rawActiveArray.bottom)
        }
        val focalLengths = c[CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS]
            ?.toList().orEmpty()
        val apertures = c[CameraCharacteristics.LENS_INFO_AVAILABLE_APERTURES]
            ?.toList().orEmpty()
        val sensitivityRange = c[CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE]
            ?: Range(100, 800)
        val exposureRange = c[CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE]
            ?: Range(1_000_000L, 33_000_000L)
        val maximumFrameDuration = c[CameraCharacteristics.SENSOR_INFO_MAX_FRAME_DURATION]
            ?: 33_333_333L
        val timestampUsesRealtime = c[CameraCharacteristics.SENSOR_INFO_TIMESTAMP_SOURCE] ==
            CameraMetadata.SENSOR_INFO_TIMESTAMP_SOURCE_REALTIME
        if (!timestampUsesRealtime) failures += QualificationFailure.SENSOR_TIMESTAMP_NOT_REALTIME
        val physicalSize = c[CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE]
        val intrinsics = c[CameraCharacteristics.LENS_INTRINSIC_CALIBRATION]?.toList()
            ?: if (focalLengths.isNotEmpty() && physicalSize != null) {
                val focal = focalLengths.first()
                listOf(
                    focal / physicalSize.width * activeArray.width(),
                    focal / physicalSize.height * activeArray.height(),
                    activeArray.width() / 2.0f,
                    activeArray.height() / 2.0f,
                    0.0f,
                )
            } else null
        if (intrinsics == null) {
            limitations += CaptureLimitation.INTRINSICS_NOT_REPORTED
            failures += QualificationFailure.INTRINSICS_UNAVAILABLE
        } else if (c[CameraCharacteristics.LENS_INTRINSIC_CALIBRATION] == null) {
            limitations += CaptureLimitation.INTRINSICS_NOT_REPORTED
        }

        val distortion = c[CameraCharacteristics.LENS_DISTORTION]?.toList()?.let { android ->
            if (android.size >= 5) {
                // Android: k1,k2,k3,t1,t2. OpenCV: k1,k2,p1,p2,k3.
                listOf(android[0], android[1], android[3], android[4], android[2])
            } else null
        }
        if (distortion == null) limitations += CaptureLimitation.DISTORTION_NOT_REPORTED
        val rawIntrinsics = if (usesMaximumResolutionPixelMode) {
            c[CameraCharacteristics.LENS_INTRINSIC_CALIBRATION_MAXIMUM_RESOLUTION]?.toList()
                ?: intrinsics?.let {
                    val scaleX = rawActiveArray.width().toFloat() / activeArray.width()
                    val scaleY = rawActiveArray.height().toFloat() / activeArray.height()
                    listOf(
                        it[0] * scaleX,
                        it[1] * scaleY,
                        it[2] * scaleX,
                        it[3] * scaleY,
                        it.getOrElse(4) { 0f } * scaleX,
                    )
                }
        } else intrinsics
        val rawDistortion = (if (usesMaximumResolutionPixelMode) {
            c[CameraCharacteristics.LENS_DISTORTION_MAXIMUM_RESOLUTION]
        } else c[CameraCharacteristics.LENS_DISTORTION])?.toList()?.let { android ->
            if (android.size >= 5) {
                listOf(android[0], android[1], android[3], android[4], android[2])
            } else null
        } ?: distortion
        if (rawSize != null && rawIntrinsics != null) {
            runCatching {
                resolveRawImageGeometry(
                    rawSize,
                    rawPixelArraySize,
                    rawActiveArray,
                    rawIntrinsics,
                )
            }.onFailure {
                failures += QualificationFailure.RAW_GEOMETRY_UNRESOLVED
            }
        }

        return InspectedCamera(
            profile = CaptureProfile(
                cameraId = cameraId,
                previewSize = previewSize,
                yuvSize = yuvSize,
                rawSize = rawSize ?: Size(1, 1),
                usesMaximumResolutionPixelMode = usesMaximumResolutionPixelMode,
                sensorOrientationDegrees = c[CameraCharacteristics.SENSOR_ORIENTATION] ?: 0,
                focalLengthsMm = focalLengths,
                availableApertures = apertures,
                minimumFocusDistanceDiopters = minimumFocusDistance,
                sensitivityRange = sensitivityRange,
                exposureTimeRangeNs = exposureRange,
                maximumFrameDurationNs = maximumFrameDuration,
                sensorTimestampUsesRealtimeClock = timestampUsesRealtime,
                intrinsicCalibration = intrinsics,
                distortion = distortion,
                rawIntrinsicCalibration = rawIntrinsics,
                rawDistortion = rawDistortion,
                activeArray = activeArray,
                rawActiveArray = rawActiveArray,
                rawPixelArraySize = rawPixelArraySize,
                supportsManualPostProcessing = manualPost,
                supportsOisOff = oisOff,
                supportsEdgeOff = edgeOff,
                supportsNoiseReductionOff = noiseOff,
                supportsTonemapGamma = tonemapGamma,
                processingControls = processingControls,
            ),
            failures = failures,
            limitations = limitations,
        )
    }

    private fun outputSizes(map: StreamConfigurationMap?, format: Int): List<Size> =
        map?.getOutputSizes(format)?.toList().orEmpty()

    private fun outputSizes(map: StreamConfigurationMap?, outputClass: Class<*>): List<Size> =
        map?.getOutputSizes(outputClass)?.toList().orEmpty()

    private fun highResolutionOutputSizes(
        map: StreamConfigurationMap?,
        format: Int,
    ): List<Size> = runCatching {
        map?.getHighResolutionOutputSizes(format)?.toList().orEmpty()
    }.getOrDefault(emptyList())

    private fun chooseTrackingSize(sizes: List<Size>, rawSize: Size): Size? {
        val targetAspect = rawSize.width.toDouble() / rawSize.height
        val usable = sizes.filter { it.width >= 640 && it.height >= 480 }
        val aspectMatched = usable.filter {
            kotlin.math.abs(it.width.toDouble() / it.height - targetAspect) <= 0.02
        }
        val candidates = aspectMatched.ifEmpty { usable }
        // This stream performs guidance and validation only. Reconstruction uses
        // the independently selected maximum-resolution RAW_SENSOR stream.
        val preferredPixels = 1280L * 960L
        return candidates.minWithOrNull(
            compareBy<Size> {
                kotlin.math.abs(it.width.toDouble() / it.height - targetAspect)
            }.thenBy { kotlin.math.abs(it.area - preferredPixels) }
        ) ?: sizes.maxByOrNull { it.area }
    }

    private data class InspectedCamera(
        val profile: CaptureProfile,
        val failures: Set<QualificationFailure>,
        val limitations: Set<CaptureLimitation>,
    )
}

private val Size.area: Long get() = width.toLong() * height.toLong()

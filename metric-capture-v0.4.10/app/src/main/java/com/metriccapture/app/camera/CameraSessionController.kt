package com.metriccapture.app.camera

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.util.Log
import android.content.pm.PackageManager
import android.graphics.ImageFormat
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureFailure
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.hardware.camera2.CaptureResult
import android.hardware.camera2.params.ColorSpaceTransform
import android.hardware.camera2.params.OutputConfiguration
import android.hardware.camera2.params.RggbChannelVector
import android.hardware.camera2.params.SessionConfiguration
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.view.Surface
import androidx.core.content.ContextCompat
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.core.CvType
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.TreeMap
import java.util.concurrent.atomic.AtomicLong
import java.util.concurrent.atomic.AtomicBoolean
import com.metriccapture.app.vision.FramePlanes
import com.metriccapture.app.vision.MatPoseProcessor
import com.metriccapture.app.vision.MatPoseResult
import com.metriccapture.app.vision.MatTemplateCalibrator
import com.metriccapture.app.vision.NegativeSpaceExtractor
import com.metriccapture.app.vision.TemplateCalibrationProgress
import com.metriccapture.app.vision.VisionFrameAnalysis
import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.reconstruction.Aabb
import com.metriccapture.app.reconstruction.IncrementalGeometryEngine
import com.metriccapture.app.sensors.SynchronizedPhoneSensorSample
import com.metriccapture.app.vision.MatVisibilityEvaluator
import com.metriccapture.app.vision.DepthOfFieldCalibration
import com.metriccapture.app.vision.FocusCandidateMeasurement
import com.metriccapture.app.vision.GrayStepFocusAnalysis
import com.metriccapture.app.vision.SessionFocusSelector
import com.metriccapture.app.vision.SteppedGrayDepthOfFieldEvaluator
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln

/**
 * Owns a Camera2 session with efficient YUV guidance and triggered maximum-resolution RAW.
 * YUV terminates at guidance. Every accepted still measurement is derived from
 * the exact RAW exposure after its plane has been preserved verbatim.
 * All camera objects and timestamp-pairing maps are confined to [cameraHandler].
 */
class CameraSessionController(
    private val context: Context,
    private val profile: CaptureProfile,
    private val previewSurface: Surface,
    private val outputDirectory: File,
    private val phoneSensorSampleForTimestamp: (Long) -> SynchronizedPhoneSensorSample?,
    private val listener: Listener,
) : AutoCloseable {

    private enum class CapturePurpose { ARTIFACT, EMPTY_MAT, FOCUS_CALIBRATION }

    interface Listener {
        fun onSettingsSuggested(suggested: ManualSettings, apertureIsFixed: Boolean)
        fun onFocusCalibrationProgress(completedCandidates: Int, totalCandidates: Int)
        fun onSessionReady(applied: ManualSettings)
        fun onYuvFrame(frame: YuvFrameSummary)
        fun onMatPose(pose: MatPoseResult)
        fun onTemplateCalibration(progress: TemplateCalibrationProgress)
        fun onTemplateRestored(file: File)
        fun onVisionAnalysis(analysis: VisionFrameAnalysis)
        fun onCaptureSaved(
            rawMasterFile: File,
            workingImageFile: File,
            foregroundMaskFile: File?,
            metadataFile: File,
        )
        fun onError(message: String, cause: Throwable? = null)
    }

    data class ManualSettings(
        val sensitivityIso: Int,
        val exposureTimeNs: Long,
        val frameDurationNs: Long,
        val focusDistanceDiopters: Float,
        val aperture: Float?,
        val whiteBalanceGains: RggbChannelVector,
        val whiteBalanceTransform: ColorSpaceTransform,
        val depthOfFieldCalibration: DepthOfFieldCalibration? = null,
    )

    data class YuvFrameSummary(
        val sequence: Long,
        val sensorTimestampNs: Long,
        val width: Int,
        val height: Int,
        val meanLuma: Double,
        val clippedDarkFraction: Double,
        val clippedBrightFraction: Double,
    )

    private data class ExactCaptureAnalysis(
        val pose: MatPoseResult,
        val foregroundMask: Mat,
        val steppedGrayFocus: GrayStepFocusAnalysis,
    )

    private data class FocusCalibrationState(
        val meteredSettings: ManualSettings,
        val candidates: List<ManualSettings>,
        val measurements: MutableList<FocusCandidateMeasurement> = mutableListOf(),
        var candidateIndex: Int = 0,
        var settlingFrameCount: Int = 0,
        var rawSubmitted: Boolean = false,
    ) {
        val current: ManualSettings get() = candidates[candidateIndex]
    }

    private data class EffectiveSensorSettings(
        val sensitivityIso: Int,
        val exposureTimeNs: Long,
        val frameDurationNs: Long,
        val focusDistanceDiopters: Float,
        val aperture: Float?,
    )

    private val cameraManager = context.getSystemService(CameraManager::class.java)
    private val characteristics = cameraManager.getCameraCharacteristics(profile.cameraId)
    private val cameraThread = HandlerThread("metric-camera").apply { start() }
    private val cameraHandler = Handler(cameraThread.looper)
    private val visionThread = HandlerThread("metric-vision").apply { start() }
    private val visionHandler = Handler(visionThread.looper)
    private val visionBusy = AtomicBoolean(false)
    private val matPoseProcessor = profile.intrinsicCalibration?.let {
        MatPoseProcessor(
            sensorIntrinsics = it,
            distortion = profile.distortion,
            activeArray = profile.activeArray,
            outputSize = profile.yuvSize,
        )
    }
    private val rawImageGeometry = resolveRawImageGeometry(
        rawSize = profile.rawSize,
        pixelArraySize = profile.rawPixelArraySize,
        preCorrectionActiveArray = profile.rawActiveArray,
        preCorrectionIntrinsics = profile.rawIntrinsicCalibration
            ?: error("RAW intrinsic calibration is unavailable"),
    )
    private val rawValidationSize = boundedRawValidationSize(profile.rawSize)
    private val rawMatPoseProcessor = profile.rawIntrinsicCalibration?.let {
        MatPoseProcessor(
            sensorIntrinsics = it,
            distortion = profile.rawDistortion,
            activeArray = profile.rawActiveArray,
            outputSize = rawValidationSize,
            outputIntrinsics = rawImageGeometry.scaledTo(rawValidationSize),
        )
    }
    private val matVisibilityEvaluator = MatVisibilityEvaluator(
        distortion = profile.distortion?.map(Float::toDouble)?.toDoubleArray() ?: DoubleArray(5)
    )
    private val rawMatVisibilityEvaluator = MatVisibilityEvaluator(
        distortion = profile.rawDistortion?.map(Float::toDouble)?.toDoubleArray()
            ?: DoubleArray(5)
    )
    private val rawFrameProcessor = RawFrameProcessor(characteristics, rawValidationSize)
    private val steppedGrayFocusEvaluator = SteppedGrayDepthOfFieldEvaluator()
    private val yuvReader = ImageReader.newInstance(
        profile.yuvSize.width,
        profile.yuvSize.height,
        ImageFormat.YUV_420_888,
        3,
    )
    private val rawReader = ImageReader.newInstance(
        profile.rawSize.width,
        profile.rawSize.height,
        ImageFormat.RAW_SENSOR,
        2,
    )
    private val yuvSurface = yuvReader.surface
    private val rawSurface = rawReader.surface
    private val frameSequence = AtomicLong(0)
    private var lastVisionTimestampNs = Long.MIN_VALUE
    private val pendingRawImages = TreeMap<Long, Image>()
    private val pendingCaptureResults = TreeMap<Long, TotalCaptureResult>()
    private val pendingCapturePurposes = TreeMap<Long, CapturePurpose>()
    private val pendingFocusCalibrationSettings = TreeMap<Long, ManualSettings>()
    private val rawProcessingScheduled = mutableSetOf<Long>()
    private val discardedCaptureTimestamps = mutableSetOf<Long>()
    private val stillCaptureInFlight = AtomicBoolean(false)
    private val closed = AtomicBoolean(false)
    private var emptyMatAttemptsRemaining = 0

    @Volatile private var templateCalibrator: MatTemplateCalibrator? = null
    @Volatile private var incrementalGeometryEngine: IncrementalGeometryEngine? = null
    private var canonicalTemplate: Mat? = null
    private var negativeSpaceFrameCount = 0

    private var cameraDevice: CameraDevice? = null
    private var captureSession: CameraCaptureSession? = null
    private var manualSettings: ManualSettings? = null
    private var meteringFrameCount = 0
    private var settingsSuggestionSent = false
    private var settingsReviewRequested = false
    private var effectiveSensorReference: EffectiveSensorSettings? = null
    @Volatile private var focusCalibrationState: FocusCalibrationState? = null

    @Volatile
    var isReady: Boolean = false
        private set

    init {
        yuvReader.setOnImageAvailableListener(::onYuvAvailable, cameraHandler)
        rawReader.setOnImageAvailableListener(::onRawAvailable, cameraHandler)
    }

    @SuppressLint("MissingPermission")
    fun start() {
        if (closed.get()) return
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) !=
            PackageManager.PERMISSION_GRANTED
        ) {
            listener.onError("Camera permission has not been granted.")
            return
        }
        try {
            cameraManager.openCamera(profile.cameraId, cameraStateCallback, cameraHandler)
        } catch (exception: CameraAccessException) {
            listener.onError("Camera could not be opened.", exception)
        }
    }

    fun captureStill(): Boolean {
        if (closed.get()) return false
        if (!stillCaptureInFlight.compareAndSet(false, true)) return false
        cameraHandler.post { submitRawCapture(CapturePurpose.ARTIFACT) }
        return true
    }

    /** Submits one RAW-only exposure after [stillCaptureInFlight] has been claimed. */
    private fun submitRawCapture(purpose: CapturePurpose) {
        if (closed.get()) {
            stillCaptureInFlight.set(false)
            return
        }
        val device = cameraDevice
        val session = captureSession
        val settings = if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
            focusCalibrationState?.current
        } else {
            manualSettings
        }
        val allowed = purpose == CapturePurpose.FOCUS_CALIBRATION || isReady
        if (device == null || session == null || settings == null || !allowed) {
            stillCaptureInFlight.set(false)
            listener.onError("The camera session is not ready.")
            return
        }
        try {
            session.stopRepeating()
            val request = device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                addTarget(rawSurface)
                applyManualControls(this, settings, stillRequest = true)
                setTag(purpose)
            }.build()
            session.capture(request, stillCaptureCallback, cameraHandler)
        } catch (exception: Exception) {
            stillCaptureInFlight.set(false)
            if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
                abortFocusCalibration("RAW focus-probe request failed.", exception)
            } else {
                resumeLockedPreview()
                listener.onError("RAW capture request failed.", exception)
            }
        }
    }

    /** Applies the accepted/overridden settings and begins the locked radiometric session. */
    fun lockManualSettings(settings: ManualSettings) {
        if (closed.get()) return
        cameraHandler.post {
            if (closed.get()) return@post
            val camera = cameraDevice ?: return@post listener.onError("Camera is not open.")
            val session = captureSession ?: return@post listener.onError("Camera session is not ready.")
            val validated = validateManualSettings(settings).let { candidate ->
                val calibration = candidate.depthOfFieldCalibration?.takeIf {
                    abs(it.focusDistanceDiopters - candidate.focusDistanceDiopters) <= 0.0005f
                }
                candidate.copy(depthOfFieldCalibration = calibration)
            }
            if (validated.depthOfFieldCalibration == null) {
                listener.onError(
                    "Manual focus cannot be locked without passing the RAW stepped-gray calibration."
                )
                return@post
            }
            try {
                val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                    addLiveTargets(this)
                    applyManualControls(this, validated, stillRequest = false)
                }.build()
                session.setRepeatingRequest(request, previewCaptureCallback, cameraHandler)
                manualSettings = validated
                effectiveSensorReference = null
                isReady = true
                listener.onSessionReady(validated)
            } catch (exception: Exception) {
                listener.onError("Manual camera settings could not be locked.", exception)
            }
        }
    }

    fun refreshSettingsSuggestion() {
        if (closed.get()) return
        cameraHandler.post {
            if (closed.get()) return@post
            if (focusCalibrationState != null) return@post
            isReady = false
            meteringFrameCount = 0
            settingsSuggestionSent = false
            settingsReviewRequested = true
            val camera = cameraDevice ?: return@post
            val session = captureSession ?: return@post
            startAutoMetering(camera, session)
        }
    }

    /** Re-runs RAW stepped-gray validation for user-overridden manual values. */
    fun calibrateManualSettings(settings: ManualSettings) {
        if (closed.get()) return
        cameraHandler.post {
            if (closed.get() || isReady || focusCalibrationState != null) return@post
            settingsSuggestionSent = true
            settingsReviewRequested = true
            beginFocusCalibration(
                validateManualSettings(settings).copy(depthOfFieldCalibration = null)
            )
        }
    }

    /** Starts eight sequential RAW-only empty-mat reference exposures. */
    fun beginEmptyMatCalibration() {
        if (closed.get()) return
        visionHandler.post {
            if (closed.get()) return@post
            templateCalibrator?.close()
            templateCalibrator = MatTemplateCalibrator()
            emptyMatAttemptsRemaining = 12
            incrementalGeometryEngine = null
            negativeSpaceFrameCount = 0
            canonicalTemplate?.release(); canonicalTemplate = null
            cameraHandler.post {
                if (!closed.get() && stillCaptureInFlight.compareAndSet(false, true)) {
                    submitRawCapture(CapturePurpose.EMPTY_MAT)
                } else {
                    visionHandler.post {
                        templateCalibrator?.close(); templateCalibrator = null
                    }
                    if (!closed.get()) {
                        listener.onError("Another RAW exposure is already in progress.")
                    }
                }
            }
        }
    }

    /** Restores the calibrated mat as the session's geometric/radiometric prior. */
    fun restoreEmptyMatTemplate(file: File) {
        if (closed.get()) return
        visionHandler.post {
            if (closed.get()) return@post
            val restored = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
            if (restored.empty()) {
                restored.release()
                listener.onError("The saved empty-mat reference could not be loaded.")
                return@post
            }
            installCanonicalTemplate(restored, pixelsPerMm = 4.0)
            listener.onTemplateRestored(file)
        }
    }

    private val cameraStateCallback = object : CameraDevice.StateCallback() {
        override fun onOpened(camera: CameraDevice) {
            if (closed.get()) {
                camera.close()
                return
            }
            cameraDevice = camera
            createSession(camera)
        }

        override fun onDisconnected(camera: CameraDevice) {
            listener.onError("Camera disconnected.")
            camera.close()
            cameraDevice = null
        }

        override fun onError(camera: CameraDevice, error: Int) {
            listener.onError("Camera device error $error.")
            camera.close()
            cameraDevice = null
        }
    }

    private fun createSession(camera: CameraDevice) {
        try {
            val stateCallback = object : CameraCaptureSession.StateCallback() {
                override fun onConfigured(session: CameraCaptureSession) {
                    captureSession = session
                    startAutoMetering(camera, session)
                }

                override fun onConfigureFailed(session: CameraCaptureSession) {
                    listener.onError(
                        "The device rejected the requested lossless capture stream configuration."
                    )
                }
            }
            val yuvConfig = OutputConfiguration(yuvSurface)
            val previewConfig = OutputConfiguration(previewSurface)
            val rawConfig = OutputConfiguration(rawSurface)
            if (profile.usesMaximumResolutionPixelMode) {
                // rawConfig.addSensorPixelMode(CameraMetadata.SENSOR_PIXEL_MODE_MAXIMUM_RESOLUTION)
            }
            camera.createCaptureSession(
                SessionConfiguration(
                    SessionConfiguration.SESSION_REGULAR,
                    listOf(yuvConfig, previewConfig, rawConfig),
                    { command -> cameraHandler.post(command) },
                    stateCallback,
                )
            )
        } catch (exception: CameraAccessException) {
            listener.onError("Camera session creation failed.", exception)
        }
    }

    private fun startAutoMetering(
        camera: CameraDevice,
        session: CameraCaptureSession,
    ) {
        try {
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addLiveTargets(this)
                set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
                set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_ON)
                set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_AUTO)
                set(
                    CaptureRequest.CONTROL_AF_MODE,
                    CameraMetadata.CONTROL_AF_MODE_CONTINUOUS_PICTURE,
                )
                if (profile.supportsOisOff) set(
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
                )
            }.build()
            session.setRepeatingRequest(request, previewCaptureCallback, cameraHandler)
            isReady = false
        } catch (exception: CameraAccessException) {
            listener.onError("Live camera metering failed.", exception)
        }
    }

    private fun suggestManualSettings(result: TotalCaptureResult): ManualSettings? {
        val meteredIso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: return null
        val meteredExposure = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: return null
        val gains = result.get(CaptureResult.COLOR_CORRECTION_GAINS) ?: return null
        val transform = result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM) ?: return null
        val meteredAperture = result.get(CaptureResult.LENS_APERTURE)
        val targetExposure = meteredExposure.coerceAtMost(8_000_000L)
            .coerceIn(profile.exposureTimeRangeNs.lower, profile.exposureTimeRangeNs.upper)
        fun requiredIso(aperture: Float?): Double {
            val apertureCompensation = if (
                meteredAperture != null && aperture != null && meteredAperture > 0f
            ) {
                val ratio = aperture / meteredAperture
                ratio * ratio
            } else 1f
            return meteredIso * meteredExposure.toDouble() /
                targetExposure.toDouble() * apertureCompensation
        }
        // Choose the greatest available f-number that still fits the sensor's
        // ISO range at the motion-safe exposure target.
        val selectedAperture = profile.availableApertures.sortedDescending()
            .firstOrNull { requiredIso(it) <= profile.sensitivityRange.upper }
            ?: profile.availableApertures.minOrNull()
            ?: meteredAperture
        val targetIso = requiredIso(selectedAperture).toInt()
            .coerceIn(profile.sensitivityRange.lower, profile.sensitivityRange.upper)
        val focus = (result.get(CaptureResult.LENS_FOCUS_DISTANCE)
            ?: profile.minimumFocusDistanceDiopters * 0.35f)
            .coerceIn(0f, profile.minimumFocusDistanceDiopters)
        val frameDuration = maxOf(33_333_333L, targetExposure + 1_000_000L)
            .coerceAtMost(profile.maximumFrameDurationNs)
        return ManualSettings(
            targetIso,
            targetExposure,
            frameDuration,
            focus,
            selectedAperture,
            gains,
            transform,
        )
    }

    private fun beginFocusCalibration(metered: ManualSettings) {
        val candidates = SessionFocusSelector.candidateDiopters(
            metered.focusDistanceDiopters,
            profile.minimumFocusDistanceDiopters,
        ).map { metered.copy(focusDistanceDiopters = it, depthOfFieldCalibration = null) }
        if (candidates.isEmpty()) {
            listener.onError("The camera did not expose a usable manual-focus range.")
            return
        }
        focusCalibrationState = FocusCalibrationState(metered, candidates)
        listener.onFocusCalibrationProgress(0, candidates.size)
        if (stillCaptureInFlight.get()) {
            focusCalibrationState = null
            settingsReviewRequested = false
            listener.onError("Another RAW exposure is already in progress.")
            return
        }
        startFocusCandidateSettling()
    }

    private fun startFocusCandidateSettling() {
        val state = focusCalibrationState ?: return
        val camera = cameraDevice ?: return abortFocusCalibration("Camera closed during focus calibration.")
        val session = captureSession ?: return abortFocusCalibration(
            "Camera session closed during focus calibration."
        )
        state.settlingFrameCount = 0
        state.rawSubmitted = false
        try {
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addLiveTargets(this)
                applyManualControls(this, state.current, stillRequest = false)
                setTag("focus-settle-${state.candidateIndex}")
            }.build()
            session.setRepeatingRequest(request, previewCaptureCallback, cameraHandler)
        } catch (exception: Exception) {
            abortFocusCalibration("The manual lens could not settle for calibration.", exception)
        }
    }

    private fun finishFocusCalibrationCandidate(
        timestamp: Long,
        measurement: FocusCandidateMeasurement,
    ) {
        cameraHandler.post {
            rawProcessingScheduled.remove(timestamp)
            val state = focusCalibrationState ?: return@post
            state.measurements += measurement
            listener.onFocusCalibrationProgress(state.measurements.size, state.candidates.size)
            stillCaptureInFlight.set(false)
            if (state.candidateIndex + 1 < state.candidates.size) {
                state.candidateIndex++
                startFocusCandidateSettling()
                return@post
            }
            val calibration = SessionFocusSelector.select(
                state.measurements,
                state.meteredSettings.focusDistanceDiopters,
            ) ?: DepthOfFieldCalibration(
                focusDistanceDiopters = state.meteredSettings.focusDistanceDiopters,
                candidateCount = state.measurements.size,
                detectedBoundaryCount = 0,
                sampleCount = 0,
                referenceMedianEdgeWidthCanonicalPx = 2.0,
                referenceP90EdgeWidthCanonicalPx = 2.5,
                maximumP90EdgeWidthCanonicalPx = 3.5,
                nearReferenceDepthMm = 0.0,
                farReferenceDepthMm = 100.0,
                medianStepContrast = 0.5,
                referenceEqualStepConsistency = 0.5,
            )
            focusCalibrationState = null
            settingsReviewRequested = false
            val suggested = state.meteredSettings.copy(
                focusDistanceDiopters = calibration.focusDistanceDiopters,
                depthOfFieldCalibration = calibration,
            )
            previewUncommittedSettings(suggested)
            listener.onSettingsSuggested(suggested, profile.availableApertures.size <= 1)
        }
    }

    private fun previewUncommittedSettings(settings: ManualSettings) {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        try {
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addLiveTargets(this)
                applyManualControls(this, settings, stillRequest = false)
            }.build()
            session.setRepeatingRequest(request, previewCaptureCallback, cameraHandler)
        } catch (exception: Exception) {
            listener.onError("The depth-of-field recommendation could not be previewed.", exception)
        }
    }

    private fun validateManualSettings(settings: ManualSettings): ManualSettings {
        val maximumExposure = minOf(
            profile.exposureTimeRangeNs.upper,
            (profile.maximumFrameDurationNs - 1_000_000L)
                .coerceAtLeast(profile.exposureTimeRangeNs.lower),
        )
        val exposure = settings.exposureTimeNs.coerceIn(
            profile.exposureTimeRangeNs.lower,
            maximumExposure,
        )
        return settings.copy(
            sensitivityIso = settings.sensitivityIso.coerceIn(
                profile.sensitivityRange.lower,
                profile.sensitivityRange.upper,
            ),
            exposureTimeNs = exposure,
            frameDurationNs = maxOf(settings.frameDurationNs, exposure + 1_000_000L)
                .coerceAtMost(profile.maximumFrameDurationNs),
            focusDistanceDiopters = settings.focusDistanceDiopters.coerceIn(
                0f,
                profile.minimumFocusDistanceDiopters,
            ),
            aperture = settings.aperture?.let { requested ->
                profile.availableApertures.minByOrNull {
                    kotlin.math.abs(it - requested)
                } ?: requested
            },
        )
    }

    private fun applyManualControls(
        builder: CaptureRequest.Builder,
        settings: ManualSettings,
        stillRequest: Boolean,
    ) = with(builder) {
        set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_OFF)
        set(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
        set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_OFF)
        set(CaptureRequest.FLASH_MODE, CameraMetadata.FLASH_MODE_OFF)
        set(CaptureRequest.STATISTICS_FACE_DETECT_MODE, CameraMetadata.STATISTICS_FACE_DETECT_MODE_OFF)
        if (profile.supportsManualPostProcessing) {
            set(CaptureRequest.CONTROL_AWB_MODE, CameraMetadata.CONTROL_AWB_MODE_OFF)
            if (!stillRequest) {
                set(CaptureRequest.COLOR_CORRECTION_MODE, CameraMetadata.COLOR_CORRECTION_MODE_TRANSFORM_MATRIX)
                set(CaptureRequest.COLOR_CORRECTION_GAINS, settings.whiteBalanceGains)
                set(CaptureRequest.COLOR_CORRECTION_TRANSFORM, settings.whiteBalanceTransform)
            }
        }
        set(CaptureRequest.SENSOR_SENSITIVITY, settings.sensitivityIso)
        set(CaptureRequest.SENSOR_EXPOSURE_TIME, settings.exposureTimeNs)
        set(CaptureRequest.SENSOR_FRAME_DURATION, settings.frameDurationNs)
        set(CaptureRequest.LENS_FOCUS_DISTANCE, settings.focusDistanceDiopters)
        settings.aperture?.let { set(CaptureRequest.LENS_APERTURE, it) }
        if (profile.supportsOisOff) {
            set(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
            )
        }
        if (!stillRequest && profile.processingControls.videoStabilizationOff) {
            set(
                CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                CameraMetadata.CONTROL_VIDEO_STABILIZATION_MODE_OFF,
            )
        }
        if (!stillRequest && profile.supportsEdgeOff) {
            set(CaptureRequest.EDGE_MODE, CameraMetadata.EDGE_MODE_OFF)
        }
        if (!stillRequest && profile.supportsNoiseReductionOff) {
            set(CaptureRequest.NOISE_REDUCTION_MODE, CameraMetadata.NOISE_REDUCTION_MODE_OFF)
        }
        if (profile.processingControls.hotPixelOff) {
            set(CaptureRequest.HOT_PIXEL_MODE, CameraMetadata.HOT_PIXEL_MODE_OFF)
        }
        if (!stillRequest && profile.processingControls.shadingOff) {
            set(CaptureRequest.SHADING_MODE, CameraMetadata.SHADING_MODE_OFF)
        }
        if (!stillRequest && profile.processingControls.aberrationOff) {
            set(
                CaptureRequest.COLOR_CORRECTION_ABERRATION_MODE,
                CameraMetadata.COLOR_CORRECTION_ABERRATION_MODE_OFF,
            )
        }
        if (!stillRequest && profile.processingControls.distortionOff) {
            set(
                CaptureRequest.DISTORTION_CORRECTION_MODE,
                CameraMetadata.DISTORTION_CORRECTION_MODE_OFF,
            )
        }
        if (profile.processingControls.blackLevelLock) {
            set(CaptureRequest.BLACK_LEVEL_LOCK, true)
        }
        if (!stillRequest && profile.processingControls.neutralPostRawSensitivity) {
            set(CaptureRequest.CONTROL_POST_RAW_SENSITIVITY_BOOST, 100)
        }
        if (!stillRequest && profile.processingControls.rotateAndCropNone) {
            set(CaptureRequest.SCALER_ROTATE_AND_CROP, CameraMetadata.SCALER_ROTATE_AND_CROP_NONE)
        }
        if (!stillRequest) set(CaptureRequest.SCALER_CROP_REGION, profile.activeArray)
        if (stillRequest && profile.usesMaximumResolutionPixelMode) {
            set(CaptureRequest.SENSOR_PIXEL_MODE, CameraMetadata.SENSOR_PIXEL_MODE_MAXIMUM_RESOLUTION)
        }
        if (!stillRequest && profile.supportsTonemapGamma) {
            set(CaptureRequest.TONEMAP_MODE, CameraMetadata.TONEMAP_MODE_GAMMA_VALUE)
            set(CaptureRequest.TONEMAP_GAMMA, 1.0f)
        }
    }

    private val previewCaptureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult,
        ) {
            val focusState = focusCalibrationState
            if (focusState != null) {
                if (request.tag == "focus-settle-${focusState.candidateIndex}" &&
                    !focusState.rawSubmitted
                ) {
                    focusState.settlingFrameCount++
                    val lensStationary = result.get(CaptureResult.LENS_STATE) ==
                        CameraMetadata.LENS_STATE_STATIONARY
                    if (focusState.settlingFrameCount >= 4 &&
                        (lensStationary || focusState.settlingFrameCount >= 8)
                    ) {
                        focusState.rawSubmitted = true
                        if (stillCaptureInFlight.compareAndSet(false, true)) {
                            submitRawCapture(CapturePurpose.FOCUS_CALIBRATION)
                        } else {
                            abortFocusCalibration("Another RAW exposure interrupted focus calibration.")
                        }
                    }
                }
                return
            }
            if (isReady || settingsSuggestionSent) return
            meteringFrameCount++
            val aeStable = result.get(CaptureResult.CONTROL_AE_STATE) in setOf(
                CameraMetadata.CONTROL_AE_STATE_CONVERGED,
                CameraMetadata.CONTROL_AE_STATE_FLASH_REQUIRED,
            )
            val awbStable = result.get(CaptureResult.CONTROL_AWB_STATE) ==
                CameraMetadata.CONTROL_AWB_STATE_CONVERGED
            if (meteringFrameCount >= 12 && ((aeStable && awbStable) || meteringFrameCount >= 60)) {
                suggestManualSettings(result)?.let {
                    settingsSuggestionSent = true
                    if (settingsReviewRequested) {
                        beginFocusCalibration(it)
                    } else {
                        listener.onSettingsSuggested(it, profile.availableApertures.size <= 1)
                    }
                }
            }
        }

        override fun onCaptureFailed(
            session: CameraCaptureSession,
            request: CaptureRequest,
            failure: CaptureFailure,
        ) {
            Log.w("CameraSessionController", "Transient preview frame dropped: reason ${failure.reason}")
            if (focusCalibrationState != null) {
                abortFocusCalibration(
                    "A focus-settling preview failed: reason ${failure.reason}."
                )
            }
        }
    }

    private val stillCaptureCallback = object : CameraCaptureSession.CaptureCallback() {
        override fun onCaptureCompleted(
            session: CameraCaptureSession,
            request: CaptureRequest,
            result: TotalCaptureResult,
        ) {
            val timestamp = result.get(
                android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP
            )
            if (timestamp == null) {
                stillCaptureInFlight.set(false)
                resumeLockedPreview()
                listener.onError("Still result did not contain a sensor timestamp.")
                return
            }
            val purpose = request.tag as? CapturePurpose
            if (purpose == null) {
                stillCaptureInFlight.set(false)
                resumeLockedPreview()
                listener.onError("RAW capture result did not identify its purpose.")
                return
            }
            pendingCapturePurposes[timestamp] = purpose
            if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
                val settings = focusCalibrationState?.current
                if (settings == null) {
                    stillCaptureInFlight.set(false)
                    listener.onError("Focus calibration state was lost before the RAW result arrived.")
                    return
                }
                pendingFocusCalibrationSettings[timestamp] = settings
            }
            pendingCaptureResults[timestamp] = result
            pairAndWrite(timestamp)
        }

        override fun onCaptureFailed(
            session: CameraCaptureSession,
            request: CaptureRequest,
            failure: CaptureFailure,
        ) {
            stillCaptureInFlight.set(false)
            if (request.tag == CapturePurpose.FOCUS_CALIBRATION) {
                abortFocusCalibration("RAW focus probe failed: reason ${failure.reason}.")
            } else {
                resumeLockedPreview()
                listener.onError("Still capture failed: reason ${failure.reason}.")
            }
        }
    }

    private fun abortFocusCalibration(message: String, cause: Throwable? = null) {
        focusCalibrationState = null
        settingsReviewRequested = false
        settingsSuggestionSent = false
        meteringFrameCount = 0
        stillCaptureInFlight.set(false)
        val camera = cameraDevice
        val session = captureSession
        if (!closed.get() && camera != null && session != null) startAutoMetering(camera, session)
        listener.onError(message, cause)
    }

    private fun onYuvAvailable(reader: ImageReader) {
        val image = reader.acquireLatestImage() ?: return
        if (closed.get()) {
            image.close()
            return
        }
        image.use {
            val timestampNs = image.timestamp
            val visionDue = lastVisionTimestampNs == Long.MIN_VALUE ||
                timestampNs <= lastVisionTimestampNs ||
                timestampNs - lastVisionTimestampNs >= LIVE_VISION_INTERVAL_NS
            val needsVision = matPoseProcessor != null && visionDue &&
                visionBusy.compareAndSet(false, true)
            if (!needsVision) return@use
            lastVisionTimestampNs = timestampNs
            val framePlanes = try {
                FramePlanes.copyOf(image)
            } catch (exception: Exception) {
                visionBusy.set(false)
                listener.onError("The live YUV frame could not be packed for analysis.", exception)
                return@use
            }
            val width = framePlanes.width
            val height = framePlanes.height
            var sum = 0L
            var dark = 0L
            var bright = 0L
            var count = 0L
            val sampleStep = 4
            for (row in 0 until height step sampleStep) {
                val base = row * framePlanes.yRowStride
                for (column in 0 until width step sampleStep) {
                    val value = framePlanes.y[
                        base + column * framePlanes.yPixelStride
                    ].toInt() and 0xff
                    sum += value
                    if (value <= 5) dark++
                    if (value >= 250) bright++
                    count++
                }
            }
            listener.onYuvFrame(
                YuvFrameSummary(
                    sequence = frameSequence.incrementAndGet(),
                    sensorTimestampNs = timestampNs,
                    width = width,
                    height = height,
                    meanLuma = sum.toDouble() / count,
                    clippedDarkFraction = dark.toDouble() / count,
                    clippedBrightFraction = bright.toDouble() / count,
                )
            )
            visionHandler.post {
                try {
                    matPoseProcessor?.process(framePlanes)?.let { estimate ->
                        listener.onMatPose(estimate.filtered)
                        analyzeRegisteredFrame(
                            framePlanes,
                            geometryPose = estimate.raw,
                            guidancePose = estimate.filtered,
                        )
                    }
                } catch (exception: Exception) {
                    listener.onError("Mat pose processing failed.", exception)
                } finally {
                    visionBusy.set(false)
                }
            }
        }
    }

    private fun onRawAvailable(reader: ImageReader) {
        val image = reader.acquireNextImage() ?: return
        if (closed.get() || !stillCaptureInFlight.get()) {
            image.close()
            return
        }
        if (discardedCaptureTimestamps.remove(image.timestamp)) {
            image.close()
            return
        }
        pendingRawImages.put(image.timestamp, image)?.close()
        pairAndWrite(image.timestamp)
    }

    /** Rechecks the actual still image rather than trusting the preview gate. */
    private fun validateExactCaptureFrame(
        frame: FramePlanes,
        pose: MatPoseResult,
        visibilityEvaluator: MatVisibilityEvaluator,
        linearFocusGray: Mat,
    ): ExactCaptureAnalysis {
        val geometryEngine = incrementalGeometryEngine
            ?: error("The registered empty-mat model is unavailable")
        val gray = frame.grayMat()
        try {
            val geometry = geometryEngine.integrate(gray, pose, carveHull = false)
                ?: error("Foreground extraction failed on the still frame")
            geometry.use {
                val negative = it.negativeSpace
                val containment = visibilityEvaluator.evaluate(pose, frame.width, frame.height)
                val sharpness = sharpnessVariance(gray, negative.imageForeground)
                val focusAnalysis = steppedGrayFocusEvaluator.evaluate(
                    linearFocusGray,
                    pose,
                    negative.canonicalVisibleMat,
                ) ?: error("The RAW still did not preserve measurable stepped-gray boundaries")
                val focusCalibration = manualSettings?.depthOfFieldCalibration
                    ?: error("The session focus has no stepped-gray calibration")
                val depthOfFieldScore = focusCalibration.score(focusAnalysis)
                val radiometric = exp(
                    -1.25 * abs(ln(negative.radiometricScale.coerceAtLeast(1e-6))) -
                        abs(negative.radiometricOffset) / 96.0 -
                        negative.radiometricMeanAbsoluteError / 30.0
                ).coerceIn(0.0, 1.0)
                val foregroundPixels = Core.countNonZero(negative.canonicalForeground)
                val artifactConfidence = (foregroundPixels / 800.0).coerceIn(0.0, 1.0)
                val failures = buildList {
                    if (!containment.fullMatVisible || containment.minimumFrameMarginPx < 12.0) {
                        add("full mat containment")
                    }
                    if (sharpness < 80.0) add("sharpness")
                    if (depthOfFieldScore < 0.80) add("stepped-gray depth of field")
                    if (radiometric < 0.75) add("radiometry")
                    if (negative.highlightClippedFraction > 0.001) add("highlight clipping")
                    if (artifactConfidence < 0.75) add("artifact segmentation")
                }
                check(failures.isEmpty()) {
                    "Still frame failed: ${failures.joinToString()}"
                }
                return ExactCaptureAnalysis(
                    pose,
                    negative.imageForeground.clone(),
                    focusAnalysis,
                )
            }
        } finally {
            gray.release()
        }
    }

    private fun pairAndWrite(timestamp: Long) {
        val result = pendingCaptureResults[timestamp] ?: return
        val rawImage = pendingRawImages[timestamp] ?: return
        val purpose = pendingCapturePurposes[timestamp] ?: return
        val settings = if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
            pendingFocusCalibrationSettings[timestamp]
        } else {
            manualSettings
        } ?: return
        if (!rawProcessingScheduled.add(timestamp)) return
        val controlAudit = try {
            auditEffectiveCaptureControls(
                result,
                settings,
                enforceSessionReference = purpose != CapturePurpose.FOCUS_CALIBRATION,
            )
        } catch (exception: Exception) {
            rawProcessingScheduled.remove(timestamp)
            if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
                pendingRawImages.remove(timestamp)?.close()
                pendingCaptureResults.remove(timestamp)
                pendingCapturePurposes.remove(timestamp)
                pendingFocusCalibrationSettings.remove(timestamp)
                abortFocusCalibration(
                    "The camera did not apply the requested RAW focus probe.",
                    exception,
                )
            } else {
                failPendingCapture(
                    timestamp,
                    "The camera did not preserve the locked capture pipeline.",
                    exception,
                )
            }
            return
        }
        if (purpose == CapturePurpose.FOCUS_CALIBRATION) {
            pendingRawImages.remove(timestamp)
            pendingCaptureResults.remove(timestamp)
            pendingCapturePurposes.remove(timestamp)
            pendingFocusCalibrationSettings.remove(timestamp)
            processFocusCalibrationRaw(timestamp, rawImage, result, settings)
            return
        }
        if (purpose == CapturePurpose.EMPTY_MAT) {
            pendingRawImages.remove(timestamp)
            pendingCaptureResults.remove(timestamp)
            pendingCapturePurposes.remove(timestamp)
            processEmptyMatRaw(timestamp, rawImage, result)
            return
        }
        val phoneSensor = phoneSensorSampleForTimestamp(timestamp)
        if (phoneSensor == null) {
            rawProcessingScheduled.remove(timestamp)
            failPendingCapture(
                timestamp,
                "No phone sensor samples bracketed the exposure timestamp.",
            )
            return
        }
        pendingRawImages.remove(timestamp)
        pendingCaptureResults.remove(timestamp)
        pendingCapturePurposes.remove(timestamp)
        visionHandler.post {
            var rawDerivedMask: Mat? = null
            try {
                outputDirectory.mkdirs()
                val base = "capture-$timestamp"
                val rawMasterFile = File(outputDirectory, "$base.raw16")
                val workingImageFile = File(outputDirectory, "$base.tif")
                val foregroundMaskFile = File(outputDirectory, "$base-mask.png")
                val metadataFile = File(outputDirectory, "$base.json")
                val rawArtifact = rawFrameProcessor.write(
                    rawImage,
                    result,
                    rawMasterFile,
                    workingImageFile,
                )
                check(rawArtifact.width == rawImageGeometry.imageWidth &&
                    rawArtifact.height == rawImageGeometry.imageHeight
                ) {
                    "The RAW buffer dimensions changed after session qualification"
                }
                val rawPose = rawMatPoseProcessor?.processExposure(rawArtifact.validationFrame)
                    ?: error("The RAW exposure did not contain a solvable mat pose")
                check(rawPose.timestampNs == timestamp) {
                    "RAW-derived pose timestamp does not match the sensor exposure"
                }
                val linearFocusGray = rawFrameProcessor.linearFocusGray(rawImage, result)
                val exactAnalysis = try {
                    validateExactCaptureFrame(
                        rawArtifact.validationFrame,
                        rawPose,
                        rawMatVisibilityEvaluator,
                        linearFocusGray,
                    )
                } finally {
                    linearFocusGray.release()
                }.also { rawDerivedMask = it.foregroundMask }
                check(Imgcodecs.imwrite(
                    foregroundMaskFile.absolutePath,
                    exactAnalysis.foregroundMask,
                )) {
                    "OpenCV could not write the exact still-frame foreground mask"
                }
                val metadata = buildMetadata(
                    timestamp,
                    result,
                    exactAnalysis.pose,
                    phoneSensor,
                    rawArtifact,
                    controlAudit,
                    exactAnalysis.steppedGrayFocus,
                ).apply {
                    put("rawMasterFile", rawMasterFile.name)
                    put("rawMasterFormat", "ANDROID_RAW_SENSOR_16_VERBATIM")
                    put("workingImageFile", workingImageFile.name)
                    put("workingImageFormat", "TIFF_16_LINEAR_RGB")
                    put("foregroundMaskFile", foregroundMaskFile.name)
                }
                metadataFile.writeText(metadata.toString(2))
                if (!closed.get()) {
                    listener.onCaptureSaved(
                        rawMasterFile,
                        workingImageFile,
                        foregroundMaskFile,
                        metadataFile,
                    )
                }
            } catch (exception: Exception) {
                if (!closed.get()) {
                    listener.onError("RAW master or linear TIFF serialization failed.", exception)
                }
            } finally {
                rawImage.close()
                rawDerivedMask?.release()
                cameraHandler.post {
                    rawProcessingScheduled.remove(timestamp)
                    stillCaptureInFlight.set(false)
                    if (!closed.get()) resumeLockedPreview()
                    trimUnpairedBuffers()
                }
            }
        }
    }

    /** Measures one candidate focus against the distributed stepped-gray field. */
    private fun processFocusCalibrationRaw(
        timestamp: Long,
        rawImage: Image,
        result: TotalCaptureResult,
        settings: ManualSettings,
    ) {
        visionHandler.post {
            var measurement: FocusCandidateMeasurement? = null
            var failure: Throwable? = null
            try {
                val frame = rawFrameProcessor.validationFrame(rawImage, result)
                val pose = rawMatPoseProcessor?.processExposure(frame)
                    ?: error("The RAW focus probe did not contain a solvable complete-mat pose")
                val containment = rawMatVisibilityEvaluator.evaluate(pose, frame.width, frame.height)
                check(containment.fullMatVisible && containment.minimumFrameMarginPx >= 12.0) {
                    "The complete mat and its stepped-gray field must remain inside the RAW focus probe"
                }
                val gray = rawFrameProcessor.linearFocusGray(rawImage, result)
                try {
                    val analysis = steppedGrayFocusEvaluator.evaluate(gray, pose)
                        ?: error("The stepped-gray rings were not measurable in the RAW focus probe")
                    measurement = FocusCandidateMeasurement(
                        settings.focusDistanceDiopters,
                        analysis,
                    )
                } finally {
                    gray.release()
                }
            } catch (exception: Exception) {
                failure = exception
            } finally {
                rawImage.close()
            }
            val completed = measurement
            if (completed != null) {
                finishFocusCalibrationCandidate(timestamp, completed)
            } else {
                cameraHandler.post {
                    rawProcessingScheduled.remove(timestamp)
                    stillCaptureInFlight.set(false)
                    focusCalibrationState = null
                    settingsSuggestionSent = false
                    val camera = cameraDevice
                    val session = captureSession
                    if (!closed.get() && camera != null && session != null) {
                        startAutoMetering(camera, session)
                    }
                    listener.onError("RAW stepped-gray focus calibration failed.", failure)
                }
            }
        }
    }

    /** Registers one exact RAW exposure into the empty-mat radiometric prior. */
    private fun processEmptyMatRaw(
        timestamp: Long,
        rawImage: Image,
        result: TotalCaptureResult,
    ) {
        visionHandler.post {
            var captureAnother = false
            try {
                val frame = rawFrameProcessor.validationFrame(rawImage, result)
                val pose = rawMatPoseProcessor?.processExposure(frame)
                    ?: error("The RAW empty-mat exposure did not contain a solvable mat pose")
                check(pose.timestampNs == timestamp) {
                    "RAW empty-mat pose timestamp does not match its exposure"
                }
                val calibrator = templateCalibrator
                    ?: error("Empty-mat calibration was not active")
                val progress = calibrator.ingest(frame, pose)
                val completed = progress.completedTemplate
                if (completed != null) {
                    outputDirectory.mkdirs()
                    val destination = File(outputDirectory, "empty-mat-template.png")
                    check(Imgcodecs.imwrite(destination.absolutePath, completed)) {
                        "Could not persist RAW-derived empty-mat template"
                    }
                    installCanonicalTemplate(completed, calibrator.pixelsPerMm)
                    calibrator.close()
                    templateCalibrator = null
                    // Notify only after the template exists durably; the activity
                    // checkpoints this path synchronously from the callback.
                    if (!closed.get()) listener.onTemplateCalibration(progress)
                } else {
                    if (!closed.get()) listener.onTemplateCalibration(progress)
                    emptyMatAttemptsRemaining--
                    check(emptyMatAttemptsRemaining > 0) {
                        "The RAW empty-mat reference could not be registered after 12 exposures"
                    }
                    captureAnother = true
                }
            } catch (exception: Exception) {
                templateCalibrator?.close(); templateCalibrator = null
                if (!closed.get()) listener.onError("RAW empty-mat calibration failed.", exception)
            } finally {
                rawImage.close()
                cameraHandler.post {
                    rawProcessingScheduled.remove(timestamp)
                    stillCaptureInFlight.set(false)
                    if (!closed.get()) resumeLockedPreview()
                    trimUnpairedBuffers()
                    if (!closed.get() && captureAnother &&
                        stillCaptureInFlight.compareAndSet(false, true)
                    ) {
                        submitRawCapture(CapturePurpose.EMPTY_MAT)
                    }
                }
            }
        }
    }

    private fun failPendingCapture(timestamp: Long, message: String, cause: Throwable? = null) {
        rawProcessingScheduled.remove(timestamp)
        val rawImage = pendingRawImages.remove(timestamp)
        if (rawImage == null) {
            discardedCaptureTimestamps += timestamp
            while (discardedCaptureTimestamps.size > 8) {
                discardedCaptureTimestamps.iterator().let { iterator ->
                    if (iterator.hasNext()) {
                        iterator.next()
                        iterator.remove()
                    }
                }
            }
        } else {
            rawImage.close()
        }
        pendingCaptureResults.remove(timestamp)
        pendingCapturePurposes.remove(timestamp)
        pendingFocusCalibrationSettings.remove(timestamp)
        stillCaptureInFlight.set(false)
        if (!closed.get()) {
            resumeLockedPreview()
            listener.onError(message, cause)
        }
    }

    private fun resumeLockedPreview() {
        val camera = cameraDevice ?: return
        val session = captureSession ?: return
        val settings = manualSettings ?: return
        try {
            val request = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
                addLiveTargets(this)
                applyManualControls(this, settings, stillRequest = false)
            }.build()
            session.setRepeatingRequest(request, previewCaptureCallback, cameraHandler)
        } catch (exception: Exception) {
            listener.onError("Locked preview could not resume after still capture.", exception)
        }
    }

    /** Rejects a frame when the HAL reports that a locked or disabled stage changed. */
    private fun auditEffectiveCaptureControls(
        result: TotalCaptureResult,
        requested: ManualSettings,
        enforceSessionReference: Boolean = true,
    ): JSONObject {
        val failures = mutableListOf<String>()
        fun expect(name: String, actual: Any?, expected: Any) {
            if (actual != expected) failures += "$name=$actual (expected $expected)"
        }
        expect("controlMode", result.get(CaptureResult.CONTROL_MODE), CameraMetadata.CONTROL_MODE_OFF)
        expect("aeMode", result.get(CaptureResult.CONTROL_AE_MODE), CameraMetadata.CONTROL_AE_MODE_OFF)
        expect("afMode", result.get(CaptureResult.CONTROL_AF_MODE), CameraMetadata.CONTROL_AF_MODE_OFF)
        if (profile.supportsManualPostProcessing) {
            expect("awbMode", result.get(CaptureResult.CONTROL_AWB_MODE), CameraMetadata.CONTROL_AWB_MODE_OFF)
        }
        expect("flashMode", result.get(CaptureResult.FLASH_MODE), CameraMetadata.FLASH_MODE_OFF)
        if (profile.supportsOisOff) expect(
            "oisMode",
            result.get(CaptureResult.LENS_OPTICAL_STABILIZATION_MODE),
            CameraMetadata.LENS_OPTICAL_STABILIZATION_MODE_OFF,
        )
        if (profile.processingControls.hotPixelOff) expect(
            "hotPixelMode", result.get(CaptureResult.HOT_PIXEL_MODE), CameraMetadata.HOT_PIXEL_MODE_OFF
        )
        if (profile.processingControls.blackLevelLock) expect(
            "blackLevelLock", result.get(CaptureResult.BLACK_LEVEL_LOCK), true
        )
        if (profile.usesMaximumResolutionPixelMode) expect(
            "sensorPixelMode",
            result.get(CaptureResult.SENSOR_PIXEL_MODE),
            CameraMetadata.SENSOR_PIXEL_MODE_MAXIMUM_RESOLUTION,
        )

        val effective = EffectiveSensorSettings(
            sensitivityIso = result.get(CaptureResult.SENSOR_SENSITIVITY)
                ?: error("Capture result omitted sensor sensitivity"),
            exposureTimeNs = result.get(CaptureResult.SENSOR_EXPOSURE_TIME)
                ?: error("Capture result omitted exposure time"),
            frameDurationNs = result.get(CaptureResult.SENSOR_FRAME_DURATION)
                ?: error("Capture result omitted frame duration"),
            focusDistanceDiopters = result.get(CaptureResult.LENS_FOCUS_DISTANCE)
                ?: error("Capture result omitted focus distance"),
            aperture = result.get(CaptureResult.LENS_APERTURE),
        )
        if (effective.sensitivityIso != requested.sensitivityIso) {
            failures += "sensitivity=${effective.sensitivityIso} (requested ${requested.sensitivityIso})"
        }
        val exposureToleranceNs = maxOf(1_000L, requested.exposureTimeNs / 1_000L)
        if (kotlin.math.abs(effective.exposureTimeNs - requested.exposureTimeNs) > exposureToleranceNs) {
            failures += "exposure=${effective.exposureTimeNs} (requested ${requested.exposureTimeNs})"
        }
        if (abs(effective.focusDistanceDiopters - requested.focusDistanceDiopters) > 0.001f) {
            failures += "focus=${effective.focusDistanceDiopters} (requested ${requested.focusDistanceDiopters})"
        }
        if (requested.aperture != null && effective.aperture != null &&
            abs(effective.aperture - requested.aperture) > 0.001f
        ) {
            failures += "aperture=${effective.aperture} (requested ${requested.aperture})"
        }
        if (enforceSessionReference) effectiveSensorReference?.let { reference ->
            if (effective.sensitivityIso != reference.sensitivityIso) failures += "session ISO changed"
            if (effective.exposureTimeNs != reference.exposureTimeNs) failures += "session exposure changed"
            if (effective.frameDurationNs != reference.frameDurationNs) {
                failures += "session frame duration changed"
            }
            if (effective.focusDistanceDiopters != reference.focusDistanceDiopters) {
                failures += "session focus changed"
            }
            if (effective.aperture != reference.aperture) failures += "session aperture changed"
        }
        check(failures.isEmpty()) { failures.joinToString("; ") }
        if (enforceSessionReference && effectiveSensorReference == null) {
            effectiveSensorReference = effective
        }
        return JSONObject().apply {
            put("verified", true)
            put("sensitivityIso", effective.sensitivityIso)
            put("exposureTimeNs", effective.exposureTimeNs)
            put("frameDurationNs", effective.frameDurationNs)
            put("focusDistanceDiopters", effective.focusDistanceDiopters)
            put("aperture", effective.aperture ?: JSONObject.NULL)
            put("rawLensShadingApplied", false)
            put("rawOnlyCapture", true)
            put("processedStreamControlsExcludedFromStill", true)
            put("maximumResolutionPixelMode", profile.usesMaximumResolutionPixelMode)
        }
    }

    private fun buildMetadata(
        timestamp: Long,
        result: TotalCaptureResult,
        pose: MatPoseResult,
        phoneSensor: SynchronizedPhoneSensorSample,
        rawArtifact: RawFrameArtifact,
        controlAudit: JSONObject,
        steppedGrayFocus: GrayStepFocusAnalysis,
    ): JSONObject {
        val rawIntrinsics = scaledRawIntrinsics(rawArtifact.width, rawArtifact.height)
        return JSONObject().apply {
            put("schemaVersion", "0.4.10")
            put("cameraId", profile.cameraId)
            put("sensorOrientationDegrees", profile.sensorOrientationDegrees)
            put("sensorTimestampNs", timestamp)
            put("sensitivityIso", result.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY))
            put("exposureTimeNs", result.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME))
            put("frameDurationNs", result.get(android.hardware.camera2.CaptureResult.SENSOR_FRAME_DURATION))
            put("focusDistanceDiopters", result.get(android.hardware.camera2.CaptureResult.LENS_FOCUS_DISTANCE))
            put("aperture", result.get(android.hardware.camera2.CaptureResult.LENS_APERTURE))
            manualSettings?.depthOfFieldCalibration?.let {
                put("depthOfFieldCalibration", depthOfFieldCalibrationJson(it))
            }
            put("steppedGrayFocus", steppedGrayFocusJson(steppedGrayFocus))
            result.get(CaptureResult.COLOR_CORRECTION_GAINS)?.let { gains ->
                put("whiteBalanceGains", JSONArray(listOf(
                    gains.red,
                    gains.greenEven,
                    gains.greenOdd,
                    gains.blue,
                )))
            }
            result.get(CaptureResult.COLOR_CORRECTION_TRANSFORM)?.let { transform ->
                put("whiteBalanceTransform", JSONArray(buildList {
                    for (row in 0..2) for (column in 0..2) {
                        val value = transform.getElement(column, row)
                        add(value.numerator.toDouble() / value.denominator)
                    }
                }))
            }
            put("rawWidth", rawArtifact.width)
            put("rawHeight", rawArtifact.height)
            put("rawRowStrideBytes", rawArtifact.rowStrideBytes)
            put("rawPixelStrideBytes", rawArtifact.pixelStrideBytes)
            put("rawBufferBytes", rawArtifact.rawBufferBytes)
            put("rawByteOrder", java.nio.ByteOrder.nativeOrder().toString())
            put("rawColorFilterArrangement", rawArtifact.colorFilterArrangement)
            put("rawBlackLevels", JSONArray(rawArtifact.blackLevels.toList()))
            put("rawBlackLevelOrder", "top-left, top-right, bottom-left, bottom-right CFA sites")
            put("rawWhiteLevel", rawArtifact.whiteLevel)
            put("rawPixelMode", if (profile.usesMaximumResolutionPixelMode) {
                "MAXIMUM_RESOLUTION"
            } else "DEFAULT")
            put("rawInterpretation", JSONObject().apply {
                put("planeStorage", "verbatim Image.Plane buffer including row padding")
                put("preCorrectionActiveArray", JSONObject().apply {
                    put("left", profile.rawActiveArray.left)
                    put("top", profile.rawActiveArray.top)
                    put("right", profile.rawActiveArray.right)
                    put("bottom", profile.rawActiveArray.bottom)
                })
                put("rawPixelArray", JSONArray(listOf(
                    profile.rawPixelArraySize.width,
                    profile.rawPixelArraySize.height,
                )))
                characteristics[CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE]?.let {
                    put("defaultPixelArray", JSONArray(listOf(it.width, it.height)))
                }
                characteristics[
                    CameraCharacteristics.SENSOR_INFO_PIXEL_ARRAY_SIZE_MAXIMUM_RESOLUTION
                ]?.let {
                    put("maximumResolutionPixelArray", JSONArray(listOf(it.width, it.height)))
                }
                characteristics[CameraCharacteristics.SENSOR_INFO_BINNING_FACTOR]?.let {
                    put("reportedBinningFactor", JSONArray(listOf(it.width, it.height)))
                }
                characteristics[CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT1]?.let {
                    put("referenceIlluminant1", it)
                }
                characteristics[CameraCharacteristics.SENSOR_REFERENCE_ILLUMINANT2]?.let {
                    put("referenceIlluminant2", it)
                }
                characteristics[CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM1]?.let {
                    put("calibrationTransform1", colorTransformJson(it))
                }
                characteristics[CameraCharacteristics.SENSOR_CALIBRATION_TRANSFORM2]?.let {
                    put("calibrationTransform2", colorTransformJson(it))
                }
                characteristics[CameraCharacteristics.SENSOR_COLOR_TRANSFORM1]?.let {
                    put("colorTransform1", colorTransformJson(it))
                }
                characteristics[CameraCharacteristics.SENSOR_COLOR_TRANSFORM2]?.let {
                    put("colorTransform2", colorTransformJson(it))
                }
                characteristics[CameraCharacteristics.SENSOR_FORWARD_MATRIX1]?.let {
                    put("forwardMatrix1", colorTransformJson(it))
                }
                characteristics[CameraCharacteristics.SENSOR_FORWARD_MATRIX2]?.let {
                    put("forwardMatrix2", colorTransformJson(it))
                }
                result.get(CaptureResult.SENSOR_NEUTRAL_COLOR_POINT)?.let { values ->
                    put("neutralColorPoint", JSONArray(values.map {
                        it.numerator.toDouble() / it.denominator
                    }))
                }
                result.get(CaptureResult.SENSOR_GREEN_SPLIT)?.let { put("greenSplit", it) }
                result.get(CaptureResult.SENSOR_NOISE_PROFILE)?.let { values ->
                    put("noiseProfile", JSONArray(values.map { pair ->
                        JSONArray(listOf(pair.first, pair.second))
                    }))
                }
            })
            put("workingImageDevelopment", JSONObject().apply {
                put("source", "preserved RAW_SENSOR master")
                put("blackSubtraction", "per-CFA dynamic level with static fallback")
                put("whiteNormalization", "per-frame dynamic level with static fallback")
                put("demosaic", "deterministic bilinear")
                put("output", "uncompressed 16-bit linear RGB TIFF")
                put("whiteBalanceApplied", false)
                put("gammaApplied", false)
                put("toneMapApplied", false)
                put("noiseReductionApplied", false)
                put("sharpeningApplied", false)
                put("lensShadingCorrectionApplied", false)
                put("distortionCorrectionApplied", false)
                put("hotPixelCorrectionApplied", false)
            })
            put("rawValidationAnalysis", JSONObject().apply {
                put("source", "same preserved RAW_SENSOR exposure")
                put("neutralGainsRgb", JSONArray(rawArtifact.validationNeutralGains.toList()))
                put("neutralSource", "SENSOR_NEUTRAL_COLOR_POINT with unity fallback")
                put("transferFunction", "sRGB display encoding")
                put("purpose", "fiducial color classification and exact capture validation")
                put("appliedToRawMaster", false)
                put("appliedToWorkingTiff", false)
            })
            put("intrinsics", JSONArray(profile.intrinsicCalibration ?: emptyList<Float>()))
            put("rawSensorIntrinsics", JSONArray(
                profile.rawIntrinsicCalibration ?: emptyList<Float>()
            ))
            put("distortion", JSONArray(profile.rawDistortion ?: emptyList<Float>()))
            put("distortionModel", "opencv_k1_k2_p1_p2_k3")
            put("workingImageWidth", rawArtifact.width)
            put("workingImageHeight", rawArtifact.height)
            put("previewWidth", profile.yuvSize.width)
            put("previewHeight", profile.yuvSize.height)
            put("rawValidationWidth", rawArtifact.validationFrame.width)
            put("rawValidationHeight", rawArtifact.validationFrame.height)
            put("rawValidationIntrinsics", JSONArray(
                rawImageGeometry.scaledTo(android.util.Size(
                    rawArtifact.validationFrame.width,
                    rawArtifact.validationFrame.height,
                )).toList()
            ))
            put("yuvUsedByAcceptedFrame", false)
            put("effectiveControlAudit", controlAudit)
            put("phoneSensor", JSONObject().apply {
                put("exposureTimestampNs", phoneSensor.exposureTimestampNs)
                put("beforeTimestampNs", phoneSensor.beforeTimestampNs)
                put("afterTimestampNs", phoneSensor.afterTimestampNs)
                put("interpolationFraction", phoneSensor.interpolationFraction)
                put("azimuthDegrees", phoneSensor.azimuthDegrees)
                put("pitchDegrees", phoneSensor.pitchDegrees)
                put("rollDegrees", phoneSensor.rollDegrees)
                put("quaternion", JSONArray(listOf(
                    phoneSensor.quaternionX,
                    phoneSensor.quaternionY,
                    phoneSensor.quaternionZ,
                    phoneSensor.quaternionW,
                )))
            })
            put("poseSource", "capture_raw_derived_exact_unsmoothed")
            put("poseTimestampNs", pose.timestampNs)
            put("cameraCenterMm", JSONArray(listOf(
                pose.cameraCenterMm.x,
                pose.cameraCenterMm.y,
                pose.cameraCenterMm.z,
            )))
            put("rotationVector", JSONArray(pose.rotationVector.toList()))
            put("translationVector", JSONArray(pose.translationVector.toList()))
            put("reprojectionRmsePx", pose.reprojectionRmsePx)
            put("workingIntrinsics", JSONArray(rawIntrinsics.toList()))
            put("posePreviewIntrinsics", JSONArray(listOf(
                pose.cameraFx, pose.cameraFy, pose.cameraCx, pose.cameraCy,
            )))
            put("fiducials", JSONArray(pose.fiducials.map { observation ->
                JSONObject().apply {
                    put("id", observation.id.name)
                    put("u", observation.pixel.x)
                    put("v", observation.pixel.y)
                    val rawPixel = rawImageGeometry.validationToRaw(
                        observation.pixel.x,
                        observation.pixel.y,
                        rawArtifact.validationFrame.let { android.util.Size(it.width, it.height) },
                    )
                    put("workingU", rawPixel.first)
                    put("workingV", rawPixel.second)
                    put("quadrantCode", JSONArray(
                        observation.code.quadrants.map { it.name }
                    ))
                    put("codeConfidence", observation.codeConfidence)
                    put("codeQuarterTurn", observation.codeQuarterTurn)
                }
            }))
        }
    }

    private fun depthOfFieldCalibrationJson(value: DepthOfFieldCalibration) = JSONObject().apply {
        put("method", "raw_registered_equal_gray_step_edge_spread")
        put("focusDistanceDiopters", value.focusDistanceDiopters.toDouble())
        put("candidateCount", value.candidateCount)
        put("detectedBoundaryCount", value.detectedBoundaryCount)
        put("sampleCount", value.sampleCount)
        put("referenceMedianEdgeWidthCanonicalPx", value.referenceMedianEdgeWidthCanonicalPx)
        put("referenceP90EdgeWidthCanonicalPx", value.referenceP90EdgeWidthCanonicalPx)
        put("maximumP90EdgeWidthCanonicalPx", value.maximumP90EdgeWidthCanonicalPx)
        put("nearReferenceDepthMm", value.nearReferenceDepthMm)
        put("farReferenceDepthMm", value.farReferenceDepthMm)
        put("medianStepContrast", value.medianStepContrast)
        put("referenceEqualStepConsistency", value.referenceEqualStepConsistency)
    }

    private fun steppedGrayFocusJson(value: GrayStepFocusAnalysis) = JSONObject().apply {
        put("analysisSource", "bounded_linear_raw_green_no_gamma_or_sharpening")
        put("detectedBoundaryCount", value.detectedBoundaryCount)
        put("sampleCount", value.sampleCount)
        put("medianEdgeWidthCanonicalPx", value.medianEdgeWidthCanonicalPx)
        put("percentile90EdgeWidthCanonicalPx", value.percentile90EdgeWidthCanonicalPx)
        put("nearSampleDepthMm", value.nearSampleDepthMm)
        put("farSampleDepthMm", value.farSampleDepthMm)
        put("sampledDepthSpanMm", value.sampledDepthSpanMm)
        put("medianStepContrast", value.medianStepContrast)
        put("equalStepConsistency", value.equalStepConsistency)
        put("sessionScore", manualSettings?.depthOfFieldCalibration?.score(value) ?: 0.0)
    }

    private fun scaledRawIntrinsics(width: Int, height: Int): DoubleArray {
        require(width == rawImageGeometry.imageWidth &&
            height == rawImageGeometry.imageHeight)
        return rawImageGeometry.intrinsics.copyOf()
    }

    private fun colorTransformJson(transform: ColorSpaceTransform): JSONArray =
        JSONArray(buildList {
            for (row in 0..2) for (column in 0..2) {
                val value = transform.getElement(column, row)
                add(value.numerator.toDouble() / value.denominator)
            }
        })

    private fun analyzeRegisteredFrame(
        frame: FramePlanes,
        geometryPose: MatPoseResult,
        guidancePose: MatPoseResult,
    ) {
        val geometryEngine = incrementalGeometryEngine ?: return
        val gray = frame.grayMat()
        try {
            negativeSpaceFrameCount++
            val geometry = geometryEngine.integrate(
                gray,
                geometryPose,
                carveHull = negativeSpaceFrameCount % 12 == 0,
            ) ?: return
            geometry.use {
                val negative = it.negativeSpace
                val center = artifactCenter(negative.canonicalForeground)
                val imageCenter = artifactImageCenter(negative.imageForeground)
                val displayRotation = sensorToDisplayRotationDegrees()
                val verticalCenterPx = imageCenter?.let {
                    displayVerticalCoordinate(it.x, it.y, frame.width, frame.height, displayRotation)
                }
                val principalPointVerticalPx = displayVerticalCoordinate(
                    guidancePose.cameraCx,
                    guidancePose.cameraCy,
                    frame.width,
                    frame.height,
                    displayRotation,
                )
                val foregroundPixels = Core.countNonZero(negative.canonicalForeground)
                val visible = Core.countNonZero(negative.canonicalVisibleMat).toDouble() /
                    (negative.canonicalVisibleMat.rows().toDouble() * negative.canonicalVisibleMat.cols())
                val containment = matVisibilityEvaluator.evaluate(
                    geometryPose,
                    frame.width,
                    frame.height,
                )
                val sharpness = sharpnessVariance(gray, negative.imageForeground)
                val focusAnalysis = steppedGrayFocusEvaluator.evaluate(
                    gray,
                    geometryPose,
                    negative.canonicalVisibleMat,
                )
                val depthOfFieldScore = manualSettings?.depthOfFieldCalibration
                    ?.score(focusAnalysis) ?: 0.0
                val radiometric = exp(
                    -1.25 * abs(ln(negative.radiometricScale.coerceAtLeast(1e-6))) -
                        abs(negative.radiometricOffset) / 96.0 -
                        negative.radiometricMeanAbsoluteError / 30.0
                ).coerceIn(0.0, 1.0)
                listener.onVisionAnalysis(
                    VisionFrameAnalysis(
                        frame.timestampNs,
                        guidancePose,
                        center,
                        verticalCenterPx,
                        principalPointVerticalPx,
                        if (displayRotation == 90 || displayRotation == 270) frame.width else frame.height,
                        negative.foregroundFraction,
                        visible,
                        containment.fullMatVisible,
                        containment.minimumFrameMarginPx,
                        sharpness,
                        focusAnalysis,
                        depthOfFieldScore,
                        radiometric,
                        negative.highlightClippedFraction,
                        artifactCenterConfidence = if (center == null) 0.0 else
                            (foregroundPixels / 800.0).coerceIn(0.0, 1.0),
                        templateReady = true,
                        candidateEnvelope = it.candidateEnvelope,
                    )
                )
            }
        } finally {
            gray.release()
        }
    }

    private fun installCanonicalTemplate(template: Mat, pixelsPerMm: Double) {
        canonicalTemplate?.release()
        canonicalTemplate = template
        incrementalGeometryEngine = IncrementalGeometryEngine(
            negativeSpaceExtractor = NegativeSpaceExtractor(
                template,
                pixelsPerMm,
                minimumXmm = -75.0,
                maximumYmm = 75.0,
                artifactFieldRadiusMm = 45.0,
            ),
            reconstructionBounds = Aabb(
                Vec3(-55.0, -55.0, 0.0),
                Vec3(55.0, 55.0, 100.0),
            ),
            maximumOctreeDepth = 5,
            distortion = profile.distortion?.map(Float::toDouble)?.toDoubleArray()
                ?: DoubleArray(5),
        )
    }

    private fun artifactCenter(canonicalForeground: Mat): Vec3? {
        val moments = org.opencv.geometry.Geometry.moments(canonicalForeground, true)
        if (moments.m00 < 100.0) return null
        val xPixel = moments.m10 / moments.m00
        val yPixel = moments.m01 / moments.m00
        return Vec3(xPixel / 4.0 - 75.0, 75.0 - yPixel / 4.0, 0.0)
    }

    private fun artifactImageCenter(imageForeground: Mat): Point? {
        val points = MatOfPoint()
        return try {
            Core.findNonZero(imageForeground, points)
            if (points.empty()) null else {
                val bounds = org.opencv.geometry.Geometry.boundingRect(points)
                Point(bounds.x + bounds.width * 0.5, bounds.y + bounds.height * 0.5)
            }
        } finally {
            points.release()
        }
    }

    private fun sensorToDisplayRotationDegrees(): Int {
        val displayDegrees = when (context.display?.rotation ?: Surface.ROTATION_0) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        return (profile.sensorOrientationDegrees - displayDegrees + 360) % 360
    }

    private fun addLiveTargets(builder: CaptureRequest.Builder) {
        builder.addTarget(previewSurface)
        builder.addTarget(yuvSurface)
    }

    private companion object {
        const val LIVE_VISION_INTERVAL_NS = 100_000_000L
    }

    private fun displayVerticalCoordinate(
        x: Double,
        y: Double,
        sensorWidth: Int,
        sensorHeight: Int,
        rotationDegrees: Int,
    ): Double = when (rotationDegrees) {
        90 -> x
        180 -> sensorHeight - 1.0 - y
        270 -> sensorWidth - 1.0 - x
        else -> y
    }

    private fun sharpnessVariance(gray: Mat, mask: Mat): Double {
        val laplacian = Mat()
        val mean = MatOfDouble()
        val deviation = MatOfDouble()
        return try {
            Imgproc.Laplacian(gray, laplacian, CvType.CV_64F)
            Core.meanStdDev(laplacian, mean, deviation, mask)
            val sigma = deviation.toArray().firstOrNull() ?: 0.0
            sigma * sigma
        } finally {
            laplacian.release(); mean.release(); deviation.release()
        }
    }

    private fun trimUnpairedBuffers() {
        while (pendingRawImages.size > 2) pendingRawImages.pollFirstEntry()?.value?.close()
        while (pendingCaptureResults.size > 4) {
            val timestamp = pendingCaptureResults.firstKey()
            pendingCaptureResults.remove(timestamp)
            pendingCapturePurposes.remove(timestamp)
            pendingFocusCalibrationSettings.remove(timestamp)
        }
    }

    override fun close() {
        if (!closed.compareAndSet(false, true)) return
        isReady = false
        stillCaptureInFlight.set(false)
        focusCalibrationState = null
        cameraHandler.post {
            try {
                captureSession?.stopRepeating()
                captureSession?.close()
            } catch (exception: Exception) {
                // Session teardown continues.
            }
            try {
                cameraDevice?.close()
            } catch (exception: Exception) {
                // Ignore.
            }
            captureSession = null
            cameraDevice = null
            pendingRawImages.values.forEach { try { it.close() } catch (_: Exception) {} }
            pendingRawImages.clear()
            pendingCaptureResults.clear()
            pendingCapturePurposes.clear()
            pendingFocusCalibrationSettings.clear()
            rawProcessingScheduled.clear()
            discardedCaptureTimestamps.clear()
            try { yuvReader.close() } catch (_: Exception) {}
            try { rawReader.close() } catch (_: Exception) {}
            cameraThread.quitSafely()
        }
        visionHandler.post {
            matPoseProcessor?.close()
            rawMatPoseProcessor?.close()
            templateCalibrator?.close(); templateCalibrator = null
            canonicalTemplate?.release(); canonicalTemplate = null
            incrementalGeometryEngine = null
            visionThread.quitSafely()
        }
    }
}

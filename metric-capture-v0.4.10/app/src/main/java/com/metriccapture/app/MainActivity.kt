package com.metriccapture.app

import android.Manifest
import android.graphics.Matrix
import android.graphics.SurfaceTexture
import android.content.pm.PackageManager
import android.os.Bundle
import android.content.Intent
import android.app.AlertDialog
import android.text.InputType
import android.widget.Button
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.LinearLayout
import android.widget.TextView
import android.widget.ArrayAdapter
import android.widget.Spinner
import android.view.Surface
import android.view.TextureView
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.metriccapture.app.camera.CameraQualifier
import com.metriccapture.app.camera.CameraSessionController
import com.metriccapture.app.camera.CaptureProfile
import com.metriccapture.app.camera.DeviceQualification
import com.metriccapture.app.compute.AndroidResourceProbe
import com.metriccapture.app.compute.ComputePermission
import com.metriccapture.app.compute.ReconstructionStage
import com.metriccapture.app.compute.ResourceGovernor
import com.metriccapture.app.compute.StageEstimate
import com.metriccapture.app.geometry.Vec3
import com.metriccapture.app.guidance.CaptureDecision
import com.metriccapture.app.guidance.AcquisitionMode
import com.metriccapture.app.guidance.GuidedCaptureCoordinator
import com.metriccapture.app.guidance.GuidedCaptureStatus
import com.metriccapture.app.guidance.GateFailure
import com.metriccapture.app.guidance.OrbitState
import android.view.ViewGroup
import android.widget.ScrollView
import android.content.res.Configuration
import com.metriccapture.app.reconstruction.Aabb
import com.metriccapture.app.reconstruction.CaptureSessionStore
import com.metriccapture.app.reconstruction.OnDeviceReconstructionEngine
import com.metriccapture.app.reconstruction.ProjectSnapshot
import com.metriccapture.app.reconstruction.ReconstructionOutcome
import com.metriccapture.app.reconstruction.ReconstructionProgress
import com.metriccapture.app.reconstruction.SessionCaptureFiles
import com.metriccapture.app.sensors.PhoneSensorTracker
import com.metriccapture.app.vision.MatPoseResult
import com.metriccapture.app.vision.TemplateCalibrationProgress
import com.metriccapture.app.vision.VisionFrameAnalysis
import java.io.File
import java.util.concurrent.Executors
import org.opencv.android.OpenCVLoader

class MainActivity : ComponentActivity() {
    private lateinit var status: TextView
    private lateinit var preview: TextureView
    private lateinit var reticleOverlay: ReticleOverlayView
    private lateinit var settingsButton: Button
    private lateinit var calibrateButton: Button
    private lateinit var startButton: Button
    private lateinit var modeSpinner: Spinner
    private lateinit var reconstructButton: Button
    private lateinit var exportButton: Button
    private var cameraController: CameraSessionController? = null
    private var coordinator: GuidedCaptureCoordinator? = null
    private lateinit var sessionStore: CaptureSessionStore
    private lateinit var phoneSensors: PhoneSensorTracker
    private val background = Executors.newSingleThreadExecutor()
    private var collectionStarted = false
    private var templateReady = false
    private var cameraLocked = false
    private var awaitingSettingsReview = false
    private var completedPly: File? = null
    private var restoredProject: ProjectSnapshot? = null
    private var previewSurface: Surface? = null
    private var pendingCameraStart: Pair<CaptureProfile, File?>? = null
    private var activeProfile: CaptureProfile? = null
    private var activeSessionDirectory: File? = null
    private var activityDestroying = false
    @Volatile private var lastAnalysis: VisionFrameAnalysis? = null

    private val permissionRequest = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) qualifyCamera() else status.text = getString(R.string.camera_permission_required)
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        if (!OpenCVLoader.initLocal()) {
            setContentView(TextView(this).apply { text = getString(R.string.opencv_initialization_failed) })
            return
        }
        phoneSensors = PhoneSensorTracker(this).also(PhoneSensorTracker::start)
        status = TextView(this).apply { textSize = 17f; text = getString(R.string.checking_camera_capabilities) }
        preview = TextureView(this).apply {
            surfaceTextureListener = object : TextureView.SurfaceTextureListener {
                override fun onSurfaceTextureAvailable(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int,
                ) {
                    pendingCameraStart?.let { (profile, directory) ->
                        startCamera(profile, directory)
                    }
                }

                override fun onSurfaceTextureSizeChanged(
                    surface: SurfaceTexture,
                    width: Int,
                    height: Int,
                ) {
                    activeProfile?.let(::configurePreviewTransform)
                }

                override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean {
                    if (isFinishing) {
                        val controller = cameraController
                        cameraController = null
                        controller?.close()
                        previewSurface?.release()
                        previewSurface = null
                    }
                    return true
                }

                override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
            }
        }
        reticleOverlay = ReticleOverlayView(this)
        settingsButton = Button(this).apply {
            text = getString(R.string.meter_and_review_settings)
            isEnabled = false
            setOnClickListener {
                awaitingSettingsReview = true
                isEnabled = false
                status.text = getString(R.string.metering_position_instruction)
                cameraController?.refreshSettingsSuggestion()
            }
        }
        calibrateButton = Button(this).apply {
            text = getString(R.string.calibrate_empty_mat)
            isEnabled = false
            setOnClickListener {
                isEnabled = false
                status.text = getString(R.string.empty_mat_instruction)
                cameraController?.beginEmptyMatCalibration()
            }
        }
        modeSpinner = Spinner(this).apply {
            adapter = ArrayAdapter(
                this@MainActivity,
                android.R.layout.simple_spinner_dropdown_item,
                listOf("Camera moves around fixed mat", "Turntable rotates mat and artifact"),
            )
        }
        startButton = Button(this).apply {
            text = getString(R.string.establish_guidance_zero)
            isEnabled = false
            setOnClickListener {
                val analysis = lastAnalysis
                if (analysis == null || analysis.artifactCenterMm == null) {
                    status.text = getString(R.string.align_artifact_center)
                    return@setOnClickListener
                }
                val verticalCenter = analysis.artifactVerticalCenterPx
                if (verticalCenter == null || kotlin.math.abs(
                        verticalCenter - analysis.principalPointVerticalPx
                    ) > analysis.displayFrameHeightPx * 0.04
                ) {
                    status.text = getString(R.string.move_phone_vertically)
                    return@setOnClickListener
                }
                if (!analysis.fullMatVisible || analysis.minimumMatFrameMarginPx < 12.0) {
                    status.text = getString(R.string.show_entire_mat)
                    return@setOnClickListener
                }
                if (!phoneSensors.establishUserGuidanceBaseline()) {
                    status.text = getString(R.string.phone_orientation_unavailable)
                    return@setOnClickListener
                }
                val mode = if (modeSpinner.selectedItemPosition == 0) {
                    AcquisitionMode.CAMERA_ORBIT
                } else {
                    AcquisitionMode.TURNTABLE
                }
                sessionStore.setAcquisitionMode(mode)
                sessionStore.autosaveGuidanceZero(mode)
                collectionStarted = true
                isEnabled = false
                modeSpinner.isEnabled = false
                reticleOverlay.showReticle = false
                coordinator?.startAtUserGuidanceZero(mode, sessionStore.captureCount)
                status.text = getString(R.string.guidance_zero_locked)
            }
        }
        reconstructButton = Button(this).apply {
            text = getString(R.string.reconstruct_metric_ply)
            isEnabled = false
            setOnClickListener { runReconstruction() }
        }
        exportButton = Button(this).apply {
            text = getString(R.string.share_ply)
            isEnabled = false
            setOnClickListener { shareCompletedPly() }
        }
        setupResponsiveLayout()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) ==
            PackageManager.PERMISSION_GRANTED
        ) qualifyCamera() else permissionRequest.launch(Manifest.permission.CAMERA)
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        setupResponsiveLayout()
        activeProfile?.let(::configurePreviewTransform)
    }

    private fun setupResponsiveLayout() {
        val isLandscape = resources.configuration.orientation == Configuration.ORIENTATION_LANDSCAPE
        
        val views = listOf(
            status, settingsButton, calibrateButton, modeSpinner, 
            startButton, reconstructButton, exportButton
        )
        
        // Ensure views are detached from previous parents
        views.forEach { (it.parent as? ViewGroup)?.removeView(it) }
        
        // We wrap the preview and reticle in a stable container so we don't have to 
        // remove/re-add the actual TextureView, which avoids surface destruction.
        val cameraContainer = (preview.parent as? ViewGroup) ?: FrameLayout(this).apply {
            addView(preview, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ))
            addView(reticleOverlay, FrameLayout.LayoutParams(
                FrameLayout.LayoutParams.MATCH_PARENT,
                FrameLayout.LayoutParams.MATCH_PARENT,
            ))
        }
        (cameraContainer.parent as? ViewGroup)?.removeView(cameraContainer)

        val controlsLayout = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(20, 0, 20, 0)
            views.forEach { addView(it) }
        }

        val rootLayout = LinearLayout(this).apply {
            setPadding(20, 20, 20, 20)
            if (isLandscape) {
                orientation = LinearLayout.HORIZONTAL
                addView(cameraContainer, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 0.7f))
                addView(ScrollView(this@MainActivity).apply {
                    addView(controlsLayout)
                }, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.MATCH_PARENT, 0.3f))
            } else {
                orientation = LinearLayout.VERTICAL
                // Force the camera frame to be landscape (4:3 ratio) in portrait mode
                val screenWidth = resources.displayMetrics.widthPixels
                val padding = 40 
                val cameraWidth = screenWidth - padding
                val cameraHeight = (cameraWidth * 0.75f).toInt() 
                
                addView(cameraContainer, LinearLayout.LayoutParams(cameraWidth, cameraHeight))
                addView(ScrollView(this@MainActivity).apply {
                    addView(controlsLayout)
                }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f))
            }
        }
        setContentView(rootLayout)
    }

    private fun qualifyCamera() {
        when (val result = CameraQualifier(this).qualify()) {
            is DeviceQualification.Supported -> chooseProject(result.profile)
            is DeviceQualification.Degraded -> chooseProject(result.profile)
            is DeviceQualification.Unsupported -> status.text = buildString {
                appendLine("This device cannot run the metric capture pipeline:")
                result.reasons.forEach { appendLine("• $it") }
            }
        }
    }

    private fun chooseProject(profile: CaptureProfile) {
        val sessions = File(getExternalFilesDir(null), "sessions").apply { mkdirs() }
        val latest = sessions.listFiles()
            .orEmpty()
            .filter { File(it, "session.json").isFile }
            .sortedByDescending(File::lastModified)
            .firstNotNullOfOrNull { directory ->
                try {
                    CaptureSessionStore(directory).projectSnapshot().let { directory to it }
                } catch (_: Exception) {
                    null
                }
            }
        if (latest == null) {
            startCamera(profile, null)
            return
        }
        val (directory, snapshot) = latest
        AlertDialog.Builder(this)
            .setTitle("Continue saved project?")
            .setMessage(
                "${snapshot.committedCaptureCount}/36 captures saved; " +
                    "stage ${snapshot.milestone.name.lowercase().replace('_', ' ')}."
            )
            .setPositiveButton("Continue") { _, _ -> startCamera(profile, directory) }
            .setNegativeButton("New project") { _, _ -> startCamera(profile, null) }
            .setCancelable(false)
            .show()
    }

    private fun startCamera(profile: CaptureProfile, existingDirectory: File?) {
        if (!preview.isAvailable) {
            pendingCameraStart = profile to existingDirectory
            status.text = "Waiting for the live camera surface…"
            return
        }
        pendingCameraStart = null
        val sessionDirectory = existingDirectory ?: File(
            getExternalFilesDir(null),
            "sessions/active-${System.currentTimeMillis()}",
        )
        activeProfile = profile
        activeSessionDirectory = sessionDirectory
        val surfaceTexture = requireNotNull(preview.surfaceTexture)
        surfaceTexture.setDefaultBufferSize(
            profile.previewSize.width,
            profile.previewSize.height,
        )
        configurePreviewTransform(profile)
        previewSurface?.release()
        previewSurface = Surface(surfaceTexture)
        sessionStore = CaptureSessionStore(sessionDirectory)
        restoredProject = existingDirectory?.let { sessionStore.projectSnapshot() }
        restoredProject?.acquisitionMode?.let { mode ->
            modeSpinner.setSelection(if (mode == AcquisitionMode.CAMERA_ORBIT) 0 else 1)
        }
        completedPly = restoredProject?.completedPly?.takeIf(File::isFile)
        val captureProbe = AndroidResourceProbe(this, sessionDirectory)
        val governor = ResourceGovernor()
        coordinator = GuidedCaptureCoordinator(
            triggerCapture = {
                val rawPixels = profile.rawSize.width.toLong() * profile.rawSize.height
                val durableCaptureBytes = rawPixels * 8L // RAW16 plus 16-bit RGB TIFF.
                val permission = governor.permission(
                    captureProbe.snapshot(),
                    ReconstructionStage.WORKING_IMAGE_GENERATION,
                    StageEstimate(96_000_000L, durableCaptureBytes + 32_000_000L, false),
                )
                if (permission is ComputePermission.Pause) {
                    runOnUiThread {
                        status.text = getString(
                            R.string.capture_paused,
                            permission.reasons.joinToString(),
                        )
                    }
                    false
                } else cameraController?.captureStill() == true
            },
            userSensorPose = phoneSensors::userRelative,
            statusListener = { updateGuidance(it) },
            distortion = profile.distortion?.map(Float::toDouble)?.toDoubleArray()
                ?: DoubleArray(5),
        )
        status.text = getString(R.string.opening_camera, profile.cameraId)
        cameraController?.close()
        cameraController = CameraSessionController(
            context = this,
            profile = profile,
            previewSurface = requireNotNull(previewSurface),
            outputDirectory = File(sessionDirectory, "captures"),
            phoneSensorSampleForTimestamp = { timestamp -> phoneSensors.sampleAt(timestamp) },
            listener = object : CameraSessionController.Listener {
                override fun onSettingsSuggested(
                    suggested: CameraSessionController.ManualSettings,
                    apertureIsFixed: Boolean,
                ) {
                    runOnUiThread {
                        if (awaitingSettingsReview) {
                            awaitingSettingsReview = false
                            showSettingsSuggestion(suggested, apertureIsFixed)
                        } else {
                            status.text = getString(R.string.place_and_meter)
                            settingsButton.isEnabled = true
                        }
                    }
                }

                override fun onFocusCalibrationProgress(
                    completedCandidates: Int,
                    totalCandidates: Int,
                ) {
                    runOnUiThread {
                        status.text = getString(
                            R.string.focus_calibration_progress,
                            completedCandidates,
                            totalCandidates,
                        )
                    }
                }

                override fun onSessionReady(applied: CameraSessionController.ManualSettings) {
                    sessionStore.autosaveCameraLocked(
                        cameraId = profile.cameraId,
                        sensitivityIso = applied.sensitivityIso,
                        exposureTimeNs = applied.exposureTimeNs,
                        focusDistanceDiopters = applied.focusDistanceDiopters,
                        aperture = applied.aperture,
                        depthOfFieldCalibration = applied.depthOfFieldCalibration,
                    )
                    runOnUiThread {
                        cameraLocked = true
                        val savedTemplate = restoredProject?.emptyMatTemplate?.takeIf(File::isFile)
                        status.text = buildString {
                            appendLine("Camera locked: efficient YUV guidance + maximum-resolution RAW capture")
                            appendLine("ISO ${applied.sensitivityIso}; exposure %.1f ms".format(
                                applied.exposureTimeNs / 1_000_000.0
                            ))
                            appendLine("Focus %.3f D; aperture %s".format(
                                applied.focusDistanceDiopters,
                                applied.aperture?.let { "f/%.1f".format(it) } ?: "fixed/unreported",
                            ))
                            applied.depthOfFieldCalibration?.let {
                                appendLine(
                                    "Gray-step DOF: ${it.sampleCount} samples across " +
                                        "%.1f mm at %.0f%% step consistency; locked for both camera views".format(
                                            it.farReferenceDepthMm - it.nearReferenceDepthMm,
                                            it.referenceEqualStepConsistency * 100.0,
                                        )
                                )
                            }
                            append(
                                if (savedTemplate != null) "Restoring the saved empty-mat reference…"
                                else "Remove the artifact, show the full mat, then calibrate."
                            )
                        }
                        calibrateButton.isEnabled = savedTemplate == null
                        settingsButton.isEnabled = false
                        if (savedTemplate != null) {
                            cameraController?.restoreEmptyMatTemplate(savedTemplate)
                        }
                    }
                }

                override fun onYuvFrame(frame: CameraSessionController.YuvFrameSummary) = Unit

                override fun onMatPose(pose: MatPoseResult) {
                    if (collectionStarted || !cameraLocked) return
                    runOnUiThread {
                        status.text = if (templateReady) {
                            "At printed 0°: align vertically with the artifact center, then back up until the whole mat is visible."
                        } else {
                            "Mat locked (%.2f px RMSE). Ready for empty-mat calibration."
                                .format(pose.reprojectionRmsePx)
                        }
                    }
                }

                override fun onTemplateCalibration(progress: TemplateCalibrationProgress) {
                    runOnUiThread {
                        status.text = getString(
                            R.string.empty_mat_progress,
                            progress.acceptedFrames,
                            progress.requiredFrames,
                        )
                        if (progress.completedTemplate != null) {
                            sessionStore.autosaveMatCalibration(
                                File(sessionStore.directory, "captures/empty-mat-template.png")
                            )
                            templateReady = true
                            status.text = getString(R.string.reference_locked)
                            startButton.isEnabled = true
                        }
                    }
                }

                override fun onTemplateRestored(file: File) {
                    runOnUiThread {
                        templateReady = true
                        val count = sessionStore.captureCount
                        if (count >= 36) {
                            status.text = getString(R.string.saved_capture_set_restored)
                            reconstructButton.isEnabled = true
                            exportButton.isEnabled = completedPly?.isFile == true
                        } else {
                            status.text = getString(R.string.saved_project_restored, count)
                            startButton.isEnabled = true
                        }
                    }
                }

                override fun onVisionAnalysis(analysis: VisionFrameAnalysis) {
                    lastAnalysis = analysis
                    if (collectionStarted) {
                        coordinator?.onAnalysis(analysis)
                    } else if (templateReady) {
                        runOnUiThread { updateSetupGuidance(analysis) }
                    }
                }

                override fun onCaptureSaved(
                    rawMasterFile: File,
                    workingImageFile: File,
                    foregroundMaskFile: File?,
                    metadataFile: File,
                ) {
                    val target = coordinator?.pendingCaptureTarget
                    val measurement = coordinator?.pendingCaptureMeasurement
                    if (target == null || measurement == null) {
                        coordinator?.onCaptureFailed()
                        onError("A frame arrived without an active guided target.")
                        return
                    }
                    try {
                        sessionStore.record(
                            target,
                            measurement,
                            SessionCaptureFiles(
                                rawMasterFile,
                                workingImageFile,
                                foregroundMaskFile,
                                metadataFile,
                            ),
                        )
                        coordinator?.onCaptureSaved()
                        if (coordinator?.isComplete == true) runOnUiThread {
                            reconstructButton.isEnabled = true
                        }
                    } catch (exception: Exception) {
                        coordinator?.onCaptureFailed()
                        onError("The accepted frame could not be committed.", exception)
                    }
                }

                override fun onError(message: String, cause: Throwable?) {
                    coordinator?.onCaptureFailed()
                    runOnUiThread {
                        if (!cameraLocked) {
                            awaitingSettingsReview = false
                            settingsButton.isEnabled = true
                        }
                        if (cameraLocked && !templateReady && !collectionStarted) {
                            calibrateButton.isEnabled = true
                        }
                        status.text = buildString {
                            append(message)
                            cause?.message?.let { append("\n$it") }
                        }
                    }
                }
            },
        ).also(CameraSessionController::start)
    }

    private fun updateSetupGuidance(analysis: VisionFrameAnalysis) {
        val artifactVerticalCenter = analysis.artifactVerticalCenterPx
        if (artifactVerticalCenter == null) {
            status.text = getString(R.string.place_artifact_keep_mat_visible)
            return
        }
        val verticalError = artifactVerticalCenter - analysis.principalPointVerticalPx
        val alignmentTolerancePx = analysis.displayFrameHeightPx * 0.04
        status.text = when {
            verticalError > alignmentTolerancePx ->
                "At printed 0°: move the phone down without changing its tilt."
            verticalError < -alignmentTolerancePx ->
                "At printed 0°: move the phone up without changing its tilt."
            !analysis.fullMatVisible || analysis.minimumMatFrameMarginPx < 12.0 ->
                "Vertical center aligned. Back up until the whole mat has a visible border."
            else ->
                "Setup aligned and whole mat visible. Press Establish user guidance zero."
        }
    }

    private fun showSettingsSuggestion(
        suggested: CameraSessionController.ManualSettings,
        apertureIsFixed: Boolean,
    ) {
        val apertureText = suggested.aperture?.let { "f/%.1f".format(it) } ?: "unreported"
        AlertDialog.Builder(this)
            .setTitle("Lock capture settings")
            .setMessage(buildString {
                appendLine("Live-metered recommendation:")
                appendLine("ISO ${suggested.sensitivityIso}")
                appendLine("Exposure %.2f ms".format(suggested.exposureTimeNs / 1_000_000.0))
                appendLine("Focus %.3f diopters".format(suggested.focusDistanceDiopters))
                suggested.depthOfFieldCalibration?.let {
                    appendLine(
                        "Stepped-gray DOF: ${it.sampleCount} RAW edge samples, " +
                            "%.1f mm measured depth span, %.0f%% equal-step consistency".format(
                                it.farReferenceDepthMm - it.nearReferenceDepthMm,
                                it.referenceEqualStepConsistency * 100.0,
                            )
                    )
                }
                append("Aperture $apertureText")
                if (apertureIsFixed) append(" (fixed by this camera)")
            })
            .setPositiveButton("Accept") { _, _ ->
                cameraController?.lockManualSettings(suggested)
            }
            .setNeutralButton("Override") { _, _ ->
                showManualOverrideDialog(suggested, apertureIsFixed)
            }
            .setCancelable(false)
            .show()
    }

    private fun showManualOverrideDialog(
        suggested: CameraSessionController.ManualSettings,
        apertureIsFixed: Boolean,
    ) {
        fun field(label: String, value: String): EditText = EditText(this).apply {
            hint = label
            setText(value)
            inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
        }
        val iso = field("ISO", suggested.sensitivityIso.toString())
        val exposure = field(
            "Exposure milliseconds",
            "%.3f".format(suggested.exposureTimeNs / 1_000_000.0),
        )
        val focus = field("Focus diopters", suggested.focusDistanceDiopters.toString())
        val aperture = field("Aperture f-number", suggested.aperture?.toString().orEmpty()).apply {
            isEnabled = !apertureIsFixed
        }
        val form = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(32, 8, 32, 8)
            addView(iso); addView(exposure); addView(focus); addView(aperture)
        }
        AlertDialog.Builder(this)
            .setTitle("Override recommendation")
            .setView(form)
            .setPositiveButton("Lock") { _, _ ->
                val exposureNs = ((exposure.text.toString().toDoubleOrNull()
                    ?: suggested.exposureTimeNs / 1_000_000.0) * 1_000_000.0).toLong()
                val overridden = suggested.copy(
                        sensitivityIso = iso.text.toString().toIntOrNull()
                            ?: suggested.sensitivityIso,
                        exposureTimeNs = exposureNs,
                        frameDurationNs = maxOf(33_333_333L, exposureNs + 1_000_000L),
                        focusDistanceDiopters = focus.text.toString().toFloatOrNull()
                            ?: suggested.focusDistanceDiopters,
                        aperture = if (apertureIsFixed) suggested.aperture else
                            aperture.text.toString().toFloatOrNull() ?: suggested.aperture,
                        depthOfFieldCalibration = null,
                    )
                awaitingSettingsReview = true
                status.text = getString(R.string.validating_override)
                cameraController?.calibrateManualSettings(overridden)
            }
            .setNegativeButton("Use recommendation") { _, _ ->
                cameraController?.lockManualSettings(suggested)
            }
            .setCancelable(false)
            .show()
    }

    private fun updateGuidance(guidance: GuidedCaptureStatus) = runOnUiThread {
        reconstructButton.isEnabled = guidance.state is OrbitState.Complete
        status.text = when (guidance.state) {
            is OrbitState.AwaitingUserGuidanceZero ->
                "At printed 0°: align vertically with the artifact center, then back up until the full mat is visible."
            is OrbitState.Complete ->
                "Two 360° passes complete (${guidance.capturedFrames} frames). Reconstruct when ready."
            else -> buildString {
                appendLine("Captured ${guidance.capturedFrames}/${guidance.totalFrames}")
                guidance.target?.let {
                    appendLine("Target: ${it.round.name.lowercase()} pass, rotation %.0f°, position %.0f°, tilt %.0f°"
                        .format(it.azimuthDegrees, it.positionAngleDegrees, it.tiltDegrees))
                }
                appendLine(
                    if (guidance.mode == AcquisitionMode.CAMERA_ORBIT)
                        "Move the phone around the fixed mat; keep the whole mat visible."
                    else
                        "Set the requested position/tilt, then keep the phone fixed while mat and artifact rotate together."
                )
                when (val decision = guidance.decision) {
                    is CaptureDecision.Accept -> append("Quality locked — capturing automatically…")
                    is CaptureDecision.Reject -> {
                        appendLine("Adjust: rotation %+.1f°, position %+.1f°, tilt %+.1f°, roll %+.1f°"
                            .format(
                                decision.correction.azimuthDeltaDegrees,
                                decision.correction.positionAngleDeltaDegrees,
                                decision.correction.tiltDeltaDegrees,
                                decision.correction.rollDeltaDegrees,
                            ))
                        if (decision.correction.additionalMatMarginPx > 0.0) {
                            appendLine("Back up or tilt until the full mat has more border.")
                        }
                        if (decision.failures.contains(GateFailure.DEPTH_OF_FIELD_INVALID)) {
                            appendLine(
                                "Move within the locked focus range while keeping the full mat visible."
                            )
                        }
                        append("Waiting on: ${decision.failures.joinToString { it.name.lowercase() }}")
                    }
                    null -> append("Move toward the displayed target.")
                }
            }
        }
    }

    private fun runReconstruction() {
        reconstructButton.isEnabled = false
        val frames = sessionStore.frames()
        val output = File(sessionStore.directory, "artifact-metric.ply")
        // Reconstruction starts from the mat's known metric negative-space volume.
        // The live YUV visual hull may improve guidance, but never constrains the model.
        val envelope = Aabb(
            Vec3(-75.0, -75.0, 0.0), Vec3(75.0, 75.0, 150.0)
        )
        sessionStore.autosaveReconstructionStarted()
        status.text = getString(R.string.starting_reconstruction)
        background.execute {
            val probe = AndroidResourceProbe(this, sessionStore.directory)
            val engine = OnDeviceReconstructionEngine(
                ResourceGovernor(),
                resourceSnapshot = probe::snapshot,
                progress = object : ReconstructionProgress {
                    override fun onStage(stage: ReconstructionStage, completed: Int, total: Int) =
                        runOnUiThread {
                            status.text = getString(
                                R.string.reconstruction_progress,
                                stage.name.lowercase(),
                                completed,
                                total,
                            )
                        }
                },
            )
            val outcome = engine.reconstruct(frames, envelope, output)
            runOnUiThread {
                status.text = when (outcome) {
                    is ReconstructionOutcome.Complete ->
                        "PLY complete: ${outcome.pointCount} points\n${outcome.plyFile.absolutePath}".also {
                            sessionStore.autosaveReconstructionComplete(outcome.plyFile)
                            completedPly = outcome.plyFile
                            exportButton.isEnabled = true
                        }
                    is ReconstructionOutcome.Paused ->
                        "Paused safely at ${outcome.stage.name.lowercase()}: ${outcome.permission.reasons.joinToString()}".also {
                            sessionStore.autosaveReconstructionPaused(
                                outcome.stage.name,
                                outcome.permission.reasons.map { it.name },
                            )
                        }
                    is ReconstructionOutcome.Failed ->
                        "Reconstruction stopped at ${outcome.stage.name.lowercase()}: ${outcome.message}".also {
                            sessionStore.autosaveReconstructionFailed(
                                outcome.stage.name,
                                outcome.message,
                            )
                        }
                }
                reconstructButton.isEnabled = outcome !is ReconstructionOutcome.Complete
            }
        }
    }

    private fun shareCompletedPly() {
        val file = completedPly ?: return
        val uri = FileProvider.getUriForFile(this, "$packageName.files", file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply {
            type = "application/octet-stream"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }, "Share metric PLY"))
    }

    /** Keeps the complete preview field visible while rotating it into display coordinates. */
    private fun configurePreviewTransform(profile: CaptureProfile) {
        if (preview.width == 0 || preview.height == 0) {
            preview.post { configurePreviewTransform(profile) }
            return
        }
        val displayDegrees = when (display?.rotation ?: Surface.ROTATION_0) {
            Surface.ROTATION_90 -> 90
            Surface.ROTATION_180 -> 180
            Surface.ROTATION_270 -> 270
            else -> 0
        }
        
        // Use full sensor width by default, matching the physical 4:3 or 16:9 ratio.
        val sourceWidth = profile.previewSize.width.toFloat()
        val sourceHeight = profile.previewSize.height.toFloat()
        
        // We always want to show the landscape aspect of the camera.
        // If the sensor is rotated 90/270 (most phones), we rotate it back to be horizontal
        // even if the phone is held vertically.
        val rotationDegrees = (profile.sensorOrientationDegrees - displayDegrees + 360) % 360
        
        val rotatedWidth = if (rotationDegrees == 90 || rotationDegrees == 270) sourceHeight else sourceWidth
        val rotatedHeight = if (rotationDegrees == 90 || rotationDegrees == 270) sourceWidth else sourceHeight
        
        // We calculate scale to fit the WIDE camera into the current preview view.
        val scale = minOf(preview.width / rotatedWidth, preview.height / rotatedHeight)
        
        val matrix = Matrix()
        val centerX = preview.width / 2f
        val centerY = preview.height / 2f
        
        // Apply the rotation needed to make the camera output appear horizontally correct
        matrix.postRotate(rotationDegrees.toFloat(), centerX, centerY)
        matrix.postScale(scale * rotatedWidth / preview.width, scale * rotatedHeight / preview.height, centerX, centerY)
        
        preview.setTransform(matrix)
    }
    }

    override fun onDestroy() {
        activityDestroying = true
        pendingCameraStart = null
        val controller = cameraController
        cameraController = null
        if (controller != null) controller.close() else previewSurface?.release()
        previewSurface = null
        if (::phoneSensors.isInitialized) phoneSensors.close()
        background.shutdown()
        super.onDestroy()
    }
}

package com.metriccapture.app.reconstruction

import com.metriccapture.app.guidance.AcceptedFrameMeasurement
import com.metriccapture.app.guidance.AcquisitionMode
import com.metriccapture.app.guidance.TargetView
import com.metriccapture.app.vision.DepthOfFieldCalibration
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption
import kotlin.math.abs

enum class ProjectMilestone {
    CREATED,
    CAMERA_LOCKED,
    MAT_CALIBRATED,
    GUIDANCE_ZERO_ESTABLISHED,
    CAPTURING_UPPER,
    UPPER_PASS_COMPLETE,
    CAPTURING_LOWER,
    CAPTURE_COMPLETE,
    RECONSTRUCTING,
    RECONSTRUCTION_PAUSED,
    RECONSTRUCTION_FAILED,
    RECONSTRUCTION_COMPLETE,
}

data class SavedCameraSettings(
    val cameraId: String,
    val sensitivityIso: Int,
    val exposureTimeNs: Long,
    val focusDistanceDiopters: Float,
    val aperture: Float?,
    val depthOfFieldCalibration: DepthOfFieldCalibration?,
)

data class ProjectSnapshot(
    val milestone: ProjectMilestone,
    val acquisitionMode: AcquisitionMode?,
    val committedCaptureCount: Int,
    val cameraSettings: SavedCameraSettings?,
    val emptyMatTemplate: File?,
    val completedPly: File?,
    val updatedAtEpochMs: Long,
)

data class SessionCaptureFiles(
    val rawMaster: File,
    val workingImage: File,
    val foregroundMask: File?,
    val metadata: File,
)

internal data class MeasuredViewAngle(
    val azimuthDegrees: Double,
    val elevationDegrees: Double,
)

/** Durable capture manifest; every accepted guided target is reconstructable after restart. */
class CaptureSessionStore(val directory: File) {
    private val manifest = File(directory, "session.json")
    private val currentSchemaVersion = 8

    init {
        directory.mkdirs()
        if (!manifest.exists()) {
            commit(JSONObject().apply {
                put("schemaVersion", currentSchemaVersion)
                put("project", newProjectState())
                put("captures", JSONArray())
            })
        } else {
            val root = readRoot()
            require(root.optInt("schemaVersion", 1) >= 6) {
                "This project predates RAW-authoritative capture geometry; start a new project"
            }
            if (root.optInt("schemaVersion", 1) < currentSchemaVersion) {
                migrateToSchema8(root)
                commit(root)
            }
        }
    }

    @Synchronized
    fun projectSnapshot(): ProjectSnapshot {
        val root = readRoot()
        val project = projectState(root)
        val settings = project.optJSONObject("cameraSettings")
        return ProjectSnapshot(
            milestone = ProjectMilestone.valueOf(
                project.optString("milestone", ProjectMilestone.CREATED.name)
            ),
            acquisitionMode = project.optString("acquisitionMode")
                .takeIf(String::isNotBlank)?.let { AcquisitionMode.valueOf(it) },
            committedCaptureCount = root.getJSONArray("captures").length(),
            cameraSettings = settings?.let {
                SavedCameraSettings(
                    cameraId = it.getString("cameraId"),
                    sensitivityIso = it.getInt("sensitivityIso"),
                    exposureTimeNs = it.getLong("exposureTimeNs"),
                    focusDistanceDiopters = it.getDouble("focusDistanceDiopters").toFloat(),
                    aperture = if (it.isNull("aperture")) null else
                        it.getDouble("aperture").toFloat(),
                    depthOfFieldCalibration = it.optJSONObject("depthOfFieldCalibration")
                        ?.toDepthOfFieldCalibration(),
                )
            },
            emptyMatTemplate = project.optString("emptyMatTemplate")
                .takeIf(String::isNotBlank)?.let { File(directory, it) },
            completedPly = project.optString("completedPly")
                .takeIf(String::isNotBlank)?.let { File(directory, it) },
            updatedAtEpochMs = project.optLong("updatedAtEpochMs", 0L),
        )
    }

    @Synchronized
    fun autosaveCameraLocked(
        cameraId: String,
        sensitivityIso: Int,
        exposureTimeNs: Long,
        focusDistanceDiopters: Float,
        aperture: Float?,
        depthOfFieldCalibration: DepthOfFieldCalibration?,
    ) = updateProject(ProjectMilestone.CAMERA_LOCKED) {
        put("cameraSettings", JSONObject().apply {
            put("cameraId", cameraId)
            put("sensitivityIso", sensitivityIso)
            put("exposureTimeNs", exposureTimeNs)
            put("focusDistanceDiopters", focusDistanceDiopters.toDouble())
            put("aperture", aperture?.toDouble() ?: JSONObject.NULL)
            put(
                "depthOfFieldCalibration",
                depthOfFieldCalibration?.toJson() ?: JSONObject.NULL,
            )
        })
    }

    @Synchronized
    fun autosaveMatCalibration(template: File) =
        updateProject(ProjectMilestone.MAT_CALIBRATED) {
            put("emptyMatTemplate", relative(template))
        }

    @Synchronized
    fun autosaveGuidanceZero(mode: AcquisitionMode) =
        updateProject(ProjectMilestone.GUIDANCE_ZERO_ESTABLISHED) {
            put("acquisitionMode", mode.name)
        }

    @Synchronized
    fun autosaveReconstructionStarted() = updateProject(ProjectMilestone.RECONSTRUCTING)

    @Synchronized
    fun autosaveReconstructionPaused(stage: String, reasons: Collection<String>) =
        updateProject(ProjectMilestone.RECONSTRUCTION_PAUSED) {
            put("reconstructionStage", stage)
            put("pauseReasons", JSONArray(reasons.toList()))
        }

    @Synchronized
    fun autosaveReconstructionFailed(stage: String, message: String) =
        updateProject(ProjectMilestone.RECONSTRUCTION_FAILED) {
            put("reconstructionStage", stage)
            put("failureMessage", message)
        }

    @Synchronized
    fun autosaveReconstructionComplete(ply: File) =
        updateProject(ProjectMilestone.RECONSTRUCTION_COMPLETE) {
            put("completedPly", relative(ply))
            remove("pauseReasons")
            remove("failureMessage")
        }

    @Synchronized
    fun setAcquisitionMode(mode: AcquisitionMode) {
        val root = readRoot()
        root.put("schemaVersion", currentSchemaVersion)
        root.put("acquisitionMode", mode.name)
        projectState(root).put("acquisitionMode", mode.name)
        commit(root)
    }

    @Synchronized
    fun record(
        target: TargetView,
        measurement: AcceptedFrameMeasurement,
        files: SessionCaptureFiles,
    ) {
        require(files.rawMaster.isFile && files.rawMaster.length() > 0L) {
            "The RAW_SENSOR master is not durable"
        }
        require(files.workingImage.isFile && files.workingImage.length() > 0L) {
            "The working image is not durable"
        }
        require(files.metadata.isFile && files.metadata.length() > 0L) {
            "The capture metadata is not durable"
        }
        require(files.foregroundMask == null ||
            (files.foregroundMask.isFile && files.foregroundMask.length() > 0L)
        ) {
            "The foreground mask is not durable"
        }
        val captureMetadata = JSONObject(files.metadata.readText())
        require(captureMetadata.optString("poseSource") ==
            "capture_raw_derived_exact_unsmoothed"
        ) {
            "The capture does not contain an exact still-frame pose"
        }
        require(
            captureMetadata.getLong("sensorTimestampNs") ==
                captureMetadata.getLong("poseTimestampNs")
        ) { "The still image and camera pose timestamps do not match" }
        require(!captureMetadata.optBoolean("yuvUsedByAcceptedFrame", true)) {
            "The accepted frame must not depend on YUV"
        }
        require(captureMetadata.getJSONObject("effectiveControlAudit")
            .optBoolean("rawOnlyCapture", false)
        ) { "The accepted exposure was not a RAW-only request" }
        require(captureMetadata.optJSONObject("depthOfFieldCalibration") != null &&
            (captureMetadata.optJSONObject("steppedGrayFocus")
                ?.optDouble("sessionScore", 0.0) ?: 0.0) >= 0.80
        ) { "The accepted exposure lacks passing stepped-gray depth-of-field evidence" }
        val calibratedFocus = captureMetadata.getJSONObject("depthOfFieldCalibration")
            .getDouble("focusDistanceDiopters")
        require(abs(calibratedFocus - captureMetadata.getDouble("focusDistanceDiopters")) <= 0.001) {
            "The accepted exposure focus differs from the session depth-of-field calibration"
        }
        val timestamp = captureMetadata.getLong("sensorTimestampNs")
        val base = "capture-$timestamp"
        require(files.rawMaster.name == "$base.raw16" &&
            files.workingImage.name == "$base.tif" &&
            files.metadata.name == "$base.json" &&
            (files.foregroundMask == null || files.foregroundMask.name == "$base-mask.png")
        ) { "Capture files do not share the exposure timestamp identity" }
        require(captureMetadata.optString("rawMasterFile") == files.rawMaster.name &&
            captureMetadata.optString("workingImageFile") == files.workingImage.name &&
            captureMetadata.optString("foregroundMaskFile") ==
                (files.foregroundMask?.name ?: "")
        ) { "Capture metadata file references do not match the durable files" }
        val cameraIndex = target.round.ordinal
        captureMetadata.apply {
            put("cameraIndex", cameraIndex)
            put("targetIndex", target.index)
            put("targetAzimuthDegrees", target.azimuthDegrees)
            put("targetPositionAngleDegrees", target.positionAngleDegrees)
            put("targetTiltDegrees", target.tiltDegrees)
        }
        writeJsonAtomically(files.metadata, captureMetadata)
        val root = readRoot()
        root.put("schemaVersion", currentSchemaVersion)
        val capture = JSONObject().apply {
            put("frameId", "camera-$cameraIndex-${target.index}")
            put("cameraIndex", cameraIndex)
            put("targetIndex", target.index)
            put("targetAzimuthDegrees", target.azimuthDegrees)
            put("targetPositionAngleDegrees", target.positionAngleDegrees)
            put("targetTiltDegrees", target.tiltDegrees)
            put("targetRadiusMm", target.radiusMm)
            put("measuredMatAzimuthDegrees", measurement.matAzimuthDegrees)
            put("measuredMatElevationDegrees", measurement.matElevationDegrees)
            put("measuredMatRadiusMm", measurement.matRadiusMm)
            put("guidanceAzimuthDegrees", measurement.guidanceAzimuthDegrees)
            put("guidancePositionAngleDegrees", measurement.guidancePositionAngleDegrees)
            put("guidanceTiltDegrees", measurement.guidanceTiltDegrees)
            put("guidanceRollDegrees", measurement.guidanceRollDegrees)
            put("minimumMatFrameMarginPx", measurement.minimumMatFrameMarginPx)
            put("rawMaster", relative(files.rawMaster))
            put("workingImage", relative(files.workingImage))
            if (files.foregroundMask != null) put("foregroundMask", relative(files.foregroundMask))
            put("metadata", relative(files.metadata))
        }
        val captures = root.getJSONArray("captures")
        val existingIndex = (0 until captures.length()).firstOrNull {
            captures.getJSONObject(it).let { existing ->
                existing.getInt("cameraIndex") == cameraIndex &&
                    existing.getInt("targetIndex") == target.index
            }
        }
        if (existingIndex == null) captures.put(capture) else captures.put(existingIndex, capture)
        val committed = captures.length()
        val milestone = captureMilestone(committed)
        projectState(root).apply {
            put("milestone", milestone.name)
            put("committedCaptureCount", committed)
            put("lastCommittedFrameId", capture.getString("frameId"))
            put("updatedAtEpochMs", System.currentTimeMillis())
        }
        commit(root)
    }

    @Synchronized
    fun frames(): List<ReconstructionFrame> {
        val captures = readRoot().getJSONArray("captures")
        val loaded = buildList {
            for (index in 0 until captures.length()) {
                val entry = captures.getJSONObject(index)
                val metadataFile = File(directory, entry.getString("metadata"))
                val workingImage = File(directory, entry.getString("workingImage"))
                val foregroundMask = entry.optString("foregroundMask")
                    .takeIf(String::isNotBlank)?.let { File(directory, it) }
                if (!metadataFile.isFile || !workingImage.isFile ||
                    (foregroundMask != null && !foregroundMask.isFile)
                ) continue
                val metadata = JSONObject(metadataFile.readText())
                if (!metadata.has("rotationVector") || !metadata.has("workingIntrinsics")) continue
                if (metadata.optString("poseSource") !=
                    "capture_raw_derived_exact_unsmoothed"
                ) continue
                if (metadata.getLong("sensorTimestampNs") != metadata.getLong("poseTimestampNs")) {
                    continue
                }
                if (metadata.optBoolean("yuvUsedByAcceptedFrame", true) ||
                    metadata.optJSONObject("effectiveControlAudit")
                        ?.optBoolean("rawOnlyCapture", false) != true
                ) continue
                if (metadata.has("cameraIndex") &&
                    metadata.getInt("cameraIndex") != entry.getInt("cameraIndex")
                ) continue
                if (metadata.has("targetIndex") &&
                    metadata.getInt("targetIndex") != entry.getInt("targetIndex")
                ) continue
                val intrinsics = metadata.getJSONArray("workingIntrinsics").toDoubles()
                val cameraCenter = metadata.getJSONArray("cameraCenterMm").toDoubles()
                add(
                    cameraCenter to ReconstructionFrame(
                        frameId = entry.getString("frameId"),
                        workingImage = workingImage,
                        imageWidth = metadata.getInt("workingImageWidth"),
                        imageHeight = metadata.getInt("workingImageHeight"),
                        foregroundMask = foregroundMask,
                        cameraIndex = entry.getInt("cameraIndex"),
                        targetIndex = entry.getInt("targetIndex"),
                        scheduledAzimuthDegrees = entry.getDouble("targetAzimuthDegrees"),
                        azimuthDegrees = 0.0,
                        elevationDegrees = 0.0,
                        cameraFx = intrinsics[0], cameraFy = intrinsics[1],
                        cameraCx = intrinsics[2], cameraCy = intrinsics[3],
                        distortion = metadata.optJSONArray("distortion")?.toDoubles()
                            ?: DoubleArray(5),
                        rotationVector = metadata.getJSONArray("rotationVector").toDoubles(),
                        translationVector = metadata.getJSONArray("translationVector").toDoubles(),
                    )
                )
            }
        }
        val angles = measuredViewAngles(loaded.map { it.first })
        return loaded.indices.map { index ->
            loaded[index].second.copy(
                azimuthDegrees = angles[index].azimuthDegrees,
                elevationDegrees = angles[index].elevationDegrees,
            )
        }
    }

    val captureCount: Int get() = readRoot().getJSONArray("captures").length()

    private fun readRoot() = JSONObject(manifest.readText())

    private fun relative(file: File): String = file.relativeTo(directory).path

    private fun newProjectState() = JSONObject().apply {
        put("milestone", ProjectMilestone.CREATED.name)
        put("createdAtEpochMs", System.currentTimeMillis())
        put("updatedAtEpochMs", System.currentTimeMillis())
        put("committedCaptureCount", 0)
    }

    private fun projectState(root: JSONObject): JSONObject =
        root.optJSONObject("project") ?: newProjectState().also { root.put("project", it) }

    private fun updateProject(
        milestone: ProjectMilestone,
        block: JSONObject.() -> Unit = {},
    ) {
        val root = readRoot()
        projectState(root).apply {
            val current = optString("milestone")
                .takeIf(String::isNotBlank)
                ?.let(ProjectMilestone::valueOf)
                ?: ProjectMilestone.CREATED
            val setupMilestone = milestone in setOf(
                ProjectMilestone.CAMERA_LOCKED,
                ProjectMilestone.MAT_CALIBRATED,
                ProjectMilestone.GUIDANCE_ZERO_ESTABLISHED,
            )
            val effective = if (setupMilestone && current.ordinal > milestone.ordinal) {
                current
            } else {
                milestone
            }
            put("milestone", effective.name)
            put("updatedAtEpochMs", System.currentTimeMillis())
            block()
        }
        root.put("schemaVersion", currentSchemaVersion)
        commit(root)
    }

    private fun commit(root: JSONObject) {
        writeJsonAtomically(manifest, root)
    }

    private fun writeJsonAtomically(destination: File, value: JSONObject) {
        destination.parentFile?.mkdirs()
        val temporary = File(destination.parentFile, "${destination.name}.tmp")
        temporary.writeText(value.toString(2))
        try {
            Files.move(
                temporary.toPath(), destination.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: Exception) {
            Files.move(
                temporary.toPath(), destination.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        }
    }

    /**
     * v0.4.4 used roundIndex for camera identity. Schema 8 adds optional saved
     * focus evidence; a fresh lock supplies it before any new capture is committed.
     */
    private fun migrateToSchema8(root: JSONObject) {
        val captures = root.optJSONArray("captures") ?: JSONArray().also {
            root.put("captures", it)
        }
        for (index in 0 until captures.length()) {
            captures.getJSONObject(index).apply {
                if (!has("cameraIndex") && has("roundIndex")) {
                    put("cameraIndex", getInt("roundIndex"))
                }
                remove("roundIndex")
            }
        }
        root.put("schemaVersion", currentSchemaVersion)
    }

    private fun JSONArray.toDoubles() = DoubleArray(length()) { getDouble(it) }

    private fun JSONObject.toDepthOfFieldCalibration() = DepthOfFieldCalibration(
        focusDistanceDiopters = getDouble("focusDistanceDiopters").toFloat(),
        candidateCount = getInt("candidateCount"),
        detectedBoundaryCount = getInt("detectedBoundaryCount"),
        sampleCount = getInt("sampleCount"),
        referenceMedianEdgeWidthCanonicalPx =
            getDouble("referenceMedianEdgeWidthCanonicalPx"),
        referenceP90EdgeWidthCanonicalPx = getDouble("referenceP90EdgeWidthCanonicalPx"),
        maximumP90EdgeWidthCanonicalPx = getDouble("maximumP90EdgeWidthCanonicalPx"),
        nearReferenceDepthMm = getDouble("nearReferenceDepthMm"),
        farReferenceDepthMm = getDouble("farReferenceDepthMm"),
        medianStepContrast = getDouble("medianStepContrast"),
        referenceEqualStepConsistency = optDouble("referenceEqualStepConsistency", 1.0),
    )

    private fun DepthOfFieldCalibration.toJson() = JSONObject().apply {
        put("method", "raw_registered_equal_gray_step_edge_spread")
        put("focusDistanceDiopters", focusDistanceDiopters.toDouble())
        put("candidateCount", candidateCount)
        put("detectedBoundaryCount", detectedBoundaryCount)
        put("sampleCount", sampleCount)
        put("referenceMedianEdgeWidthCanonicalPx", referenceMedianEdgeWidthCanonicalPx)
        put("referenceP90EdgeWidthCanonicalPx", referenceP90EdgeWidthCanonicalPx)
        put("maximumP90EdgeWidthCanonicalPx", maximumP90EdgeWidthCanonicalPx)
        put("nearReferenceDepthMm", nearReferenceDepthMm)
        put("farReferenceDepthMm", farReferenceDepthMm)
        put("medianStepContrast", medianStepContrast)
        put("referenceEqualStepConsistency", referenceEqualStepConsistency)
    }

}

internal fun captureMilestone(committedCaptureCount: Int): ProjectMilestone {
    require(committedCaptureCount in 0..36)
    return when {
        committedCaptureCount >= 36 -> ProjectMilestone.CAPTURE_COMPLETE
        committedCaptureCount == 18 -> ProjectMilestone.UPPER_PASS_COMPLETE
        committedCaptureCount > 18 -> ProjectMilestone.CAPTURING_LOWER
        committedCaptureCount > 0 -> ProjectMilestone.CAPTURING_UPPER
        else -> ProjectMilestone.GUIDANCE_ZERO_ESTABLISHED
    }
}

/**
 * A complete, evenly distributed orbit has its XY center at the mean measured
 * camera center. These angles are retained as pose diagnostics without borrowing
 * the guidance baseline or assuming that the artifact lies exactly at mat (0,0).
 * Sparse candidate selection uses the separately stored capture schedule.
 */
internal fun measuredViewAngles(cameraCenters: List<DoubleArray>): List<MeasuredViewAngle> {
    if (cameraCenters.isEmpty()) return emptyList()
    require(cameraCenters.all { it.size >= 3 })
    val centerX = cameraCenters.map { it[0] }.average()
    val centerY = cameraCenters.map { it[1] }.average()
    return cameraCenters.map { camera ->
        val x = camera[0] - centerX
        val y = camera[1] - centerY
        MeasuredViewAngle(
            azimuthDegrees = com.metriccapture.app.geometry.wrapDegrees(
                Math.toDegrees(kotlin.math.atan2(y, x))
            ),
            elevationDegrees = Math.toDegrees(
                kotlin.math.atan2(camera[2], kotlin.math.sqrt(x * x + y * y))
            ),
        )
    }
}

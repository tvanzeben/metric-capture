package com.metriccapture.app.reconstruction

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.file.Files
import java.nio.file.StandardCopyOption

data class ReconstructionCheckpoint(
    val sparseComplete: Boolean = false,
    val sparseEdgeCount: Int = 0,
    val refinedCameraPoses: Map<String, RefinedCameraPose> = emptyMap(),
    val completedDensePairIds: Set<String> = emptySet(),
)

data class RefinedCameraPose(
    val rotationVector: DoubleArray,
    val translationVector: DoubleArray,
)

class ReconstructionCheckpointStore(private val directory: File) {
    private val stateFile = File(directory, "reconstruction-state.json")
    val voxelFile = File(directory, "voxel-fusion.bin")

    fun load(): ReconstructionCheckpoint {
        if (!stateFile.exists()) return ReconstructionCheckpoint()
        val json = JSONObject(stateFile.readText())
        // Sparse graph and refined-pose semantics changed in schema 3. An older
        // voxel checkpoint must be rebuilt instead of silently bypassing them.
        if (json.optInt("schemaVersion", 1) < 3) return ReconstructionCheckpoint()
        val completed = json.optJSONArray("completedDensePairIds") ?: JSONArray()
        val refined = json.optJSONObject("refinedCameraPoses") ?: JSONObject()
        return ReconstructionCheckpoint(
            sparseComplete = json.optBoolean("sparseComplete", false),
            sparseEdgeCount = json.optInt("sparseEdgeCount", 0),
            refinedCameraPoses = buildMap {
                val keys = refined.keys()
                while (keys.hasNext()) {
                    val frameId = keys.next()
                    val pose = refined.getJSONObject(frameId)
                    put(
                        frameId,
                        RefinedCameraPose(
                            pose.getJSONArray("rotationVector").toDoubles(),
                            pose.getJSONArray("translationVector").toDoubles(),
                        ),
                    )
                }
            },
            completedDensePairIds = buildSet {
                for (index in 0 until completed.length()) add(completed.getString(index))
            },
        )
    }

    fun save(checkpoint: ReconstructionCheckpoint, fusion: VoxelFusion) {
        directory.mkdirs()
        val temporaryVoxel = File(directory, "voxel-fusion.bin.tmp")
        val temporaryState = File(directory, "reconstruction-state.json.tmp")
        fusion.save(temporaryVoxel)
        temporaryState.writeText(JSONObject().apply {
            put("schemaVersion", 3)
            put("sparseComplete", checkpoint.sparseComplete)
            put("sparseEdgeCount", checkpoint.sparseEdgeCount)
            put("refinedCameraPoses", JSONObject().apply {
                checkpoint.refinedCameraPoses.toSortedMap().forEach { (frameId, pose) ->
                    put(frameId, JSONObject().apply {
                        put("rotationVector", JSONArray(pose.rotationVector.toList()))
                        put("translationVector", JSONArray(pose.translationVector.toList()))
                    })
                }
            })
            put("completedDensePairIds", JSONArray(checkpoint.completedDensePairIds.sorted()))
        }.toString(2))
        replace(temporaryVoxel, voxelFile)
        replace(temporaryState, stateFile)
    }

    fun clearAfterSuccess() {
        stateFile.delete()
        voxelFile.delete()
    }

    private fun replace(source: File, destination: File) {
        try {
            Files.move(
                source.toPath(), destination.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
                StandardCopyOption.ATOMIC_MOVE,
            )
        } catch (_: Exception) {
            Files.move(
                source.toPath(), destination.toPath(),
                StandardCopyOption.REPLACE_EXISTING,
            )
        }
    }

    private fun JSONArray.toDoubles() = DoubleArray(length()) { getDouble(it) }
}

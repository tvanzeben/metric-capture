package com.metriccapture.app.compute

import android.app.ActivityManager
import android.content.Context
import android.os.BatteryManager
import android.os.PowerManager
import android.os.StatFs
import java.io.File
import kotlin.math.max
import kotlin.math.min

enum class ReconstructionStage {
    WORKING_IMAGE_GENERATION,
    SPARSE_FEATURES,
    FEATURE_TRACKING,
    TRIANGULATION,
    BUNDLE_ADJUSTMENT,
    VISUAL_HULL,
    DENSE_DEPTH,
    DEPTH_FUSION,
    MESHING,
    TEXTURE_PROJECTION,
    PLY_EXPORT,
}

enum class ConstraintReason {
    THERMAL_MODERATE,
    THERMAL_SEVERE,
    LOW_BATTERY,
    MEMORY_PRESSURE,
    INSUFFICIENT_STORAGE,
}

data class ResourceSnapshot(
    val availableMemoryBytes: Long,
    val lowMemory: Boolean,
    val freeStorageBytes: Long,
    val batteryPercent: Int,
    val charging: Boolean,
    val thermalStatus: Int,
    val processorCount: Int,
)

data class StageEstimate(
    val peakMemoryBytes: Long,
    val temporaryStorageBytes: Long,
    val thermallyIntensive: Boolean,
)

sealed interface ComputePermission {
    data class Run(
        val workerCount: Int,
        val tileEdgePixels: Int,
        val geometryLongEdgePixels: Int,
        val memoryBudgetBytes: Long,
    ) : ComputePermission

    data class Throttle(
        val workerCount: Int,
        val tileEdgePixels: Int,
        val geometryLongEdgePixels: Int,
        val memoryBudgetBytes: Long,
        val reasons: Set<ConstraintReason>,
    ) : ComputePermission

    data class Pause(val reasons: Set<ConstraintReason>) : ComputePermission
}

class ResourceGovernor(
    private val reserveStorageBytes: Long = 1_000_000_000L,
    private val minimumBatteryPercent: Int = 25,
) {
    fun permission(
        snapshot: ResourceSnapshot,
        stage: ReconstructionStage,
        estimate: StageEstimate,
    ): ComputePermission {
        val reasons = linkedSetOf<ConstraintReason>()
        if (snapshot.freeStorageBytes - reserveStorageBytes < estimate.temporaryStorageBytes) {
            reasons += ConstraintReason.INSUFFICIENT_STORAGE
        }
        if (snapshot.lowMemory || estimate.peakMemoryBytes > snapshot.availableMemoryBytes * 45 / 100) {
            reasons += ConstraintReason.MEMORY_PRESSURE
        }
        if (!snapshot.charging && snapshot.batteryPercent < minimumBatteryPercent &&
            stage in intensiveStages
        ) reasons += ConstraintReason.LOW_BATTERY
        if (snapshot.thermalStatus >= PowerManager.THERMAL_STATUS_SEVERE) {
            reasons += ConstraintReason.THERMAL_SEVERE
        } else if (snapshot.thermalStatus >= PowerManager.THERMAL_STATUS_MODERATE) {
            reasons += ConstraintReason.THERMAL_MODERATE
        }

        val hardStop = reasons.any {
            it == ConstraintReason.INSUFFICIENT_STORAGE ||
                it == ConstraintReason.THERMAL_SEVERE ||
                (it == ConstraintReason.LOW_BATTERY && stage in intensiveStages)
        }
        if (hardStop) return ComputePermission.Pause(reasons)

        val memoryBudget = min(
            1_500_000_000L,
            max(192_000_000L, snapshot.availableMemoryBytes * 35 / 100),
        )
        val normalWorkers = max(1, min(4, snapshot.processorCount / 2))
        return if (reasons.isEmpty()) {
            ComputePermission.Run(
                workerCount = normalWorkers,
                tileEdgePixels = 384,
                geometryLongEdgePixels = 2048,
                memoryBudgetBytes = memoryBudget,
            )
        } else {
            ComputePermission.Throttle(
                workerCount = 1,
                tileEdgePixels = 192,
                geometryLongEdgePixels = 1280,
                memoryBudgetBytes = min(memoryBudget, 512_000_000L),
                reasons = reasons,
            )
        }
    }

    companion object {
        private val intensiveStages = setOf(
            ReconstructionStage.BUNDLE_ADJUSTMENT,
            ReconstructionStage.DENSE_DEPTH,
            ReconstructionStage.DEPTH_FUSION,
            ReconstructionStage.MESHING,
            ReconstructionStage.TEXTURE_PROJECTION,
        )
    }
}

class AndroidResourceProbe(
    context: Context,
    private val storageDirectory: File,
) {
    private val activityManager = context.getSystemService(ActivityManager::class.java)
    private val batteryManager = context.getSystemService(BatteryManager::class.java)
    private val powerManager = context.getSystemService(PowerManager::class.java)

    fun snapshot(): ResourceSnapshot {
        val memory = ActivityManager.MemoryInfo().also(activityManager::getMemoryInfo)
        val stat = StatFs(storageDirectory.absolutePath)
        val battery = batteryManager.getIntProperty(BatteryManager.BATTERY_PROPERTY_CAPACITY)
        val charging = batteryManager.isCharging
        return ResourceSnapshot(
            availableMemoryBytes = memory.availMem,
            lowMemory = memory.lowMemory,
            freeStorageBytes = stat.availableBytes,
            batteryPercent = battery.coerceIn(0, 100),
            charging = charging,
            thermalStatus = powerManager.currentThermalStatus,
            processorCount = Runtime.getRuntime().availableProcessors(),
        )
    }
}

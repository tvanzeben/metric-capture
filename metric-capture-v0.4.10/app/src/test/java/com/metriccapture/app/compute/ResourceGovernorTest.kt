package com.metriccapture.app.compute

import android.os.PowerManager
import org.junit.Assert.assertTrue
import org.junit.Test

class ResourceGovernorTest {
    private val estimate = StageEstimate(
        peakMemoryBytes = 500_000_000,
        temporaryStorageBytes = 800_000_000,
        thermallyIntensive = true,
    )

    @Test
    fun `severe thermal state pauses dense reconstruction`() {
        val permission = ResourceGovernor().permission(
            snapshot = healthy().copy(thermalStatus = PowerManager.THERMAL_STATUS_SEVERE),
            stage = ReconstructionStage.DENSE_DEPTH,
            estimate = estimate,
        )
        assertTrue(permission is ComputePermission.Pause)
    }

    @Test
    fun `moderate thermal state throttles rather than discarding work`() {
        val permission = ResourceGovernor().permission(
            snapshot = healthy().copy(thermalStatus = PowerManager.THERMAL_STATUS_MODERATE),
            stage = ReconstructionStage.DENSE_DEPTH,
            estimate = estimate,
        )
        assertTrue(permission is ComputePermission.Throttle)
    }

    private fun healthy() = ResourceSnapshot(
        availableMemoryBytes = 3_000_000_000,
        lowMemory = false,
        freeStorageBytes = 10_000_000_000,
        batteryPercent = 80,
        charging = false,
        thermalStatus = PowerManager.THERMAL_STATUS_NONE,
        processorCount = 8,
    )
}

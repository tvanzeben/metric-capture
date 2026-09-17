package com.metriccapture.app.reconstruction

import org.junit.Assert.assertEquals
import org.junit.Test

class ProjectMilestoneTest {
    @Test
    fun `capture commits advance durable project milestones`() {
        assertEquals(ProjectMilestone.GUIDANCE_ZERO_ESTABLISHED, captureMilestone(0))
        assertEquals(ProjectMilestone.CAPTURING_UPPER, captureMilestone(1))
        assertEquals(ProjectMilestone.CAPTURING_UPPER, captureMilestone(17))
        assertEquals(ProjectMilestone.UPPER_PASS_COMPLETE, captureMilestone(18))
        assertEquals(ProjectMilestone.CAPTURING_LOWER, captureMilestone(19))
        assertEquals(ProjectMilestone.CAPTURING_LOWER, captureMilestone(35))
        assertEquals(ProjectMilestone.CAPTURE_COMPLETE, captureMilestone(36))
    }
}

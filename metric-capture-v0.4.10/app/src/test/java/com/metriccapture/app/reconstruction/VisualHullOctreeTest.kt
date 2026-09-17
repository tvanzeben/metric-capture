package com.metriccapture.app.reconstruction

import com.metriccapture.app.geometry.Vec3
import org.junit.Assert.assertTrue
import org.junit.Test

class VisualHullOctreeTest {
    @Test
    fun `visible background removes candidate volume`() {
        val hull = VisualHullOctree(
            Aabb(Vec3(-10.0, -10.0, 0.0), Vec3(10.0, 10.0, 20.0)),
            maximumDepth = 3,
        )
        val background = BinaryMask(100, 100, ByteArray(10_000))
        val camera = CameraProjection(
            fx = 100.0,
            fy = 100.0,
            cx = 50.0,
            cy = 50.0,
            rotationVector = doubleArrayOf(0.0, 0.0, 0.0),
            translationVector = doubleArrayOf(0.0, 0.0, 100.0),
        )

        hull.integrate(background, camera)

        assertTrue(hull.candidateLeafBounds().isEmpty())
    }
}

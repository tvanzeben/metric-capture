package com.metriccapture.app.vision

import org.opencv.core.Point3

enum class FiducialId { A, B, C, D, E, F, G, H }
enum class QuadrantTone { BLACK, WHITE, YELLOW, CYAN, UNKNOWN }

/** Clockwise artwork order: top-left, top-right, bottom-right, bottom-left. */
data class FiducialCode(val quadrants: List<QuadrantTone>) {
    init { require(quadrants.size == 4) }
}

/** Coordinates follow the printed mat: +X right, +Y toward C, Z above the mat. */
object MatModel {
    val pointsMm: Map<FiducialId, Point3> = mapOf(
        FiducialId.A to Point3(0.0, -60.0, 0.0),
        FiducialId.B to Point3(60.0, 0.0, 0.0),
        FiducialId.C to Point3(0.0, 60.0, 0.0),
        FiducialId.D to Point3(-60.0, 0.0, 0.0),
        FiducialId.E to Point3(-25.0, -25.0, 0.0),
        FiducialId.F to Point3(25.0, -25.0, 0.0),
        FiducialId.G to Point3(25.0, 25.0, 0.0),
        FiducialId.H to Point3(-25.0, 25.0, 0.0),
    )

    val expectedCode: Map<FiducialId, FiducialCode> = mapOf(
        FiducialId.A to code(
            QuadrantTone.CYAN, QuadrantTone.BLACK,
            QuadrantTone.CYAN, QuadrantTone.WHITE,
        ),
        FiducialId.B to code(
            QuadrantTone.YELLOW, QuadrantTone.WHITE,
            QuadrantTone.YELLOW, QuadrantTone.BLACK,
        ),
        FiducialId.C to code(
            QuadrantTone.BLACK, QuadrantTone.WHITE,
            QuadrantTone.BLACK, QuadrantTone.WHITE,
        ),
        FiducialId.D to code(
            QuadrantTone.BLACK, QuadrantTone.WHITE,
            QuadrantTone.BLACK, QuadrantTone.WHITE,
        ),
        FiducialId.E to code(
            QuadrantTone.WHITE, QuadrantTone.BLACK,
            QuadrantTone.WHITE, QuadrantTone.BLACK,
        ),
        FiducialId.F to code(
            QuadrantTone.WHITE, QuadrantTone.BLACK,
            QuadrantTone.WHITE, QuadrantTone.BLACK,
        ),
        FiducialId.G to code(
            QuadrantTone.CYAN, QuadrantTone.WHITE,
            QuadrantTone.CYAN, QuadrantTone.BLACK,
        ),
        FiducialId.H to code(
            QuadrantTone.YELLOW, QuadrantTone.BLACK,
            QuadrantTone.YELLOW, QuadrantTone.WHITE,
        ),
    )

    private fun code(vararg quadrants: QuadrantTone) = FiducialCode(quadrants.toList())
}

/** Rotation-invariant local score; global mat geometry resolves the cyclic orientation. */
internal fun cyclicCodeMatch(observed: FiducialCode, expected: FiducialCode): Double =
    (0 until 4).maxOf { rotation -> codeMatch(observed, expected, rotation) }

/** Exact spatial code score under one quarter-turn shared by the whole mat. */
internal fun codeMatch(
    observed: FiducialCode,
    expected: FiducialCode,
    quarterTurns: Int,
): Double = observed.quadrants.indices.count { index ->
    val observedTone = observed.quadrants[index]
    observedTone != QuadrantTone.UNKNOWN &&
        observedTone == expected.quadrants[(index + quarterTurns.mod(4)) % 4]
} / 4.0

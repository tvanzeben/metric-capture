package com.metriccapture.app.guidance

sealed interface OrbitState {
    data object AwaitingUserGuidanceZero : OrbitState
    data class Collecting(val roundIndex: Int, val nextIndex: Int) : OrbitState
    data class RoundReview(val completedRoundIndex: Int) : OrbitState
    data class Complete(val rounds: List<RoundDefinition>) : OrbitState
}

/** Deterministic multi-round acquisition state; it does not contain reconstruction pose. */
class OrbitController {
    private var definitions: List<RoundDefinition> = emptyList()
    var state: OrbitState = OrbitState.AwaitingUserGuidanceZero
        private set

    fun start(rounds: List<RoundDefinition>, completedTargets: Int = 0) {
        require(state is OrbitState.AwaitingUserGuidanceZero)
        require(rounds.isNotEmpty())
        require(rounds.map { it.round }.distinct().size == rounds.size)
        definitions = rounds.toList()
        require(completedTargets in 0..definitions.sumOf(RoundDefinition::targetCount))
        var remaining = completedTargets
        for ((roundIndex, definition) in definitions.withIndex()) {
            if (remaining < definition.targetCount) {
                state = OrbitState.Collecting(roundIndex, remaining)
                return
            }
            remaining -= definition.targetCount
        }
        state = OrbitState.Complete(definitions.toList())
    }

    fun currentTarget(): TargetView? {
        val collecting = state as? OrbitState.Collecting ?: return null
        val definition = definitions[collecting.roundIndex]
        return TargetView(
            round = definition.round,
            index = collecting.nextIndex,
            azimuthDegrees = definition.targetAzimuth(collecting.nextIndex),
            positionAngleDegrees = definition.targetPositionAngleDegrees,
            tiltDegrees = definition.targetTiltDegrees,
            radiusMm = definition.targetRadiusMm,
        )
    }

    fun acceptCurrentTarget() {
        val collecting = state as? OrbitState.Collecting ?: error("No active target")
        val definition = definitions[collecting.roundIndex]
        val next = collecting.nextIndex + 1
        state = if (next >= definition.targetCount) {
            OrbitState.RoundReview(collecting.roundIndex)
        } else {
            collecting.copy(nextIndex = next)
        }
    }

    fun continueAfterRoundReview() {
        val review = state as? OrbitState.RoundReview ?: error("No round to review")
        val nextRound = review.completedRoundIndex + 1
        state = if (nextRound >= definitions.size) {
            OrbitState.Complete(definitions.toList())
        } else {
            OrbitState.Collecting(roundIndex = nextRound, nextIndex = 0)
        }
    }

    val totalTargets: Int get() = definitions.sumOf(RoundDefinition::targetCount)
}

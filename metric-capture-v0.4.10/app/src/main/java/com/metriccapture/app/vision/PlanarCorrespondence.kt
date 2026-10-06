package com.metriccapture.app.vision

import kotlin.math.abs
import kotlin.math.sqrt

internal enum class TargetColorClass { CYAN, YELLOW, MONO }

/**
 * Color is the only part of a decoded code that reliably identifies a target.
 * A/G share one cyan pattern, B/H share one yellow pattern, and C/D/E/F share
 * one diagonal black/white pattern whose quarter-turn phase cannot be read
 * reliably from a symmetric crosshair. Identity within a color class therefore
 * has to come from mat geometry, not from the quadrant order.
 */
internal fun colorClassOf(code: FiducialCode): TargetColorClass = when {
    QuadrantTone.CYAN in code.quadrants -> TargetColorClass.CYAN
    QuadrantTone.YELLOW in code.quadrants -> TargetColorClass.YELLOW
    else -> TargetColorClass.MONO
}

internal data class PlanarCandidate(
    /** Same length and order as the input points; null means that detection is not used. */
    val ids: List<FiducialId?>,
    /** Plane homography residual RMS in pixels; NaN when exactly four points were assigned. */
    val homographyRmsPx: Double,
)

/**
 * Generates fiducial assignments without camera intrinsics.
 *
 * 1. Every detection is assigned only to ids of its own color class, each id at most once.
 * 2. Mirrored assignments are rejected: the camera views the mat from above, so every
 *    non-degenerate triangle of mat points keeps a consistent winding in the image.
 * 3. With five or more points, the remaining candidates are ranked by how well a single
 *    plane homography maps the mat model onto the detections. The correct assignment fits
 *    to sub-pixel accuracy; a single wrong swap is off by tens to hundreds of pixels.
 *
 * With exactly four points a homography fits any assignment exactly, so those candidates
 * are returned unranked and the caller's PnP reprojection error has to decide.
 */
internal object PlanarCorrespondence {

    private val idsByClass = mapOf(
        TargetColorClass.CYAN to listOf(FiducialId.A, FiducialId.G),
        TargetColorClass.YELLOW to listOf(FiducialId.B, FiducialId.H),
        TargetColorClass.MONO to listOf(FiducialId.C, FiducialId.D, FiducialId.E, FiducialId.F),
    )

    /** Extra detections allowed per class beyond its id count, to tolerate one spurious blob. */
    private const val SPARE_PER_CLASS = 1

    fun candidates(
        xs: DoubleArray,
        ys: DoubleArray,
        classes: List<TargetColorClass>,
        quality: DoubleArray,
        modelXmm: Map<FiducialId, Double>,
        modelYmm: Map<FiducialId, Double>,
        maxResults: Int = 4,
    ): List<PlanarCandidate> {
        val n = xs.size
        require(ys.size == n && classes.size == n && quality.size == n)

        // Per class, keep the strongest detections only.
        val perClassIndices = TargetColorClass.values().associateWith { cls ->
            val ids = idsByClass.getValue(cls)
            (0 until n).filter { classes[it] == cls }
                .sortedByDescending { quality[it] }
                .take(ids.size + SPARE_PER_CLASS)
        }
        val perClassOptions = TargetColorClass.values().map { cls ->
            injectiveAssignments(perClassIndices.getValue(cls), idsByClass.getValue(cls))
        }

        val results = mutableListOf<PlanarCandidate>()
        for (cyan in perClassOptions[0]) for (yellow in perClassOptions[1]) for (mono in perClassOptions[2]) {
            val pairs = cyan + yellow + mono
            if (pairs.size < 4) continue
            val modelX = DoubleArray(pairs.size) { modelXmm.getValue(pairs[it].second) }
            val modelY = DoubleArray(pairs.size) { modelYmm.getValue(pairs[it].second) }
            val imageX = DoubleArray(pairs.size) { xs[pairs[it].first] }
            val imageY = DoubleArray(pairs.size) { ys[pairs[it].first] }
            if (!windingConsistent(modelX, modelY, imageX, imageY)) continue
            val rms = if (pairs.size >= 5) {
                homographyRms(modelX, modelY, imageX, imageY) ?: continue
            } else {
                Double.NaN
            }
            val ids = arrayOfNulls<FiducialId>(n)
            for ((index, id) in pairs) ids[index] = id
            results += PlanarCandidate(ids.toList(), rms)
        }

        // More assigned points first (a fuller fit is more trustworthy), then best homography fit.
        return results
            .sortedWith(
                compareByDescending<PlanarCandidate> { c -> c.ids.count { it != null } }
                    .thenBy { if (it.homographyRmsPx.isNaN()) Double.MAX_VALUE else it.homographyRmsPx }
            )
            .take(maxResults)
    }

    /**
     * All injective maps from detections to ids where the number of assigned detections
     * is min(detections, ids). Each map is a list of (detectionIndex, id) pairs.
     */
    private fun injectiveAssignments(
        detections: List<Int>,
        ids: List<FiducialId>,
    ): List<List<Pair<Int, FiducialId>>> {
        val target = minOf(detections.size, ids.size)
        if (target == 0) return listOf(emptyList())
        val out = mutableListOf<List<Pair<Int, FiducialId>>>()
        fun recurse(position: Int, used: Set<FiducialId>, current: List<Pair<Int, FiducialId>>) {
            val remainingDetections = detections.size - position
            if (current.size == target) {
                out += current
                return
            }
            if (current.size + remainingDetections < target) return
            val detection = detections[position]
            for (id in ids) if (id !in used) {
                recurse(position + 1, used + id, current + (detection to id))
            }
            // Leave this detection unassigned (spurious blob).
            recurse(position + 1, used, current)
        }
        recurse(0, emptySet(), emptyList())
        return out
    }

    /**
     * The mat model uses +Y toward C with +X to the right (y-up); image coordinates are y-down.
     * Seen from above without mirroring, every triangle's winding therefore flips sign between
     * model and image. Near-degenerate image triangles are skipped as uninformative.
     */
    internal fun windingConsistent(
        modelX: DoubleArray, modelY: DoubleArray,
        imageX: DoubleArray, imageY: DoubleArray,
    ): Boolean {
        val n = modelX.size
        var informative = 0
        for (i in 0 until n) for (j in i + 1 until n) for (k in j + 1 until n) {
            val model = cross(modelX, modelY, i, j, k)
            val image = cross(imageX, imageY, i, j, k)
            val imageScale = maxOf(
                sq(imageX[j] - imageX[i]) + sq(imageY[j] - imageY[i]),
                sq(imageX[k] - imageX[i]) + sq(imageY[k] - imageY[i]),
            )
            if (abs(model) < 1e-6 || abs(image) < 0.02 * imageScale) continue
            informative++
            if (model * image > 0.0) return false
        }
        return informative > 0
    }

    /** Least-squares plane homography (model mm -> image px); returns pixel RMS or null if singular. */
    internal fun homographyRms(
        modelX: DoubleArray, modelY: DoubleArray,
        imageX: DoubleArray, imageY: DoubleArray,
    ): Double? {
        val n = modelX.size
        val (sx, sy, ss) = normalization(modelX, modelY)
        val (dx, dy, ds) = normalization(imageX, imageY)
        // Normal equations for the 8 unknowns with h33 = 1, in normalized coordinates.
        val ata = Array(8) { DoubleArray(8) }
        val atb = DoubleArray(8)
        for (p in 0 until n) {
            val x = (modelX[p] - sx) * ss
            val y = (modelY[p] - sy) * ss
            val u = (imageX[p] - dx) * ds
            val v = (imageY[p] - dy) * ds
            val rowU = doubleArrayOf(x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y)
            val rowV = doubleArrayOf(0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y)
            for (r in 0 until 8) {
                atb[r] += rowU[r] * u + rowV[r] * v
                for (c in 0 until 8) ata[r][c] += rowU[r] * rowU[c] + rowV[r] * rowV[c]
            }
        }
        val h = solveLinear(ata, atb) ?: return null
        var sumSq = 0.0
        for (p in 0 until n) {
            val x = (modelX[p] - sx) * ss
            val y = (modelY[p] - sy) * ss
            val w = h[6] * x + h[7] * y + 1.0
            if (abs(w) < 1e-12) return null
            val u = (h[0] * x + h[1] * y + h[2]) / w
            val v = (h[3] * x + h[4] * y + h[5]) / w
            val px = u / ds + dx
            val py = v / ds + dy
            sumSq += sq(px - imageX[p]) + sq(py - imageY[p])
        }
        return sqrt(sumSq / n)
    }

    /** Hartley normalization: centroid to origin, mean distance sqrt(2). Returns (cx, cy, scale). */
    private fun normalization(x: DoubleArray, y: DoubleArray): Triple<Double, Double, Double> {
        val cx = x.average()
        val cy = y.average()
        val meanDistance = x.indices.map { sqrt(sq(x[it] - cx) + sq(y[it] - cy)) }.average()
        val scale = if (meanDistance > 1e-12) sqrt(2.0) / meanDistance else 1.0
        return Triple(cx, cy, scale)
    }

    /** Gaussian elimination with partial pivoting; null if the system is singular. */
    private fun solveLinear(a: Array<DoubleArray>, b: DoubleArray): DoubleArray? {
        val n = b.size
        val m = Array(n) { r -> a[r].copyOf() }
        val v = b.copyOf()
        for (col in 0 until n) {
            var pivot = col
            for (r in col + 1 until n) if (abs(m[r][col]) > abs(m[pivot][col])) pivot = r
            if (abs(m[pivot][col]) < 1e-12) return null
            if (pivot != col) {
                val tmp = m[pivot]; m[pivot] = m[col]; m[col] = tmp
                val tv = v[pivot]; v[pivot] = v[col]; v[col] = tv
            }
            for (r in col + 1 until n) {
                val factor = m[r][col] / m[col][col]
                if (factor == 0.0) continue
                for (c in col until n) m[r][c] -= factor * m[col][c]
                v[r] -= factor * v[col]
            }
        }
        val x = DoubleArray(n)
        for (r in n - 1 downTo 0) {
            var sum = v[r]
            for (c in r + 1 until n) sum -= m[r][c] * x[c]
            x[r] = sum / m[r][r]
        }
        return x
    }

    private fun cross(x: DoubleArray, y: DoubleArray, i: Int, j: Int, k: Int): Double =
        (x[j] - x[i]) * (y[k] - y[i]) - (y[j] - y[i]) * (x[k] - x[i])

    private fun sq(value: Double) = value * value
}

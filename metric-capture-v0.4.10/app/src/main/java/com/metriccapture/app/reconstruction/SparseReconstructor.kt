package com.metriccapture.app.reconstruction

import org.opencv.geometry.Geometry
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfByte
import org.opencv.core.MatOfDMatch
import org.opencv.core.MatOfDouble
import org.opencv.core.MatOfKeyPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.features.BFMatcher
import org.opencv.features.SIFT
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

/**
 * Incremental sparse model driven by the known capture schedule. Features are
 * extracted once, successful edges are merged into multi-image tracks, and a
 * final post-pass/global refinement uses the complete two-camera image set.
 */
class SparseReconstructor(
    private val ratioThreshold: Double = 0.72,
    private val maximumFeatures: Int = 3_000,
    private val maximumPostPassAttempts: Int = 36,
) {
    private val sift = SIFT.create(maximumFeatures)
    private val matcher = BFMatcher.create(Core.NORM_L2, false)
    private val planner = ScheduleMatchPlanner()

    fun reconstruct(frames: List<ReconstructionFrame>): SparseReconstructionResult {
        require(frames.size >= 2)
        require(frames.map(ReconstructionFrame::frameId).distinct().size == frames.size)
        require(frames.groupBy(ReconstructionFrame::cameraIndex).values.all { camera ->
            camera.map(ReconstructionFrame::targetIndex).distinct().size == camera.size
        }) { "Capture schedule contains duplicate target indices" }

        val features = linkedMapOf<String, FrameFeatures>()
        val attempted = linkedSetOf<String>()
        val successful = mutableListOf<MatchedFeatureEdge>()

        fun attempt(candidate: ScheduledMatchCandidate) {
            if (!attempted.add(candidate.id)) return
            val first = features[candidate.firstFrameId] ?: return
            val second = features[candidate.secondFrameId] ?: return
            match(candidate, first, second)?.let(successful::add)
        }

        try {
            frames.forEach { frame ->
                extractFeatures(frame)?.let { features[it.frameId] = it }
            }
            require(features.size >= 2) { "Too few frames produced reusable sparse features" }

            // Replay capture order: each arriving image is connected to several
            // already accumulated schedule neighbors, never a discovered nearest pose.
            planner.incrementalCandidates(frames).forEach(::attempt)

            // Revisit the graph after both cameras exist. One wider cross-camera
            // connection is attempted per 60-degree sector even if the initial
            // graph already supplies enough connective tissue.
            val postCandidates = planner.postPassCandidates(frames, attempted)
            val frameById = frames.associateBy(ReconstructionFrame::frameId)
            val mandatoryPost = linkedMapOf<Int, ScheduledMatchCandidate>()
            postCandidates.filter(ScheduledMatchCandidate::crossCamera).forEach { candidate ->
                val upper = listOf(
                    frameById.getValue(candidate.firstFrameId),
                    frameById.getValue(candidate.secondFrameId),
                ).minBy(ReconstructionFrame::cameraIndex)
                val sector = (upper.scheduledAzimuthDegrees / 60.0).toInt().coerceIn(0, 5)
                mandatoryPost.putIfAbsent(sector, candidate)
            }
            mandatoryPost.values.forEach(::attempt)

            val minimumCrossCamera = planner.minimumUsefulCrossCameraEdges(frames)
            var postAttempts = mandatoryPost.size
            for (candidate in postCandidates) {
                if (postAttempts >= maximumPostPassAttempts) break
                val successfulCrossCamera = successful.count { it.candidate.crossCamera }
                if (successfulCrossCamera >= minimumCrossCamera && candidate.crossCamera &&
                    candidate !in mandatoryPost.values
                ) continue
                if (candidate.id in attempted) continue
                attempt(candidate)
                postAttempts++
            }

            val tracks = buildTracks(features, successful)
            val refinement = GlobalTrackRefiner().refine(frames, tracks)
            return SparseReconstructionResult(
                points = refinement.points,
                matchedEdges = successful.map { it.diagnostic },
                featureTrackCount = refinement.acceptedTrackCount,
                crossCameraEdgeCount = successful.count { it.candidate.crossCamera },
                crossCameraTrackCount = refinement.crossCameraAcceptedTrackCount,
                refinedFrames = frames.map { refinement.refinedFrames[it.frameId] ?: it },
            )
        } finally {
            features.values.forEach(FrameFeatures::close)
        }
    }

    private data class FeatureColor(val red: Int, val green: Int, val blue: Int)

    private data class FrameFeatures(
        val frame: ReconstructionFrame,
        val pixels: Array<Point>,
        val undistortedPixels: Array<Point>,
        val colors: Array<FeatureColor>,
        val descriptors: Mat,
    ) : AutoCloseable {
        val frameId: String get() = frame.frameId
        override fun close() = descriptors.release()
    }

    private data class IndexedFeatureMatch(val firstIndex: Int, val secondIndex: Int)

    private data class MatchedFeatureEdge(
        val candidate: ScheduledMatchCandidate,
        val matches: List<IndexedFeatureMatch>,
        val diagnostic: SparseMatchEdgeResult,
    )

    private fun extractFeatures(frame: ReconstructionFrame): FrameFeatures? {
        val reduction = when {
            maxOf(frame.imageWidth, frame.imageHeight) > 12_000 -> 8
            maxOf(frame.imageWidth, frame.imageHeight) > 6_000 -> 4
            maxOf(frame.imageWidth, frame.imageHeight) > 3_000 -> 2
            else -> 1
        }
        val readFlag = when (reduction) {
            8 -> Imgcodecs.IMREAD_REDUCED_COLOR_8
            4 -> Imgcodecs.IMREAD_REDUCED_COLOR_4
            2 -> Imgcodecs.IMREAD_REDUCED_COLOR_2
            else -> Imgcodecs.IMREAD_COLOR
        }
        val color = Imgcodecs.imread(frame.workingImage.absolutePath, readFlag)
        if (color.empty()) {
            color.release()
            return null
        }
        val gray = Mat()
        val mask = loadMask(frame.foregroundMask, color.size())
        val keypoints = MatOfKeyPoint()
        val descriptors = Mat()
        var keepDescriptors = false
        return try {
            Imgproc.cvtColor(color, gray, Imgproc.COLOR_BGR2GRAY)
            sift.detectAndCompute(gray, mask, keypoints, descriptors)
            val keys = keypoints.toArray()
            if (keys.size < 8 || descriptors.rows() < 8) {
                return null
            }
            val scaleX = color.cols().toDouble() / frame.imageWidth
            val scaleY = color.rows().toDouble() / frame.imageHeight
            val pixels = Array(keys.size) { index -> Point(
                keys[index].pt.x / scaleX,
                keys[index].pt.y / scaleY,
            ) }
            val undistorted = undistortToPixel(pixels, frame)
            val colors = Array(keys.size) { index ->
                val x = keys[index].pt.x.toInt().coerceIn(0, color.cols() - 1)
                val y = keys[index].pt.y.toInt().coerceIn(0, color.rows() - 1)
                val bgr = color[y, x] ?: doubleArrayOf(127.0, 127.0, 127.0)
                FeatureColor(
                    red = bgr[2].toInt().coerceIn(0, 255),
                    green = bgr[1].toInt().coerceIn(0, 255),
                    blue = bgr[0].toInt().coerceIn(0, 255),
                )
            }
            FrameFeatures(frame, pixels, undistorted, colors, descriptors).also {
                keepDescriptors = true
            }
        } finally {
            if (!keepDescriptors) descriptors.release()
            color.release(); gray.release(); mask.release(); keypoints.release()
        }
    }

    private fun match(
        candidate: ScheduledMatchCandidate,
        first: FrameFeatures,
        second: FrameFeatures,
    ): MatchedFeatureEdge? {
        val forwardRows = mutableListOf<MatOfDMatch>()
        val reverseRows = mutableListOf<MatOfDMatch>()
        return try {
            matcher.knnMatch(first.descriptors, second.descriptors, forwardRows, 2)
            matcher.knnMatch(second.descriptors, first.descriptors, reverseRows, 2)
            val forward = ratioMatches(forwardRows)
            val reverse = ratioMatches(reverseRows).associateBy { it.firstIndex }
            val mutual = forward.filter { match ->
                reverse[match.secondIndex]?.secondIndex == match.firstIndex
            }
            if (mutual.size < 8) return null
            val firstPoints = MatOfPoint2f(*mutual.map {
                first.undistortedPixels[it.firstIndex]
            }.toTypedArray())
            val secondPoints = MatOfPoint2f(*mutual.map {
                second.undistortedPixels[it.secondIndex]
            }.toTypedArray())
            val inlierMask = MatOfByte()
            val fundamental = Geometry.findFundamentalMat(
                firstPoints, secondPoints, Geometry.FM_RANSAC, 1.25, 0.995, inlierMask
            )
            try {
                if (fundamental.empty()) return null
                val maskValues = inlierMask.toArray()
                val inliers = mutual.indices.filter {
                    it < maskValues.size && maskValues[it].toInt() != 0
                }.map(mutual::get)
                if (inliers.size < 8) return null
                MatchedFeatureEdge(
                    candidate = candidate,
                    matches = inliers,
                    diagnostic = SparseMatchEdgeResult(
                        firstFrameId = candidate.firstFrameId,
                        secondFrameId = candidate.secondFrameId,
                        phase = candidate.phase,
                        crossCamera = candidate.crossCamera,
                        rawMatchCount = forwardRows.size,
                        ratioMatchCount = mutual.size,
                        geometricInlierCount = inliers.size,
                    ),
                )
            } finally {
                firstPoints.release(); secondPoints.release(); inlierMask.release()
                fundamental.release()
            }
        } finally {
            forwardRows.forEach(MatOfDMatch::release)
            reverseRows.forEach(MatOfDMatch::release)
        }
    }

    /** Returned indices are queryIndex -> trainIndex. */
    private fun ratioMatches(rows: List<MatOfDMatch>): List<IndexedFeatureMatch> =
        rows.mapNotNull { row ->
            val values = row.toArray()
            if (values.size >= 2 && values[0].distance < ratioThreshold * values[1].distance) {
                IndexedFeatureMatch(values[0].queryIdx, values[0].trainIdx)
            } else null
        }

    private fun buildTracks(
        features: Map<String, FrameFeatures>,
        edges: List<MatchedFeatureEdge>,
    ): List<MultiViewFeatureTrack> {
        val orderedFeatures = features.values.sortedWith(
            compareBy<FrameFeatures> { it.frame.cameraIndex }.thenBy { it.frame.targetIndex }
        )
        val offsets = mutableMapOf<String, Int>()
        var totalFeatures = 0
        orderedFeatures.forEach {
            offsets[it.frameId] = totalFeatures
            totalFeatures += it.pixels.size
        }
        val disjoint = DisjointFeatureSet(totalFeatures)
        val used = BooleanArray(totalFeatures)
        val support = IntArray(totalFeatures)
        edges.forEach { edge ->
            val firstOffset = offsets.getValue(edge.candidate.firstFrameId)
            val secondOffset = offsets.getValue(edge.candidate.secondFrameId)
            edge.matches.forEach { match ->
                val firstNode = firstOffset + match.firstIndex
                val secondNode = secondOffset + match.secondIndex
                disjoint.union(firstNode, secondNode)
                used[firstNode] = true; used[secondNode] = true
                support[firstNode]++; support[secondNode]++
            }
        }
        val components = mutableMapOf<Int, MutableList<Int>>()
        used.indices.filter { used[it] }.forEach { node ->
            components.getOrPut(disjoint.find(node)) { mutableListOf() } += node
        }
        val nodeLookup = buildList(totalFeatures) {
            orderedFeatures.forEach { feature ->
                feature.pixels.indices.forEach { featureIndex -> add(feature to featureIndex) }
            }
        }
        return components.values.mapNotNull { nodes ->
            val selected = nodes.groupBy { nodeLookup[it].first.frameId }.values.map { sameFrame ->
                sameFrame.maxBy { support[it] }
            }
            if (selected.size < 2) return@mapNotNull null
            MultiViewFeatureTrack(selected.map { node ->
                val (feature, featureIndex) = nodeLookup[node]
                val color = feature.colors[featureIndex]
                TrackObservation(
                    frameId = feature.frameId,
                    featureIndex = featureIndex,
                    pixel = feature.pixels[featureIndex],
                    undistortedPixel = feature.undistortedPixels[featureIndex],
                    red = color.red,
                    green = color.green,
                    blue = color.blue,
                )
            })
        }
    }

    private class DisjointFeatureSet(size: Int) {
        private val parent = IntArray(size) { it }
        private val rank = ByteArray(size)

        fun find(value: Int): Int {
            var root = value
            while (parent[root] != root) root = parent[root]
            var current = value
            while (parent[current] != current) {
                val next = parent[current]
                parent[current] = root
                current = next
            }
            return root
        }

        fun union(first: Int, second: Int) {
            var firstRoot = find(first)
            var secondRoot = find(second)
            if (firstRoot == secondRoot) return
            if (rank[firstRoot] < rank[secondRoot]) {
                val temporary = firstRoot
                firstRoot = secondRoot
                secondRoot = temporary
            }
            parent[secondRoot] = firstRoot
            if (rank[firstRoot] == rank[secondRoot]) rank[firstRoot]++
        }
    }

    private fun undistortToPixel(pixels: Array<Point>, frame: ReconstructionFrame): Array<Point> {
        val source = MatOfPoint2f(*pixels)
        val destination = MatOfPoint2f()
        val intrinsic = cameraMatrix(frame)
        val distortion = MatOfDouble(*frame.distortion)
        val identity = Mat()
        return try {
            Geometry.undistortPoints(
                source, destination, intrinsic, distortion, identity, intrinsic
            )
            destination.toArray()
        } finally {
            source.release(); destination.release(); intrinsic.release()
            distortion.release(); identity.release()
        }
    }

    private fun cameraMatrix(frame: ReconstructionFrame): Mat =
        Mat.eye(3, 3, CvType.CV_64F).apply {
            put(0, 0, frame.cameraFx); put(1, 1, frame.cameraFy)
            put(0, 2, frame.cameraCx); put(1, 2, frame.cameraCy)
        }

    private fun loadMask(file: java.io.File?, size: org.opencv.core.Size): Mat {
        if (file == null || !file.exists()) return Mat.ones(size, CvType.CV_8UC1)
        val loaded = Imgcodecs.imread(file.absolutePath, Imgcodecs.IMREAD_GRAYSCALE)
        if (loaded.empty()) {
            loaded.release()
            return Mat.ones(size, CvType.CV_8UC1)
        }
        if (loaded.size() == size) return loaded
        return Mat().also {
            Imgproc.resize(loaded, it, size, 0.0, 0.0, Imgproc.INTER_NEAREST)
            loaded.release()
        }
    }
}

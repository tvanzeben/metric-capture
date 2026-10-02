# Metric Capture v0.4.10

Metric Capture is for guided, metric artifact
photogrammetry with the calibrated A–H mat. Capture, reconstruction,
checkpointing, and PLY export run on the phone. No Google Cloud component is
used or required.

This package contains Android source only. It does not contain a prebuilt APK,
SDK, dependency cache, or generated build directory. It uses the included
Gradle 9.5 wrapper, Android SDK Platform 37.0, Android Gradle Plugin 9.3.2,
JDK 17-compatible bytecode, and OpenCV 5.0.0.1. The preceding v0.4.9 source
compiled and started on the Pixel 6. The v0.4.10 live-preview changes still
require an Android Studio build and device run.

## Open and run in Android Studio

1. Extract the ZIP to a normal folder. Do not open the project from inside the
   ZIP.
2. In Android Studio choose **Open** and select the extracted project folder—the
   folder containing `settings.gradle.kts`, `gradlew`, and the `app` directory.
3. Trust the project when prompted. Android Studio should detect the included
   Gradle 9.5 wrapper and begin Gradle sync.
4. If sync reports a missing Android SDK component, open **Tools > SDK Manager**
   and install:
   - Android SDK Platform 37.0;
   - Android SDK Platform-Tools;
   - Android SDK Build-Tools 36.0.0;
   - Google USB Driver only when Windows does not already recognize the Pixel.
5. Leave Android Studio's Gradle JVM management in its current mode. Android
   Studio Panda 1 and newer use **Gradle Daemon JVM criteria**; if Android Studio
   offers to migrate this imported project to that mode, accept the migration.
   The included Gradle 9.5 wrapper runs on JVM 17 through JVM 26, including JVM
6. The Java 17 entries in `app/build.gradle.kts` are the app bytecode target,
   not an instruction to replace Android Studio's Gradle JVM.
7. Allow Gradle sync to download Android Gradle Plugin 9.3.2, its built-in
   Kotlin compiler, OpenCV, AndroidX, and the other declared dependencies.
8. Test on the physical RAW-capable phone rather than an emulator. On the phone,
   enable Developer options and USB debugging, connect it, accept the debugging
   authorization, select the `app` run configuration and the phone in Android
   Studio's toolbar, then click **Run**. Grant camera permission when Metric
   Capture opens.

The project requires Android 12 / API 31 or newer. Camera qualification is a
runtime check: installation alone does not establish that a device exposes the
strict RAW path and controls required by Metric Capture.

## Session workflow

1. At the printed 0° position, place the artifact on the mat and show the full
   mat from an oblique view to the live camera. Then
   tap **Meter and review camera settings**. The app meters the scene and
   proposes ISO, exposure time, focus distance, and aperture. Before presenting
   the proposal, the app automatically takes up to three temporary RAW-only
   focus probes around the metered distance. It registers the mat, detects the
   existing equal gray-step boundaries, and selects the single focus distance
   only after their local contrasts agree with the mat's equal-increment pattern.
   It then selects the single focus distance with the best 90th-percentile
   edge-spread width across the sampled mat depths.
   The probes are bounded analysis frames and are neither saved nor reconstructed.
   A fixed-aperture phone reports its single aperture. A user override is run
   through the same RAW focus validation before it can be locked.
2. The app locks AE, AWB, AF, focus, exposure, ISO, aperture when adjustable,
   OIS, edge enhancement, noise reduction, and gamma-1 tone mapping according
   to the qualified Camera2 capabilities. The selected focus and its measured
   gray-step envelope remain fixed for both camera/view sets. Remove the artifact without moving
   the mat and capture the eight-exposure RAW-only registered empty-mat reference.
3. Replace the artifact and choose an acquisition mode:
   - **Camera orbit:** the mat and artifact remain fixed while the user moves
     the phone.
   - **Turntable:** the phone remains fixed during each pass while the mat and
     artifact rotate rigidly together.
4. At the mat's printed 0° position, align the phone vertically with the
   artifact center, then back up until the entire mat is visible. The printed
   degree marks are instructions for the user; the software never reads or
   OCRs them. Press **Establish user guidance zero**.
5. The baseline-adjusted guidance frame persists for the entire acquisition:
   - upper pass target: 55° camera-position change and 45° downward phone tilt;
   - lower pass target: 25° camera-position change and 25° downward phone tilt.
   These are standardized guidance targets. The actual camera-to-mat pose is
   solved continuously and stored at every accepted exposure.
6. Each pass contains 18 accepted frames at 20° rotational intervals. The upper
   pass begins at 0° and the lower pass at 10°, giving 36 frames whose combined
   rotational positions occur every 10°. This preserves the combined angular
   density previously supplied by three 12-frame, 30°-step passes.
   In reconstruction these are explicitly `cameraIndex = 0` and
   `cameraIndex = 1`: two photogrammetric cameras/view sets. “Upper” and
   “lower round” remain guidance language only.
7. Every exposure requires the projected 150 mm mat perimeter to lie inside the
   sensor frame with a safety margin. The user may tilt or back up as directed;
   the measured deviation is retained. Pose quality, expected-view agreement,
   overlap, artifact sharpness, stepped-gray depth-of-field coverage, mat radiometry, and four-frame temporal
   stability must also pass before the lossless still capture fires. The app then
   captures the selected sensor's largest exposed `RAW_SENSOR` frame in a
   RAW-only request. In every pixel mode, the app builds a bounded-size
   validation frame directly from that exact RAW exposure. A separate bounded
   linear RAW green-channel image measures gray-step edge spread without white
   balance, gamma, tone mapping, or sharpening. Failed PnP, mat containment,
   artifact sharpness, depth-of-field coverage, radiometry, clipping,
   segmentation, or effective-control audits reject the frame.
8. After both passes, tap **Reconstruct metric PLY**. The resource governor may
   throttle or checkpoint reconstruction when memory, storage, battery, or
   thermal conditions are unsafe. Tap again to resume after conditions improve.

## Automatic project saving

Every acquisition uses an app-private project directory. `session.json` is
atomically replaced at each durable milestone so an interrupted write cannot
leave a partially written project manifest. The saved project state includes:

- accepted manual ISO, exposure, focus distance, aperture, camera ID, and the
  RAW stepped-gray depth-of-field calibration;
- the registered empty-mat template path;
- acquisition mode and guidance-zero establishment;
- every accepted exposure and its exact frame ID, schedule target, preserved
  RAW_SENSOR master, full-resolution 16-bit linear TIFF, mask, metadata,
  measured pose, and synchronized sensor record;
- upper-pass completion, lower-pass progress, and capture completion;
- reconstruction start, pause reasons, failure state, and completed PLY path.

Capture commits are idempotent by scheduled frame ID. If a capture callback is
replayed, its manifest entry is replaced rather than counted twice. Image files
are written before the manifest commit, so a frame appears in the project only
after the RAW master, TIFF, mask, and exact-pose metadata exist. Each accepted
frame is stored as `capture-<sensor timestamp>.raw16`, `.tif`, `-mask.png`, and
`.json`. Sparse/refined-pose and
dense-fusion checkpoints remain separate binary reconstruction checkpoints and
are updated after the corresponding completed work and retained until final PLY
export succeeds.

## Guidance and reconstruction are separate

One phone sensor stream has two consumers:

| Consumer | Values | Lifetime | Used by reconstruction |
|---|---|---|---|
| User guidance | Phone attitude adjusted by the person-relative session baseline, plus scene-pose changes relative to the setup view | Entire acquisition | No |
| Capture record | Exact still-frame Camera2/visual pose plus quaternion SLERP from the two unmodified rotation-vector samples bracketing the exposure | Per exposure | Camera/scene pose only |

The baseline itself is never written into the per-frame pose metadata and is
never applied to a projection matrix. Reconstruction reads the capture-time
intrinsics, rotation vector, and translation vector. Planned 55°/45° and
25°/25° values determine where the interface asks the user to go; they never
replace the measured `K[R|t]` values.

Phone inertial sensors measure attitude. Translation and camera-position angle
come from the live metric mat pose; integrating accelerometer noise is not used
as a substitute for position. Both measurements are expressed relative to the
same persistent user-guidance start only inside the guidance controller.

The live YUV PnP pose is Kalman-filtered for stable instructions. Every still
capture runs an independent full fiducial detection on a bounded derivative of
the exact RAW buffer. Its unsmoothed `R`, `t`, and camera center are written to
metadata. A nearby preview pose is never substituted.

## Executable pipeline

- Camera2 live auto-metering followed by user-confirmed manual lock; efficient
  direct `TextureView` display, bounded `YUV_420_888` tracking, and triggered
  maximum-resolution `RAW_SENSOR` capture. The display surface receives Camera2
  buffers continuously and never waits for OpenCV. Recognition samples a packed
  YUV frame at no more than 10 Hz; guidance overlays use the newest completed
  measurement without controlling display cadence.
  Runtime qualification requires RAW, Bayer interpretation metadata, manual
  sensor control, controllable OIS/hot-pixel behavior, intrinsics, and the
  realtime timestamp source needed to pair exposures with rotation-vector samples.
- Up to three sequential RAW-only focus probes with explicit lens-settling
  frames; automatic detection of the mat's existing equal gray-step rings;
  registered 10–90% edge-spread measurements around their visible circumference;
  metric camera-depth association from `R,t`; worst-case candidate selection;
  and one focus lock retained across both photogrammetric camera/view sets.
- Default and maximum-resolution stream maps are queried independently. Every
  accepted still and every empty-mat reference exposure is a RAW-only request.
  Both live display and YUV analysis are paused for the RAW-only exposure and
  then resumed; validation is always derived from the exact RAW buffer.
- Perspective-tolerant contour/ellipse target proposals bounded to a 1280 px
  detection edge, with a Hough-circle fallback, ellipse-normalized
  saddle/quadrant sampling, and full-resolution `cornerSubPix`
  refinement. Candidates are rejected unless the complete refinement window
  lies inside the image, and OpenCV refinement storage is released on every
  path. Separate black/white/yellow/cyan quadrant sampling; one shared
  quarter-turn code hypothesis for the whole A–H layout; target-size/ring and
  geometric assignment; planar PnP plus LM refinement; temporal
  optical flow; and constant-velocity smoothing confined to live guidance.
- Eight sequential RAW-derived registered views form the empty-mat template;
  robust per-frame radiometric fit, negative-space
  subtraction, connected artifact mask, and projected full-mat containment.
- The stepped-gray negative space is also the depth-of-field instrument. Ring
  radii are detected from the registered artwork rather than hard-coded. At an
  oblique view, near-side and far-side portions occupy different camera-space
  depths. Visible portions outside the artifact mask produce a distributed
  focus measurement. Because the printed gray changes are equal, the app also
  rejects boundary sets whose robust local contrasts are inconsistent. The exact
  RAW still must remain within the calibrated edge-spread and equal-step
  envelopes before it can be committed. No additional focus artwork is needed.
- `IncrementalGeometryEngine` live visual-hull carving runs intermittently to
  control guidance load. Reconstruction instead begins from the mat's known
  150 × 150 mm negative-space footprint and a conservative 0–150 mm height;
  live YUV hull state never constrains reconstruction.
- Automatic 36-view capture state machine supporting camera-orbit and turntable
  motion. It permits viewpoint-dependent silhouette changes; it does not treat
  a changing silhouette as evidence that the artifact moved off the mat.
- Exact-timestamp verbatim RAW plane, complete RAW interpretation/provenance
  metadata, streamed 16-bit linear RGB TIFF, still-frame foreground mask,
  unsmoothed PnP pose, SLERP-synchronized phone attitude, JSON, and atomic
  session-manifest record for every accepted exposure.
- The bounded RAW-derived analysis copy applies only the inverse reported
  sensor neutral point and an sRGB display transfer so the finished mat's
  yellow/cyan/black/white quadrant codes remain classifiable. Those gains are
  recorded and never alter the RAW master or linear TIFF.
- Schedule-planned masked SIFT matching, cached descriptors, mutual 0.72 ratio
  checks, epipolar RANSAC, accumulated multi-image feature tracks, multi-view
  metric DLT, global track/camera refinement, rectified SGBM, hull-bounded
  voxel fusion, RGB sampling, and binary Stanford PLY output in millimeters.

### Fiducial code model

Quadrants are stored clockwise as top-left, top-right, bottom-right,
bottom-left in mat artwork coordinates:

| Target | Ring | Printed quadrant code |
|---|---|---|
| A | Outer | cyan, black, cyan, white |
| B | Outer | yellow, white, yellow, black |
| C | Outer | black, white, black, white |
| D | Outer | black, white, black, white |
| E | Inner | white, black, white, black |
| F | Inner | white, black, white, black |
| G | Inner | cyan, white, cyan, black |
| H | Inner | yellow, black, yellow, white |

The decoder samples each quadrant independently and tests one common
quarter-turn for all eight targets. Ring, printed target scale, complete mat
layout, and PnP reprojection then resolve target identity; locally symmetric
black/white pairs are not incorrectly claimed to be unique from color alone.
The accepted common quarter-turn, complete quadrant code, and confidence are
written into each still-frame metadata record.

## Schedule-driven sparse reconstruction

The capture grid is part of the reconstruction record: upper-pass targets are
`0°, 20°, 40° … 340°`; lower-pass targets are `10°, 30°, 50° … 350°`.
`SparseReconstructor` uses those scheduled targets to decide which comparisons
are worth computing. It does not rediscover candidate pairs from noisy measured
azimuths and therefore has no nearest-angle tie to resolve.

- SIFT features and foreground colors are computed once per frame and retained
  only for the sparse stage. Each same-pass frame connects to nearby accumulated
  schedule neighbors. Each lower-pass frame is offered four upper-pass
  candidates at nominal separations of 10°, 10°, 30°, and 30°.
- A connection is retained only after mutual ratio matching and fundamental-
  matrix RANSAC. Successful edges are union into tracks spanning two or more
  images, so a new lower-pass view contributes to one accumulated feature graph
  rather than producing an isolated pair cloud.
- Complete one-to-one cross-pass coverage is not required. For the 18+18 grid,
  reconstruction requires at least six geometrically valid cross-pass edges
  and at least 20 accepted cross-pass feature tracks overall.
- After all 36 views are available, an explicitly bounded post-pass tries
  previously untried schedule-supported connections out to 70°. At least one
  wider cross-pass candidate is revisited in each 60° sector. The resulting
  full graph is triangulated and refined globally before dense stereo begins.
- Global refinement alternates multi-view DLT for every track with LM updates
  for every camera using all tracks visible in that camera. Corrections are
  clamped to the exact still-frame mat-PnP priors, preserving millimeter scale,
  axes, and the `Z = 0` mat datum. Refined camera poses are checkpointed and are
  also used by dense stereo after resume.

## Current boundaries

- Reconstruction never consumes the ISP-produced live YUV as its image master.
  The app first copies the `RAW_SENSOR` plane verbatim, including its row padding,
  and records CFA, stride, active-array, black/white-level, color-transform,
  noise-profile, pixel-mode, exposure, and sensor metadata. It then performs
  explicit black subtraction, white-level normalization, and deterministic
  bilinear demosaicing into an uncompressed 16-bit linear RGB TIFF. No white
  balance, gamma, tone map, sharpening, denoising, shading correction,
  distortion correction, or hot-pixel repair is added to that derivative.
- The YUV stream is deliberately modest in resolution and exists only for
  prospective live display, tracking, instructions, and trigger gating. It does
  not supply the empty-mat template, accepted pose, accepted mask, containment
  result, reconstruction image, or reconstruction volume.
- Each RAW-only still result is audited against the sensor and RAW-affecting
  controls: manual AE/AF state, ISO, exposure, frame duration, focus, aperture,
  OIS, flash, RAW hot-pixel mode, black-level lock, and pixel mode. Processed
  edge/noise/tone/crop/WB controls remain confined to the YUV preview request.
  A reported override rejects the capture. Cameras that cannot expose acceptable
  RAW, disable RAW hot-pixel correction/OIS, or report RAW lens-shading already
  applied to do not qualify for this strict pipeline.
- Sparse and dense reconstruction decode bounded-resolution TIFF views from disk
  and scale their intrinsics/feature coordinates back to the full-resolution
  coordinate system. This preserves the full master while bounding peak RAM.
- Global refinement is implemented as bounded alternating multi-view DLT and
  all-camera OpenCV LM pose updates. It is not a single joint Ceres normal-equation solve over every camera and point.
- Metric scale and `Z = 0` come directly from the known millimeter mat points
  used by every camera pose. A free Guayama similarity transform is not applied
  because it would re-estimate a scale and datum already fixed by the mat.
- Export is a colored binary PLY point cloud with confidence values. Surface
  faces and a texture atlas are outside v0.4.10.
- The latest saved project can be reopened. Its mat reference, acquisition
  mode, completed frames, next scheduled target, reconstruction checkpoints,
  and completed PLY are restored. A restarted process deliberately requires a
  fresh person-relative sensor zero before capture continues; that baseline
  remains guidance-only and is never reconstructed.
- Projects created before schema 6 cannot be continued because they do not have
  mandatory RAW-derived still geometry and the matching interpretation record;
  the app leaves them on disk and starts a new compatible project instead.
  Schema-6 projects are migrated through schema 8 by translating their former
  `roundIndex` capture field into the correct `cameraIndex` identity. Schema-7
  projects remain readable; new captures require the schema-8 stepped-gray
  focus evidence.
- `ARTIFACT_MOVED` is an optional safeguard rather than a release requirement.
  A future implementation must measure artifact motion relative to the mat;
  the removed fixed-silhouette IoU test remains excluded because silhouettes
  correctly change with viewpoint.

## Build and install

Install Android SDK Platform 37.0, Android SDK Build Tools 36.0.0, and Platform
Tools. The included wrapper supplies Gradle 9.5 and supports JVM 17 through JVM
1. From the project root:

```bash
./gradlew --no-daemon testDebugUnitTest lintDebug assembleDebug
adb install -r app/build/outputs/apk/debug/app-debug.apk
adb logcat --pid=$(adb shell pidof -s com.metriccapture.app)
```

Alternatively, open the extracted project root in Android Studio, select the
installed Pixel as the run target, allow Gradle sync to finish, and run the
`app` configuration. The first session should be a short empty-mat/device test
before attempting all 36 captures.

The first connected-device test must verify that the selected phone camera accepts
the efficient YUV preview plus selected RAW stream configuration, produces a
readable full-resolution TIFF and preserved RAW master, reports every requested
manual/processing control as applied, and decodes the ellipse-normalized
fiducials at both pass angles. RAW is mandatory for the strict pipeline.
The device test must also confirm that the printed gray steps yield at least two
stable detected boundaries, that focus probes report at least 20 distributed
samples over a 20 mm camera-depth span, and that deliberate near/far defocus is
rejected by both the live prospective gate and the exact RAW still audit.

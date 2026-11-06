# Problem Statement 2: Architecture for Location and Dimension Verification

## Goal
Beyond "is part X present?", decide whether each present part is **at the correct location (pose)** and has **correct dimensions** relative to the CAD model.

## Core idea
Move from 2D image comparison to **pose-aware comparison against the CAD geometry**, with tolerances per part. Do this in three stages: (1) per-part localisation in the real image, (2) pose estimation against CAD, (3) deviation measurement and tolerance check.

```
 Real image(s) ─► Pre-process ─┐
 (calibrated cam)               ├─► Part-level segmentation ─► Presence ─► Pose estimation ─► Deviation ─► Verdict
 CAD (STEP/mesh) ─► Render +   ─┘        (instance masks)        (PS1)     (6-DoF per part)  & metrology   + report
 mask + part map    GT per-part                                                                 vs tolerance
```

## 1. Inputs and calibration (what makes dimensions measurable)
- **Camera calibration**: intrinsics (checkerboard/Charuco), lens distortion; fixed or known mount.
- **Scale reference**: a fiducial (ArUco/Charuco board) in the scene, or a known datum on the fixture, so pixels map to mm.
- **Native CAD**: use the STEP/mesh, not only a PNG. Rendered views, masks, depth and normals can be regenerated for any pose.
- **Capture**: fixed lighting and background; ideally 2 or more views, or a depth sensor (stereo/ToF/structured light) for true 3D deviation.

## 2. Per-part localisation in the real image
Options, in increasing robustness:
1. **Template/render matching** (the PS1 approach): per-part rendered silhouette matched to real edges. Works for rigid, textured-enough parts.
2. **Promptable segmentation**: use the CAD-derived per-part box or mask as a prompt for SAM/SAM2 to get a real-image instance mask for each part. Overlap or occlusion is handled by per-part prompts plus a decoder that knows the expected neighbours.
3. **Learned detector/segmenter** trained on synthetic data: render the CAD with domain randomisation (lighting, textures, backgrounds, small pose jitter, occlusions) and fine-tune an instance-segmentation model (Mask2Former / YOLO-seg) with one class per part. Fine-tune with a few real images. This removes the need for hand-tuned thresholds and scales to many parts.

## 3. Pose estimation (location and orientation)
- **Global alignment**: estimate the assembly pose first (PnP on fiducials, or silhouette/edge-based registration of the whole CAD to the image). Everything is then expressed in the **assembly frame**, so camera placement does not matter.
- **Per-part refinement**: for each present part, estimate 6-DoF pose relative to the assembly frame:
  - classical: render-and-compare with edge chamfer / differentiable silhouette loss (initialised from the CAD pose);
  - learned: FoundationPose / MegaPose-style CAD-conditioned pose estimators, or keypoint regression trained on synthetic renders then PnP-RANSAC;
  - 3D: with depth or point cloud, run ICP / Generalized-ICP of the part's CAD point cloud onto the segmented points.
- Output per part: `T_actual` (4x4), a confidence and a residual.

## 4. Deviation and dimension measurement
**Location**
- `ΔT = T_cad⁻¹ · T_actual`, giving translation error (mm) and rotation error (deg) per part and per axis.
- Compare against per-part tolerance, with a looser tolerance for floating parts and a tighter one for fixed interfaces.
- Also check *relative* error to mating parts, which is often more meaningful than absolute error.

**Dimensions**
- 2D: fit measured features (length, width, hole diameter, spacing) from the aligned part mask via subpixel edges, convert with scale, compare to the CAD value for the same feature, which is pre-computed from the model's geometry.
- 3D: with a point cloud, compute point-to-surface distance of the aligned scan to the CAD mesh and give a per-part deviation heatmap, RMS and max error.
- Because the CAD is metric and the camera is calibrated, scale error appears as a systematic residual after the rigid fit, which flags a wrong-size part (for example the wrong variant).

**Wrong part vs. wrong location**: if the best-fit pose still leaves a high residual, flag **wrong variant / dimension**. If the residual is low but `ΔT` is high, flag **misplaced**.

## 5. Decision layer
Each part receives a status:
`ABSENT` | `PRESENT-OK` | `PRESENT-MISPLACED` (with Δ values) | `PRESENT-WRONG-DIMENSION` | `UNCERTAIN` (low confidence, occluded → request another view).
Scores are calibrated on known-good and known-bad examples (ROC) rather than set by hand. Thresholds come from engineering tolerances, not from the model.

## 6. Outputs
- Overlay: CAD ghost on the real image, with colour by deviation (green/amber/red) and arrows for translation.
- Per-part table: pose error, dimension error, confidence, verdict.
- Machine-readable JSON for MES/QMS integration, plus image archive for traceability.

## 7. Practical system design
| Layer | Choice |
|---|---|
| Capture | Fixed camera(s), diffuse lighting, fiducials; optional depth camera |
| CAD services | Offline render of per-part masks/depth, feature dimensions extracted from STEP |
| Models | SAM2 for masks, synthetic-trained segmenter and pose network, ICP for refinement |
| Inference | GPU service (e.g. REST/gRPC), roughly 1 to 3 s per assembly |
| Data | Synthetic data generator (Blender/BlenderProc) + small labelled real set; active learning on UNCERTAIN cases |
| Ops | Versioned CAD to model mapping, drift monitoring, human-in-the-loop review UI |

## 8. Risks and mitigations
- **Occlusion between parts** → multi-view capture; reason in 3D; UNCERTAIN state.
- **Symmetric parts** (pose ambiguity) → symmetry-aware loss/metrics, or check against features that break symmetry.
- **Single 2D view** gives poor depth accuracy → add depth/second camera if out-of-plane tolerance matters.
- **Domain gap** (CAD render vs. real material) → domain randomisation, edge/silhouette cues in place of colour, DINOv2 features.
- **Accuracy ceiling**: measurement precision is bounded by pixel size (mm/pixel) and calibration quality; state this limit alongside the tolerance.

## 9. Phased roadmap
1. **Now**: PS1 presence check (this repo) on aligned images.
2. **Phase 2**: add camera calibration and fiducials; 2D pose and dimension deviation from silhouettes, with per-part tolerances.
3. **Phase 3**: synthetic-trained segmenter and 6-DoF pose; ICP refinement.
4. **Phase 4**: multi-view/depth, deviation heatmaps, MES integration, review UI.

# RoGER-SLAM
RoGER-SLAM DEMO and Module code

📺 Videos & Demo

https://github.com/user-attachments/assets/c0129b2c-675d-4269-b131-ff9a2fbdcc20

RoGER-SLAM is a robust 3D Gaussian Splatting–based SLAM system designed to handle severe noise, low illumination, and real sensor degradations.
It introduces three key innovations that significantly improve tracking stability, geometric consistency, and photometric robustness compared with existing 3DGS-SLAM frameworks such as SplaTAM and GS-SLAM.

🔧 Released Modules & Integration Notes

This repository provides the core modules proposed in RoGER-SLAM, allowing researchers to easily migrate our ideas to other 3DGS-SLAM baselines such as SplaTAM or GS-SLAM.
The released components include:

Multi-Scale Importance Gating

Dynamic Keyframe Selection

Visibility-Accumulation–Based Pruning

SP-RoFusion (Structure-Preserving Robust Fusion)

Adaptive Tracking Objective

CLIP Encoder for Robust Feature Extraction

A key element shared across these modules is our recomputed Gaussian importance score.
To support this, we provide a modified differentiable Gaussian renderer that returns an additional re-ordered per-pixel depth map. This enables importance accumulation across frames and is required for multi-view fusion, pruning, and keyframe scheduling.

When integrated together, these modules produce over 50% improvement on clean Replica sequences, achieved purely through enhanced structural consistency and tracking stability—before applying CLIP enhancement. After CLIP feature fusion, the system gains strong robustness against severe noise and low-light degradations.

All core modules are now released.
Upon acceptance of the paper, we will gradually open-source the full SLAM system, including the complete pipeline, training scripts, and evaluation tools.

🌟 Key Features

🔷 1. Structure-Preserving Robust Fusion (SP-RoFusion)

Uses rendered image as low-pass reference

Incorporates depth & edge priors to restore geometric structure

Prevents structural collapse under low-light + noise

🔷 2. Adaptive Tracking Objective

Automatically balances color vs. depth residuals

Removes the non-universality of fixed weighting

Stabilizes gradient updates under illumination fluctuations

🔷 3. CLIP-Based Enhancement Module (Auto-Activated)

Triggered only under low-light or heavy noise

Provides high-level semantic supervision on illumination


🔷 4. Efficient Gaussian Pruning

Visibility-accumulation pruning reduces redundant Gaussians

Achieves 1.4M Gaussians vs. 5.0M in SplaTAM

Improves efficiency & memory footprint

🙏 Acknowledgments

We gratefully acknowledge the authors of SplaTAM for releasing their excellent 3DGS-SLAM framework, which serves as an important baseline and reference implementation for this project. Their open-source contribution has significantly facilitated research progress in Gaussian Splatting–based SLAM systems.

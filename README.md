# RoGER-SLAM
RoGER-SLAM DEMO and Module code

📺 Videos & Demo
https://github.com/user-attachments/assets/c0129b2c-675d-4269-b131-ff9a2fbdcc20

RoGER-SLAM is a robust 3D Gaussian Splatting–based SLAM system designed to handle severe noise, low illumination, and real sensor degradations.
It introduces three key innovations that significantly improve tracking stability, geometric consistency, and photometric robustness compared with existing 3DGS-SLAM frameworks such as SplaTAM and GS-SLAM.

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


📈 Experimental Results
Replica Benchmark
Method	ATE ↓	PSNR ↑	Notes
SplaTAM	0.36	33.1	baseline fails under noise
RoGER-SLAM	0.24	34.8	robust structure + adaptive tracking

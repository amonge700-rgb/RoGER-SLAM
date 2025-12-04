def compute_single_frame_importance_for_mapping(params, dataset, fid, renderer):
    with torch.no_grad():
        color, depth, intrinsics, pose = dataset[fid]
        w2c = torch.linalg.inv(pose)
        intrinsics = intrinsics[:3, :3]
        cam = setup_camera(
            color.shape[1],
            color.shape[0],
            intrinsics.detach().cpu().numpy(),
            w2c.detach().cpu().numpy()
        )
        c2w = torch.inverse(w2c)
        cam_center = c2w[:3, 3]

        dists = torch.norm(params['means3D'] - cam_center[None, :], dim=1) + 1e-6
        beta = 0.05
        weights = torch.exp(-beta * dists)
        weights = weights / weights.max()

        transformed = transform_to_frame(params, fid, gaussians_grad=False, camera_grad=False)
        rendervar = transformed_params2rendervar(params, transformed)
        this_renderer = renderer(raster_settings=cam)
        _, _, accum_weights, _, _ = this_renderer(
            means3D=rendervar["means3D"],
            means2D=rendervar["means2D"],
            colors_precomp=rendervar["colors_precomp"],
            opacities=rendervar["opacities"],
            scales=rendervar["scales"],
            rotations=rendervar["rotations"],
        )

        accum_weights = accum_weights.squeeze()
        imp_score = weights * accum_weights

    print(f"[Keyframe] Frame {fid} imp_score max/min/mean:", imp_score.max().item(), imp_score.min().item(), imp_score.mean().item())
    return imp_score

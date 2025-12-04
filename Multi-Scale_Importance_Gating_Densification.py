# --- [Multi-Scale Importance Gating Densification] ---
            if config['mapping']['add_new_gaussians'] and time_idx > 0:

                # Step 1: use importance to judge densify
                if config['mapping'].get('use_imp_score_filter_for_densify', False):
                    from utils.gs_helpers import compute_single_frame_importance_for_mapping

                    imp_score = compute_single_frame_importance_for_mapping(
                        params=params,
                        dataset=dataset,
                        fid=time_idx,
                        renderer=Renderer
                    )
                    imp_mean = imp_score.mean().item()
                    print(f"[DENSIFY FILTER] Frame {time_idx} importance score = {imp_mean:.4f}")

                    if imp_mean < config['mapping'].get('densify_imp_thres', 0.01):
                        print(f"[SKIP DENSIFY] Frame {time_idx} rejected by importance filter.")
                        # 如果 importance 太低，跳过 densification
                        continue

                # Step 2: prepare input
                if seperate_densification_res:
                    densify_color, densify_depth, _, _ = densify_dataset[time_idx]
                    densify_color = densify_color.permute(2, 0, 1) / 255
                    densify_depth = densify_depth.permute(2, 0, 1)
                    densify_curr_data = {
                        'cam': densify_cam, 'im': densify_color, 'depth': densify_depth,
                        'id': time_idx, 'intrinsics': densify_intrinsics,
                        'w2c': first_frame_w2c, 'iter_gt_w2c_list': curr_gt_w2c
                    }
                else:
                    densify_curr_data = curr_data

                # Step 3:  densify
                params, variables = add_new_gaussians(
                    params, variables, densify_curr_data,
                    config['mapping']['sil_thres'], time_idx,
                    config['mean_sq_dist_method'], config['gaussian_distribution']
                )

        if config['mapping']['use_dynamic_keyframe_selection']:
            from utils.gs_helpers import compute_single_frame_importance_for_keyframe  # 单帧 imp_score 计算函数

            # 计算当前帧的重要性
            imp_score = compute_single_frame_importance_for_keyframe(
                params=params,
                dataset=dataset,
                fid=time_idx,
                renderer=Renderer
            )
            imp_mean = imp_score.mean().item()

            # 加入缓存
            frame_score_buffer.append((time_idx, imp_mean))

            # 每 5 帧触发一次关键帧选择
            if (time_idx + 1) % config['mapping']['dynamic_keyframe_eval_every'] == 0:
                # 选择 imp_score 最大的帧
                top_frame = max(frame_score_buffer, key=lambda x: x[1])
                selected_fid = top_frame[0]

                if selected_fid not in keyframe_time_indices:
                    with torch.no_grad():
                        curr_cam_rot = F.normalize(params['cam_unnorm_rots'][..., selected_fid].detach())
                        curr_cam_tran = params['cam_trans'][..., selected_fid].detach()
                        curr_w2c = torch.eye(4, device='cuda')
                        curr_w2c[:3, :3] = build_rotation(curr_cam_rot)
                        curr_w2c[:3, 3] = curr_cam_tran

                        color, depth, *_ = dataset[selected_fid]

                        color = color.permute(2, 0, 1) / 255
                        depth = depth.permute(2, 0, 1)

                        curr_keyframe = {
                            'id': selected_fid,
                            'est_w2c': curr_w2c,
                            'color': color,
                            'depth': depth,
                            'selected_for_mapping': True
                        }
                        keyframe_list.append(curr_keyframe)
                        keyframe_time_indices.append(selected_fid)
                        print(f"[DYNAMIC] Frame {selected_fid} selected as keyframe.")

                        if config['use_wandb']:
                            wandb_run.log({"Dynamic Keyframe/Selected": 1}, step=time_idx)

                # 清空 buffer，准备下一轮
                frame_score_buffer.clear()

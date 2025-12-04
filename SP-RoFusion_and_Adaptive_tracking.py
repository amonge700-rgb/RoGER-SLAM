def get_loss(params, curr_data, variables, iter_time_idx, loss_weights, use_sil_for_loss,
             sil_thres, use_l1, ignore_outlier_depth_loss, tracking=False, 
             mapping=False, do_ba=False, plot_dir='/home/d1/SplaTAM/tracking_plot', visualize_tracking_loss=True, tracking_iteration=None):
    # Initialize Loss Dictionary
    losses = {}

    if tracking:
        # Get current frame Gaussians, where only the camera pose gets gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx, 
                                             gaussians_grad=False,
                                             camera_grad=True)
    elif mapping:
        if do_ba:
            # Get current frame Gaussians, where both camera pose and Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=True)
        else:
            # Get current frame Gaussians, where only the Gaussians get gradient
            transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                                 gaussians_grad=True,
                                                 camera_grad=False)
    else:
        # Get current frame Gaussians, where only the Gaussians get gradient
        transformed_gaussians = transform_to_frame(params, iter_time_idx,
                                             gaussians_grad=True,
                                             camera_grad=False)

    # Initialize Render Variables
    rendervar = transformed_params2rendervar(params, transformed_gaussians)
    depth_sil_rendervar = transformed_params2depthplussilhouette(params, curr_data['w2c'],
                                                                 transformed_gaussians)

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, accum_weights, accum_weights_count, accum_max_count = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, _, _,= Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]
    presence_sil_mask = (silhouette > sil_thres)
    depth_sq = depth_sil[2, :, :].unsqueeze(0)
    uncertainty = depth_sq - depth**2
    uncertainty = uncertainty.detach()

    # Mask with valid depth values (accounts for outlier depth values)
    nan_mask = (~torch.isnan(depth)) & (~torch.isnan(uncertainty))
    if ignore_outlier_depth_loss:
        depth_error = torch.abs(curr_data['depth'] - depth) * (curr_data['depth'] > 0)
        mask = (depth_error < 10*depth_error.median())
        mask = mask & (curr_data['depth'] > 0)
    else:
        mask = (curr_data['depth'] > 0)
    mask = mask & nan_mask
    # Mask with presence silhouette mask (accounts for empty space)
    if tracking and use_sil_for_loss:
        mask = mask & presence_sil_mask

    # Depth loss
    if use_l1:
        mask = mask.detach()
        if tracking:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].sum()
        else:
            losses['depth'] = torch.abs(curr_data['depth'] - depth)[mask].mean()

    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        # --------- param ---------
        gamma_im = 1 #0.1
        gamma_depth = 2 #0.1
        lambda_ratio = 0.5  # 
        target_ratio = 2  #  L_depth / L_im ≈ 2）
        opacity_thresh = 0.99
        eps = 1e-6
        # ----------------------------

        # residual compute
        color_residual = torch.abs(curr_data['im'] - im)  # (3, H, W)
        color_residual_mean = color_residual.mean(dim=0)  # (H, W)
        depth_residual = torch.abs(curr_data['depth'] - depth)[0]  # (H, W)

        # mask
        sil_mask = (silhouette > opacity_thresh)

        # adaptive
        w_im = gamma_im / (color_residual_mean + gamma_im)
        w_depth = gamma_depth / (depth_residual + gamma_depth)

        # weight
        L_im = (w_im * color_residual_mean)[sil_mask].sum()
        L_depth = (w_depth * depth_residual)[sil_mask].sum()

        # regulation
        ratio_reg = ((L_depth / (L_im + eps)) - target_ratio) ** 2

        # 写入损失字典
        losses['im'] = L_im
        losses['depth'] = L_depth
        losses['ratio'] = lambda_ratio * ratio_reg

        # RGB Loss 
    if tracking and (use_sil_for_loss or ignore_outlier_depth_loss):
        color_mask = torch.tile(mask, (3, 1, 1))
        color_mask = color_mask.detach()
        losses['im'] = torch.abs(curr_data['im'] - im)[color_mask].sum()
    elif tracking:
        losses['im'] = torch.abs(curr_data['im'] - im).sum()
    else:
        losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))



# === Fuse ===
        rgb = curr_data['im']
        pred_rgb = im
        gt_depth = curr_data['depth'].unsqueeze(0)
        pred_depth = depth

        gray = rgb.mean(dim=0, keepdim=True)
        sobel_x = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], device=rgb.device, dtype=rgb.dtype).view(1, 1, 3,
                                                                                                              3) / 8.0
        sobel_y = sobel_x.transpose(2, 3)
        grad_x = F.conv2d(gray.unsqueeze(0), sobel_x, padding=1)
        grad_y = F.conv2d(gray.unsqueeze(0), sobel_y, padding=1)
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2).squeeze(0)

        # 替代 normalize，保留结构
        rgb = rgb / (rgb.mean() + 1e-6)
        depth = gt_depth.squeeze(0).squeeze(0) / (gt_depth.mean() + 1e-6)
        grad = grad.squeeze(0) / (grad.mean() + 1e-6)

        depth_3ch = depth.unsqueeze(0).expand(3, -1, -1)
        grad_3ch = grad.unsqueeze(0).expand(3, -1, -1)

        fused_gt = 0.6 * rgb + 0.3 * depth_3ch + 0.1 * grad_3ch


# === Loss function ===
        color_loss = F.l1_loss(pred_rgb, rgb)
        depth_loss = F.l1_loss(pred_depth, gt_depth)
        illumination_loss = F.l1_loss(pred_rgb, fused_gt)

        illum_weight = min(illumination_loss.item() / (color_loss.item() + 1e-6), 0.1)

        depth_loss = F.l1_loss(pred_depth, gt_depth.squeeze(1))

        losses['color'] = color_loss
        losses['depth'] = depth_loss
        losses['illumination'] = illum_weight * illumination_loss

    if tracking:
        weighted_losses = {
            'im': losses['im'],
            'depth': losses['depth'],
            'ratio': losses['ratio'],
        }
        loss = losses['im'] + losses['depth'] + losses['ratio']
    elif mapping:
        weighted_losses = {
            'im': losses['im'],
            'depth': losses['depth'],
            'illumination': losses['illumination'],
        }
        loss = 0.5*losses['im']+ losses['depth'] + losses['illumination']

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


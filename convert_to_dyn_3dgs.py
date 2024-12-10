import torch
import os
import sys
from argparse import ArgumentParser, Namespace
from arguments import ModelParams
from scene import Scene
from scene.deform_model import DeformModel
from scene.gaussian_model import GaussianModel
import numpy as np


def main(model_params):

    deform = DeformModel(
        K=model_params.K,
        deform_type=model_params.deform_type,
        is_blender=model_params.is_blender,
        skinning=model_params.skinning,
        hyper_dim=model_params.hyper_dim,
        node_num=model_params.node_num,
        pred_opacity=model_params.pred_opacity,
        pred_color=model_params.pred_color,
        use_hash=model_params.use_hash,
        hash_time=model_params.hash_time,
        d_rot_as_res=model_params.d_rot_as_res,
        local_frame=model_params.local_frame,
        progressive_brand_time=model_params.progressive_brand_time,
        max_d_scale=model_params.max_d_scale,
    )
    deform.load_weights(model_params.run_path, iteration=-1)
    print("Loaded deform model")

    # print(deform.deform.network)
    nodes = deform.deform.nodes
    print("nodes", nodes.shape)

    print("skinning", deform.deform.skinning)  # As skin model, discarding KNN weighting (False)
    print("with_node_weight", deform.deform.with_node_weight)  # activates node weight (True)

    print("_node_radius", deform.deform._node_radius.shape)  # torch.exp actvn
    print("_node_weight", deform.deform._node_weight.shape)  # torch.sigmoid actvn
    # exit(0)

    gs_fea_dim = model_params.hyper_dim
    gaussians = GaussianModel(
        model_params.sh_degree,
        fea_dim=gs_fea_dim,
        with_motion_mask=model_params.gs_with_motion_mask,
    )
    print("Gaussian model instantiated")

    scene = Scene(model_params, gaussians, load_iteration=-1, shuffle=False)

    # bg_color = [1, 1, 1] if model_params.white_background else [0, 0, 0]
    # background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    def get_gaussians_at_t(gaussians, deform, t=1.0):
        
        print(t)
        frame_id = torch.tensor([t], dtype=torch.float32, device="cuda")

        time_input = deform.deform.expand_time(frame_id)
        # print("time_input", time_input.shape)

        d_values = deform.step(
            gaussians.get_xyz.detach(),
            time_input,
            feature=gaussians.feature,
            motion_mask=gaussians.motion_mask,
        )
        d_means, d_quats, d_scales, d_opacities, d_rgbs = (
            d_values["d_xyz"],
            d_values["d_rotation"],
            d_values["d_scaling"],
            d_values["d_opacity"],
            d_values["d_color"],
        )

        deformed_gaussians = {}
        
        if d_means is not None:
            means_canonical = gaussians._xyz
            deformed_gaussians["means"] = means_canonical + d_means

        if d_quats is not None:
            quats_canonical = gaussians._rotation
            deformed_gaussians["quats"] = quats_canonical + d_quats
            
        if d_scales is not None:
            scales_canonical = gaussians._scaling
            deformed_gaussians["scales"] = scales_canonical + d_scales
        
        if d_opacities is not None:
            opacities_canonical = gaussians._opacity
            deformed_gaussians["opacities"] = opacities_canonical + d_opacities
        
        if d_rgbs is not None:
            rgbs_canonical = gaussians._features_dc.squeeze(1)
            deformed_gaussians["rgbs"] = rgbs_canonical + d_rgbs
        
        return deformed_gaussians

    sequence_lenght = 200
    ts = np.arange(sequence_lenght) / sequence_lenght
    means = []
    quats = []
    scales = []
    opacities = []
    rgbs = []
    for t in ts:
        deformed_gaussians = get_gaussians_at_t(gaussians, deform, t=t)
        if "means" in deformed_gaussians:
            means.append(deformed_gaussians["means"])
        if "quats" in deformed_gaussians:
            quats.append(deformed_gaussians["quats"])
        if "scales" in deformed_gaussians:
            scales.append(deformed_gaussians["scales"])
        if "opacities" in deformed_gaussians:
            opacities.append(deformed_gaussians["opacities"])
        if "rgbs" in deformed_gaussians:
            rgbs.append(deformed_gaussians["rgbs"])
    # concatenate and transpose
    if len(means) > 0:
        means = torch.stack(means, dim=1)  # (N, T, 3)
        means = torch.transpose(means, 0, 1)  # (T, N, 3)
    else:
        means = None
    if len(quats) > 0:
        quats = torch.stack(quats, dim=1)  # (N, T, 4)
        quats = torch.transpose(quats, 0, 1)  # (T, N, 4)
    else:
        quats = None
    if len(scales) > 0:
        scales = torch.stack(scales, dim=1)  # (N, T, 3)
        scales = torch.transpose(scales, 0, 1)  # (T, N, 3)
    else:
        scales = None
    if len(opacities) > 0:
        opacities = torch.stack(opacities, dim=1)  # (N, T)
        opacities = torch.transpose(opacities, 0, 1)  # (T, N)
    else:
        opacities = None
    if len(rgbs) > 0:
        rgbs = torch.stack(rgbs, dim=1)  # (N, T, 3)
        rgbs = torch.transpose(rgbs, 0, 1)  # (T, N, 3)
    else:
        rgbs = None

    means3D = means.detach().cpu().numpy() if means is not None else gaussians._xyz.detach().cpu().numpy()
    rgb_colors = rgbs if rgbs is not None else gaussians._features_dc.squeeze(1)
    rgb_colors = torch.sigmoid(rgb_colors).detach().cpu().numpy()
    unnorm_rotations = quats.detach().cpu().numpy() if quats is not None else gaussians._rotation.detach().cpu().numpy()
    logit_opacities = opacities.detach().cpu().numpy() if opacities is not None else gaussians._opacity.detach().cpu().numpy()
    log_scales = scales.detach().cpu().numpy() if scales is not None else gaussians._scaling.detach().cpu().numpy()

    params = {
        "means3D": means3D,
        "rgb_colors": rgb_colors,
        "unnorm_rotations": unnorm_rotations,
        "logit_opacities": logit_opacities,
        "log_scales": log_scales,
    }

    # save as npz
    scene_path = os.path.join(model_params.run_path, "params.npz")
    np.savez(scene_path, **params)
    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Conversion script parameters")
    model_params = ModelParams(parser)
    
    dataset_path = "/home/stefano/Data/d-nerf/jumpingjacks"
    run_path = "outputs/d-nerf/jumpingjacks/2024-11-18-110151/"
    
    args = parser.parse_args(sys.argv[1:])
    
    model_params = model_params.extract(args)
    
    print("dataset_path:", model_params.dataset_path)
    print("run_path:", model_params.run_path)
    
    main(model_params)
    print("Done!")

# TODO Add this to config mamba_dim_hidden,args.mamba_d_state,args.mamba_d_conv,args.mamba_expand
# TODO Train one epoch function DONE
# TODO complete the evaluate function DONE 
# TODO complete the config file 
#  test all configurations 
#  Complete checkpoint saving function to save dict differently now DONE 
import argparse

import os
from easydict import EasyDict 

import numpy as np
import pkg_resources
import torch
import wandb
from torch import optim
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

from utils.learning import load_model, AverageMeter, decay_lr_exponentially, load_model_mamba, decay_lr_exponentially_multi_model
from utils.tools import count_param_numbers
from utils.data import Augmenter2D


# Everything will be same except we will add another model specifically mamba to predict the residuals
# the args parser must ensure the pretrained weights are passed, for first experiment we will train mamba on top of a pretrained model later if required move to one of the following or test them one by one by increasing the config
# 1. Train both models together
# 2. Finetune pretrained motion AGformer with full training of mamba head employee different lrs for the pretrained motion AGformer and mamba head

DEBUG =True
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    
    parser.add_argument('--pretrained-agformer', type=str, default=None,
                    help='Path to pretrained AGFormer weights for mamba_head_only mode')
    
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=16, type=int, help='Number of CPU cores')
    parser.add_argument('--use-wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None, type=str)
    parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    opts = parser.parse_args()
    return opts


def train_one_epoch(args:EasyDict, model_agformer: torch.nn.Module, model_mamba_head: torch.nn.Module, gate: torch.nn.Module, train_loader:DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, losses: dict):

    # Putting all three models in Train
    
    if args.training_mode == 'mamba_head_only':
        model_agformer.eval()  # keep frozen in eval mode regardless
    else:
        model_agformer.train()

    model_mamba_head.train()
    # gate.train()


    for x, y in tqdm(train_loader):
        batch_size = x.shape[0]
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            if args.root_rel:
                y = y - y[..., 0:1, :]
            else:
                y[..., 2] = y[..., 2] - y[:, 0:1, 0:1, 2]  # Place the depth of first frame root to be 0

        # Get initial Predictions from AGFormer
        pred = model_agformer(x)  # (N, T, 17, 3)

        if args.mamba_input == "pred":
            mamba_in = pred.detach()
        elif args.mamba_input == "both":
            mamba_in = torch.cat([pred.detach(), x], dim=-1)
        elif args.mamba_input == "raw":
            mamba_in = x
        else:
            print(f"[ERROR] Mamba input type {args.mamba_input} not recognized, please choose from ['pred', 'both', 'raw']")
            exit()
        mamba_out = model_mamba_head(mamba_in) # (N, T, 17, 3)

        # gate_val = gate(pred.detach()) 
     
        # mamba_res = mamba_out * gate_val
        pred_final = pred + (0.1 * mamba_out) # For now we will not use the gate and directly add the mamba output as residual, later we can experiment with using the gate as well
        

        # loss_3d_pos = loss_mpjpe(pred, y)
        # loss_3d_scale = n_mpjpe(pred, y)
        # loss_3d_velocity = loss_velocity(pred, y)
        # loss_lv = loss_limb_var(pred)
        # loss_lg = loss_limb_gt(pred, y)
        # loss_a = loss_angle(pred, y)
        # loss_av = loss_angle_velocity(pred, y)

        # loss_total_agformer = loss_3d_pos + \
        #             args.lambda_scale * loss_3d_scale + \
        #             args.lambda_3d_velocity * loss_3d_velocity + \
        #             args.lambda_lv * loss_lv + \
        #             args.lambda_lg * loss_lg + \
        #             args.lambda_a * loss_a + \
        #             args.lambda_av * loss_av


        residual_gt = (y - pred).detach()
        residual_loss = loss_mpjpe(mamba_out, residual_gt)

        loss_3d_pos_mamba = loss_mpjpe(pred_final, y)
        loss_3d_scale_mamba = n_mpjpe(pred_final, y)
        loss_3d_velocity_mamba = loss_velocity(pred_final, y)
        loss_lv_mamba = loss_limb_var(pred_final)
        loss_lg_mamba = loss_limb_gt(pred_final, y)
        loss_a_mamba = loss_angle(pred_final, y)
        loss_av_mamba = loss_angle_velocity(pred_final, y)

        loss_total_mamba = loss_3d_pos_mamba + \
                    args.lambda_scale * loss_3d_scale_mamba + \
                    args.lambda_3d_velocity * loss_3d_velocity_mamba + \
                    args.lambda_lv * loss_lv_mamba + \
                    args.lambda_lg * loss_lg_mamba + \
                    args.lambda_a * loss_a_mamba + \
                    args.lambda_av * loss_av_mamba + \
                    residual_loss * 0.1

        optimizer.zero_grad()

        loss_total =  loss_total_mamba

        # losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        # losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        # losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        # losses['lv'].update(loss_lv.item(), batch_size)
        # losses['lg'].update(loss_lg.item(), batch_size)
        # losses['angle'].update(loss_a.item(), batch_size)
        # losses['angle_velocity'].update(loss_av.item(), batch_size)
        # losses['total'].update(loss_total.item(), batch_size)
        
        losses['3d_pose'].update(loss_3d_pos_mamba.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale_mamba.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity_mamba.item(), batch_size)
        losses['lv'].update(loss_lv_mamba.item(), batch_size)
        losses['lg'].update(loss_lg_mamba.item(), batch_size)
        losses['angle'].update(loss_a_mamba.item(), batch_size)
        losses['angle_velocity'].update(loss_av_mamba.item(), batch_size)
        losses['total'].update(loss_total_mamba.item(), batch_size)
        losses['residual_loss'].update(residual_loss.item(), batch_size)
        

        loss_total.backward()
        optimizer.step()



def evaluate(args, model_agformer, model_mamba_head, gate, test_loader, datareader, device):
    print("[INFO] Evaluation")
    results_all_agformer = []
    results_all_mamba = []
    model_mamba_head.eval()
    model_agformer.eval()
    # gate.eval()

    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1_agformer = model_agformer(x)

                if args.mamba_input == "pred":
                    mamba_in_1 = predicted_3d_pos_1_agformer.detach()
                elif args.mamba_input == "both":
                    mamba_in_1 = torch.cat([predicted_3d_pos_1_agformer.detach(), x], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in_1 = x
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized")
                    exit()

                residuals_1 = model_mamba_head(mamba_in_1)
                # gate_val_1 = gate(predicted_3d_pos_1_agformer.detach())
                predicted_3d_pos_1 = predicted_3d_pos_1_agformer + residuals_1 * 0.1

                predicted_3d_pos_flip_agformer = model_agformer(batch_input_flip)

                if args.mamba_input == "pred":
                    mamba_in_flip = predicted_3d_pos_flip_agformer.detach()
                elif args.mamba_input == "both":
                    mamba_in_flip = torch.cat([predicted_3d_pos_flip_agformer.detach(), batch_input_flip], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in_flip = batch_input_flip
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized")
                    exit()

                residuals_flip = model_mamba_head(mamba_in_flip)
                # gate_val_flip = gate(predicted_3d_pos_flip_agformer.detach())
                predicted_3d_pos_flip = predicted_3d_pos_flip_agformer + residuals_flip * 0.1

                predicted_3d_pos_mamba   = (predicted_3d_pos_1 + flip_data(predicted_3d_pos_flip)) / 2
                predicted_3d_pos_ag_only = (predicted_3d_pos_1_agformer + flip_data(predicted_3d_pos_flip_agformer)) / 2
            else:
                predicted_3d_pos_ag_only = model_agformer(x)

                if args.mamba_input == "pred":
                    mamba_in = predicted_3d_pos_ag_only.detach()
                elif args.mamba_input == "both":
                    mamba_in = torch.cat([predicted_3d_pos_ag_only.detach(), x], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in = x
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized")
                    exit()

                residuals = model_mamba_head(mamba_in)
                # gate_val = gate(predicted_3d_pos_ag_only.detach())
                predicted_3d_pos_mamba = predicted_3d_pos_ag_only + residuals * 0.1

            if args.root_rel:
                predicted_3d_pos_mamba[:, :, 0, :]   = 0
                predicted_3d_pos_ag_only[:, :, 0, :] = 0
            else:
                y[:, 0, 0, 2] = 0

            results_all_agformer.append(predicted_3d_pos_ag_only.cpu().numpy())
            results_all_mamba.append(predicted_3d_pos_mamba.cpu().numpy())

    # ── Concatenate & denormalize ──────────────────────────────────────────
    results_all_agformer = datareader.denormalize(np.concatenate(results_all_agformer))
    results_all_mamba    = datareader.denormalize(np.concatenate(results_all_mamba))

    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts     = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames       = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips  = frames[split_id_test]
    gt_clips     = gts[split_id_test]

    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips  = frame_clips[:, :-1]
        gt_clips     = gt_clips[:, :-1]

    assert len(results_all_agformer) == len(results_all_mamba) == len(action_clips)

    # ── Accumulators — one set per model ──────────────────────────────────
    e1_all_ag    = np.zeros(num_test_frames)
    e1_all_mamba = np.zeros(num_test_frames)
    e2_all_ag    = np.zeros(num_test_frames)
    e2_all_mamba = np.zeros(num_test_frames)
    acc_err_all_ag    = np.zeros(num_test_frames - 2)
    acc_err_all_mamba = np.zeros(num_test_frames - 2)
    jpe_all_ag    = np.zeros((num_test_frames, args.num_joints))
    jpe_all_mamba = np.zeros((num_test_frames, args.num_joints))
    oc = np.zeros(num_test_frames)

    action_names = sorted(set(datareader.dt_dataset['test']['action']))

    # dicts for agformer-only
    results_ag         = {a: [] for a in action_names}
    results_proc_ag    = {a: [] for a in action_names}
    results_acc_ag     = {a: [] for a in action_names}
    results_joints_ag  = [{a: [] for a in action_names} for _ in range(args.num_joints)]

    # dicts for mamba
    results_mamba      = {a: [] for a in action_names}
    results_proc_mamba = {a: [] for a in action_names}
    results_acc_mamba  = {a: [] for a in action_names}
    results_joints_mamba = [{a: [] for a in action_names} for _ in range(args.num_joints)]

    block_list = ['s_09_act_05_subact_02', 's_09_act_10_subact_02', 's_09_act_13_subact_01']

    # ── Per-clip accumulation ─────────────────────────────────────────────
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue

        frame_list = frame_clips[idx]
        action     = action_clips[idx][0]
        factor     = factor_clips[idx][:, None, None]
        gt         = gt_clips[idx]

        pred_ag    = results_all_agformer[idx] * factor
        pred_mamba = results_all_mamba[idx]    * factor

        # root-relative
        gt         = gt         - gt[:, 0:1, :]
        pred_ag    = pred_ag    - pred_ag[:, 0:1, :]
        pred_mamba = pred_mamba - pred_mamba[:, 0:1, :]

        # MPJPE
        e1_all_ag[frame_list]    += calculate_mpjpe(pred_ag,    gt)
        e1_all_mamba[frame_list] += calculate_mpjpe(pred_mamba, gt)

        # P-MPJPE
        e2_all_ag[frame_list]    += calculate_p_mpjpe(pred_ag,    gt)
        e2_all_mamba[frame_list] += calculate_p_mpjpe(pred_mamba, gt)

        # Per-joint error
        jpe_ag    = calculate_jpe(pred_ag,    gt)
        jpe_mamba = calculate_jpe(pred_mamba, gt)
        for j in range(args.num_joints):
            jpe_all_ag[frame_list, j]    += jpe_ag[:, j]
            jpe_all_mamba[frame_list, j] += jpe_mamba[:, j]

        # Acceleration error
        acc_err_all_ag[frame_list[:-2]]    += calculate_acc_err(pred_ag,    gt)
        acc_err_all_mamba[frame_list[:-2]] += calculate_acc_err(pred_mamba, gt)

        oc[frame_list] += 1

    # ── Per-frame aggregation ─────────────────────────────────────────────
    for idx in range(num_test_frames):
        if oc[idx] == 0:
            continue
        action = actions[idx]
        o = oc[idx]

        results_ag[action].append(e1_all_ag[idx] / o)
        results_proc_ag[action].append(e2_all_ag[idx] / o)
        results_acc_ag[action].append(acc_err_all_ag[idx] / o)

        results_mamba[action].append(e1_all_mamba[idx] / o)
        results_proc_mamba[action].append(e2_all_mamba[idx] / o)
        results_acc_mamba[action].append(acc_err_all_mamba[idx] / o)

        for j in range(args.num_joints):
            results_joints_ag[j][action].append(jpe_all_ag[idx, j] / o)
            results_joints_mamba[j][action].append(jpe_all_mamba[idx, j] / o)

    # ── Final means ───────────────────────────────────────────────────────
    def _mean_over_actions(per_action_dict):
        return np.mean([np.mean(per_action_dict[a]) for a in action_names])

    e1_ag     = _mean_over_actions(results_ag)
    e2_ag     = _mean_over_actions(results_proc_ag)
    acc_ag    = _mean_over_actions(results_acc_ag)

    e1_mb     = _mean_over_actions(results_mamba)
    e2_mb     = _mean_over_actions(results_proc_mamba)
    acc_mb    = _mean_over_actions(results_acc_mamba)

    joint_errors_ag    = np.array([_mean_over_actions(results_joints_ag[j])    for j in range(args.num_joints)])
    joint_errors_mamba = np.array([_mean_over_actions(results_joints_mamba[j]) for j in range(args.num_joints)])

    assert round(e1_ag, 4) == round(np.mean(joint_errors_ag),    4), \
        f"AG MPJPE {e1_ag:.4f} != mean joint errors {np.mean(joint_errors_ag):.4f}"
    assert round(e1_mb, 4) == round(np.mean(joint_errors_mamba), 4), \
        f"Mamba MPJPE {e1_mb:.4f} != mean joint errors {np.mean(joint_errors_mamba):.4f}"

    # ── Print ─────────────────────────────────────────────────────────────
    print('--- AGFormer only ---')
    print(f'Protocol #1 Error (MPJPE):  {e1_ag:.2f} mm')
    print(f'Acceleration error:         {acc_ag:.2f} mm/s²')
    print(f'Protocol #2 Error (P-MPJPE):{e2_ag:.2f} mm')
    print('--- AGFormer + Mamba ---')
    print(f'Protocol #1 Error (MPJPE):  {e1_mb:.2f} mm')
    print(f'Acceleration error:         {acc_mb:.2f} mm/s²')
    print(f'Protocol #2 Error (P-MPJPE):{e2_mb:.2f} mm')
    print('----------')

    return e1_ag, e2_ag, joint_errors_ag, acc_ag, e1_mb, e2_mb, joint_errors_mamba, acc_mb



def save_checkpoint(checkpoint_path, epoch, optimizer, model_agformer, model_mamba_head, gate, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'optimizer': optimizer.state_dict(),
        'model_agformer': model_agformer.state_dict(),
        'model_mamba_head': model_mamba_head.state_dict(),
        'gate': gate.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)



from typing import Tuple
from argparse import Namespace
def load_datasets_and_dataloaders(args:EasyDict, opts:Namespace) -> Tuple[DataLoader, DataLoader, DataReaderH36M]:
    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    common_loader_params = {
        'batch_size': args.batch_size,
        'num_workers': opts.num_cpus - 1,
        'pin_memory': True,
        'prefetch_factor': (opts.num_cpus - 1) // 3,
        'persistent_workers': True
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_params)

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1,
                                data_stride_train=args.n_frames // 3, data_stride_test=args.n_frames,
                                dt_root='data/motion3d', dt_file=args.dt_file)  # Used for H36m evaluation

    return train_loader, test_loader, datareader

def train(args, opts):
    print_args(args)
    create_directory_if_not_exists(opts.new_checkpoint)
    train_loader, test_loader, datareader = load_datasets_and_dataloaders(args, opts)


        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_agformer = load_model(args)
    model_mamba_head, gate = load_model_mamba(args) #TODO Return 2 models from this function a MLP Gate, and A mamba head
    if args.training_mode == 'mamba_head_only':
        if opts.pretrained_agformer is None:
            print("[ERROR] mamba_head_only mode requires --pretrained-agformer path")
            exit()
        ag_ckpt = torch.load(opts.pretrained_agformer,  map_location=lambda storage, loc: storage)
        state_dict = ag_ckpt.get('model', ag_ckpt.get('model_agformer', ag_ckpt))
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model_agformer.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Loaded pretrained AGFormer from {opts.pretrained_agformer}")

    lr_decay = args.lr_decay


        

    if torch.cuda.is_available():
        model_agformer = torch.nn.DataParallel(model_agformer)
        model_mamba_head = torch.nn.DataParallel(model_mamba_head)
        gate = torch.nn.DataParallel(gate)

    model_agformer.to(device)
    model_mamba_head.to(device)
    gate.to(device)

    n_params = count_param_numbers(model_agformer)
    print(f"[INFO] Number of parameters: {n_params:,}")
    params_mamba = count_param_numbers(model_mamba_head)
    print(f"[INFO] Number of parameters in Mamba head: {params_mamba:,}")
    params_gate = count_param_numbers(gate)
    print(f"[INFO] Number of parameters in MLP gate: {params_gate:,}")




    optimizer = torch.optim.AdamW([
    {
        "params": filter(lambda p: p.requires_grad, model_agformer.parameters()),
        "lr": args.learning_rate_motion_former,
        "weight_decay": args.wd_agformer
    },
    {
        "params": filter(lambda p: p.requires_grad, model_mamba_head.parameters()),
        "lr": args.learning_rate_mamba,
        "weight_decay": args.wd_mamba
    },
    {
        "params": filter(lambda p: p.requires_grad, gate.parameters()),
        "lr": args.learning_rate_gate,
        "weight_decay": args.wd_gate
    },
])
    

    epoch_start = 0
    min_mpjpe = float('inf')  # Used for storing the best model
    wandb_id = opts.wandb_run_id if opts.wandb_run_id is not None else wandb.util.generate_id()

    if opts.checkpoint:
        checkpoint_path = os.path.join(opts.checkpoint, opts.checkpoint_dir if opts.checkpoint_dir else "latest_epoch.pth.tr")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

            model_agformer.load_state_dict(checkpoint['model_agformer'], strict=True)
            model_mamba_head.load_state_dict(checkpoint['model_mamba_head'], strict=True)
            gate.load_state_dict(checkpoint['gate'], strict=True)

            print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
            
            if opts.resume:
                epoch_start = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint['optimizer'])
                min_mpjpe = checkpoint['min_mpjpe']
                if 'wandb_id' in checkpoint and opts.wandb_run_id is None:
                    wandb_id = checkpoint['wandb_id']
        else:
            print("[WARN] Checkpoint path is empty. Starting from the beginning")
            opts.resume = False

    if not opts.eval_only:
        if opts.resume:
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        project='MotionMetaFormer',
                        resume="must",
                        settings=wandb.Settings(start_method='fork'))
        else:
            print(f"Run ID: {wandb_id}")
            if opts.use_wandb:
                wandb.init(id=wandb_id,
                        name=opts.wandb_name,
                        project='MotionMetaFormer',
                        settings=wandb.Settings(start_method='fork'))
                wandb.config.update({"run_id": wandb_id})
                wandb.config.update(args)
                installed_packages = {d.project_name: d.version for d in pkg_resources.working_set}
                wandb.config.update({'installed_packages': installed_packages})


    
    checkpoint_path_latest = os.path.join(opts.new_checkpoint, 'latest_epoch.pth.tr')
    checkpoint_path_best = os.path.join(opts.new_checkpoint, 'best_epoch.pth.tr')

    if args.training_mode == 'joint':
        print('[INFO] This mode will train both models simultaneously with the same learning rate, make sure to set the learning rate in config accordingly')


    elif args.training_mode == 'mamba_head_only':
        print('[INFO] This mode will train only the mamba head on top of a pretrained motion AGFormer model, make sure to set the learning rate in config accordingly')
        model_agformer.eval() # Set AGFormer to eval mode since we are not training it
        for p in model_agformer.parameters():
            p.requires_grad = False


    else:
        print(f"[ERROR] Training mode {args.training_mode} not recognized, please choose from ['joint', 'mamba_head_only']")
        exit()

    for epoch in range(epoch_start, args.epochs):
        if opts.eval_only:
            evaluate(args, model_agformer, model_mamba_head,gate, test_loader, datareader, device) # This function now takes in 2 models instead of just one
            exit()

        print(f"[INFO] epoch {epoch}")
        loss_names = [
            '3d_pose', 
            '3d_scale',
            '2d_proj',
            'lg', 
            'lv', 
            '3d_velocity', 
            'angle', 
            'angle_velocity', 
            'total',
            'residual_loss'
            # "3d_pose_mamba",
            # "3d_scale_mamba",
            # "3d_velocity_mamba",
            # "lv_mamba",
            # "lg_mamba",
            # "angle_mamba",
            # "angle_velocity_mamba",
            # "total_mamba",
            ]
        losses = {name: AverageMeter() for name in loss_names}

        

        train_one_epoch(args,model_agformer, model_mamba_head,gate, train_loader, optimizer, device, losses)# This function now takes in 2 models instead of just one

        # Unpack all 8 return values
        e1_ag, e2_ag, joints_error_ag, acc_ag, mpjpe, p_mpjpe, joints_error, acceleration_error = \
            evaluate(args, model_agformer, model_mamba_head, gate, test_loader, datareader, device)

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, optimizer, model_agformer, model_mamba_head, gate, min_mpjpe, wandb_id)
        save_checkpoint(checkpoint_path_latest, epoch, optimizer, model_agformer, model_mamba_head, gate, min_mpjpe, wandb_id)

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]

        if opts.use_wandb:
            wandb.log({
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,
                'train/residual_loss': losses['residual_loss'].avg,

                # Mamba (final model) metrics
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors,

                # AGFormer-only metrics (for comparison)
                'eval_agformer/mpjpe': e1_ag,
                'eval_agformer/acceleration_error': acc_ag,
                'eval_agformer/p-mpjpe': e2_ag,
                'eval_agformer/upper_body_error': np.mean(joints_error_ag[H36M_UPPER_BODY_JOINTS]),
                'eval_agformer/lower_body_error': np.mean(joints_error_ag[H36M_LOWER_BODY_JOINTS]),
            }, step=epoch + 1)

        new_lrs = decay_lr_exponentially_multi_model(optimizer=optimizer, lr_decay=lr_decay)

        for lr in new_lrs:
            print(f'[INFO] Learning rate decayed to {lr:.6e}')

        if opts.use_wandb:
            artifact = wandb.Artifact(f'model', type='model')
            artifact.add_file(checkpoint_path_latest)
            artifact.add_file(checkpoint_path_best)
            wandb.log_artifact(artifact)


def main():
    opts = parse_args()
    set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = False
    args = get_config(opts.config)

    
    
    train(args, opts)


if __name__ == '__main__':
    main()

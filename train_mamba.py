
# TODO Add this to config mamba_dim_hidden,args.mamba_d_state,args.mamba_d_conv,args.mamba_expand
# TODO Train one epoch function DONE
# TODO complete the evaluate function DONE 
# TODO complete the config file 
#  test all configurations 
#  Complete checkpoint saving function to save dict differently now DONE 
import argparse
import os

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

from utils.learning import load_model, AverageMeter, decay_lr_exponentially, load_model_mamba
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
    model_agformer.train()
    model_mamba_head.train()
    gate.train()


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

        gate_val = gate(pred.detach()) 
        mamba_res = mamba_out * gate_val
        pred_final = pred.detach() + mamba_res
        

        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss_lv = loss_limb_var(pred)
        loss_lg = loss_limb_gt(pred, y)
        loss_a = loss_angle(pred, y)
        loss_av = loss_angle_velocity(pred, y)

        loss_total_agformer = loss_3d_pos + \
                    args.lambda_scale * loss_3d_scale + \
                    args.lambda_3d_velocity * loss_3d_velocity + \
                    args.lambda_lv * loss_lv + \
                    args.lambda_lg * loss_lg + \
                    args.lambda_a * loss_a + \
                    args.lambda_av * loss_av


        # residual_gt = (y - pred).detach()

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
                    args.lambda_av * loss_av_mamba

        optimizer.zero_grad()

        loss_total = loss_total_agformer + loss_total_mamba

        losses['3d_pose'].update(loss_3d_pos.item(), batch_size)
        losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
        losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
        losses['lv'].update(loss_lv.item(), batch_size)
        losses['lg'].update(loss_lg.item(), batch_size)
        losses['angle'].update(loss_a.item(), batch_size)
        losses['angle_velocity'].update(loss_av.item(), batch_size)
        losses['total'].update(loss_total.item(), batch_size)
        
        losses['3d_pose_mamba'].update(loss_3d_pos_mamba.item(), batch_size)
        losses['3d_scale_mamba'].update(loss_3d_scale_mamba.item(), batch_size)
        losses['3d_velocity_mamba'].update(loss_3d_velocity_mamba.item(), batch_size)
        losses['lv_mamba'].update(loss_lv_mamba.item(), batch_size)
        losses['lg_mamba'].update(loss_lg_mamba.item(), batch_size)
        losses['angle_mamba'].update(loss_a_mamba.item(), batch_size)
        losses['angle_velocity_mamba'].update(loss_av_mamba.item(), batch_size)
        losses['total_mamba'].update(loss_total_mamba.item(), batch_size)
        

        loss_total.backward()
        optimizer.step()




def evaluate(args, model_agformer,model_mamba_head,gate, test_loader, datareader, device):
    print("[INFO] Evaluation")
    results_all = []
    model_mamba_head.eval()
    model_agformer.eval()
    gate.eval()
    with torch.no_grad():
        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)

            if args.flip:
                batch_input_flip = flip_data(x)
                predicted_3d_pos_1_agformer = model_agformer(x)  # prediction of AGFORMER on original input
                if args.mamba_input == "pred":
                    mamba_in_1 = predicted_3d_pos_1_agformer.detach()
                elif args.mamba_input == "both":
                    mamba_in_1 = torch.cat([predicted_3d_pos_1_agformer.detach(), x], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in_1 = x
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized, please choose from ['pred', 'both', 'raw']")
                    exit()
                residuals_1 = model_mamba_head(mamba_in_1)
                gate_val_1 = gate(predicted_3d_pos_1_agformer.detach())
                mamba_res_1 = residuals_1 * gate_val_1
            
                predicted_3d_pos_1= predicted_3d_pos_1_agformer + mamba_res_1

                predicted_3d_pos_flip_agformer = model_agformer(batch_input_flip) # Prediction of AGFORMER on flipped input
                if args.mamba_input == "pred":
                    mamba_in_flip = predicted_3d_pos_flip_agformer.detach()
                elif args.mamba_input == "both":
                    mamba_in_flip = torch.cat([predicted_3d_pos_flip_agformer.detach(), batch_input_flip], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in_flip = batch_input_flip
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized, please choose from ['pred', 'both', 'raw']")
                    exit()

                residuals_flip = model_mamba_head(mamba_in_flip)
                gate_val_flip = gate(predicted_3d_pos_flip_agformer.detach())
                mamba_res_flip = residuals_flip * gate_val_flip
                predicted_3d_pos_flip = predicted_3d_pos_flip_agformer + mamba_res_flip

                predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_flip) / 2 # Average Prediction of AGFORMER on original and flipped input

            else:
                predicted_3d_pos_agformer = model_agformer(x)
                if args.mamba_input == "pred":
                    mamba_in = predicted_3d_pos_agformer
                elif args.mamba_input == "both":
                    mamba_in = torch.cat([predicted_3d_pos_agformer, x], dim=-1)
                elif args.mamba_input == "raw":
                    mamba_in = x
                else:
                    print(f"[ERROR] Mamba input type {args.mamba_input} not recognized, please choose from ['pred', 'both', 'raw']")
                    exit()
                residuals = model_mamba_head(mamba_in)
                gate_val = gate(predicted_3d_pos_agformer)
                mamba_res = residuals * gate_val
                predicted_3d_pos = predicted_3d_pos_agformer + mamba_res

                
            if args.root_rel:
                predicted_3d_pos[:, :, 0, :] = 0  # [N,T,17,3]
            else:
                y[:, 0, 0, 2] = 0

            results_all.append(predicted_3d_pos.cpu().numpy())

    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    if args.add_velocity:
        action_clips = action_clips[:, :-1]
        factor_clips = factor_clips[:, :-1]
        frame_clips = frame_clips[:, :-1]
        gt_clips = gt_clips[:, :-1]

    assert len(results_all) == len(action_clips)

    e1_all = np.zeros(num_test_frames)
    jpe_all = np.zeros((num_test_frames, args.num_joints))
    e2_all = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    oc = np.zeros(num_test_frames)
    results = {}
    results_procrustes = {}
    results_joints = [{} for _ in range(args.num_joints)]
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
        for joint_idx in range(args.num_joints):
            results_joints[joint_idx][action] = []

    block_list = ['s_09_act_05_subact_02',
                  's_09_act_10_subact_02',
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:, None, None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor

        # Root-relative Errors
        pred = pred - pred[:, 0:1, :]
        gt = gt - gt[:, 0:1, :]
        err1 = calculate_mpjpe(pred, gt)
        jpe = calculate_jpe(pred, gt)
        for joint_idx in range(args.num_joints):
            jpe_all[frame_list, joint_idx] += jpe[:, joint_idx]
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err
        e1_all[frame_list] += err1
        err2 = calculate_p_mpjpe(pred, gt)
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            results_procrustes[action].append(err2)
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_accelaration[action].append(acc_err)
            for joint_idx in range(args.num_joints):
                jpe = jpe_all[idx, joint_idx] / oc[idx]
                results_joints[joint_idx][action].append(jpe)
    final_result_procrustes = []
    final_result_joints = [[] for _ in range(args.num_joints)]
    final_result_acceleration = []
    final_result = []

    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
        for joint_idx in range(args.num_joints):
            final_result_joints[joint_idx].append(np.mean(results_joints[joint_idx][action]))

    joint_errors = []
    for joint_idx in range(args.num_joints):
        joint_errors.append(
            np.mean(np.array(final_result_joints[joint_idx]))
        )
    joint_errors = np.array(joint_errors)
    e1 = np.mean(np.array(final_result))
    assert round(e1, 4) == round(np.mean(joint_errors), 4), f"MPJPE {e1:.4f} is not equal to mean of joint errors {np.mean(joint_errors):.4f}"
    acceleration_error = np.mean(np.array(final_result_acceleration))
    e2 = np.mean(np.array(final_result_procrustes))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Acceleration error:', acceleration_error, 'mm/s^2')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('----------')
    return e1, e2, joint_errors, acceleration_error


def save_checkpoint(checkpoint_path, epoch, lr, optimizer, model_agformer, model_mamba_head, gate, min_mpjpe, wandb_id):
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_agformer': model_agformer.state_dict(),
        'model_mamba_head': model_mamba_head.state_dict(),
        'gate': gate.state_dict(),
        'min_mpjpe': min_mpjpe,
        'wandb_id': wandb_id,
    }, checkpoint_path)



from typing import Tuple
from easydict import EasyDict 
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


    # TODO Define these params in config
    lr_motion_former = args.learning_rate_motion_former 
    lr_mamba = args.learning_rate_mamba
    lr_gate = args.learning_rate_gate
    

    # TODO Define these params in config
    lr_decay = args.lr_decay_agformer
    lr_decay_mamba = args.lr_decay_mamba
    lr_decay_gate = args.lr_decay_gate

    # TODO Define these params in config
    weight_decay_motion_former = args.weight_decay
    weight_decay_mamba = args.weight_decay_mamba
    weight_decay_gate = args.weight_decay_gate


    optimizer = torch.optim.AdamW([
    {
        "params": filter(lambda p: p.requires_grad, model_agformer.parameters()),
        "lr": args.lr_agformer,
        "weight_decay": args.wd_agformer
    },
    {
        "params": filter(lambda p: p.requires_grad, model_mamba_head.parameters()),
        "lr": args.lr_mamba,
        "weight_decay": args.wd_mamba
    },
    {
        "params": filter(lambda p: p.requires_grad, gate.parameters()),
        "lr": args.lr_gate,
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
                lr = checkpoint['lr']
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
        loss_names = ['3d_pose', '3d_scale', '2d_proj', 'lg', 'lv', '3d_velocity', 'angle', 'angle_velocity', 'total']
        losses = {name: AverageMeter() for name in loss_names}

        train_one_epoch(args,model_agformer, model_mamba_head,gate, train_loader, optimizer, device, losses)# This function now takes in 2 models instead of just one

        mpjpe, p_mpjpe, joints_error, acceleration_error = evaluate(args, model_agformer, model_mamba_head,gate, test_loader, datareader, device)# This function now takes in 2 models instead of just one

        if mpjpe < min_mpjpe:
            min_mpjpe = mpjpe
            save_checkpoint(checkpoint_path_best, epoch, lr, optimizer, model_mamba_head, min_mpjpe, wandb_id) # This function now saves the mamba head model 
        save_checkpoint(checkpoint_path_latest, epoch, lr, optimizer, model_mamba_head, min_mpjpe, wandb_id) # This function now saves the mamba head model 

        joint_label_errors = {}
        for joint_idx in range(args.num_joints):
            joint_label_errors[f"eval_joints/{H36M_JOINT_TO_LABEL[joint_idx]}"] = joints_error[joint_idx]
        if opts.use_wandb:
            wandb.log({
                'lr': lr,
                'train/loss_3d_pose': losses['3d_pose'].avg,
                'train/loss_3d_scale': losses['3d_scale'].avg,
                'train/loss_3d_velocity': losses['3d_velocity'].avg,
                'train/loss_2d_proj': losses['2d_proj'].avg,
                'train/loss_lg': losses['lg'].avg,
                'train/loss_lv': losses['lv'].avg,
                'train/loss_angle': losses['angle'].avg,
                'train/angle_velocity': losses['angle_velocity'].avg,
                'train/total': losses['total'].avg,

                'train/loss_3d_pose_with_mamba': losses['3d_pose_mamba'].avg,
                'train/loss_3d_scale_with_mamba': losses['3d_scale_mamba'].avg,
                'train/loss_3d_velocity_with_mamba': losses['3d_velocity_mamba'].avg,
                'train/loss_2d_proj_with_mamba': losses['2d_proj_mamba'].avg,
                'train/loss_lg_with_mamba': losses['lg_mamba'].avg,
                'train/loss_lv_with_mamba': losses['lv_mamba'].avg,
                'train/loss_angle_with_mamba': losses['angle_mamba'].avg,
                'train/angle_velocity_with_mamba': losses['angle_velocity_mamba'].avg,
                'train/total_with_mamba': losses['total_mamba'].avg,

                
                
                'eval/mpjpe': mpjpe,
                'eval/acceleration_error': acceleration_error,
                'eval/min_mpjpe': min_mpjpe,
                'eval/p-mpjpe': p_mpjpe,
                'eval_additional/upper_body_error': np.mean(joints_error[H36M_UPPER_BODY_JOINTS]),
                'eval_additional/lower_body_error': np.mean(joints_error[H36M_LOWER_BODY_JOINTS]),
                'eval_additional/1_DF_error': np.mean(joints_error[H36M_1_DF]),
                'eval_additional/2_DF_error': np.mean(joints_error[H36M_2_DF]),
                'eval_additional/3_DF_error': np.mean(joints_error[H36M_3_DF]),
                **joint_label_errors
            }, step=epoch + 1)



        # TODO This needs to change to show decay in all three learning rates
        lr = decay_lr_exponentially(lr, lr_decay, optimizer)

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

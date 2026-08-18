import logger
import psutil
import os.path as osp

from tqdm import tqdm
from options import Options
from ckpt_util import load_config
from data.videoloader import create_dataloader
from wan.util import instantiate_from_config, default, rank_zero_print

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


MAXM_VRAM = 40960           # Default GPU: A100-SXM4-40GB

def get_configurations():
    parser = Options(eval=False)
    opt = parser.get_options()
    opt.mode = "train"
    device_num = torch.cuda.device_count()
    configs = load_config(default(opt.config_file, opt.model_config_file))

    if opt.dynamic_lr:
        base_lr = configs.model.base_learning_rate
        opt.lr = base_lr * opt.batch_size * opt.acumulate_batch_size * len(opt.gpus)
    parser.print_options(opt)
    return opt, configs, device_num


def get_system_memory_usage_gb():
    memory = psutil.virtual_memory()
    return memory.used / (1024 ** 3), memory.used / memory.total * 100.


if __name__ == '__main__':
    opt, configs, device_num = get_configurations()

    # setup data loader
    dataloader, data_size = create_dataloader(opt, configs.dataloader, device_num)
    # dataloader, data_size = create_test_dataloader()

    # setup huggingface accelerator EARLY to enable distributed-aware loading
    projection_config = ProjectConfiguration(project_dir=opt.ckpt_path)
    accelerator = Accelerator(
        mixed_precision = opt.precision,
        log_with = "wandb",
        project_config = projection_config,
        gradient_accumulation_steps = opt.acumulate_batch_size if hasattr(opt, 'accumulate_batch_size') else 1,
    )

    # For FSDP with sync_module_states: non-main ranks skip transformer weight loading
    # (weights will be broadcast from rank 0 during accelerator.prepare())
    from accelerate.utils import DistributedType
    from omegaconf import OmegaConf
    if (accelerator.distributed_type == DistributedType.FSDP
            and not accelerator.is_main_process):
        OmegaConf.update(configs, "model.params._skip_model_loading", True, force_add=True)

    # Stagger I/O: local_rank 0 per node loads first, warming OS page cache
    # This dramatically reduces shared filesystem contention in multi-node setups
    with accelerator.local_main_process_first():
        model = instantiate_from_config(configs.model)

    params = model.get_trainable_params()
    optimizer = torch.optim.AdamW(params, lr=opt.lr)
    if accelerator.is_main_process:
        tqdm.write(
            f"Total parameter number: {sum(p.numel() for m in model.model_list for p in m.parameters()) / 1000 ** 3:.2f} B, "
            f"trainable parameter number: {sum(p.numel() for p in params) / 1000 ** 3:.2f} B !!!"
        )

    if opt.pretrained is not None:
        with accelerator.local_main_process_first():
            model.init_from_ckpt(opt.pretrained, make_it_fit=opt.fitting_model)

    # Prepare with explicit tensor contiguous state for DeepSpeed
    model.train()
    if isinstance(model, torch.nn.Module):
        model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    else:
        model.model, optimizer, dataloader = accelerator.prepare(model.model, optimizer, dataloader)

    # for n, p in model.model.named_parameters():
    #     if p.requires_grad:
    #         print(n, p.shape)

    # resume training
    if opt.load_checkpoint:
        r"""
        To avoid the default weights_only=True since 2.6.0,
        set load_kwargs={"weights_only": False} explicitly
        """
        accelerator.load_state(opt.load_checkpoint, load_kwargs={"weights_only": False})
        if accelerator.is_main_process:
            tqdm.write(f"Loaded checkpoint from {opt.load_checkpoint}")

    # setup loggers
    # TODO: Check if there are alternative methods in huggingface libraries
    vars_opt = vars(opt)
    batch_per_epoch = len(dataloader)
    ckpt_callback = logger.CustomCheckpoint(**vars_opt)
    # vis_logger must be created on ALL ranks to avoid NCCL deadlock during log_images
    # (FSDP forward pass in log_images requires collective ops from all ranks)
    vis_logger = logger.ImageLogger(**configs.logger, **vars_opt, accelerator=accelerator)
    if accelerator.is_main_process:
        cli_logger = logger.ConsoleLogger(batch_per_epoch=batch_per_epoch, **vars_opt)
        pbar_epoch = tqdm(initial=opt.start_epoch, total=opt.epoch, desc="Training process")

    # start training
    global_step = 0
    model.training = True
    model.on_train_start(accelerator)
    ckpt_callback.on_train_start(accelerator)
    accelerator.init_trackers(
        project_name="animate-training",
        config={
            "learning_rate": opt.lr,
            "batch_size": opt.batch_size,
            "epochs": opt.epoch,
            "model_config": opt.config_file,
        },
        init_kwargs={"wandb": {"name": opt.name}}
    )
    for epoch in range(opt.start_epoch, opt.epoch):
        if accelerator.is_main_process:
            pbar_iter = tqdm(total=batch_per_epoch, desc=f"Current epoch {epoch}, process")

        for idx, batch in enumerate(dataloader):
            # forward and backward
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                loss = model.training_step(batch)
                accelerator.backward(loss)
                optimizer.step()

            # batch end callbacks
            model.on_train_batch_end()

            if accelerator.sync_gradients:
                loss_value = accelerator.gather(loss.detach()).mean().item()
                loss_dict = {"loss": loss_value}
                ckpt_callback.on_train_batch_end(accelerator, global_step, idx)

                # vis_logger.on_train_batch_end must run on ALL ranks to avoid NCCL deadlock
                # (FSDP forward pass in log_images requires collective ops from all ranks)
                # The logger internally only saves/logs on main process
                training_state = {
                    "max_epoch": opt.epoch,
                    "learning_rate": opt.lr,
                    "loss_dict": loss_dict,
                    "batch": batch,
                    "batch_idx": idx,
                    "global_step": global_step,
                    "current_epoch": epoch,
                }
                vis_logger.on_train_batch_end(model, **training_state)

                if accelerator.is_main_process:
                    # Enhanced logging with more metrics for tensorboard
                    logging_dict = {
                        "train/loss": loss_value,
                        "train/learning_rate": optimizer.param_groups[0]['lr'],
                        "train/epoch": epoch,
                        "system/gpu_memory_allocated_mb": torch.cuda.memory_allocated() / (1024 ** 2),
                        "system/gpu_memory_reserved_mb": torch.cuda.memory_reserved() / (1024 ** 2),
                    }

                    accelerator.log(logging_dict, step=global_step)
                    cli_logger.on_train_batch_end(**training_state)
                    pbar_iter.set_postfix(loss_dict)
                    pbar_iter.update(1)
                    del logging_dict
                del training_state

            global_step += 1
            del loss_dict

        if accelerator.sync_gradients:
            # epoch end callbacks
            ckpt_callback.on_train_epoch_end(accelerator, epoch)
            if accelerator.is_main_process:
                cli_logger.on_train_epoch_end(epoch, global_step, opt.lr)
                pbar_epoch.update(1)
                pbar_iter.close()

    accelerator.save_state(f"{osp.join(opt.ckpt_path, 'final')}")
    accelerator.end_training()
    if accelerator.is_main_process:
        pbar_epoch.close()
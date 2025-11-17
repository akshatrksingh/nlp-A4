import os
import torch
import transformers
from transformers import T5ForConditionalGeneration, T5Config
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
import wandb
from load_data import TOKENIZER, PAD_IDX 

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def setup_wandb(args):
    pass

def initialize_model(args):
    model_name = "google-t5/t5-small"

    if args.finetune:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        cfg = T5Config.from_pretrained(model_name)
        model = T5ForConditionalGeneration(cfg)

    # Make sure model & tokenizer are fully aligned
    model.resize_token_embeddings(len(TOKENIZER))

    model.config.pad_token_id = PAD_IDX
    model.config.decoder_start_token_id = PAD_IDX   # we use pad as BOS in load_data.py
    if model.config.eos_token_id is None:
        model.config.eos_token_id = TOKENIZER.eos_token_id

    # OPTIONAL: freeze some bottom encoder layers (helps stability, and is an "architecture choice")
    # You can tune this number or add an arg; here we freeze the first 2 encoder blocks by default.
    num_freeze = getattr(args, "freeze_encoder_layers", 0)  # if you add this arg in train_t5.py
    if num_freeze > 0:
        freeze_encoder_bottom_k_layers(model, k=num_freeze)

    model.to(DEVICE)
    return model

def mkdir(dirpath):
    if not os.path.exists(dirpath):
        try:
            os.makedirs(dirpath)
        except FileExistsError:
            pass

def save_model(checkpoint_dir, model, best):
    sub = "best" if best else "last"
    outdir = os.path.join(checkpoint_dir, sub)
    mkdir(outdir)
    model.save_pretrained(outdir)

def load_model_from_checkpoint(args, best):

    sub = "best" if best else "last"
    ckpt_dir = os.path.join(args.checkpoint_dir, sub)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt_dir}")
    model = T5ForConditionalGeneration.from_pretrained(ckpt_dir).to(DEVICE)
    model.eval()
    return model

def initialize_optimizer_and_scheduler(args, model, epoch_length):
    optimizer = initialize_optimizer(args, model)
    scheduler = initialize_scheduler(args, optimizer, epoch_length)
    return optimizer, scheduler

def initialize_optimizer(args, model):
    if args.optimizer_type != "diff_llrd":
        # fallback to vanilla AdamW
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
    base_lr = args.learning_rate
    wd = args.weight_decay
    decay_rate = 0.9          # typical decay factor
    no_decay = ("bias", "layer_norm.weight", "layernorm.weight")

    # ---------------------------------------------
    # Collect parameter groups per layer index
    # ---------------------------------------------
    optimizer_groups = []

    def add_param_group(params, lr, weight_decay):
        if len(params) > 0:
            optimizer_groups.append({
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay,
            })

    # Helper: put parameters into decay/no-decay lists
    def split_decay(param_dict):
        decay, no_decay_list = [], []
        for name, p in param_dict.items():
            if any(nd in name.lower() for nd in no_decay):
                no_decay_list.append(p)
            else:
                decay.append(p)
        return decay, no_decay_list

    # ---------------------------------------------
    # Handle encoder layers: encoder.block[i]
    # ---------------------------------------------
    if hasattr(model, "encoder") and hasattr(model.encoder, "block"):
        num_enc = len(model.encoder.block)
        for i, block in enumerate(model.encoder.block):
            # deeper layers → larger LR
            layer_lr = base_lr * (decay_rate ** (num_enc - 1 - i))

            enc_params = {n: p for n, p in block.named_parameters() if p.requires_grad}
            dec, nde = split_decay(enc_params)

            add_param_group(dec, layer_lr, wd)
            add_param_group(nde, layer_lr, 0.0)

    # ---------------------------------------------
    # Handle decoder layers: decoder.block[i]
    # decoder usually gets higher LR
    # ---------------------------------------------
    if hasattr(model, "decoder") and hasattr(model.decoder, "block"):
        num_dec = len(model.decoder.block)
        for i, block in enumerate(model.decoder.block):
            layer_lr = base_lr * (decay_rate ** (num_dec - 1 - i)) * 2.0  # ×2 over encoder

            dec_params = {n: p for n, p in block.named_parameters() if p.requires_grad}
            d, nd = split_decay(dec_params)

            add_param_group(d, layer_lr, wd)
            add_param_group(nd, layer_lr, 0.0)

    # ---------------------------------------------
    # LM head (highest LR)
    # ---------------------------------------------
    if hasattr(model, "lm_head"):
        head_params = {n: p for n, p in model.lm_head.named_parameters()}
        d, nd = split_decay(head_params)
        add_param_group(d, base_lr * 3.0, wd)
        add_param_group(nd, base_lr * 3.0, 0.0)

    # ---------------------------------------------
    # Finally, construct optimizer
    # ---------------------------------------------
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        lr=base_lr,
        eps=1e-8,
        betas=(0.9, 0.999),
    )

    # Optional: print group summary for debugging
    print(f"[Optimizer] Created {len(optimizer_groups)} parameter groups (LLRD enabled).")

    return optimizer
        
def initialize_scheduler(args, optimizer, epoch_length):
    num_training_steps = epoch_length * args.max_n_epochs
    num_warmup_steps = epoch_length * args.num_warmup_epochs

    if args.scheduler_type == "none":
        return None
    elif args.scheduler_type == "cosine":
        return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    elif args.scheduler_type == "linear":
        return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    else:
        raise NotImplementedError

def get_parameter_names(model, forbidden_layer_types):
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    result += list(model._parameters.keys())
    return result

def freeze_encoder_bottom_k_layers(model, k=2):
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "block"):
        return

    encoder_blocks = model.encoder.block
    k = min(k, len(encoder_blocks))

    for i in range(k):
        for param in encoder_blocks[i].parameters():
            param.requires_grad = False

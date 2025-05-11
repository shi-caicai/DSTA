import os
from tqdm import tqdm
import torch,gc
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from src.model import llama_model_path
from src.dataset.my_dataset import MyDataset
from src.model.graph_llm import GraphLLM
from src.utils.evaluate import cal_result
from src.config import parse_args_llama
from src.utils.ckpt import _save_checkpoint, _reload_best_model
from src.utils.collate import collate_fn
from src.utils.seed import seed_everything
from src.utils.lr_schedule import adjust_learning_rate

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:5860'

def main(args):
    seed_everything(seed=args.seed)
    print(args)

    dataset = MyDataset(args)
    idx_split = dataset.get_idx_split()

    train_dataset = [dataset[i] for i in idx_split['train']]
    val_dataset = [dataset[i] for i in idx_split['val']]
    test_dataset = [dataset[i] for i in idx_split['test']]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, pin_memory=True,shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False, collate_fn=collate_fn)

    # Build Model
    if args.llm_model_name == 'llama2-13b':
        args.gnn_hidden_dim = 1280
    args.llm_model_path = llama_model_path[args.llm_model_name]
    model = GraphLLM(args=args, init_prompt=dataset.prompt)

    # Set Optimize
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}, ],
        betas=(0.9, 0.95)
    )
    trainable_params, all_param = model.print_trainable_params()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")

    # Training
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss, accum_loss = 0., 0.
        for step, batch in enumerate(train_loader):
            # print(batch['id'])
            # print("\n")
            optimizer.zero_grad()
            loss = model(batch)
            if not loss.requires_grad or torch.isnan(loss) or torch.isinf(loss):
                continue
            loss.backward()

            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)

            if (step + 1) % args.grad_steps == 0:
                adjust_learning_rate(optimizer.param_groups[0], args.lr, step / len(train_loader) + epoch, args)

            optimizer.step()
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                accum_loss = 0.

            progress_bar.update(1)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()

        print(f"Epoch: {epoch}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader)}")

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()

        val_loss = 0.
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            val_loss = val_loss/len(val_loader)
            print(f"Epoch: {epoch}|{args.num_epochs}: Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, optimizer, epoch, args, is_best=True)
            best_epoch = epoch

        print(f'Epoch {epoch} Val Loss {val_loss} Best Val Loss {best_val_loss} Best Epoch {best_epoch}')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()

    # Evaluating
    if args.rationale_yes_no:
        args.max_new_tokens = 300
    model = _reload_best_model(model, args)
    model.eval()
    eval_output = []
    progress_bar_test = tqdm(range(len(test_loader)))
    for step, batch in enumerate(test_loader):
        with torch.no_grad():
            output = model.inference(batch)
            eval_output.append(output)

        progress_bar_test.update(1)

    # Post-processing & compute metrics
    acc, f1_real, f1_fake, macro_f1 = cal_result(eval_output)
    print(f'\nTest Acc {acc} | F1 real {f1_real} | F1 fake {f1_fake} | Macro F1 {macro_f1}')


if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    args = parse_args_llama()
    args.dataset = 'gossipcop'
    args.llm_model_name = 'llama2-13b'
    args.llm_frozen = 'False'
    args.seed = 1
    args.batch_size = 8
    args.gnn_model_name = 'gat'
    main(args)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # 手动触发垃圾回收
    gc.collect()
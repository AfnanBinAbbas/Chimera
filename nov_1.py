import argparse
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import Callback, EarlyStopping
from sklearn.metrics import ndcg_score

# ----------------------------------------------------------------------
# Assume these modules exist in your project.
from src.models import Chimera
from src.dataset import load_json
from src.utils import get_abs_path

# ----------------------------------------------------------------------
# 1. Gradient Reversal Layer
# ----------------------------------------------------------------------
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

def grad_reverse(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)

# ----------------------------------------------------------------------
# 2. Cross‑Domain Chimera Model
# ----------------------------------------------------------------------
class CrossDomainChimera(pl.LightningModule):
    def __init__(self, cfg, embedding_dict, num_domains, rca_dim,
                 domain_lambda=0.1, lambd_max=1.0, warmup_epochs=10):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Base anomaly detection model
        self.base_model = Chimera(cfg, embedding_dict)
        self.rca_dim = rca_dim   # fix RCA dimension (e.g., 20)

        # Determine hidden size
        hidden_dim = getattr(cfg, 'hidden_dim', 256)
        if hasattr(self.base_model, 'hidden_dim'):
            hidden_dim = self.base_model.hidden_dim

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_domains)
        )

        self.num_domains = num_domains
        self.domain_lambda = domain_lambda
        self.lambd_max = lambd_max
        self.warmup_epochs = warmup_epochs

    def set_domain_lambda(self, value):
        self.domain_lambda = value

    def forward(self, batch):
        return self.base_model(batch)

    def _get_features(self, seqs_list):
        """Extract shared representation from the encoder."""
        max_len = max(len(seq) for seq in seqs_list)
        padded = torch.zeros(len(seqs_list), max_len, dtype=torch.long)
        for i, seq in enumerate(seqs_list):
            padded[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        embedded = self.base_model.embedding(padded)
        encoder = self.base_model.ad_encoder
        features = encoder(embedded)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            features = features[:, -1, :]
        return features

    def training_step(self, batch, batch_idx):
        seqs_list, domain_labels, (ad_label, rca_label) = batch

        # Anomaly loss
        base_out = self.base_model((seqs_list, ad_label, rca_label))
        anomaly_logits, anomaly_score, _, _ = base_out[:4]
        loss_anomaly = F.binary_cross_entropy_with_logits(
            anomaly_logits[:, 1] - anomaly_logits[:, 0],
            ad_label.float()
        )

        # Domain adversarial loss
        features = self._get_features(seqs_list)
        reversed_features = grad_reverse(features, self.domain_lambda)
        domain_logits = self.domain_classifier(reversed_features)
        loss_domain = F.cross_entropy(domain_logits, domain_labels)

        total_loss = loss_anomaly + self.domain_lambda * loss_domain
        self.log('train_loss', total_loss, prog_bar=True)
        self.log('loss_anomaly', loss_anomaly)
        self.log('loss_domain', loss_domain)
        return total_loss

    def validation_step(self, batch, batch_idx):
        seqs_list, domain_labels, (ad_label, rca_label) = batch
        base_out = self.base_model((seqs_list, ad_label, rca_label))
        anomaly_logits, anomaly_score, _, _ = base_out[:4]
        loss_anomaly = F.binary_cross_entropy_with_logits(
            anomaly_logits[:, 1] - anomaly_logits[:, 0],
            ad_label.float()
        )
        self.log('val_loss', loss_anomaly, prog_bar=True)
        return loss_anomaly

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.epochs)
        return [optimizer], [scheduler]

    def adapt_few_shot(self, support_loader, num_steps=100, lr=1e-4):
        """Fine‑tune task heads on target support set."""
        # Freeze encoder and domain classifier
        encoder = self.base_model.ad_encoder
        for param in encoder.parameters():
            param.requires_grad = False
        for param in self.domain_classifier.parameters():
            param.requires_grad = False

        trainable_params = []
        if hasattr(self.base_model, 'anomaly_head'):
            trainable_params += list(self.base_model.anomaly_head.parameters())
        if hasattr(self.base_model, 'rca_head'):
            trainable_params += list(self.base_model.rca_head.parameters())
        if hasattr(self.base_model, 'classifier'):
            trainable_params += list(self.base_model.classifier.parameters())
        if hasattr(self.base_model, 'rca_classifier'):
            trainable_params += list(self.base_model.rca_classifier.parameters())

        optimizer = torch.optim.Adam(trainable_params, lr=lr)

        self.train()
        for step in range(num_steps):
            for batch in support_loader:
                seqs_list, _, (ad_label, rca_label) = batch
                optimizer.zero_grad()
                base_out = self.base_model((seqs_list, ad_label, rca_label))
                anomaly_logits, anomaly_score, _, _ = base_out[:4]

                loss_anomaly = F.binary_cross_entropy_with_logits(
                    anomaly_logits[:, 1] - anomaly_logits[:, 0],
                    ad_label.float()
                )
                loss_rca = F.binary_cross_entropy_with_logits(anomaly_score, rca_label.float())
                loss = loss_anomaly + loss_rca
                loss.backward()
                optimizer.step()
                break
        for param in encoder.parameters():
            param.requires_grad = True

    def save(self, path):
        torch.save(self.state_dict(), get_abs_path(path, f'{self.__class__.__name__}_model.bin'))

    def load(self, path, device='cuda'):
        self.load_state_dict(torch.load(get_abs_path(path, f'{self.__class__.__name__}_model.bin'),
                                        map_location=device))

# ----------------------------------------------------------------------
# 3. Domain Lambda Scheduler
# ----------------------------------------------------------------------
class DomainLambdaCallback(Callback):
    def __init__(self, model, lambd_max, warmup_epochs):
        super().__init__()
        self.model = model
        self.lambd_max = lambd_max
        self.warmup_epochs = warmup_epochs

    def on_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        frac = min(epoch / max(self.warmup_epochs, 1), 1.0)
        new_lambda = frac * self.lambd_max
        self.model.set_domain_lambda(new_lambda)
        print(f"[DomainLambda] epoch={epoch}  lambda={new_lambda:.4f}")

# ----------------------------------------------------------------------
# 4. Cross‑domain Data Loaders with fixed RCA dimension
# ----------------------------------------------------------------------
class CrossDomainDataset(Dataset):
    def __init__(self, domain_specs, rca_dim):
        self.rca_dim = rca_dim
        self.data = []
        for spec in domain_specs:
            # Normal sequences
            with open(spec['n_fp'], 'r') as f:
                n_lines = f.read().strip().split('\n')
            for line in n_lines:
                line = line.strip()
                if not line:
                    continue
                # Remove leading label (e.g., '0:' or '1:')
                if line[0].isdigit() and len(line) > 1 and line[1] == ':':
                    line = line[2:]
                tokens = line.split()
                seq = []
                for token in tokens:
                    if ':' in token:
                        idx_part = token.split(':')[0]
                        try:
                            seq.append(int(idx_part))
                        except ValueError:
                            continue
                    else:
                        try:
                            seq.append(int(token))
                        except ValueError:
                            continue
                if seq:
                    self.data.append((seq, 0, spec['domain_id']))

            # Anomalous sequences
            with open(spec['an_fp'], 'r') as f:
                an_lines = f.read().strip().split('\n')
            for line in an_lines:
                line = line.strip()
                if not line:
                    continue
                if line[0].isdigit() and len(line) > 1 and line[1] == ':':
                    line = line[2:]
                tokens = line.split()
                seq = []
                for token in tokens:
                    if ':' in token:
                        idx_part = token.split(':')[0]
                        try:
                            seq.append(int(idx_part))
                        except ValueError:
                            continue
                    else:
                        try:
                            seq.append(int(token))
                        except ValueError:
                            continue
                if seq:
                    self.data.append((seq, 1, spec['domain_id']))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, ad_label, domain = self.data[idx]
        # Dummy RCA label of fixed dimension (e.g., 20)
        rca_label = [0.0] * self.rca_dim
        return seq, domain, ad_label, rca_label

def collate_fn(batch):
    seqs, domains, ad_labels, rca_labels = zip(*batch)
    domains = torch.tensor(domains, dtype=torch.long)
    ad_labels = torch.tensor(ad_labels, dtype=torch.float)
    rca_labels = torch.tensor(rca_labels, dtype=torch.float)
    return list(seqs), domains, (ad_labels, rca_labels)

def get_cross_domain_loader(domain_specs, batch_size, rca_dim, num_workers=4):
    dataset = CrossDomainDataset(domain_specs, rca_dim)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, collate_fn=collate_fn)

def get_cross_domain_eval_loader(data_fp, domain_id, batch_size, rca_dim, num_workers=4):
    """Loader for evaluation without labels (uses dummy RCA of correct size)."""
    lines = open(data_fp).read().strip().split('\n')
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and len(line) > 1 and line[1] == ':':
            line = line[2:]
        tokens = line.split()
        seq = []
        for token in tokens:
            if ':' in token:
                idx_part = token.split(':')[0]
                try:
                    seq.append(int(idx_part))
                except ValueError:
                    continue
            else:
                try:
                    seq.append(int(token))
                except ValueError:
                    continue
        if seq:
            data.append((seq, domain_id))
    class EvalDataset(Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, idx): return data[idx][0], data[idx][1]
    return DataLoader(EvalDataset(), batch_size=batch_size, shuffle=False,
                      num_workers=num_workers,
                      collate_fn=lambda b: ([x[0] for x in b], torch.tensor([x[1] for x in b])))

def get_few_shot_loader(n_fp, an_fp, domain_id, k_shots, rca_dim):
    lines_n = open(n_fp).read().strip().split('\n')
    lines_an = open(an_fp).read().strip().split('\n')
    data = []
    for line in lines_n[:k_shots]:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and len(line) > 1 and line[1] == ':':
            line = line[2:]
        tokens = line.split()
        seq = []
        for token in tokens:
            if ':' in token:
                idx_part = token.split(':')[0]
                try:
                    seq.append(int(idx_part))
                except ValueError:
                    continue
            else:
                try:
                    seq.append(int(token))
                except ValueError:
                    continue
        if seq:
            data.append((seq, 0, domain_id))
    for line in lines_an[:k_shots]:
        line = line.strip()
        if not line:
            continue
        if line[0].isdigit() and len(line) > 1 and line[1] == ':':
            line = line[2:]
        tokens = line.split()
        seq = []
        for token in tokens:
            if ':' in token:
                idx_part = token.split(':')[0]
                try:
                    seq.append(int(idx_part))
                except ValueError:
                    continue
            else:
                try:
                    seq.append(int(token))
                except ValueError:
                    continue
        if seq:
            data.append((seq, 1, domain_id))
    class FewShotDataset(Dataset):
        def __len__(self): return len(data)
        def __getitem__(self, idx):
            seq, ad_label, dom = data[idx]
            rca_label = [0.0] * rca_dim
            return seq, dom, (ad_label, rca_label)
    return DataLoader(FewShotDataset(), batch_size=len(data), shuffle=True,
                      collate_fn=lambda b: ([x[0] for x in b], torch.tensor([x[1] for x in b]),
                                            (torch.tensor([x[2][0] for x in b], dtype=torch.float),
                                             torch.tensor([x[2][1] for x in b], dtype=torch.float))))

# ----------------------------------------------------------------------
# 5. Evaluation
# ----------------------------------------------------------------------
def run_eval(model, n_eval_loader, an_eval_loader, device, rca_dim):
    model.eval()
    with torch.no_grad():
        tp = tn = fp = fn = 0
        start = time.time()

        for batch in n_eval_loader:
            seqs_list, domains = batch
            ad_label = torch.zeros(len(seqs_list), device=device)
            rca_label = torch.zeros(len(seqs_list), rca_dim, device=device)
            out, score, _, _ = model((seqs_list, ad_label, rca_label))
            for i in range(len(out)):
                if out[i][1].item() >= out[i][0].item():
                    fp += 1
                else:
                    tn += 1

        for batch in an_eval_loader:
            seqs_list, domains = batch
            ad_label = torch.ones(len(seqs_list), device=device)
            rca_label = torch.zeros(len(seqs_list), rca_dim, device=device)
            out, score, _, _ = model((seqs_list, ad_label, rca_label))
            for i in range(len(out)):
                if out[i][1].item() >= out[i][0].item():
                    tp += 1
                else:
                    fn += 1

        print(f"tp:{tp} fn:{fn} fp:{fp} tn:{tn}")

        p = tp/(tp+fp) if tp>0 else 0
        r = tp/(tp+fn) if tp>0 else 0
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0
        acc = (tp+tn)/(tp+tn+fp+fn)
        print(f"P={p:.4f} R={r:.4f} F1={f1:.4f} Acc={acc:.4f}\n")
        print(f"Inference time: {time.time()-start:.2f}s")

# ----------------------------------------------------------------------
# 6. Argument parsing
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard_device", default='cuda', type=str)
    parser.add_argument("--gpu_index", default=0, type=int)
    parser.add_argument("--load_checkpoint", action='store_true')
    parser.add_argument("--model_save_path", default='checkpoint', type=str)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--warmup_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--accumulate_grad_batches", default=1, type=int)
    parser.add_argument("--mode", default='cd_train', type=str,
                        choices=['cd_train', 'cd_eval', 'few_shot'])
    parser.add_argument("--dataset", default='BGL', type=str)
    parser.add_argument("--source_datasets", nargs='+', default=['BGL'], type=str)
    parser.add_argument("--target_dataset", default='GAIA', type=str)
    parser.add_argument("--num_domains", default=2, type=int)
    parser.add_argument("--domain_lambda", default=0.1, type=float)
    parser.add_argument("--domain_lambda_max", default=1.0, type=float)
    parser.add_argument("--shots", default=50, type=int)
    parser.add_argument("--adaptation_steps", default=200, type=int)
    parser.add_argument("--adaptation_lr", default=1e-4, type=float)
    # Added RCA dimension argument (default 20, matching original Chimera)
    parser.add_argument("--rca_dim", default=20, type=int,
                        help="Number of RCA classes (default 20)")
    args = parser.parse_args()
    if args.hard_device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.hard_device = 'cpu'
    args.device = torch.device(args.hard_device)
    return args

# ----------------------------------------------------------------------
# 7. Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    # Load embedding dictionary (used only for base model)
    emb_dict = load_json(f'data/{args.source_datasets[0]}/emd_dict.json')

    # Use fixed RCA dimension (from args, default 20)
    rca_dim = args.rca_dim
    print(f"Using RCA dimension: {rca_dim}")

    if args.mode == 'cd_train':
        # Build domain specs
        train_specs = []
        val_specs = []
        for i, ds in enumerate(args.source_datasets):
            train_specs.append({
                'n_fp': f'data/{ds}/n_train.txt',
                'an_fp': f'data/{ds}/an_train.txt',
                'domain_id': i
            })
            val_specs.append({
                'n_fp': f'data/{ds}/n_dev.txt',
                'an_fp': f'data/{ds}/an_dev.txt',
                'domain_id': i
            })
        train_loader = get_cross_domain_loader(train_specs, args.batch_size, rca_dim)
        val_loader = get_cross_domain_loader(val_specs, args.batch_size, rca_dim)

        model = CrossDomainChimera(args, emb_dict, num_domains=len(args.source_datasets),
                                   rca_dim=rca_dim,
                                   domain_lambda=args.domain_lambda,
                                   lambd_max=args.domain_lambda_max,
                                   warmup_epochs=args.warmup_epochs).to(args.device)

        lambda_callback = DomainLambdaCallback(model, args.domain_lambda_max, args.warmup_epochs)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='min')
        trainer = pl.Trainer(
            max_epochs=args.epochs,
            accelerator='gpu' if args.hard_device=='cuda' else 'cpu',
            devices=[args.gpu_index] if args.hard_device=='cuda' else 1,
            accumulate_grad_batches=args.accumulate_grad_batches,
            enable_progress_bar=False,
            callbacks=[lambda_callback, early_stop]
        )
        start = time.time()
        trainer.fit(model, train_loader, val_loader)
        print(f"Training time: {time.time()-start:.1f}s")
        model.save(args.model_save_path)

    elif args.mode == 'cd_eval':
        model = CrossDomainChimera(args, emb_dict, num_domains=len(args.source_datasets),
                                   rca_dim=rca_dim).to(args.device)
        # Load checkpoint
        ckpt_path = get_abs_path(args.model_save_path, f'CrossDomainChimera_model.bin')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please train the model first using --mode cd_train")
        model.load(args.model_save_path, device=args.device)

        target_domain_id = len(args.source_datasets)
        n_eval = get_cross_domain_eval_loader(f'data/{args.target_dataset}/n_test.txt',
                                              target_domain_id, args.batch_size, rca_dim, 4)
        an_eval = get_cross_domain_eval_loader(f'data/{args.target_dataset}/an_test.txt',
                                               target_domain_id, args.batch_size, rca_dim, 4)

        run_eval(model, n_eval, an_eval, args.device, rca_dim)

    elif args.mode == 'few_shot':
        model = CrossDomainChimera(args, emb_dict, num_domains=len(args.source_datasets),
                                   rca_dim=rca_dim).to(args.device)
        ckpt_path = get_abs_path(args.model_save_path, f'CrossDomainChimera_model.bin')
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Please train the model first using --mode cd_train")
        model.load(args.model_save_path, device=args.device)

        target_domain_id = len(args.source_datasets)
        support_loader = get_few_shot_loader(
            f'data/{args.target_dataset}/n_train.txt',
            f'data/{args.target_dataset}/an_train.txt',
            target_domain_id, args.shots, rca_dim
        )
        model.adapt_few_shot(support_loader, num_steps=args.adaptation_steps, lr=args.adaptation_lr)

        n_eval = get_cross_domain_eval_loader(f'data/{args.target_dataset}/n_test.txt',
                                              target_domain_id, args.batch_size, rca_dim, 4)
        an_eval = get_cross_domain_eval_loader(f'data/{args.target_dataset}/an_test.txt',
                                               target_domain_id, args.batch_size, rca_dim, 4)
        run_eval(model, n_eval, an_eval, args.device, rca_dim)

if __name__ == '__main__':
    main()
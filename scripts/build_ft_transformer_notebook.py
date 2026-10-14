"""Build the self-contained Kaggle FT-Transformer training notebook."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import nbformat as nbf
import pandas as pd

OUTPUT_PATH = Path("notebooks/kaggle_ev_purchase_ft_transformer.ipynb")
EXPECTED_SPLIT_CONTENT_SHA256 = "5a6a1965ff57c5214035bc4e01ae51e306419b9c05aa04cae8faad9d53fd2206"
SOURCE_HASH_PLACEHOLDER = "__FULL_NOTEBOOK_SOURCE_SHA256__"
RAW_PREDICTORS = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]


def validate_completed_fold_items(
    items: object, n_repeats: int = 3, n_splits: int = 5
) -> set[tuple[int, int]]:
    """Validate serialized completed-fold identities before trusting their count."""
    if not isinstance(items, list):
        raise ValueError("completed_folds must be a list")
    normalized: list[tuple[int, int]] = []
    canonical = {(repeat, fold) for repeat in range(n_repeats) for fold in range(n_splits)}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid completed fold entry: {item!r}")
        repeat, fold = item
        if type(repeat) is not int or type(fold) is not int:
            raise ValueError(f"Completed fold values must be integers: {item!r}")
        if (repeat, fold) not in canonical:
            raise ValueError(f"Completed fold is outside the canonical domain: {item!r}")
        normalized.append((repeat, fold))
    if len(normalized) != len(set(normalized)):
        raise ValueError("completed_folds contains duplicate entries")
    return set(normalized)


def split_content_sha256(manifest: pd.DataFrame) -> str:
    """Hash canonical assignments independently of parquet writer metadata."""
    columns = ["row_index", "repeat", "fold"]
    array = manifest[columns].to_numpy(dtype="<i8", copy=True)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def notebook_source_sha256(notebook: nbf.NotebookNode) -> str:
    """Hash every notebook cell source using a self-normalized source marker."""
    payload = []
    for cell in notebook.cells:
        source = re.sub(
            r'PIPELINE_SOURCE_SHA256 = "[0-9a-f]{64}"',
            f'PIPELINE_SOURCE_SHA256 = "{SOURCE_HASH_PLACEHOLDER}"',
            cell.source,
        )
        payload.append({"cell_type": cell.cell_type, "source": source})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


RESUME_DISCOVERY_CODE = r'''def _resume_state_matches(
    state_path, expected_fingerprint, expected_fingerprint_sha256
):
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        state.get("run_fingerprint") == expected_fingerprint
        and state.get("run_fingerprint_sha256") == expected_fingerprint_sha256
    )


def select_resume_artifact_dir(
    kaggle_input_root,
    explicit_dir,
    expected_fingerprint,
    expected_fingerprint_sha256,
):
    """Select one fingerprint-compatible resume root regardless of dataset slug."""
    if explicit_dir is not None:
        explicit_root = Path(explicit_dir).resolve()
        state_path = explicit_root / "run_state.json"
        if not state_path.is_file() or not _resume_state_matches(
            state_path, expected_fingerprint, expected_fingerprint_sha256
        ):
            raise RuntimeError(
                f"Explicit resume artifact fingerprint mismatch: {explicit_root}"
            )
        return explicit_root

    input_root = Path(kaggle_input_root)
    if not input_root.exists():
        return None
    compatible = []
    for state_path in sorted(input_root.rglob("run_state.json")):
        if _resume_state_matches(
            state_path, expected_fingerprint, expected_fingerprint_sha256
        ):
            compatible.append(state_path.parent.resolve())
    compatible = sorted(set(compatible))
    if len(compatible) > 1:
        raise RuntimeError(
            "Multiple fingerprint-compatible resume artifact directories found; "
            "set RESUME_INPUT_DIR explicitly"
        )
    return compatible[0] if compatible else None
'''


MODEL_CODE = r"""class TabularDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, index):
        item = {"x_num": self.x_num[index], "x_cat": self.x_cat[index]}
        if self.y is not None:
            item["y"] = self.y[index]
        return item


class FTTransformer(nn.Module):
    def __init__(
        self,
        n_num_features,
        cat_cardinalities,
        d_token=48,
        n_blocks=2,
        n_heads=4,
        d_ffn=96,
        dropout=0.1,
    ):
        super().__init__()
        self.num_weight = nn.Parameter(torch.empty(n_num_features, d_token))
        self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_token))
        nn.init.kaiming_uniform_(self.num_weight)
        self.cat_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, d_token) for cardinality in cat_cardinalities]
        )
        for embedding in self.cat_embeddings:
            nn.init.normal_(embedding.weight, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token,
            nhead=n_heads,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks)
        self.head = nn.Sequential(nn.LayerNorm(d_token), nn.Linear(d_token, 1))

    def forward(self, x_num, x_cat):
        num_tokens = x_num.unsqueeze(-1) * self.num_weight.unsqueeze(0)
        num_tokens = num_tokens + self.num_bias.unsqueeze(0)
        cat_tokens = [
            embedding(x_cat[:, index]).unsqueeze(1)
            for index, embedding in enumerate(self.cat_embeddings)
        ]
        tokens = torch.cat([num_tokens, *cat_tokens], dim=1)
        cls = self.cls_token.expand(len(x_num), -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1))
        return self.head(encoded[:, 0]).squeeze(-1)


def make_model(cardinalities, dropout=MODEL_DROPOUT):
    return FTTransformer(
        n_num_features=len(NUMERIC_PREDICTORS),
        cat_cardinalities=cardinalities,
        d_token=D_TOKEN,
        n_blocks=N_BLOCKS,
        n_heads=N_HEADS,
        d_ffn=D_FFN,
        dropout=dropout,
    ).to(DEVICE)
"""


TRAINING_CODE = r"""def transform_fold(train_index, valid_index):
    numeric_scaler = StandardScaler()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    fold_train = X_train_raw.iloc[train_index]
    fold_valid = X_train_raw.iloc[valid_index]

    x_train_num = numeric_scaler.fit_transform(fold_train[NUMERIC_PREDICTORS]).astype("float32")
    x_valid_num = numeric_scaler.transform(fold_valid[NUMERIC_PREDICTORS]).astype("float32")
    x_test_num = numeric_scaler.transform(X_test_raw[NUMERIC_PREDICTORS]).astype("float32")

    x_train_cat = encoder.fit_transform(fold_train[CATEGORICAL_PREDICTORS]).astype("int64") + 1
    x_valid_cat = encoder.transform(fold_valid[CATEGORICAL_PREDICTORS]).astype("int64") + 1
    x_test_cat = encoder.transform(X_test_raw[CATEGORICAL_PREDICTORS]).astype("int64") + 1
    cardinalities = [len(categories) + 1 for categories in encoder.categories_]
    return (
        x_train_num,
        x_train_cat,
        x_valid_num,
        x_valid_cat,
        x_test_num,
        x_test_cat,
        cardinalities,
        numeric_scaler,
        encoder,
    )


def make_loader(
    dataset,
    shuffle,
    seed,
    batch_size=BATCH_SIZE,
    generator_state=None,
    return_generator=False,
):
    generator = torch.Generator()
    if generator_state is None:
        generator.manual_seed(seed)
    else:
        generator.set_state(generator_state)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=NUM_WORKERS > 0,
        generator=generator,
    )
    return (loader, generator) if return_generator else loader


def make_grad_scaler():
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def amp_autocast():
    try:
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast(dtype=torch.float16)


def torch_load_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def assert_finite_loss_and_gradients(model, loss):
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss detected: {loss.item()}")
    missing = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            missing.append(name)
        elif not torch.isfinite(param.grad).all():
            raise FloatingPointError(f"Non-finite gradient detected in {name}")
    if missing:
        raise RuntimeError(f"Missing gradients for trainable parameters: {missing}")


def validate_run_fingerprint(payload, source, quarantine=True):
    observed_fingerprint = payload.get("run_fingerprint")
    observed_sha256 = payload.get("run_fingerprint_sha256")
    valid = (
        observed_fingerprint == RUN_FINGERPRINT
        and observed_sha256 == RUN_FINGERPRINT_SHA256
        and hash_json(observed_fingerprint) == observed_sha256
    )
    if not valid:
        if quarantine and str(source).startswith(str(OUTPUT_DIR)):
            quarantine_artifact(source, "run fingerprint mismatch")
        raise RuntimeError(
            f"Resume provenance mismatch in {source}: "
            f"expected={RUN_FINGERPRINT_SHA256}, observed={observed_sha256}"
        )


def validate_fold_checkpoint(payload, source, repeat, fold):
    validate_run_fingerprint(payload, source)
    if payload.get("repeat") != repeat or payload.get("fold") != fold:
        quarantine_artifact(source, "fold checkpoint identity mismatch")
        raise RuntimeError(
            f"Fold checkpoint identity mismatch in {source}: "
            f"expected={(repeat, fold)}, observed={(payload.get('repeat'), payload.get('fold'))}"
        )


def require_integer_series(series, name):
    if series.isna().any() or not pd.api.types.is_integer_dtype(series.dtype):
        raise RuntimeError(f"{name} must contain non-null integers")
    return series.to_numpy(dtype=np.int64, copy=True)


def validate_persisted_fold_metric(observed, expected, source):
    exact_fields = (
        "run_fingerprint",
        "run_fingerprint_sha256",
        "fold_identity",
        "repeat",
        "fold",
        "seed",
        "n_validation",
        "checkpoint",
        "preprocessor",
    )
    for field in exact_fields:
        if observed.get(field) != expected[field]:
            quarantine_artifact(source, f"fold metric {field} mismatch")
            raise RuntimeError(f"Persisted fold metric {field} mismatch in {source}")
    for field in ("roc_auc", "pr_auc", "log_loss", "best_validation_loss"):
        if not np.isclose(observed.get(field, np.nan), expected[field], rtol=0.0, atol=1e-12):
            quarantine_artifact(source, f"fold metric {field} mismatch")
            raise RuntimeError(f"Persisted fold metric {field} mismatch in {source}")


def capture_rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(rng_state):
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def clone_checkpoint_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: clone_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_checkpoint_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_checkpoint_value(item) for item in value)
    return value


def optimizer_step(model, optimizer, scaler, loss):
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    assert_finite_loss_and_gradients(model, loss)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
    if not torch.isfinite(grad_norm):
        raise FloatingPointError(f"Non-finite gradient norm: {grad_norm}")
    scaler.step(optimizer)
    scaler.update()
    return float(grad_norm.detach().cpu())


def run_one_batch_smoke_test(train_loader, cardinalities, seed):
    seed_everything(seed)
    model = make_model(cardinalities)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scaler = make_grad_scaler()
    criterion = nn.BCEWithLogitsLoss()
    batch = next(iter(train_loader))
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    with amp_autocast():
        logits = model(
            batch["x_num"].to(DEVICE, non_blocking=True),
            batch["x_cat"].to(DEVICE, non_blocking=True),
        )
        target = batch["y"].to(DEVICE, non_blocking=True)
        loss = criterion(logits, target)
    optimizer_step(model, optimizer, scaler, loss)
    first_optimizer_step_verified = any(
        not torch.equal(before[name], value.detach()) for name, value in model.named_parameters()
    )
    if not first_optimizer_step_verified:
        raise RuntimeError("First optimizer step did not change any model parameter")
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    return {"loss": float(loss.detach().cpu()), "first_optimizer_step_verified": True}


def run_tiny_overfit_test(x_num, x_cat, target, cardinalities, seed):
    seed_everything(seed)
    available_zero = np.flatnonzero(target == 0)
    available_one = np.flatnonzero(target == 1)
    n_per_class = min(32, len(available_zero), len(available_one))
    if n_per_class < 2:
        raise RuntimeError("Tiny overfit preflight requires both classes")
    class_zero = available_zero[:n_per_class]
    class_one = available_one[:n_per_class]
    selected = np.concatenate([class_zero, class_one])
    selected = selected[np.random.default_rng(seed).permutation(len(selected))]
    dataset = TabularDataset(x_num[selected], x_cat[selected], target[selected])
    count = len(selected)
    loader = make_loader(dataset, shuffle=False, seed=seed, batch_size=count)
    batch = next(iter(loader))
    model = make_model(cardinalities, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    accuracy = 0.0
    for _step in range(100):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(
            batch["x_num"].to(DEVICE, non_blocking=True),
            batch["x_cat"].to(DEVICE, non_blocking=True),
        )
        target_batch = batch["y"].to(DEVICE, non_blocking=True)
        loss = criterion(logits, target_batch)
        loss.backward()
        assert_finite_loss_and_gradients(model, loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        accuracy = float(((logits.detach() >= 0) == target_batch.bool()).float().mean().cpu())
        if losses[-1] < 0.15 and accuracy >= 0.98:
            break
    minimum_loss = min(losses)
    tail_loss = float(np.mean(losses[-5:]))
    if minimum_loss >= min(losses[0] * 0.50, 0.35) or accuracy < 0.95:
        raise RuntimeError(
            "Tiny overfit preflight failed float32 memorization: "
            f"initial={losses[0]:.6f}, minimum={minimum_loss:.6f}, "
            f"tail={tail_loss:.6f}, accuracy={accuracy:.3f}"
        )
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "class_counts": {"0": len(class_zero), "1": len(class_one)},
        "initial_loss": losses[0],
        "minimum_loss": minimum_loss,
        "tail_mean_loss": tail_loss,
        "accuracy": accuracy,
        "steps": len(losses),
    }


@torch.inference_mode()
def predict_loader(model, loader):
    model.eval()
    predictions = []
    for batch in loader:
        with amp_autocast():
            logits = model(
                batch["x_num"].to(DEVICE, non_blocking=True),
                batch["x_cat"].to(DEVICE, non_blocking=True),
            )
        predictions.append(torch.sigmoid(logits).float().cpu().numpy())
    result = np.concatenate(predictions)
    if not np.isfinite(result).all() or not ((result >= 0.0) & (result <= 1.0)).all():
        raise FloatingPointError("Predictions are non-finite or outside [0, 1]")
    return result


def fit_fold(
    repeat,
    fold,
    train_index,
    valid_index,
    transformed,
    started_at,
):
    (
        x_train_num,
        x_train_cat,
        x_valid_num,
        x_valid_cat,
        x_test_num,
        x_test_cat,
        cardinalities,
        numeric_scaler,
        encoder,
    ) = transformed
    fold_seed = BASE_SEED + repeat * 100 + fold
    seed_everything(fold_seed)
    train_dataset = TabularDataset(x_train_num, x_train_cat, y_true[train_index])
    valid_loader = make_loader(
        TabularDataset(x_valid_num, x_valid_cat, y_true[valid_index]), False, fold_seed
    )
    test_loader = make_loader(
        TabularDataset(x_test_num, x_test_cat), False, fold_seed
    )

    model = make_model(cardinalities)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    amp_scaler = make_grad_scaler()
    criterion = nn.BCEWithLogitsLoss()
    fold_checkpoint = CHECKPOINT_DIR / f"fold_r{repeat}_f{fold}.pt"
    preprocess_path = CHECKPOINT_DIR / f"preprocess_r{repeat}_f{fold}.pkl"
    start_epoch = 0
    best_loss = float("inf")
    best_model_state = None
    history = []
    resume_generator_state = None

    if fold_checkpoint.exists():
        checkpoint = torch_load_compat(fold_checkpoint, map_location=DEVICE)
        validate_fold_checkpoint(checkpoint, fold_checkpoint, repeat, fold)
        if checkpoint.get("status") == "completed":
            best_model_state = checkpoint["best_model_state"]
            model.load_state_dict(best_model_state)
            valid_probability = predict_loader(model, valid_loader)
            test_probability = predict_loader(model, test_loader)
            atomic_pickle_save(
                {"scaler": numeric_scaler, "encoder": encoder}, preprocess_path
            )
            return {
                "valid_probability": valid_probability,
                "test_probability": test_probability,
                "best_loss": float(checkpoint["best_loss"]),
                "seed": fold_seed,
                "checkpoint": str(fold_checkpoint),
                "preprocessor": str(preprocess_path),
            }, "completed"
        if checkpoint.get("status") == "in_progress":
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            amp_scaler.load_state_dict(checkpoint["scaler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_loss = float(checkpoint["best_loss"])
            best_model_state = checkpoint["best_model_state"]
            history = checkpoint["history"]
            restore_rng_state(checkpoint["rng_state"])
            resume_generator_state = checkpoint["dataloader_generator_state"]
            print(f"Resuming repeat={repeat}, fold={fold} at epoch={start_epoch}")

    for epoch in range(start_epoch, EPOCHS):
        train_loader, train_generator = make_loader(
            train_dataset,
            True,
            fold_seed + epoch,
            generator_state=resume_generator_state if epoch == start_epoch else None,
            return_generator=True,
        )
        epoch_start_payload = {
            "status": "in_progress",
            "run_fingerprint": RUN_FINGERPRINT,
            "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
            "repeat": repeat,
            "fold": fold,
            "epoch": epoch - 1,
            "model_state": clone_checkpoint_value(model.state_dict()),
            "optimizer_state": clone_checkpoint_value(optimizer.state_dict()),
            "scheduler_state": clone_checkpoint_value(scheduler.state_dict()),
            "scaler_state": clone_checkpoint_value(amp_scaler.state_dict()),
            "best_loss": best_loss,
            "best_model_state": clone_checkpoint_value(best_model_state),
            "history": clone_checkpoint_value(history),
            "rng_state": clone_checkpoint_value(capture_rng_state()),
            "dataloader_generator_state": train_generator.get_state().clone(),
        }
        atomic_torch_save(epoch_start_payload, fold_checkpoint)
        if time.time() - started_at >= MAX_RUNTIME_SECONDS - TIME_BUFFER_SECONDS:
            return None, "time_budget_graceful_exit"

        model.train()
        train_loss_sum = 0.0
        for batch in train_loader:
            if time.time() - started_at >= MAX_RUNTIME_SECONDS - TIME_BUFFER_SECONDS:
                return None, "time_budget_graceful_exit"
            optimizer.zero_grad(set_to_none=True)
            x_num = batch["x_num"].to(DEVICE, non_blocking=True)
            x_cat = batch["x_cat"].to(DEVICE, non_blocking=True)
            target = batch["y"].to(DEVICE, non_blocking=True)
            with amp_autocast():
                loss = criterion(model(x_num, x_cat), target)
            optimizer_step(model, optimizer, amp_scaler, loss)
            train_loss_sum += float(loss.detach().cpu()) * len(target)

        if time.time() - started_at >= MAX_RUNTIME_SECONDS - TIME_BUFFER_SECONDS:
            return None, "time_budget_graceful_exit"
        valid_probability = predict_loader(model, valid_loader)
        valid_loss = float(log_loss(y_true[valid_index], valid_probability, labels=[0, 1]))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_sum / len(train_index),
                "valid_loss": valid_loss,
            }
        )
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_model_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
        scheduler.step()
        atomic_torch_save(
            {
                "status": "in_progress",
                "run_fingerprint": RUN_FINGERPRINT,
                "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
                "repeat": repeat,
                "fold": fold,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": amp_scaler.state_dict(),
                "best_loss": best_loss,
                "best_model_state": best_model_state,
                "history": history,
                "rng_state": capture_rng_state(),
                "dataloader_generator_state": torch.Generator()
                .manual_seed(fold_seed + epoch + 1)
                .get_state(),
            },
            fold_checkpoint,
        )

    if best_model_state is None:
        raise RuntimeError("No finite best checkpoint was captured")
    model.load_state_dict(best_model_state)
    valid_probability = predict_loader(model, valid_loader)
    test_probability = predict_loader(model, test_loader)
    final_checkpoint = {
        "status": "completed",
        "run_fingerprint": RUN_FINGERPRINT,
        "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
        "repeat": repeat,
        "fold": fold,
        "seed": fold_seed,
        "best_loss": best_loss,
        "best_model_state": best_model_state,
        "cardinalities": cardinalities,
        "history": history,
        "rng_state": capture_rng_state(),
        "dataloader_generator_state": torch.Generator()
        .manual_seed(fold_seed + EPOCHS)
        .get_state(),
    }
    atomic_torch_save(final_checkpoint, fold_checkpoint)
    atomic_pickle_save({"scaler": numeric_scaler, "encoder": encoder}, preprocess_path)
    del model, optimizer
    torch.cuda.empty_cache()
    return {
        "valid_probability": valid_probability,
        "test_probability": test_probability,
        "best_loss": best_loss,
        "seed": fold_seed,
        "checkpoint": str(fold_checkpoint),
        "preprocessor": str(preprocess_path),
    }, "completed"
"""


def build_notebook() -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    notebook.cells = [
        nbf.v4.new_markdown_cell(
            """# EV Purchase FT-Transformer: Canonical 5 x 3 GPU Experiment

This self-contained Kaggle training notebook reconstructs and authenticates the
canonical persisted-equivalent repeated stratified split before fitting a
pure-PyTorch FT-Transformer. It uses only the 13 base raw predictors and writes
all artifacts under `/kaggle/working/ft_transformer_5x3`.

**Runtime contract:** enable a Kaggle GPU accelerator. The notebook fails
immediately if CUDA is unavailable rather than silently running the full
experiment on CPU. The default run is 15 folds (5 folds x 3 repeats), 12 epochs
each, with an 8.5-hour graceful time budget.

**Incomplete-run policy:** OOF rows are published only for fully completed
folds. Do not describe an incomplete run as a completed CV experiment and do
not use or claim a submission unless all 15 folds finish.

**Artifact handoff:** after `Save Version` / `Run All`, download the entire
`ft_transformer_5x3` output folder. To resume in a later Kaggle session, attach
that folder as a Kaggle Dataset. The notebook automatically searches every
arbitrary dataset slug and nested folder for the unique fingerprint-compatible run.
Set `RESUME_INPUT_DIR` only as an explicit override. Resume files are staged
and verified automatically."""
        ),
        nbf.v4.new_markdown_cell("## 1. Imports, deterministic configuration, and CUDA guard"),
        nbf.v4.new_code_cell(
            f'''import gc
import hashlib
import json
import os
import pickle
import platform
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this Kaggle FT-Transformer experiment")

torch.use_deterministic_algorithms(True)
DEVICE = torch.device("cuda")
OUTPUT_DIR = Path("/kaggle/working/ft_transformer_5x3")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

RAW_PREDICTORS = {RAW_PREDICTORS!r}
TARGET_COLUMN = "Will_Buy_EV"
ID_COLUMN = "id"
TARGET_MAPPING = {{"No": 0, "Yes": 1}}
NUMERIC_PREDICTORS = RAW_PREDICTORS[:7]
CATEGORICAL_PREDICTORS = RAW_PREDICTORS[7:]
N_SPLITS = 5
N_REPEATS = 3
TOTAL_FOLDS = N_SPLITS * N_REPEATS
SPLIT_RANDOM_STATE = 42
BASE_SEED = 42
EXPECTED_SPLIT_CONTENT_SHA256 = "{EXPECTED_SPLIT_CONTENT_SHA256}"
PIPELINE_SOURCE_SHA256 = "{SOURCE_HASH_PLACEHOLDER}"
EPOCHS = 12
BATCH_SIZE = 4096
NUM_WORKERS = 2
PIN_MEMORY = True
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0
D_TOKEN = 48
N_BLOCKS = 2
N_HEADS = 4
D_FFN = 96
MODEL_DROPOUT = 0.1
MAX_RUNTIME_SECONDS = 8.5 * 60 * 60
TIME_BUFFER_SECONDS = 10 * 60
RESUME_INPUT_DIR = None


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def hash_json(payload):
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


def split_content_sha256(manifest):
    columns = ["row_index", "repeat", "fold"]
    array = manifest[columns].to_numpy(dtype="<i8", copy=True)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_completed_fold_items(items, n_repeats=N_REPEATS, n_splits=N_SPLITS):
    if not isinstance(items, list):
        raise ValueError("completed_folds must be a list")
    normalized = []
    canonical = {{
        (repeat, fold) for repeat in range(n_repeats) for fold in range(n_splits)
    }}
    for item in items:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid completed fold entry: {{item!r}}")
        repeat, fold = item
        if type(repeat) is not int or type(fold) is not int:
            raise ValueError(f"Completed fold values must be integers: {{item!r}}")
        if (repeat, fold) not in canonical:
            raise ValueError(f"Completed fold is outside the canonical domain: {{item!r}}")
        normalized.append((repeat, fold))
    if len(normalized) != len(set(normalized)):
        raise ValueError("completed_folds contains duplicate entries")
    return set(normalized)


def atomic_json_dump(payload, path):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def atomic_pickle_save(payload, path):
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle)
    temporary.replace(path)


def atomic_csv_dump(frame, path):
    temporary = Path(str(path) + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def directory_file_hashes(root):
    root = Path(root)
    return dict(
        (str(path.relative_to(root)), sha256_file(path))
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


def quarantine_artifact(path, reason):
    path = Path(path)
    if not path.exists():
        return None
    quarantine_dir = OUTPUT_DIR / "quarantine"
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    destination = quarantine_dir / f"{{path.name}}.{{int(time.time_ns())}}.stale"
    path.replace(destination)
    atomic_json_dump(
        {{"source": str(path), "quarantined": str(destination), "reason": reason}},
        Path(str(destination) + ".json"),
    )
    return destination


def quarantine_complete_only_outputs():
    for stale_name in ("submission.csv", "test_predictions.csv"):
        quarantine_artifact(OUTPUT_DIR / stale_name, "run is incomplete")
    stale_manifest_path = OUTPUT_DIR / "training_manifest.json"
    if stale_manifest_path.exists():
        stale_manifest = json.loads(stale_manifest_path.read_text(encoding="utf-8"))
        if stale_manifest.get("status") == "completed":
            quarantine_artifact(stale_manifest_path, "completed manifest from stale run")


{RESUME_DISCOVERY_CODE}


def restore_attached_resume_artifacts():
    if (OUTPUT_DIR / "run_state.json").exists():
        return None
    source_dir = select_resume_artifact_dir(
        Path("/kaggle/input"),
        RESUME_INPUT_DIR,
        RUN_FINGERPRINT,
        RUN_FINGERPRINT_SHA256,
    )
    if source_dir is None:
        return None
    source_state = json.loads((source_dir / "run_state.json").read_text(encoding="utf-8"))
    validate_run_fingerprint(source_state, source_dir / "run_state.json", quarantine=False)
    stage_parent = OUTPUT_DIR.parent
    stage_dir = Path(tempfile.mkdtemp(prefix=".ft_resume_stage_", dir=stage_parent))
    try:
        source_hashes = directory_file_hashes(source_dir)
        shutil.copytree(source_dir, stage_dir, dirs_exist_ok=True)
        staged_hashes = directory_file_hashes(stage_dir)
        if staged_hashes != source_hashes:
            raise RuntimeError("Staged resume copy failed file hash verification")
        staged_state = json.loads((stage_dir / "run_state.json").read_text(encoding="utf-8"))
        validate_run_fingerprint(staged_state, stage_dir / "run_state.json", quarantine=False)
        existing_files = [path for path in OUTPUT_DIR.rglob("*") if path.is_file()]
        if existing_files:
            raise RuntimeError("Working output directory is not empty; refusing resume overwrite")
        shutil.rmtree(OUTPUT_DIR)
        stage_dir.replace(OUTPUT_DIR)
    except Exception:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        raise
    print("Restored attached resume artifacts from:", source_dir)
    return str(source_dir)


RUNTIME_COMPATIBILITY = dict(
    python=sys.version,
    numpy=np.__version__,
    torch=torch.__version__,
    cuda=torch.version.cuda,
    cudnn=torch.backends.cudnn.version(),
    gpu_name=torch.cuda.get_device_name(0),
    gpu_count=torch.cuda.device_count(),
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
)

CONFIG_PAYLOAD = {{
    "base_seed": BASE_SEED,
    "seed_formula": "base_seed + repeat * 100 + fold",
    "n_splits": N_SPLITS,
    "n_repeats": N_REPEATS,
    "split_random_state": SPLIT_RANDOM_STATE,
    "target_mapping": TARGET_MAPPING,
    "numeric_predictors": NUMERIC_PREDICTORS,
    "categorical_predictors": CATEGORICAL_PREDICTORS,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
    "pin_memory": PIN_MEMORY,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "optimizer": "AdamW",
    "max_grad_norm": MAX_GRAD_NORM,
    "amp_dtype": "float16",
    "amp_scaler": "torch.amp.GradScaler(cuda)",
    "scheduler": "CosineAnnealingLR(T_max=epochs)",
    "best_state_criterion": "minimum best_validation_loss",
    "deterministic_algorithms": True,
    "cublas_workspace_config": ":4096:8",
    "required_device": "cuda",
    "max_runtime_seconds": MAX_RUNTIME_SECONDS,
    "time_buffer_seconds": TIME_BUFFER_SECONDS,
    "preprocessing": {{
        "numeric": "fold-local StandardScaler",
        "categorical": "fold-local OrdinalEncoder with unknown index 0",
    }},
    "raw_predictors": RAW_PREDICTORS,
    "model": {{
        "d_token": D_TOKEN,
        "n_blocks": N_BLOCKS,
        "n_heads": N_HEADS,
        "d_ffn": D_FFN,
        "dropout": MODEL_DROPOUT,
    }},
}}


RESTORED_RESUME_DIR = None
seed_everything(BASE_SEED)
print("PyTorch:", torch.__version__)
print("CUDA device:", torch.cuda.get_device_name(0))'''
        ),
        nbf.v4.new_markdown_cell(
            "## 2. Load data and authenticate the canonical split assignments"
        ),
        nbf.v4.new_code_cell(
            r"""def find_data_directory():
    candidates = [
        Path("/kaggle/input/playground-series-s6e9"),
        Path("/kaggle/input/competitions/playground-series-s6e9"),
    ]
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(path.parent for path in kaggle_input.rglob("train.csv"))
    for candidate in candidates:
        required_names = ("train.csv", "test.csv", "sample_submission.csv")
        if all((candidate / name).exists() for name in required_names):
            return candidate
    raise FileNotFoundError("Attach the Playground Series s6e9 competition data")


DATA_DIR = find_data_directory()
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_SUBMISSION_PATH)

required_train = [ID_COLUMN, *RAW_PREDICTORS, TARGET_COLUMN]
required_test = [ID_COLUMN, *RAW_PREDICTORS]
if list(train_df.columns) != required_train:
    raise ValueError(f"Unexpected train schema: {list(train_df.columns)}")
if list(test_df.columns) != required_test:
    raise ValueError(f"Unexpected test schema: {list(test_df.columns)}")
if train_df[required_train].isna().any().any() or test_df[required_test].isna().any().any():
    raise ValueError("This fixed preprocessing contract requires non-missing raw inputs")
if set(train_df[TARGET_COLUMN].unique()) != set(TARGET_MAPPING):
    raise ValueError("Target labels must be exactly {'No', 'Yes'}")

X_train_raw = train_df[RAW_PREDICTORS].copy()
X_test_raw = test_df[RAW_PREDICTORS].copy()
assert list(X_train_raw.columns) == RAW_PREDICTORS
y_true = train_df[TARGET_COLUMN].map(TARGET_MAPPING).to_numpy(dtype=np.int64)

splitter = RepeatedStratifiedKFold(
    n_splits=N_SPLITS,
    n_repeats=N_REPEATS,
    random_state=SPLIT_RANDOM_STATE,
)
assignment_parts = []
dummy = np.zeros((len(train_df), 1), dtype=np.uint8)
for split_index, (_train_index, valid_index) in enumerate(splitter.split(dummy, y_true)):
    repeat, fold = divmod(split_index, N_SPLITS)
    assignment_parts.append(
        pd.DataFrame(
            {
                "row_index": valid_index.astype("int64"),
                "repeat": np.full(len(valid_index), repeat, dtype="int64"),
                "fold": np.full(len(valid_index), fold, dtype="int64"),
            }
        )
    )
split_manifest = pd.concat(assignment_parts, ignore_index=True)
split_manifest = split_manifest.sort_values(
    ["repeat", "fold", "row_index"], kind="stable"
).reset_index(drop=True)
assert len(split_manifest) == len(train_df) * N_REPEATS
assert split_manifest.groupby(["repeat", "row_index"]).size().eq(1).all()
observed_split_hash = split_content_sha256(split_manifest)
if observed_split_hash != EXPECTED_SPLIT_CONTENT_SHA256:
    raise RuntimeError(
        "Canonical split assignment hash mismatch. Verify competition train row order and labels: "
        f"expected={EXPECTED_SPLIT_CONTENT_SHA256}, observed={observed_split_hash}"
    )

DATA_HASHES = {
    "train_csv_sha256": sha256_file(TRAIN_PATH),
    "test_csv_sha256": sha256_file(TEST_PATH),
    "sample_submission_csv_sha256": sha256_file(SAMPLE_SUBMISSION_PATH),
}
RUN_FINGERPRINT = {
    "experiment_name": "ft_transformer_5x3",
    "data_hashes": DATA_HASHES,
    "split_content_sha256": observed_split_hash,
    "config_sha256": hash_json(CONFIG_PAYLOAD),
    "full_notebook_source_sha256": PIPELINE_SOURCE_SHA256,
    "runtime_compatibility": RUNTIME_COMPATIBILITY,
}
RUN_FINGERPRINT_SHA256 = hash_json(RUN_FINGERPRINT)
print("Data directory:", DATA_DIR)
print("Canonical split content SHA-256:", observed_split_hash)
print("Run fingerprint SHA-256:", RUN_FINGERPRINT_SHA256)
print("Assignments:", len(split_manifest))"""
        ),
        nbf.v4.new_markdown_cell("## 3. FT-Transformer and dataset definitions"),
        nbf.v4.new_code_cell(MODEL_CODE),
        nbf.v4.new_markdown_cell(
            "## 4. Fold-local preprocessing, preflight tests, AMP training, and best-state restore"
        ),
        nbf.v4.new_code_cell(TRAINING_CODE),
        nbf.v4.new_markdown_cell(
            """## 5. Resume-aware 15-fold orchestration

The state file and per-fold checkpoints are updated atomically. A fold
contributes OOF rows only after all epochs finish and the best validation-loss
state is restored. If the time budget is reached, the current checkpoint is
retained and the notebook exits gracefully without fabricating missing OOF
rows."""
        ),
        nbf.v4.new_code_cell(
            r"""started_at = time.time()
RESTORED_RESUME_DIR = restore_attached_resume_artifacts()
STATE_PATH = OUTPUT_DIR / "run_state.json"
if STATE_PATH.exists():
    run_state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    validate_run_fingerprint(run_state, STATE_PATH)
else:
    run_state = {
        "status": "in_progress",
        "run_fingerprint": RUN_FINGERPRINT,
        "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
        "completed_folds": [],
        "stop_reason": None,
        "preflight": None,
    }

completed_folds = validate_completed_fold_items(run_state["completed_folds"])
fold_metric_rows = []
stop_reason = None
quarantine_complete_only_outputs()

for fold_key in list(completed_folds):
    repeat, fold = fold_key
    recoverable_checkpoint_path = CHECKPOINT_DIR / f"fold_r{repeat}_f{fold}.pt"
    expected_paths = (
        OUTPUT_DIR / f"fold_oof_r{repeat}_f{fold}.csv",
        OUTPUT_DIR / f"fold_test_r{repeat}_f{fold}.csv",
        OUTPUT_DIR / f"fold_metric_r{repeat}_f{fold}.json",
        recoverable_checkpoint_path,
        CHECKPOINT_DIR / f"preprocess_r{repeat}_f{fold}.pkl",
    )
    if all(path.exists() for path in expected_paths):
        continue
    if not recoverable_checkpoint_path.exists():
        raise RuntimeError(f"Completed fold {fold_key} has no recoverable checkpoint")
    recoverable_checkpoint = torch_load_compat(recoverable_checkpoint_path, map_location="cpu")
    validate_fold_checkpoint(
        recoverable_checkpoint, recoverable_checkpoint_path, repeat, fold
    )
    if recoverable_checkpoint.get("status") == "completed":
        completed_folds.remove(fold_key)
        run_state["completed_folds"] = [list(item) for item in sorted(completed_folds)]
        atomic_json_dump(run_state, STATE_PATH)
        print(
            "Regenerating missing fold outputs from authenticated completed checkpoint:",
            fold_key,
        )
    else:
        raise RuntimeError(f"Completed fold {fold_key} checkpoint is not completed")

for repeat in range(N_REPEATS):
    for fold in range(N_SPLITS):
        fold_key = (repeat, fold)
        fold_oof_path = OUTPUT_DIR / f"fold_oof_r{repeat}_f{fold}.csv"
        fold_test_path = OUTPUT_DIR / f"fold_test_r{repeat}_f{fold}.csv"
        fold_metric_path = OUTPUT_DIR / f"fold_metric_r{repeat}_f{fold}.json"
        fold_checkpoint_path = CHECKPOINT_DIR / f"fold_r{repeat}_f{fold}.pt"
        if fold_key in completed_folds:
            required_fold_artifacts = (
                fold_oof_path,
                fold_test_path,
                fold_metric_path,
                fold_checkpoint_path,
                CHECKPOINT_DIR / f"preprocess_r{repeat}_f{fold}.pkl",
            )
            if not all(path.exists() for path in required_fold_artifacts):
                raise RuntimeError(f"Completed fold {fold_key} is missing resumable artifacts")
            metric_row = json.loads(fold_metric_path.read_text(encoding="utf-8"))
            validate_run_fingerprint(metric_row, fold_metric_path)
            checkpoint_payload = torch_load_compat(fold_checkpoint_path, map_location="cpu")
            validate_fold_checkpoint(
                checkpoint_payload, fold_checkpoint_path, repeat, fold
            )
            if checkpoint_payload.get("status") != "completed":
                raise RuntimeError(f"Fold checkpoint is not completed: {fold_checkpoint_path}")
            resumed_oof = pd.read_csv(fold_oof_path)
            resumed_test = pd.read_csv(fold_test_path)
            if list(resumed_oof.columns) != [
                "row_index",
                "repeat",
                "fold",
                "probability",
                "target",
            ]:
                raise RuntimeError(f"Invalid resumed OOF schema for fold {fold_key}")
            if list(resumed_test.columns) != ["row_index", "repeat", "fold", "probability"]:
                raise RuntimeError(f"Invalid resumed test schema for fold {fold_key}")
            for frame in (resumed_oof, resumed_test):
                resumed_repeat = require_integer_series(frame["repeat"], "repeat")
                resumed_fold = require_integer_series(frame["fold"], "fold")
                if set(resumed_repeat) != {repeat} or set(resumed_fold) != {fold}:
                    raise RuntimeError(f"Invalid resumed fold labels for fold {fold_key}")
            if metric_row.get("fold_identity") != {"repeat": repeat, "fold": fold}:
                quarantine_artifact(fold_metric_path, "fold identity mismatch")
                raise RuntimeError(f"Resumed fold identity mismatch for {fold_key}")
            expected_oof_meta = metric_row.get("oof_prediction", {})
            expected_test_meta = metric_row.get("test_prediction", {})
            metadata_checks = (
                (fold_oof_path, resumed_oof, expected_oof_meta),
                (fold_test_path, resumed_test, expected_test_meta),
            )
            for prediction_path, frame, metadata in metadata_checks:
                if (
                    metadata.get("sha256") != sha256_file(prediction_path)
                    or metadata.get("row_count") != len(frame)
                    or metadata.get("schema") != list(frame.columns)
                ):
                    quarantine_artifact(prediction_path, "prediction metadata mismatch")
                    raise RuntimeError(f"Resumed prediction metadata mismatch: {prediction_path}")
            expected_index = split_manifest.loc[
                (split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold),
                "row_index",
            ].to_numpy(dtype=np.int64)
            observed_index = require_integer_series(resumed_oof["row_index"], "row_index")
            if not np.array_equal(observed_index, expected_index):
                raise RuntimeError(f"Resumed OOF row indices do not match fold {fold_key}")
            observed_target = require_integer_series(resumed_oof["target"], "target")
            if not np.array_equal(
                observed_target, y_true[expected_index]
            ):
                raise RuntimeError(f"Resumed OOF targets do not match fold {fold_key}")
            observed_test_index = require_integer_series(
                resumed_test["row_index"], "test row_index"
            )
            if not np.array_equal(
                observed_test_index,
                np.arange(len(test_df), dtype=np.int64),
            ):
                raise RuntimeError(f"Resumed test rows do not match fold {fold_key}")
            for frame in (resumed_oof, resumed_test):
                probability = frame["probability"].to_numpy(dtype=float)
                if not np.isfinite(probability).all() or not (
                    (probability >= 0.0) & (probability <= 1.0)
                ).all():
                    raise RuntimeError(f"Invalid resumed probabilities for fold {fold_key}")
            expected_metric_row = {
                "run_fingerprint": RUN_FINGERPRINT,
                "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
                "fold_identity": {"repeat": repeat, "fold": fold},
                "repeat": repeat,
                "fold": fold,
                "seed": BASE_SEED + repeat * 100 + fold,
                "n_validation": len(expected_index),
                "roc_auc": float(
                    roc_auc_score(resumed_oof["target"], resumed_oof["probability"])
                ),
                "pr_auc": float(
                    average_precision_score(
                        resumed_oof["target"], resumed_oof["probability"]
                    )
                ),
                "log_loss": float(
                    log_loss(
                        resumed_oof["target"],
                        resumed_oof["probability"],
                        labels=[0, 1],
                    )
                ),
                "best_validation_loss": float(checkpoint_payload["best_loss"]),
                "checkpoint": str(fold_checkpoint_path),
                "preprocessor": str(CHECKPOINT_DIR / f"preprocess_r{repeat}_f{fold}.pkl"),
            }
            validate_persisted_fold_metric(metric_row, expected_metric_row, fold_metric_path)
            fold_metric_rows.append(
                {
                    **expected_metric_row,
                    "oof_prediction": expected_oof_meta,
                    "test_prediction": expected_test_meta,
                }
            )
            continue

        valid_index = split_manifest.loc[
            (split_manifest["repeat"] == repeat) & (split_manifest["fold"] == fold),
            "row_index",
        ].to_numpy(dtype=np.int64)
        train_mask = np.ones(len(train_df), dtype=bool)
        train_mask[valid_index] = False
        train_index = np.flatnonzero(train_mask)
        transformed = transform_fold(train_index, valid_index)

        if run_state["preflight"] is None:
            preflight_loader = make_loader(
                TabularDataset(transformed[0], transformed[1], y_true[train_index]),
                True,
                BASE_SEED,
            )
            smoke_result = run_one_batch_smoke_test(preflight_loader, transformed[6], BASE_SEED)
            overfit_result = run_tiny_overfit_test(
                transformed[0], transformed[1], y_true[train_index], transformed[6], BASE_SEED
            )
            run_state["preflight"] = {
                "one_batch_smoke": smoke_result,
                "tiny_overfit": overfit_result,
            }
            atomic_json_dump(run_state, STATE_PATH)

        result, fold_status = fit_fold(
            repeat, fold, train_index, valid_index, transformed, started_at
        )
        if fold_status != "completed":
            stop_reason = fold_status
            run_state["stop_reason"] = stop_reason
            atomic_json_dump(run_state, STATE_PATH)
            break

        oof_fold = pd.DataFrame(
            {
                "row_index": valid_index,
                "repeat": repeat,
                "fold": fold,
                "probability": result["valid_probability"],
                "target": y_true[valid_index],
            }
        )[["row_index", "repeat", "fold", "probability", "target"]]
        test_fold = pd.DataFrame(
            {
                "row_index": np.arange(len(test_df), dtype=np.int64),
                "repeat": repeat,
                "fold": fold,
                "probability": result["test_probability"],
            }
        )
        metric_row = {
            "run_fingerprint": RUN_FINGERPRINT,
            "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
            "repeat": repeat,
            "fold": fold,
            "seed": result["seed"],
            "n_validation": len(valid_index),
            "roc_auc": float(roc_auc_score(oof_fold["target"], oof_fold["probability"])),
            "pr_auc": float(average_precision_score(oof_fold["target"], oof_fold["probability"])),
            "log_loss": float(log_loss(oof_fold["target"], oof_fold["probability"], labels=[0, 1])),
            "best_validation_loss": result["best_loss"],
            "checkpoint": result["checkpoint"],
            "preprocessor": result["preprocessor"],
        }
        atomic_csv_dump(oof_fold, fold_oof_path)
        atomic_csv_dump(test_fold, fold_test_path)
        metric_row["fold_identity"] = {"repeat": repeat, "fold": fold}
        metric_row["oof_prediction"] = {
            "sha256": sha256_file(fold_oof_path),
            "row_count": len(oof_fold),
            "schema": list(oof_fold.columns),
        }
        metric_row["test_prediction"] = {
            "sha256": sha256_file(fold_test_path),
            "row_count": len(test_fold),
            "schema": list(test_fold.columns),
        }
        atomic_json_dump(metric_row, fold_metric_path)
        fold_metric_rows.append(metric_row)
        completed_folds.add(fold_key)
        run_state["completed_folds"] = [list(item) for item in sorted(completed_folds)]
        atomic_json_dump(run_state, STATE_PATH)
        gc.collect()
        torch.cuda.empty_cache()
    if stop_reason is not None:
        break

run_complete = len(completed_folds) == TOTAL_FOLDS
run_state["status"] = "completed" if run_complete else "incomplete"
run_state["stop_reason"] = stop_reason
run_state["elapsed_seconds"] = time.time() - started_at
atomic_json_dump(run_state, STATE_PATH)
print(f"Completed folds: {len(completed_folds)}/{TOTAL_FOLDS}")
print("Run status:", run_state["status"], "stop reason:", stop_reason)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 6. OOF metrics, provenance, inventory, and conditional submission

Fold, repeat, and global metrics are emitted for the evidence actually
completed. The full OOF completeness contract and optional submission are
enabled only when all 15 best-checkpoint fold predictions exist. An incomplete
run remains explicitly labelled incomplete."""
        ),
        nbf.v4.new_code_cell(
            r"""# OOF includes only fully completed folds; in-progress checkpoints
# never contribute rows.
completed_fold_keys = sorted(completed_folds)
oof_parts = [
    pd.read_csv(OUTPUT_DIR / f"fold_oof_r{r}_f{f}.csv")
    for r, f in completed_fold_keys
]
test_parts = [
    pd.read_csv(OUTPUT_DIR / f"fold_test_r{r}_f{f}.csv")
    for r, f in completed_fold_keys
]
oof_df = (
    pd.concat(oof_parts, ignore_index=True)
    if oof_parts
    else pd.DataFrame(columns=["row_index", "repeat", "fold", "probability", "target"])
)
oof_df = oof_df[["row_index", "repeat", "fold", "probability", "target"]]
atomic_csv_dump(oof_df, OUTPUT_DIR / "oof_predictions.csv")

FOLD_METRIC_COLUMNS = [
    "run_fingerprint_sha256",
    "repeat",
    "fold",
    "seed",
    "n_validation",
    "roc_auc",
    "pr_auc",
    "log_loss",
    "best_validation_loss",
    "checkpoint",
    "preprocessor",
]
fold_metrics_df = pd.DataFrame(fold_metric_rows, columns=FOLD_METRIC_COLUMNS)
if not fold_metrics_df.empty:
    fold_metrics_df = fold_metrics_df.sort_values(["repeat", "fold"])
atomic_csv_dump(fold_metrics_df, OUTPUT_DIR / "fold_metrics.csv")
repeat_metric_rows = []
for repeat, repeat_oof in oof_df.groupby("repeat"):
    repeat_metric_rows.append(
        {
            "repeat": int(repeat),
            "completed_folds": int(repeat_oof["fold"].nunique()),
            "n_predictions": len(repeat_oof),
            "roc_auc": float(roc_auc_score(repeat_oof["target"], repeat_oof["probability"])),
            "pr_auc": float(
                average_precision_score(repeat_oof["target"], repeat_oof["probability"])
            ),
            "log_loss": float(
                log_loss(repeat_oof["target"], repeat_oof["probability"], labels=[0, 1])
            ),
        }
    )
repeat_metrics_df = pd.DataFrame(
    repeat_metric_rows,
    columns=["repeat", "completed_folds", "n_predictions", "roc_auc", "pr_auc", "log_loss"],
)
atomic_csv_dump(repeat_metrics_df, OUTPUT_DIR / "repeat_metrics.csv")

global_metrics = {
    "status": "completed" if run_complete else "incomplete",
    "completed_folds": len(completed_folds),
    "expected_folds": TOTAL_FOLDS,
    "completed_oof_rows": len(oof_df),
}
if not oof_df.empty:
    available_oof = oof_df.groupby("row_index", sort=True, as_index=False).agg(
        probability=("probability", "mean"), target=("target", "first")
    )
    global_metrics["available_unique_rows"] = len(available_oof)
    global_metrics["available_prediction_metrics"] = {
        "roc_auc": float(roc_auc_score(available_oof["target"], available_oof["probability"])),
        "pr_auc": float(
            average_precision_score(available_oof["target"], available_oof["probability"])
        ),
        "log_loss": float(
            log_loss(available_oof["target"], available_oof["probability"], labels=[0, 1])
        ),
        "scope": "mean of available fully completed-fold predictions per row",
    }
if run_complete:
    assert len(oof_df) == len(train_df) * N_REPEATS
    assert oof_df.groupby(["repeat", "row_index"]).size().eq(1).all()
    mean_oof = oof_df.groupby("row_index", sort=True, as_index=False).agg(
        probability=("probability", "mean"), target=("target", "first")
    )
    assert len(mean_oof) == len(train_df)
    global_metrics.update(
        {
            "roc_auc": float(roc_auc_score(mean_oof["target"], mean_oof["probability"])),
            "pr_auc": float(average_precision_score(mean_oof["target"], mean_oof["probability"])),
            "log_loss": float(log_loss(mean_oof["target"], mean_oof["probability"], labels=[0, 1])),
        }
    )
atomic_json_dump(global_metrics, OUTPUT_DIR / "global_metrics.json")

training_manifest = {
    "experiment_name": "ft_transformer_5x3",
    "status": "completed" if run_complete else "incomplete",
    "feature_policy": "base",
    "target_column": TARGET_COLUMN,
    "target_mapping": TARGET_MAPPING,
    "data_hashes": DATA_HASHES,
    "split_content_sha256": observed_split_hash,
    "split_hash_algorithm": "sha256(canonical [row_index,repeat,fold] little-endian int64 bytes)",
    "run_fingerprint": RUN_FINGERPRINT,
    "run_fingerprint_sha256": RUN_FINGERPRINT_SHA256,
    "config_sha256": hash_json(CONFIG_PAYLOAD),
    "source_sha256": PIPELINE_SOURCE_SHA256,
    "source_scope": "all notebook cell sources with source-hash placeholder normalization",
    "config": CONFIG_PAYLOAD,
    "completed_folds": [list(item) for item in completed_fold_keys],
    "expected_fold_count": TOTAL_FOLDS,
    "oof_schema": ["row_index", "repeat", "fold", "probability", "target"],
    "preflight": run_state["preflight"],
    "environment": {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
    },
    "device": {
        "type": str(DEVICE),
        "name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(0)),
    },
    "runtime": {
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "time_buffer_seconds": TIME_BUFFER_SECONDS,
        "elapsed_seconds": run_state["elapsed_seconds"],
        "stop_reason": stop_reason,
        "restored_resume_dir": RESTORED_RESUME_DIR,
    },
}
atomic_json_dump(training_manifest, OUTPUT_DIR / "training_manifest.json")

if run_complete:
    averaged_test = np.mean(
        [part.sort_values("row_index")["probability"].to_numpy() for part in test_parts], axis=0
    )
    if not np.isfinite(averaged_test).all() or not (
        (averaged_test >= 0) & (averaged_test <= 1)
    ).all():
        raise FloatingPointError("Invalid complete-run test probabilities")
    submission = sample_submission.copy()
    assert list(submission.columns) == [ID_COLUMN, TARGET_COLUMN]
    assert len(submission) == len(test_df)
    assert submission[ID_COLUMN].equals(test_df[ID_COLUMN])
    submission[TARGET_COLUMN] = averaged_test
    atomic_csv_dump(submission, OUTPUT_DIR / "submission.csv")
    test_predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(test_df)),
            "probability": averaged_test,
        }
    )
    atomic_csv_dump(test_predictions, OUTPUT_DIR / "test_predictions.csv")
    print("All 15 folds complete; submission.csv was generated from best checkpoint states.")
else:
    print("INCOMPLETE RUN: no submission.csv generated; resume using checkpoints.")


def artifact_row_count(path):
    if path.suffix == ".csv":
        return int(sum(1 for _line in path.open(encoding="utf-8")) - 1)
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return len(payload) if isinstance(payload, (list, dict)) else 1
    return None


inventory = []
for artifact_path in sorted(path for path in OUTPUT_DIR.rglob("*") if path.is_file()):
    if artifact_path.name == "artifact_inventory.json" or artifact_path.name.endswith(".tmp"):
        continue
    inventory.append(
        {
            "path": str(artifact_path.relative_to(OUTPUT_DIR)),
            "sha256": sha256_file(artifact_path),
            "byte_count": artifact_path.stat().st_size,
            "row_count": artifact_row_count(artifact_path),
        }
    )
atomic_json_dump(
    {
        "status": "completed" if run_complete else "incomplete",
        "artifact_count": len(inventory),
        "artifacts": inventory,
    },
    OUTPUT_DIR / "artifact_inventory.json",
)
print("Artifacts:", OUTPUT_DIR)
print("Global metrics:", global_metrics)"""
        ),
        nbf.v4.new_markdown_cell(
            """## 7. Kaggle handoff checklist

1. Confirm the final status printed above is `completed` and the completed fold
   count is 15 before using `submission.csv`.
2. In Kaggle's Output panel, download the complete
   `/kaggle/working/ft_transformer_5x3` folder, including checkpoints, OOF
   predictions, metrics, manifest, and inventory.
3. For a resumed session, attach the previous output as a private Kaggle Dataset.
   The notebook discovers the unique fingerprint-compatible root and atomically
   stages the verified files; use `RESUME_INPUT_DIR` only to resolve ambiguity.
4. Compare or blend locally using `oof_predictions.csv` keyed by `row_index`,
   `repeat`, and `fold`; authenticate artifacts with `training_manifest.json`
   and `artifact_inventory.json` first."""
        ),
    ]
    for index, cell in enumerate(notebook.cells):
        identity = f"{index}:{cell.cell_type}:{cell.source}".encode()
        cell.id = hashlib.sha256(identity).hexdigest()[:16]
    source_sha256 = notebook_source_sha256(notebook)
    for cell in notebook.cells:
        cell.source = cell.source.replace(SOURCE_HASH_PLACEHOLDER, source_sha256)
    return notebook


def main() -> None:
    notebook = build_notebook()
    nbf.validate(notebook)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT_PATH)
    print(f"Generated {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

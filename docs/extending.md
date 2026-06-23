# Extending the template

The engine (`src/engine.py`) is task-agnostic: it builds the model, data, loss,
optimizer, scheduler and metrics from config via `hydra.utils.instantiate`. Adding a
task means **writing classes + config, never editing the engine.**

## 1. A model

Any `torch.nn.Module`. Put the class under `src/models/`, then add config:

```yaml
# configs/model/my_model.yaml
_target_: src.models.my_model.MyModel
hidden_dim: 256
num_classes: 10
```

The engine calls `instantiate(cfg.model)` and `.to(device)`. It is wrapped in DDP
automatically when launched under `torchrun`.

## 2. A dataset (the "data module" contract)

`cfg.data` must instantiate an object exposing four attributes the engine relies on:

```python
class MyData:
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = ...  # torch.utils.data.Dataset
        self.val_dataset = ...    # torch.utils.data.Dataset
```

```yaml
# configs/data/my_data.yaml
_target_: src.data.my_data.MyData
data_dir: ${oc.env:DATA_DIR,./data}   # config-driven path, never hard-coded
batch_size: 128
num_workers: 4
```

You do **not** build the `DataLoader` — the engine does, so it can inject the seeded
`generator` and `worker_init_fn=seed_worker` for reproducible shuffling/augmentation.
Batches must be `(inputs, targets)`.

## 3. A loss

Usually already in `torch.nn`:

```yaml
# configs/loss/cross_entropy.yaml
_target_: torch.nn.CrossEntropyLoss
```

## 4. Optimizer / scheduler

Provided generically (`configs/optimizer/`, `configs/scheduler/`). The engine
completes them with the live objects — `instantiate(cfg.optimizer,
params=model.parameters())` and `instantiate(cfg.scheduler, optimizer=optimizer)` —
so the configs only carry hyperparameters. Add your own the same way.

## 5. Metrics (optional)

`cfg.metrics` instantiates a dict of callables `metric(outputs, targets) -> float`:

```yaml
# configs/metrics/classification.yaml
accuracy:
  _target_: src.metrics.Accuracy
```

Validation reports `val/loss` plus each metric as `val/<name>`.

## 6. Tie it together with an experiment config

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - override /model: my_model
  - override /data: my_data
  - override /loss: cross_entropy
  - override /metrics: classification

trainer:
  epochs: 20
```

Run it:

```bash
python -m src.train +experiment=my_experiment
```

See the `example` branch for all of the above, filled in for a ViT on FashionMNIST.

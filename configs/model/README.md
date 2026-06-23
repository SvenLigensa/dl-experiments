# `model` config group

This branch (`main`) is **task-independent** and ships no model, so `model` defaults
to `???` (a mandatory-missing value) in `configs/config.yaml`. Running without
selecting one fails fast with a clear Hydra error — that is expected.

To add a model, drop a YAML here whose `_target_` points at any `torch.nn.Module`,
e.g. `configs/model/vit.yaml`:

```yaml
_target_: src.models.vit.ViT
img_size: 28
patch_size: 7
num_classes: 10
```

The engine calls `hydra.utils.instantiate(cfg.model)` — no engine code changes
needed. See a worked example on the `example` branch and `docs/extending.md`.

# `loss` config group

Task-independent on `main`, so `loss` defaults to `???`. Point `_target_` at any
callable that returns `loss(outputs, targets) -> scalar tensor`. Many live in
`torch.nn` already, so no custom code is needed:

```yaml
# configs/loss/cross_entropy.yaml
_target_: torch.nn.CrossEntropyLoss
```

The engine instantiates it via `hydra.utils.instantiate(cfg.loss)`.

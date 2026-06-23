# `data` config group

Task-independent on `main`, so `data` defaults to `???`. Add a YAML whose `_target_`
instantiates a **data module** object exposing `train_dataset`, `val_dataset`,
`batch_size`, and `num_workers` (the contract the engine relies on — see
`src/engine.py`).

Keep the data location **config-driven** (hard-coded absolute paths are the #1 reason
a repo fails to move between machines). Use an interpolation with an env-var fallback:

```yaml
_target_: src.data.fashion_mnist.FashionMNISTData
data_dir: ${oc.env:DATA_DIR,./data}
batch_size: 128
num_workers: 4
```

On Colab `data_dir` is typically a Drive mount; on a cluster it is scratch vs home.
See `docs/extending.md` and the `example` branch.

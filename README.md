# pypots-pyrregular

Minimal, publication-oriented setup for running customized deep learning model on PYRREGULAR datasets.
The repository supports a single-dataset sweep on PYRREGULAR.
The upstream PyPOTS repository is kept as a Git submodule;
only the MASCIT files required by the runner live in this repository.

## Setup

```bash
git clone --recurse-submodules https://github.com/yoom618/pypots-pyrregular.git
cd pypots-pyrregular
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e ./PyPOTS
python -m pip install causal-conv1d --no-build-isolation  # only when using Mamba
python -m pip install mamba-ssm --no-build-isolation  # only when using Mamba
```

If the repository was cloned without submodules, initialize PyPOTS with:

```bash
git submodule update --init --recursive
```

`requirements.txt` first installs the dependencies declared by the pinned `PyPOTS` submodule and then installs `pyrregular` and the runner dependency `psutil`.
The editable PyPOTS package and the two MASCIT CUDA extensions are installed separately because the extensions require `--no-build-isolation`.


## Example

- Run one dataset with MASCIT (MambaSL with mask)
- Use training loss as validation metric (as in the InceptionTime paper)

```bash
bash scripts/run_mascit_train_loss_only_single_dataset.sh ABF
```
OR
```bash
python scripts/run_mascit_train_loss_only_single_dataset.py \
  --dataset ABF \
  --dataset-dir dataset \
  --sweep-config configs/mascit_sweep_128x16.json \
  --device cuda
```

- Use `--help` to see the full sweep controls.
- Results are written under `testing_results/mascit_train_loss_only_single_dataset/` unless `--output-dir` is supplied.
- The runner uses `dataset/` under the repository root for the PYRREGULAR cache by default. Use `--dataset-dir` to select another location.
- The default sweep space is defined in `configs/mascit_sweep_128x16.json`. Use `--sweep-config` to select another JSON grid without editing the runner.


## Related Publication
```bibtex
@inproceedings{jung2026mascit,
  title={MASCIT: A Mask-Aware State Space Classifier for Naturally Irregular Time Series},
  author={Yoo-Min Jung and Hyeon-Gi Kim and Jonghun Park},
  booktitle={Proceedings of the 26th Asia-Pacific Industrial Engineering and Management Systems Conference (APIEMS)},
  year={2026},
  url={https://arxiv.org/abs/2609.34409}
}
```

---
name: kjob-install
description: Install, upgrade or troubleshoot the kjob CLI (ksub, ksrun, kqueue, ksinfo ...) inside a Kubeflow Notebook terminal on a Kubernetes GPU cluster. Use when the user asks to install/set up/update kjob, when a kjob command is "not found", or before submitting GPU jobs on a notebook that does not have kjob yet.
---

# Installing kjob in a Kubeflow Notebook

kjob submits a script from a Kubeflow Notebook as a Kubernetes PyTorchJob that uses the same image, working
directory and shared volumes as the notebook. Commands are Slurm-like (ksub = sbatch, ksrun = srun, kqueue = squeue,
ksinfo = sinfo). Reply to the user in their language (most users write Traditional Chinese).

## 1. Check where you are

Run these first and report anything unexpected to the user instead of guessing:

```bash
cat /var/run/secrets/kubernetes.io/serviceaccount/namespace   # must exist: we are inside a Notebook pod
python3 --version                                            # 3.8 or newer
python3 -m pip --version
```

If the namespace file does not exist, you are not in a Kubeflow Notebook (e.g. a personal laptop or the cluster host).
kjob jobs are created *from* a Notebook; tell the user to open a terminal in their Notebook (JupyterLab ->
File -> New -> Terminal, or the terminal of the remote desktop image) and continue there.

## 2. Install

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/main.tar.gz
export PATH="$HOME/.local/bin:$PATH"
grep -q 'HOME/.local/bin' ~/.bashrc || echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

* The archive URL needs no `git`. With git available this is equivalent:
  `python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@main'`.
* Inside an activated virtualenv/conda env: drop `--user` and use that env's `python -m pip`.
* `~/.local` is on the notebook workspace volume, so the install survives notebook restarts.
* It installs only kjob and the `kubernetes` Python client. It does not install or change PyTorch/CUDA.

## 3. Verify

```bash
ksub --help | head -5     # command exists
kinfo                     # notebook, namespace, inherited volumes, GPU quota
ksinfo                    # cluster GPU types and free GPUs (needs the administrator's status service)
```

* Your AI tool blocked or asked to approve the command (no kjob output, a "denied"/"blocked"/approval message): the
  command never ran, so this is not a kjob or cluster problem. Tell the user exactly that, and see "When a command
  fails" in the `kjob-jobs` skill for the commands to allow. Do not work around it.
* `kinfo` fails with `Kubernetes API error (403 ...)`: the Notebook's service account lacks permissions (Notebooks
  created in the Kubeflow UI use `default-editor`, which has them). Only the cluster administrator can fix this; tell
  the user and quote the message.
* `ksinfo` says "cluster-wide view is not available": the administrator has not deployed the status service.
  Job submission still works; only the cluster overview is missing.

## 4. AI skills

The first kjob command (step 3) installs the `kjob-install` and `kjob-jobs` skills automatically into
`~/.agents/skills` (Codex, OpenCode, Gemini CLI, Cursor and other Agent Skills tools) and links each one from
`~/.claude/skills` (Claude Code reads only that), and updates them when kjob is upgraded (copies the user edited
are kept).
For any other location: `kjob install-skills --dir <its skills dir>`.
Start a new assistant session if the skills do not show up yet, then use `kjob-jobs` to submit and monitor jobs.

## Upgrade / uninstall

```bash
python3 -m pip install --user --upgrade --force-reinstall --no-deps https://github.com/MarkJiang-maji/kjob/archive/refs/heads/main.tar.gz
python3 -m pip uninstall kjob
```

## Troubleshooting

| Symptom | What to do |
|---|---|
| `ksub: command not found` after install | `export PATH="$HOME/.local/bin:$PATH"`, or run `python3 -m kjob ...` |
| `Can not perform a '--user' install` | you are in a virtualenv: install without `--user` |
| pip cannot reach GitHub / PyPI | ask the administrator for the wheel file, then `python3 -m pip install --user kjob-*.whl` |
| `externally-managed-environment` | the notebook image's Python is system-managed; use a virtualenv on `/home/jovyan` |

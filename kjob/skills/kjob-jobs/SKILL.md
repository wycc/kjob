---
name: kjob-jobs
description: Submit, monitor, debug and cancel GPU training jobs with kjob (ksub, ksrun, kqueue, kshow, klogs, kexec, kwait, kcancel, kacct, ksinfo) from a Kubeflow Notebook on a Kubernetes GPU cluster. Use when the user wants to run a script on GPUs "as a job", check GPU availability or the queue, see whose jobs are running, follow job output, or find out why a job is pending or failed.
---

# Running jobs with kjob

kjob runs a script as a PyTorchJob that copies the Notebook: same image, same working directory, same shared volumes
(`/home/jovyan` and group/shared NFS folders). The Notebook itself does not need a GPU. If kjob is not installed
(`ksub: command not found`), use the `kjob-install` skill first. Reply in the user's language.

## Ground rules for the assistant

* Only touch the user's own jobs. Never cancel, clean or modify other users' workloads; `kqueue -A` is read-only.
* `kexec NAME` without a command opens an interactive shell and needs a terminal; as an assistant always pass a
  command (`kexec NAME nvidia-smi`, `kexec NAME ls /home/jovyan/out`). Do not kill or modify the training from inside.
* Ask before `kcancel --all-mine`, `kcancel --all-notebooks` or anything that deletes several jobs.
* Request what the script needs, not the maximum: GPUs are shared with the whole lab. Prefer `-G 1` unless the code
  is multi-GPU (DDP/torchrun). Always set a time limit with `-t` for long runs.
* Do not run heavy training directly in the Notebook terminal; submit it.
* Report the job name, GPU type and log path to the user after submitting.

## When a command fails: your tool's permission, or kjob?

Before explaining any failure, decide which of these it is, and tell the user which one in plain words. Users
otherwise conclude that kjob or the cluster is broken when only the assistant was not allowed to run the command.

| What you see | Cause | Tell the user |
|---|---|---|
| No kjob output at all; your tool reports the command was denied, rejected, blocked, needs approval, or "the user doesn't want to proceed" (Claude Code also: "denied by the auto mode classifier") | **Your own tool's permission system.** The command never ran; kjob and the cluster are not involved | "This kjob command did not run: Claude Code (or your AI tool) blocked it, so it is not a kjob or cluster problem." Then say how to allow it (below) |
| `kjob: cannot reach the Kubernetes API ...` while you run in a sandbox | Probably your tool's sandbox blocking network access | Same as above: the sandbox blocked kjob's connection; ask to allow kjob outside the sandbox. Outside a sandbox it is a cluster or network problem |
| `kjob: Kubernetes API error (403 ...)` with `cannot ... resource` | The Notebook's service account lacks a Kubernetes permission (kjob did run) | Quote the message. Notebooks from the Kubeflow UI use `default-editor`, which has every permission; otherwise only the administrator can fix it |
| `kjob: Kubernetes API error (403 ...)` with `exceeded quota` | Namespace quota full | Wait for other jobs or request less |
| `... has no access to the requested GPUs ...` | Restricted GPU type not enabled for this namespace | Choose another type in `ksinfo`, or ask the administrator |
| Any other line starting with `kjob:` | kjob checked something and refused (bad path, name in use, unknown GPU type ...) | Explain the message and fix the command |

When your tool blocked the command:

* Do not try to get around it (no `python -m kjob`, `kubectl`, scripts or other commands that do the same thing).
  Say what was blocked and wait for the user.
* Commands that only read: `kinfo`, `ksinfo`, `kqueue`, `kshow`, `klogs`, `kacct`. Commands that change the cluster:
  `ksub`, `ksrun` (create jobs; `ksrun` deletes its job at the end), `kwait` (deletes the job when it ends), `kexec`
  (runs a command inside a job), `kcancel`, `kclean` (delete jobs). An approval prompt for the second group is normal.
* Claude Code: the user can approve the prompt, run `/permissions`, or add rules to `~/.claude/settings.json`, e.g.
  `{"permissions": {"allow": ["Bash(kinfo:*)", "Bash(ksinfo:*)", "Bash(kqueue:*)", "Bash(kshow:*)", "Bash(klogs:*)",
  "Bash(kacct:*)", "Bash(ksub:*)", "Bash(ksrun:*)", "Bash(kwait:*)", "Bash(kexec:*)"]}}`. Leave `kcancel` and
  `kclean` out so deleting still asks. Other tools (Codex, Gemini CLI, ...) have their own approval settings.

## Workflow

1. **Look at the environment**

   ```bash
   kinfo        # namespace, image, volumes the job will see, GPU quota used/hard
   ksinfo       # GPU types: total / used / free / waiting, per node; restricted nodes and whether we may use them
   ```

   Pick a GPU type with free GPUs that fits the model (memory per GPU is shown). `restricted ... no access` means jobs
   from this namespace cannot run there: do not choose it.

2. **Prepare the script** so it runs unattended:
   * it lives on a shared volume (under `/home/jovyan` or a group/shared folder), and so do its data and outputs
     (the Notebook's `/tmp` is not visible to jobs);
   * no `input()`, no GUI; print progress (`print(..., flush=True)` or `python -u`);
   * save checkpoints/results to files under `/home/jovyan/...`;
   * extra packages: install them with `pip install --user` in the Notebook (they go to `~/.local`, which jobs see)
     or use a virtualenv on `/home/jovyan` and pass `--python /home/jovyan/venv/bin/python`.
   * multi-GPU: write a `.sh` that calls `torchrun --standalone --nproc_per_node=N train.py` and submit with `-G N`.

3. **Submit**

   ```bash
   ksub -J my-train -G 1 -g v100 -c 4 --mem 32Gi --shm 8Gi -t 12:00:00 /home/jovyan/project/train.py --epochs 50
   ```

   Prints `Submitted job my-train  (log: /home/jovyan/kjob-logs/my-train.log)`. Use `ksrun` instead of `ksub` for
   short runs where the user wants to watch output live (Ctrl-C cancels the job).

4. **Monitor**

   ```bash
   kqueue              # this namespace's jobs: STATE, POD phase, NODE, GPUs, age
   klogs my-train -f   # follow output (Ctrl-C only stops watching)
   kwait my-train      # follow until it ends, then delete the finished job; exit code = script's
   kexec my-train nvidia-smi   # run a command inside the running job (GPU utilisation, files, processes)
   kshow my-train      # details: resources, constraints, conditions, pod, recent warning events
   kacct               # history of submitted jobs with exit codes and run time
   ```

5. **Results** are wherever the script wrote them; the full output is also in `~/kjob-logs/NAME.log`.

## Options of ksub / ksrun

| Option | Default | Meaning |
|---|---|---|
| `-J, --job-name` | script name + suffix | job name (lowercase letters, digits, `-`) |
| `-G, --gpus` | 1 | GPUs; `0` = CPU-only |
| `-g, --gpu-type` | any | short name (`v100`, `a5000`; `kinfo` lists the ones this cluster has, never suggest a type `ksinfo` does not show), a GPU_TYPE from `ksinfo` or an unambiguous part of it (case-insensitive), a node name, or a raw product label; comma list = any of them. Ambiguous or unknown names are rejected |
| `-w, --nodelist` / `-x, --exclude` | - | run on / avoid specific nodes |
| `-c, --cpus` | 1 | CPU cores |
| `--mem` | 2Gi | memory (Kubernetes units: `32Gi`) |
| `--shm` | same as `--mem` | `/dev/shm` size limit; its use counts toward `--mem`, so raise `--mem` for DataLoaders with many workers |
| `-t, --time` | none | time limit counted from when the pod is placed on a node (image download and start-up included, queueing not): `30` (min), `2:00:00`, `1-12:00:00`, `12h`, `2d`. At the limit the script gets SIGTERM and 30 s to save |
| `--python` | notebook's python3 | interpreter for `.py` (e.g. a venv on `/home/jovyan`) |
| `--env KEY=VALUE` | - | extra environment variable, repeatable |
| `-D, --chdir` | current dir | working directory |
| `-o, --log-dir` | `~/kjob-logs` | where the output log is written |
| `--ttl` | 3600 | seconds a finished, unwatched job object is kept |
| `--live-progress` | off | show `\r` progress bars (tqdm) at once in ksrun/kwait/klogs; the pod log is then framed and reads badly in kubectl logs / the Kubeflow UI. Off: plain pod log, progress may appear only at the next newline |

Script files may also carry options at the top (before the first command): `#KJOB -G 2 -g v100 --mem 64Gi` or
Slurm-style `#SBATCH --gres=gpu:v100:2 --mem=64G -t 1-00:00:00 -c 8 -J name`. Unsupported `#SBATCH` options
(`-p`, `-A`, `-o`, `--mail-*`, `--ntasks` ...) are ignored with a warning. Command-line options override them.

## Cluster view

```bash
ksinfo                    # all GPU types and nodes
ksinfo -g rtxpro6000      # one type/node, plus who is running or waiting there
kqueue -A                 # every GPU workload in the cluster: user (namespace), kind, GPUs, type, node, wait reason
kqueue -A --state pending # only the waiting ones (the queue)
kqueue -u mxxxxxxx        # one user's (namespace's) workloads
```

## Why is my job Pending? (`kshow NAME`, WAITING_FOR column)

| Message | Meaning / fix |
|---|---|
| `Insufficient nvidia.com/gpu` | all GPUs of that type are busy: wait, or choose another type with free GPUs in `ksinfo` |
| `didn't match Pod's node affinity/selector` | the `-g`/`-w` combination matches no node; check names in `ksinfo` |
| `untolerated taint {...}` | restricted node; this namespace has no access (newer kjob refuses such a job when it is submitted) |
| `exceeded quota` / `exceeds namespace quota` | namespace GPU/CPU/memory quota is used up (Notebooks count too): stop an idle GPU Notebook or wait |
| `Insufficient memory` / `cpu` | lower `--mem` / `-c` |

## Why did it fail?

* `exit code 137, OOMKilled`: out of memory: raise `--mem`. (137 without OOMKilled: killed after the 30 s stop grace period.)
* `Bus error` / `DataLoader worker ... exited unexpectedly`: shared memory ran out. `/dev/shm` defaults to the `--mem` limit and its use counts toward it: raise `--mem` (and `--shm` if it was set).
* `CUDA out of memory`: smaller batch, or a GPU type with more memory.
* `No such file or directory`: script or data not on a shared volume, or a relative path; use absolute paths.
* `ModuleNotFoundError`: package installed somewhere the job cannot see; see step 2.
* `log write failed` / `LOG_ERROR`: the Pod is retained. Save `klogs NAME` to writable shared storage before
  `--ttl`, `kcancel`, or `kclean` removes it. Exit 74 means log storage failed; existing script errors keep their code.
* `log stream ended but completion could not be confirmed`: the job is retained. Check `kshow NAME`, then retry `kwait NAME`.
* Stopped at the time limit: "stopped at the time limit" / `DeadlineExceeded` (ksrun/kwait/kshow; they exit with 124 even if the
  script saved and exited 0), `STOPPED` in kacct (`CANCELLED`: deleted by kcancel before it could log an end). Training code should handle
  SIGTERM (save a checkpoint) since it gets 30 s before being killed; resubmit with a larger `-t` and
  resume from a checkpoint.

## Cancel and clean up

```bash
kcancel my-train          # delete one job (the log file stays)
kclean                    # delete this notebook's finished jobs
```

# kjob 新手上手

這份教學帶你從安裝到提交第一個 GPU 工作，大約 10 分鐘。所有指令都在 **Notebook 的 Terminal** 執行
（JupyterLab：File → New → Terminal；遠端桌面 image：開啟終端機）。Notebook 本身不需要 GPU。

完整選項見 [docs/USER_GUIDE.md](docs/USER_GUIDE.md)。

## 1. 安裝

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/cgu-public.tar.gz
export PATH="$HOME/.local/bin:$PATH"
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

更新到最新版：重新執行第一行。使用虛擬環境時拿掉 `--user`。

## 2. 看看自己的環境和叢集

```bash
kinfo
```

```text
notebook  mynb-0  (namespace m1461023, node hs5800-2)
image     cguaicadmin/newlab-newpytorch:V1.0.8
workdir   /home/jovyan  (jobs start here unless -D is given)
volumes inherited by jobs:
  /home/jovyan             nfs 10.100.1.35:/Public/m1461023-mynb-workspace-pvc-...
  /mnt/groups/mylab        nfs ...
quota (used / hard):
  requests.nvidia.com/gpu      0 / 8
```

`volumes inherited by jobs` 列出的路徑，工作裡都看得到；程式和資料要放在這些地方。`quota` 是你的 namespace
能同時使用的 GPU 數（Notebook 自己占的 GPU 也算在內）。

```bash
ksinfo
```

```text
GPU_TYPE                                      NODES  GPUS  USED  FREE  WAITING_GPUS  GPU_MEM  NOTE
NVIDIA-RTX-A5000                              1      10    10    0     -             24GB
NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition  1      8     4     4     -             96GB     restricted (your namespace: no access)
Tesla-V100-SXM2-16GB                          4      32    2     30    8 (1 jobs)    16GB
```

選 `FREE` 有空位、記憶體夠的型號。`restricted` 表示需要另外開通，`no access` 的型號提交了也只會一直排隊。
想看某一種 GPU 是誰在用：`ksinfo -g rtxpro6000`；看全叢集排隊：`kqueue -A`。

> 若顯示 `cluster-wide view is not available`，代表管理員尚未啟用叢集狀態服務；提交工作不受影響。

## 3. 建立測試程式（整段複製貼上）

下面這段會在 `~/kjob-demo` 建立兩個檔案：GPU 測試程式 `gpu_test.py`，以及把資源設定寫在開頭的 `train.sh`。
整段複製到 Terminal 貼上即可：

```bash
mkdir -p ~/kjob-demo && cd ~/kjob-demo
cat > gpu_test.py <<'PY'
import os, time, torch
print("job", os.environ.get("KJOB_NAME"), "on node", os.environ.get("KJOB_NODE"), flush=True)
print("GPUs:", torch.cuda.device_count(), "x", torch.cuda.get_device_name(0), flush=True)
x = torch.randn(4096, 4096, device="cuda")
for step in range(5):
    (x @ x).sum().item()
    print(f"step {step + 1}/5 ok", flush=True)
    time.sleep(1)
PY
cat > train.sh <<'SH'
#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 10
set -e
cd /home/jovyan/kjob-demo
python3 -u gpu_test.py
SH
ls ~/kjob-demo
```

`train.sh` 開頭的 `#SBATCH` 是國網的寫法：1 張 GPU、2 個 CPU、8G 記憶體、最多 10 分鐘。

## 4. 提交並即時看輸出：`ksrun`

```bash
ksrun ~/kjob-demo/train.sh
```

```text
kjob: submitted train-1a2b3c4d5e6f (log: /home/jovyan/kjob-logs/train-1a2b3c4d5e6f.log)
kjob: train-1a2b3c4d5e6f running on hs5800-3
[kjob 2026-10-15T09:28:54Z] start train-1a2b3c4d5e6f on node hs5800-3
job train-1a2b3c4d5e6f on node hs5800-3
GPUs: 1 x Tesla V100-SXM2-16GB
step 1/5 ok
...
kjob: train-1a2b3c4d5e6f Succeeded (exit code 0)
```

想指定 GPU 型號就在指令加 `-g`，例如 `ksrun -g v100 ~/kjob-demo/train.sh`；名稱用 `ksinfo` 裡看到的
（例如 `a5000`、`a30`、`rtxpro6000`）。命令列的選項比檔案開頭的設定優先。
`ksrun` 結束後會自動刪除工作；按 Ctrl-C 會取消工作。

## 5. 背景提交：`ksub`

長時間的訓練用 `ksub`，提交後就可以關掉瀏覽器：

```bash
ksub -J demo1 ~/kjob-demo/train.sh
```

```text
Submitted job demo1  (log: /home/jovyan/kjob-logs/demo1.log)
```

`-t`（或檔案裡的 `#SBATCH -t`）是時間上限，從開始執行算起，時間到工作會被停止（格式：`30` 分鐘、`2:00:00`、
`1-12:00:00`、`12h`）。停止前程式會先收到 SIGTERM，有 30 秒可以存 checkpoint。

## 6. 查看工作

```bash
kqueue              # 自己的工作：狀態、節點、GPU 數
klogs demo1 -f      # 即時看輸出（還在排隊會先等），Ctrl-C 只停止觀看，工作繼續跑
kshow demo1         # 詳細資訊；排隊太久時看這裡的 event 找原因
kexec demo1 nvidia-smi   # 在執行中的工作裡跑指令，例如看 GPU 使用率；只打 kexec demo1 會開 shell 進去
kwait demo1         # 等到結束，回傳程式的結束代碼
kacct               # 歷史紀錄：是否成功、結束代碼、跑了多久
```

```text
NAME   SUBMITTED(UTC)    GPU  GPU_TYPE/NODE  STATE      EXIT  ELAPSED
demo1  2026-10-15 09:31  1    any            COMPLETED  0     0m07s
```

輸出永久保存在 `~/kjob-logs/demo1.log`；程式寫到 `/home/jovyan` 的檔案就在你的 Notebook 裡。

## 7. 換成自己的程式

### 多張 GPU 的範例（整段複製貼上）

建立一個用 2 張 GPU 的 PyTorch DDP 訓練範例並提交。`train_ddp.sh` 裡已附上常用的選項，`##SBATCH`（兩個 `#`）
和 `# source ...` 代表先不使用，要用時拿掉一個 `#` 即可（國網也是這個慣例）：

```bash
mkdir -p ~/kjob-demo && cd ~/kjob-demo
cat > ddp_train.py <<'PY'
import os, torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
dist.init_process_group("nccl")
rank, world, local = dist.get_rank(), dist.get_world_size(), int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local)
model = DDP(torch.nn.Linear(1024, 1024).cuda(), device_ids=[local])
opt = torch.optim.SGD(model.parameters(), lr=0.01)
for step in range(20):
    loss = model(torch.randn(256, 1024, device="cuda")).pow(2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if rank == 0 and step % 5 == 0:
        print(f"step {step:2d}  loss {loss.item():.4f}  ({world} GPUs)", flush=True)
if rank == 0:
    print(f"done on {world} x {torch.cuda.get_device_name(local)}", flush=True)
dist.destroy_process_group()
PY
cat > train_ddp.sh <<'SH'
#!/bin/bash
#SBATCH --gres=gpu:2                # GPU 張數（任何型號）
##SBATCH --gres=gpu:v100:2          # 指定型號：拿掉一個 #；型號名稱看 ksinfo（例如 a5000、a30）
##SBATCH -w hs5800-1                # 指定節點：拿掉一個 #；節點名稱看 ksinfo
#SBATCH -c 4                        # CPU 核心
#SBATCH --mem=16G                   # 記憶體
#SBATCH -t 30                       # 時間上限（分鐘），也可寫 2:00:00、1-00:00:00
#KJOB --shm 4Gi                     # /dev/shm；DataLoader worker 多時加大
set -e
cd /home/jovyan/kjob-demo
# source /home/jovyan/venv/bin/activate     # 用自己的虛擬環境：拿掉開頭的 #
NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")   # 拿到幾張就開幾個程序
torchrun --standalone --nproc_per_node="$NGPU" ddp_train.py
SH
ksrun ~/kjob-demo/train_ddp.sh
```

最後會看到 `done on 2 x ...`。過程中若出現 `[W socket.cpp] ... Address family not supported` 的警告，是 PyTorch 在
沒有 IPv6 的環境下的提示，可以忽略。

要指定 GPU 型號、節點、用虛擬環境，就把上面 `train_ddp.sh` 裡對應那行開頭多的 `#` 拿掉。其他改法（多種型號擇一、
換張數、跑更久、套用到自己的程式）見 [USER_GUIDE 的「腳本範本與常見改法」](docs/USER_GUIDE.md#腳本範本與常見改法)。

## 8. 取消與清理

```bash
kcancel demo1      # 取消／刪除一個工作（日誌檔保留）
kclean             # 刪除這個 Notebook 已結束的工作物件
```

## 9. 常見問題

| 狀況 | 處理 |
|---|---|
| 一直 `Pending` | `kshow 名稱` 看 event。`Insufficient nvidia.com/gpu`：該型號滿了，等候或換型號；`exceeded quota`：namespace GPU 配額用完（關掉閒置的 GPU Notebook）；`untolerated taint`：沒有該節點權限 |
| `exit code 137, OOMKilled` | 記憶體不足，加大 `--mem` |
| `DeadlineExceeded` | 超過 `-t` 時間上限，加大 `-t` 或從 checkpoint 續跑 |
| `Bus error`、DataLoader 錯誤 | 加大 `--shm` |
| `No such file or directory` | 檔案不在共用磁碟或用了相對路徑 |
| `ModuleNotFoundError` | 套件裝在工作看不到的地方，見第 7 節 |
| `ksub: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |

## 讓 AI 幫你提交

第 1 步之後第一次執行 kjob 指令時，skill 就自動裝到 `~/.agents/skills`，並在 `~/.claude/skills` 放連結（畫面會出現一行
`kjob: installed AI skills ...`），Claude Code、Codex、OpenCode 等工具都讀得到。

之後在 Notebook 裡使用 Claude Code 等 AI 工具時，可以直接說「用 kjob 幫我把 train.py 交到有空的 V100 上，
2 張 GPU、跑 12 小時」，AI 會依照 skill 的規則選型號、設定資源、提交並回報工作名稱。你仍應確認 AI 選的資源合理。

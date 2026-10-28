# kjob 使用手冊（完整功能）

新手請先看 [TRIAL_GUIDE.md](../TRIAL_GUIDE.md)。本手冊列出所有指令與選項。

* [運作方式](#運作方式)
* [安裝與更新](#安裝與更新)
* [提交工作：ksub / ksrun](#提交工作ksub--ksrun)
* [腳本內設定：#SBATCH / #KJOB](#腳本內設定sbatch--kjob)
* [腳本範本與常見改法](#腳本範本與常見改法)
* [查看與管理自己的工作](#查看與管理自己的工作)
* [查看整個叢集：ksinfo / kqueue -A](#查看整個叢集ksinfo--kqueue--a)
* [GPU 型號、受限節點與共享 GPU](#gpu-型號受限節點與共享-gpu)
* [排隊與錯誤排解](#排隊與錯誤排解)
* [AI 助手 skill](#ai-助手-skill)
* [Python API](#python-api)
* [檔案與環境變數](#檔案與環境變數)
* [限制](#限制)

## 運作方式

`ksub` 讀取你目前 Notebook Pod 的設定，建立一個 Kubeflow PyTorchJob：

* **image、工作目錄、使用者／群組設定**與 Notebook 相同。主容器依 `kubectl.kubernetes.io/default-container`
  annotation、`notebook-name` label 判斷，不看容器順序；判斷不出來時直接報錯。
* Notebook 主容器的**環境變數**（`env`、`envFrom`）、`runtimeClassName`、Notebook 本身設定的 **toleration** 也會帶過去。
  例外：PodDefault 會再加一次的項目、Kubernetes 自動加的 `node.kubernetes.io/*` toleration，以及 kjob 自己決定的
  `CUDA_VISIBLE_DEVICES`、`NVIDIA_VISIBLE_DEVICES` 等變數。`--env` 指定的值優先。
* **磁碟**：Notebook 主容器掛載的磁碟都會帶過去（`/home/jovyan`、群組資料夾、共用 NFS）。底層是 NFS 的
  ReadWriteOnce 工作磁碟會改用 NFS 直接掛載，所以工作可以跑在別的節點。
  * 不會複製：service account token、Istio sidecar、namespace PodDefault 會自動加上的磁碟（PodDefault 會再加一次）、
    舊的群組資料夾（`gs-*`、`ns-*`，由 PodDefault 加上目前的版本）。讀不到 PodDefault 時（沒有權限或沒有安裝），
    這些磁碟照 Notebook 原樣帶過去，`kinfo` 會註明。
  * `emptyDir`（例如 `/dev/shm`）在工作裡是新的空目錄；Notebook 的 `/tmp` 工作看不到。
  * `kinfo` 會列出實際會帶過去的內容。
* namespace 的 **PodDefault**（例如共用 NFS、受限 GPU 節點的 toleration）也會套用到工作。
* 工作的輸出用 `tee` 同時寫到 Pod log 和 `~/kjob-logs/NAME.log`。
* 工作結束後 Pod 保留供查看，工作物件在 `--ttl` 秒後自動刪除；`ksrun`、`kwait` 會在結束時直接刪除。
  日誌保存失敗或無法確認完成時會保留工作，方便檢查；`--ttl` 仍適用。

## 安裝與更新

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/main.tar.gz
export PATH="$HOME/.local/bin:$PATH"
```

| 情況 | 指令 |
|---|---|
| 有 git，想用 git 安裝 | `python3 -m pip install --user --upgrade 'git+https://github.com/MarkJiang-maji/kjob.git@main'` |
| 固定某個版本 | 把網址中的 `refs/heads/main` 換成 commit SHA：`.../archive/<sha>.tar.gz` |
| 虛擬環境內 | 啟用環境後執行 `python -m pip install --upgrade ...`（不加 `--user`） |
| 沒有網路，用管理員給的 wheel | `python3 -m pip install --user kjob-0.5.0-py3-none-any.whl` |
| 從原始碼開發 | `git clone ...; python3 -m pip install --user -e kjob` |
| 移除 | `python3 -m pip uninstall kjob` |

需要 Python 3.8 以上與 `kubernetes` Python 套件（pip 會自動安裝）。每個指令也可以寫成 `kjob 子指令`，例如
`kjob submit` = `ksub`、`kjob sinfo` = `ksinfo`、`python3 -m kjob queue` = `kqueue`。

## 提交工作：ksub / ksrun

```text
ksub  [選項] 腳本 [腳本參數...]     提交後返回（≈ sbatch）
ksrun [選項] 腳本 [腳本參數...]     提交、即時輸出、結束後刪除工作、回傳程式的結束代碼（≈ srun）
```

| 選項 | 預設 | 說明 |
|---|---|---|
| `-J, --job-name NAME` | 腳本名 + 隨機字尾 | 工作名稱，只能用小寫英數與 `-`，最多 50 字元；同名工作已存在會失敗 |
| `-G, --gpus N` | `1` | GPU 數；`0` 代表只用 CPU（工作看不到任何 GPU） |
| `-g, --gpu-type TYPE[,TYPE]` | 任何整張 GPU | GPU 型號，見 [GPU 型號](#gpu-型號受限節點與共享-gpu) |
| `-w, --nodelist NODE` | — | 指定節點（Kubernetes 節點名稱，`ksinfo` 的 NODE 欄） |
| `-x, --exclude NODE[,NODE]` | — | 避開節點 |
| `-c, --cpus N` | `1` | CPU 核心數（requests = limits） |
| `--mem SIZE` | `2Gi` | 記憶體上限，Kubernetes 格式：`512Mi`、`32Gi`。超過會被 OOMKilled |
| `--shm SIZE` | 同 `--mem` | `/dev/shm` 上限。只是上限、不會預先占用，用量算在 `--mem` 裡；DataLoader worker 多時加大 `--mem` |
| `-t, --time TIME` | 無 | 時間上限，從工作排到節點開始算：包含下載 image、啟動容器的時間，不含排隊時間。時間到程式先收到 SIGTERM，有 30 秒存檔，之後強制結束。格式：`30`（分）、`30:00`（分:秒）、`2:00:00`、`1-12`（天-時）、`1-12:30:00`、`90m`、`12h`、`2d` |
| `--python PATH` | 目前的 python3 | 執行 `.py` 用的直譯器，例如 `/home/jovyan/venv/bin/python` |
| `--env KEY=VALUE` | — | 額外環境變數，可重複；`KJOB_NAME`、`KJOB_NODE`、`PYTHONUNBUFFERED`、`NVIDIA_VISIBLE_DEVICES`、`CUDA_MPS_*` 等保留變數不能覆蓋 |
| `-D, --chdir DIR` | 目前目錄 | 工作的工作目錄 |
| `-o, --log-dir DIR` | `~/kjob-logs` | 日誌存放處，必須在可寫入的共用磁碟 |
| `--ttl SECONDS` | `3600` | 工作結束且沒人監看時，工作物件保留多久才自動刪除 |
| `--live-progress` | 關 | tqdm 這類只用 `\r` 更新的進度條即時原地顯示。預設不開：Pod log 保持一般文字，`kubectl logs`、Kubeflow UI 可直接閱讀，但進度條可能要到換行才出現。開啟後 Pod log 改用 `[kjob-output:...]` 傳輸標記，請用 `ksrun`／`kwait`／`klogs` 看；`~/kjob-logs` 的日誌檔一律是原始輸出 |

腳本類型：`.py` 用 Python 執行，`.sh` 用 bash，其他檔案直接執行（需要執行權限）；也可以直接給 image 裡的指令，
例如 `ksrun -G 1 nvidia-smi`。腳本名稱之後的所有參數都交給腳本：`ksub -G 1 train.py -c config.yaml` 的 `-c` 是給
train.py 的。工作名稱會轉成小寫英數與 `-`，數字開頭時自動加 `j-`。

停止工作（`-t` 時間到、`kcancel`、`ksrun` 按 Ctrl-C）時，程式會收到 SIGTERM，有 30 秒可以存 checkpoint，之後被強制結束。
Python 可以用 `signal.signal(signal.SIGTERM, handler)` 接住它。SIGTERM 會同時送給程式開的子行程（例如 DataLoader
worker），所以主程式可能先看到 worker 結束的錯誤；存檔的 handler 要寫在主行程，並盡快存。

提交前的檢查（不通過就不會建立工作）：

* 腳本、工作目錄、日誌目錄必須在共用磁碟，日誌目錄必須可寫。
* 要求的 CPU／記憶體／GPU 超過 namespace 配額的**上限**時直接報錯（暫時被占滿則會排隊等候）。
* GPU 數、記憶體格式、環境變數名稱。

`ksrun` 的行為：

* 排隊時顯示排程訊息（例如 GPU 不足），開始後即時輸出，結束時印出 `Succeeded/Failed (exit code N)`。
* 結束代碼就是程式的結束代碼，可用在 shell 腳本：`ksrun ... && echo ok`。
  日誌寫入失敗則回傳 74（程式本身已失敗時保留原代碼），`kacct` 顯示 `LOG_ERROR`。
* `kacct` 查詢失敗時顯示 `UNCONFIRMED`；離線結果只依已存紀錄推算。
* 主容器退出即可完成；log stream 結束後仍無法確認結果，最多再等 30 秒便回報錯誤。用 `kwait NAME` 重接。
  這 30 秒不是訓練時限或 API timeout。
* Ctrl-C 會**取消**工作。連線中斷時工作不受影響，之後用 `kqueue`、`kwait` 查看。

## 腳本內設定：#SBATCH / #KJOB

在腳本開頭（第一個指令之前）的註解行設定選項，命令列選項優先：

```bash
#!/bin/bash
#KJOB -G 2 -g v100 --shm 16Gi          # kjob 的選項，原樣使用
#SBATCH --mem=64G -t 1-00:00:00         # 國網語法，轉換如下表
python3 train.py
```

| #SBATCH | 轉成 |
|---|---|
| `-J`、`--job-name` | `-J` |
| `-c`、`--cpus-per-task` | `-c` |
| `--mem=64G`（無單位 = MB） | `--mem 64Gi` |
| `-t`、`--time` | `-t` |
| `--gres=gpu:N`、`--gres=gpu:TYPE:N` | `-G N`（`-g TYPE`） |
| `--gpus=N`、`--gpus=TYPE:N`、`-G`、`--gpus-per-node` | `-G N`（`-g TYPE`） |
| `-w`、`--nodelist`、`-x`、`--exclude`、`-D`、`--chdir` | 同名選項 |
| `--time=UNLIMITED` | 不設時間上限 |
| `--mem=0`（國網的「整台記憶體」） | 忽略並顯示警告，請明確寫 `--mem` |
| 其他（`-p`、`-A`、`-o`、`-e`、`-N`、`-n`、`--ntasks`、`--mail-*` ...） | 忽略並顯示警告 |

短選項可以連寫（`-c4`、`-Jname`）。命令列的 `-G 0` 會一併取消腳本裡指定的 GPU 型號。

## 腳本範本與常見改法

新手範例（1 張與 2 張 GPU）見 [TRIAL_GUIDE.md](../TRIAL_GUIDE.md) 第 3、7 步。自己的訓練程式建議寫成 `.sh`，
資源寫在開頭（`#SBATCH` 是國網語法，`#KJOB` 寫 kjob 選項；`##SBATCH` 兩個 `#` 表示這行先不生效）：

```bash
#!/bin/bash
#SBATCH --gres=gpu:v100:2
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#KJOB --shm 16Gi
set -e
cd /home/jovyan/project
source /home/jovyan/venv/bin/activate          # 若使用自己的虛擬環境
torchrun --standalone --nproc_per_node=2 train.py --epochs 100
```

```bash
ksub train.sh                 # 依照檔案開頭的設定提交
ksub -G 1 train.sh            # 命令列的選項優先，這次只要 1 張
```

| 想要 | 在 `.sh` 開頭這樣寫（或提交時加在 `ksub` 後面） |
|---|---|
| 指定 GPU 型號 | `#SBATCH --gres=gpu:v100:2`（或 `ksub -g v100 ...`） |
| 幾種型號都可以 | `ksub -g v100,a5000 ...` |
| 指定節點 | `#SBATCH -w node-01`（或 `ksub -w node-01 ...`） |
| 換 GPU 張數 | `#SBATCH --gres=gpu:4`；`torchrun --nproc_per_node` 也要跟著改（TRIAL_GUIDE 的範例會自動偵測） |
| 只用 1 張 | `ksub -G 1 train.sh`（命令列優先於檔案開頭的設定） |
| 跑更久 | `#SBATCH -t 1-00:00:00`（1 天） |
| 用自己的虛擬環境 | 腳本裡 `source /home/jovyan/venv/bin/activate`。環境要建在 `/home/jovyan` 底下，例如 `python3 -m venv --system-site-packages /home/jovyan/venv`（沿用 image 的 PyTorch，再 `pip install` 自己要的套件） |
| 傳環境變數給程式 | `ksub --env WANDB_MODE=offline ...` |
| 暫時停用某行設定 | 開頭改成 `##SBATCH`（兩個 `#`） |

要點：

* 路徑用絕對路徑，程式、資料、輸出放在 `kinfo` 列出的磁碟上。
* 多張 GPU 要程式本身支援（DDP / `torchrun`），否則只會用到第一張；`--nproc_per_node` 要等於 GPU 數（可用 `NGPU=$(python3 -c "import torch; print(torch.cuda.device_count())")` 自動取得）。
* PyTorch DataLoader 的 worker 透過 `/dev/shm` 傳資料；它的上限預設跟 `--mem` 一樣，用量也算在 `--mem` 裡，worker 多時加大 `--mem`。
* 只在 Notebook 裝的套件：用 `pip install --user`（裝在 `~/.local`），或在 `/home/jovyan` 建虛擬環境。

## 查看與管理自己的工作

| 指令 | 說明 |
|---|---|
| `kqueue` | 自己 namespace 由 kjob 建立的工作：NAME、STATE、提交的 NOTEBOOK、POD 狀態、NODE、GPU、CPU、AGE、COMMAND |
| `kqueue --all` | 也列出 namespace 內不是 kjob 建立的 PyTorchJob |
| `kqueue -w` | 每 5 秒更新 |
| `kshow NAME` | 詳細資訊：提交時間、指令、工作目錄、日誌、資源、節點限制、時間上限、狀態條件、Pod、最近的警告事件。工作已刪除時改顯示歷史紀錄 |
| `klogs NAME` | 印出輸出；Pod 已不在時改讀日誌檔 |
| `klogs NAME -f` | 即時跟隨輸出；還在排隊時會先顯示排隊原因並等它開始；Ctrl-C 只停止觀看 |
| `klogs NAME -o DIR` | 日誌不在預設位置時指定目錄 |
| `kexec NAME 指令...` | 在執行中的工作容器裡跑一個指令並回傳它的結束代碼，例如 `kexec NAME nvidia-smi` 看 GPU 使用率（≈ 國網 `srun --jobid=ID`） |
| `kexec NAME` | 開一個互動 shell 進到執行中的工作（≈ `srun --jobid=ID --pty bash`），`exit` 離開，工作不受影響；需要在終端機裡執行 |
| `kwait NAME` | 等到結束（排隊中會顯示原因），回傳程式的結束代碼（`-t` 時間到被停止時回傳 124），結束後刪除工作物件 |
| `kwait NAME --keep` | 同上，但不刪除工作物件 |
| `kacct` | 最近 20 筆提交紀錄：提交時間、GPU、型號/節點、STATE（COMPLETED/FAILED/STOPPED/CANCELLED/RUNNING/PENDING/ENDED?）、EXIT、ELAPSED |
| `kacct -n 0` | 全部紀錄（`-n N` 最近 N 筆） |
| `kcancel NAME [NAME...]` | 取消／刪除工作 |
| `kcancel --finished` | 刪除這個 Notebook 提交、已結束的工作 |
| `kcancel --all-mine` | 刪除這個 Notebook 提交的**所有**工作（包含執行中） |
| `kcancel ... --all-notebooks` | 配合上兩項，包含同 namespace 其他 Notebook 提交的工作 |
| `kclean [--all-notebooks]` | 等於 `kcancel --finished` |
| `kinfo` | Notebook 名稱、namespace、image、工作目錄、會帶過去的磁碟、沒帶過去的磁碟與原因、配額使用量 |

`kacct` 的 `STOPPED` 表示被時間上限或 `kcancel` 停止（日誌有記到）；`CANCELLED` 表示在結束前被 `kcancel` 刪除、日誌沒來得及記到（例如剛啟動就取消）；`ENDED?` 表示工作已不存在、日誌也沒有結束代碼（例如被別處刪除，或日誌寫不進去）。
每次提交有自己的 run id，寫在日誌的 start 行，所以同名工作重交時，舊工作的結果不會算到新工作上。
同一個名稱重複使用時，依時間把日誌裡的每次執行對到它之前最近的那次提交；從未開始的提交顯示 `ENDED?`。
工作在別的節點寫日誌，Notebook 端的 NFS 快取可能要 30–60 秒才看到最新內容，剛結束的工作可能暫時顯示 `ENDED?`。

## 查看整個叢集：ksinfo / kqueue -A

Notebook 本身沒有讀取其他 namespace 的權限，這些資訊來自管理員啟用的叢集狀態服務（每 30 秒更新，畫面上會寫
`published Ns ago`）。管理員帳號則直接讀取即時資料（`live`）。

### ksinfo

```text
ksinfo              # 每種 GPU 型號一列，加上每個節點一列
ksinfo -g v100      # 只看某型號或節點，並列出在上面跑／在等它的工作
ksinfo -w           # 每 10 秒更新
```

| 欄位 | 意義 |
|---|---|
| GPUS / USED / FREE | 總數 / 已被申請 / 可用（節點故障或停用排程時不算可用） |
| WAITING_GPUS | 正在等這種 GPU 的工作要求的 GPU 總數與工作數 |
| GPU_MEM | 每張 GPU 的記憶體 |
| NOTE | `restricted`：受限節點；`(your namespace: allowed / no access)`：你的 namespace 能不能用 |
| CPU / MEM_GiB | 節點上已被申請 / 可分配的 CPU 和記憶體 |
| STATE | `up`、`down`（節點故障）、`drain`（停用排程） |

### kqueue -A

| 指令 | 說明 |
|---|---|
| `kqueue -A` | 叢集上所有使用 GPU 的工作與 Notebook，以及所有 PyTorchJob |
| `kqueue -A --state pending` | 只看排隊中的（`--state running` 只看執行中） |
| `kqueue -A -g rtxpro6000` | 只看某型號／節點 |
| `kqueue -u mxxxxxxx[,dxxxxxxx]` | 只看某些 namespace（使用者） |
| `kqueue -A -w` | 每 5 秒更新 |

欄位：USER（namespace，`*` 為自己的）、KIND（`Notebook`、`kjob`、`PyTorchJob`、`Pod`）、NAME、STATE、GPU、
GPU_TYPE（排隊中的顯示它要求的型號，沒指定則為 `any`）、NODE、TIME（執行或等待時間）、WAITING_FOR（排程器說明的等待原因）。
最後一行是執行中／排隊中的總數。

## GPU 型號、受限節點與共享 GPU

`-g` 可以用：

1. 簡稱（不分大小寫）：`v100`、`a5000`；有 A30 的叢集另有 `a30`（整張）、`a30-shared`（MPS 共享）。`kinfo` 只列出這個叢集有的簡稱。
2. `ksinfo` GPU_TYPE 欄的名稱，不分大小寫、可省略符號：`Tesla-V100-SXM2-16GB`、`rtx-a5000`。也可以只寫其中一段，
   但必須只對應到一種型號：`v100-sxm2`、`a5000`、`rtxpro6000`、`blackwell` 可以；`rtx` 同時符合 `NVIDIA-RTX-A5000`
   與 `NVIDIA-RTX-PRO-6000-Blackwell-Server-Edition` 會報錯，請寫清楚或用逗號列出。找不到的名稱直接報錯，不會提交。
3. 節點名稱（`ksinfo` NODE 欄）：等於 `-w 節點名稱`。節點名稱同時是它型號的一段時（例如 `rtxpro6000`）當成型號；
   沒有型號標籤的 GPU 節點以節點名稱當型號名稱，只能單獨使用。
4. 原始的 `nvidia.com/gpu.product` 標籤值（例如 `Tesla-V100-SXM2-16GB`）。
5. 逗號分隔多個型號：`-g v100,a5000`、`-g v100,rtxpro6000` 表示任一種都可以。

第 2、3 種需要叢集狀態服務（`ksinfo` 能用時就可以）。`ksinfo` 能用時，所有名稱都會先檢查：叢集裡沒有的型號、
`-w` 指定的不是 GPU 節點、或 `-g` 與 `-w` 互相矛盾（例如 `-g a5000 -w node-01`）都會直接報錯，不會提交一個永遠
排不到的工作。`-G 0` 不能搭配 `-g`。

不指定 `-g` 時，排程器會挑任何一張空的整張 GPU（自動避開 MPS 共享節點）。

**受限節點**（`ksinfo` 顯示 `restricted`）：只有管理員開通的 namespace 能使用，開通後工作會自動帶上權限，直接用
`-g` 或 `-w` 指定即可。沒開通的 namespace 提交後會一直排隊，原因是 `untolerated taint`。

**MPS 共享 GPU**（`-g a30-shared`，只在有設定 MPS 的叢集）：多個工作分用一張 GPU。只能 `-G 1`，不能與整張 GPU
型號混用，目前只允許 root（UID 0）Notebook 使用，且 MPS 服務需由管理員確認正常。一般建議使用整張 GPU。

**配額**：namespace 配額（`kinfo` 的 quota）計算所有 Pod，包括 Notebook 本身占的 GPU 與排隊中的工作。
GPU 型號不同但共用同一個 `nvidia.com/gpu` 配額。

## 排隊與錯誤排解

排隊原因看 `kshow NAME`、`kwait` 的訊息或 `kqueue -A` 的 WAITING_FOR：

| 訊息 | 意義 | 處理 |
|---|---|---|
| `Insufficient nvidia.com/gpu` | 該型號沒有空的 GPU | 等候，或換 `ksinfo` 中有空位的型號 |
| `didn't match Pod's node affinity/selector` | `-g`/`-w` 組合找不到符合的節點 | 確認 `ksinfo` 中的名稱 |
| `untolerated taint` | 受限節點，沒有權限 | 改用其他型號，或請管理員開通 |
| `exceeded quota` | namespace 配額已滿 | 關閉閒置的 GPU Notebook、取消其他工作，或等候 |
| `Insufficient cpu` / `memory` | 節點 CPU／記憶體不夠 | 降低 `-c`、`--mem` |
| `ErrImagePull` | image 拉不下來 | 回報管理員 |

執行失敗：

| 現象 | 原因／處理 |
|---|---|
| `exit code 137, OOMKilled` | 記憶體超過 `--mem`（只有 137 而沒有 OOMKilled：被強制結束，例如停止後 30 秒內沒結束） |
| `Bus error`、`DataLoader worker ... killed`、`exited unexpectedly` | `/dev/shm` 不夠：加大 `--mem`（`/dev/shm` 上限預設同 `--mem`）；若有指定 `--shm` 也一起加大 |
| `CUDA out of memory` | 降低 batch size，或換記憶體大的 GPU |
| `No such file or directory` | 檔案不在共用磁碟、或用了相對路徑 |
| `ModuleNotFoundError` | 套件只裝在 Notebook 系統目錄；改用 `pip install --user` 或 `/home/jovyan` 上的虛擬環境 + `--python` |
| `stopped at the time limit`、`DeadlineExceeded`（ksrun／kwait／kshow 顯示） | 超過 `-t` 時間上限；程式有收到 SIGTERM，可在 handler 存 checkpoint 後續跑。即使程式存完檔正常結束，`ksrun`／`kwait` 也回傳 124，不當成成功 |
| `cannot write the log file ...; the script was not started` | 日誌目錄不存在、被同名檔案擋住或沒有寫入權限，程式沒有執行。檢查 `-o` 指定的目錄（預設 `~/kjob-logs`） |
| `log write failed`、`LOG_ERROR` | 日誌未存完整，Pod 會保留。先用 `klogs NAME > /可寫的共用磁碟/NAME.log` 取回，再 `kcancel NAME`。請在 `--ttl` 到期或執行 `kclean` 前取回。 |
| `log stream ended but completion could not be confirmed` | 監看結束但沒有可靠終態，工作沒有被取消。用 `kshow NAME` 看狀態或 `kwait NAME` 重新接上。 |
| `cuda: False` | 用了 `-G 0`，或 image／環境沒有 CUDA 版 PyTorch |
| `kjob: Kubernetes API error (403 Forbidden): ... cannot create resource "pytorchjobs" ...` | Notebook 的帳號沒有這個權限。訊息裡會寫是哪個帳號、缺什麼。Kubeflow 介面建立的 Notebook 用 `default-editor`，權限是齊的；不是的話重新從介面建 Notebook，否則請管理員處理（[ADMIN.md](ADMIN.md) 有權限清單） |
| `kjob: Kubernetes API error (403 Forbidden): exceeded quota` | namespace 配額用完（Notebook 也算在內）：看 `kinfo`，等其他工作結束或少要一點 |
| `has no access to the requested GPUs on ...` | 指定的是受限 GPU，這個 namespace 沒開通（`ksinfo` 的 NOTE 是 `no access`）：換型號或請管理員開通 |
| `kjob: cannot reach the Kubernetes API` | 連不到叢集。之前已送出的工作不受影響（用 `kqueue` 確認）；在 AI 工具的 sandbox 裡執行時，多半是 sandbox 擋住網路 |
| `kjob: unexpected error ...` | kjob 的問題：用 `KJOB_DEBUG=1 ksub ...` 重跑取得完整訊息並回報 |
| `no kernel image is available`、`sm_120 is not compatible` | PyTorch 不支援這張卡。V100 和 RTX PRO 6000 都要能跑（例如 `-g v100,rtxpro6000`）請用 `torch==2.8.0`（`--index-url https://download.pytorch.org/whl/cu128`，Python 3.9 以上的 venv）：2.7 以前不支援 RTX PRO 6000，較新的版本（實測 2.11）不支援 V100，不指定版本就會裝到它 |

## AI 助手 skill

| Skill | 內容 |
|---|---|
| [kjob-install](../kjob/skills/kjob-install/SKILL.md) | 在 Notebook 安裝、驗證、更新 kjob，常見安裝問題 |
| [kjob-jobs](../kjob/skills/kjob-jobs/SKILL.md) | 選 GPU、準備腳本、提交、監看、排錯；規定 AI 只能動使用者自己的工作 |

第一次執行任何 kjob 指令時，skill 會自動安裝，不論 AI 工具裝了沒：

| 位置 | 內容 | 誰會讀 |
|---|---|---|
| `~/.agents/skills` | skill 檔案本身 | Codex、OpenCode、Gemini CLI、Cursor 等支援 Agent Skills 標準的工具 |
| `~/.claude/skills` | 指向上面的連結（symlink），不是第二份 | Claude Code（它只讀這裡） |

升級 kjob 後也會自動更新，你改過的 skill 檔不會被覆蓋。設定 `KJOB_NO_SKILLS=1` 可關閉。

```bash
kjob install-skills                       # 重新安裝／強制覆蓋（同上面的位置與連結）
kjob install-skills --dir 其他工具的skills目錄
```

不支援 skill 的 AI 工具，可以把這兩個 `SKILL.md` 的內容貼給它當說明。

### 用 AI 執行時常見的問題

AI 工具（Claude Code、Codex 等）執行指令前有自己的權限機制。被它擋下時，**kjob 根本沒有執行**，跟 kjob 或叢集無關。
skill 會要求 AI 明確告訴你是不是這種情況。

| 看到的狀況 | 原因 | 處理 |
|---|---|---|
| 沒有任何 `kjob:` 開頭的輸出，AI 說指令被拒絕、需要核准、`denied`、`blocked`，或 Claude Code 的 `denied by the auto mode classifier` | AI 工具的權限機制擋下了指令 | 按下核准，或把 kjob 指令加入允許清單（下面） |
| `kjob: cannot reach the Kubernetes API`，而 AI 在 sandbox 裡執行 | sandbox 擋住了連到叢集的網路 | 讓 kjob 指令在 sandbox 外執行，或在 sandbox 設定允許連線 |
| `kjob: Kubernetes API error (403 ...)` | kjob 有執行，是 Notebook 的帳號沒有權限 | 見下方錯誤排解表 |

哪些指令會改動叢集：

| 只讀取（可以放心允許） | 會建立、刪除或進入工作 |
|---|---|
| `kinfo`、`ksinfo`、`kqueue`、`kshow`、`klogs`、`kacct` | `ksub`、`ksrun`（建立工作；`ksrun` 結束時刪除）、`kwait`（結束時刪除工作）、`kexec`（在工作裡執行指令）、`kcancel`、`kclean`（刪除工作） |

Claude Code 可以在 `/permissions` 設定，或寫進 `~/.claude/settings.json`（`kcancel`、`kclean` 不放進去，刪除時仍會先問）：

```json
{"permissions": {"allow": ["Bash(kinfo:*)", "Bash(ksinfo:*)", "Bash(kqueue:*)", "Bash(kshow:*)", "Bash(klogs:*)",
                           "Bash(kacct:*)", "Bash(ksub:*)", "Bash(ksrun:*)", "Bash(kwait:*)", "Bash(kexec:*)"]}}
```

其他 AI 工具請用它自己的核准設定。

## Python API

```python
from kjob import JobSpec, KJob

job = KJob(JobSpec(name="probe-1", script="/home/jovyan/linear_probe.py", args=["--lr", "0.01"],
                   gpus=1, cpus=4, memory="24Gi", shm="8Gi", gpu_types=["v100"], time_limit=7200,
                   interpreter="/home/jovyan/venv/bin/python"))
code = job.run()          # 提交、即時輸出、刪除；回傳結束代碼
# 或 job.submit(); ...; job.follow(); job.delete()
```

`JobSpec` 欄位：`name`、`script`、`args`、`gpus`、`cpus`、`memory`、`shm`、`gpu_types`、`node`、`exclude_nodes`、
`interpreter`、`env`、`workdir`、`log_dir`、`ttl_seconds`、`time_limit`。

## 檔案與環境變數

| 位置 | 內容 |
|---|---|
| `~/kjob-logs/NAME.log` | 每個工作的完整輸出，第一行是開始時間與節點，最後一行是結束代碼 |
| `~/.kjob/history.jsonl` | 提交紀錄（`kacct`、`kshow` 使用） |

工作內可用的環境變數：`KJOB_NAME`（工作名稱）、`KJOB_NODE`（執行的節點）、`PYTHONUNBUFFERED=1`。
GPU 工作的 `CUDA_VISIBLE_DEVICES` 由系統設定；`-G 0` 的工作看不到 GPU。

## 限制

* 沿用 Notebook 的 image，不能另外指定；只裝在 Notebook 系統目錄的套件工作看不到。
* 單一 Pod：一個工作就是一台節點上的一個容器，多 GPU 請在腳本內用 `torchrun`，不支援跨節點分散式訓練。
* 不支援 job array、`--dependency`、`sattach`、email 通知。
* 排隊由 Kubernetes 排程器處理，不是嚴格先到先得，也沒有公平分享（fair share）。
* kjob 的檢查是方便使用者的提示，不取代叢集本身的權限與配額控制。

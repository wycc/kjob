# kjob

在 Kubeflow Notebook 裡用一行指令把程式交給叢集的 GPU 跑，指令仿照國網（Slurm）。不用寫 YAML，也不用設定環境：

* **環境自動複製**：工作使用和 Notebook 相同的 image、工作目錄、共用磁碟、環境變數與使用者身分，程式、資料、
  虛擬環境在工作裡都是同樣的路徑。
* **GPU 自動分配**：不指定型號就排到任何有空的整張 GPU，沒空就排隊、有空自動開始；Notebook 本身不用占 GPU。
* **自動收尾**：輸出即時顯示並存到 `~/kjob-logs/工作名稱.log`；`ksrun` 結束就刪除工作，`ksub` 在 `--ttl` 秒後
  自動刪除；`-t` 時間到先送 SIGTERM，程式有 30 秒存檔。
  日誌保存失敗時，`ksrun`／`kwait` 會回報錯誤並保留已結束的 Pod，可用 `klogs` 取回輸出；原有 `--ttl` 仍適用。
  Pod log 是一般文字，`kubectl logs`、Kubeflow UI 都能直接看；tqdm 這類只用 `\r` 更新的進度條可能要到換行才出現，
  要即時原地更新請加 `--live-progress`（這時 Pod log 改用傳輸標記，只有 kjob 讀得好看；保存的日誌檔不受影響）。

## 安裝（在 Notebook 的 Terminal）

```bash
python3 -m pip install --user --upgrade https://github.com/MarkJiang-maji/kjob/archive/refs/heads/main.tar.gz
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
kinfo      # 顯示工作會繼承的 image、磁碟、GPU 配額
```

## 範例

```bash
# 交到任何有空的 GPU，即時看輸出，結束自動刪除
ksrun -J ft -c 4 --mem 24Gi train.py --epochs 10

# V100 或 RTX PRO 6000 都可以，4 張卡 DDP，背景執行，最多 1 天
ksub -J pretrain -g v100,rtxpro6000 -G 4 -c 16 --mem 64Gi --shm 16Gi -t 1-00:00:00 \
     torchrun --standalone --nproc_per_node=4 pretrain.py --epochs 30
kqueue -w                 # 看自己的工作，每 5 秒更新
kwait pretrain            # 等它結束，回傳結束代碼

# 用自己的虛擬環境
ksub -g v100 --python /home/jovyan/venv-cu128/bin/python train.py

# 看叢集與紀錄
ksinfo                    # 每種 GPU 有幾張、空幾張、多少人在排隊
kqueue -A                 # 全叢集誰在用、誰在等、等什麼
kacct                     # 自己提交過的工作、結束代碼、執行時間
```

## 指令

| 國網 / Slurm | kjob | 用途 |
|---|---|---|
| `srun` | `ksrun [選項] 腳本 [參數]` | 提交並即時顯示輸出，結束後刪除工作，回傳腳本的結束代碼（時間到為 124）；Ctrl-C 取消 |
| `sbatch` | `ksub [選項] 腳本 [參數]` | 提交後立即返回 |
| `sbatch --wait` | `kwait 名稱` | 接上已提交的工作，等它結束並回傳結束代碼 |
| `squeue` | `kqueue [-w] [-A]` | 自己的工作；`-A` 全叢集 |
| `sinfo` | `ksinfo [-g 型號]` | 每種 GPU、每個節點的使用與排隊 |
| `scontrol show job` | `kshow 名稱` | 單一工作的資源、狀態、事件 |
| `tail -f` | `klogs 名稱 [-f]` | 輸出；工作刪除後改讀日誌檔 |
| `srun --jobid --pty` | `kexec 名稱 [指令]` | 在執行中的工作裡跑指令或開 shell |
| `sacct` | `kacct` | 歷史紀錄與結束代碼 |
| `scancel` | `kcancel 名稱...` | 取消工作 |
| — | `kclean` | 刪除已結束的工作（日誌保留） |
| — | `kinfo` | 工作會繼承的 image、磁碟、配額 |

提交選項（`ksub` / `ksrun`）：

| 選項 | 預設 | 意思 |
|---|---|---|
| `-J, --job-name` | 腳本名 + 隨機字尾 | 工作名稱 |
| `-G, --gpus` | 1 | GPU 數；`0` 只用 CPU |
| `-g, --gpu-type` | 任何有空的 | `v100`、`a5000`、`rtxpro6000`…，逗號表示任一種都可以 |
| `-w, --nodelist` / `-x, --exclude` | — | 指定／避開節點 |
| `-c, --cpus` / `--mem` / `--shm` | 1 / 2Gi / 同 `--mem` | CPU、記憶體、`/dev/shm` 上限（用量算在記憶體內） |
| `-t, --time` | 無 | 時間上限，從排到節點開始算（含下載 image，不含排隊），例如 `2:00:00`、`1-00:00:00`、`12h` |
| `--python` | 目前的 python3 | 執行 `.py` 的直譯器 |
| `--env KEY=VALUE` | — | 額外環境變數，可重複 |
| `-D, --chdir` / `-o, --log-dir` | 目前目錄 / `~/kjob-logs` | 工作目錄／日誌位置 |
| `--ttl` | 3600 | 結束且沒人監看的工作保留幾秒 |
| `--live-progress` | 關 | 進度條（`\r` 更新）即時原地顯示；Pod log 改用傳輸標記，`kubectl logs`／Kubeflow UI 會看到標記格式 |

`.py` 用 Python 執行、`.sh` 用 bash，其他直接執行。選項也可以寫在腳本開頭的 `#SBATCH` 或 `#KJOB` 行。

## 工作從 Notebook 帶過去什麼

* image、工作目錄、使用者／群組、環境變數（`env`、`envFrom`）、`runtimeClassName`、Notebook 本身的 toleration。
* Notebook 主容器掛載的磁碟；NFS 上的 ReadWriteOnce 工作磁碟改用 NFS 直接掛載，所以工作可以跑在別台節點。
* namespace 的 PodDefault（群組資料夾、受限 GPU 節點的權限等）會再套用到工作上。
* 不會帶：`emptyDir`（例如 `/dev/shm`，在工作裡是新的）、Notebook 的 `/tmp`、service account token。`kinfo` 列出實際內容。

## AI 助手

第一次執行 kjob 指令時會自動安裝 AI skill（Claude Code、Codex 等），之後直接說「用 kjob 把 train.py 交到有空的 GPU 上跑」即可。
AI 工具擋下指令時 kjob 並沒有執行，要允許哪些指令見 [USER_GUIDE](docs/USER_GUIDE.md#用-ai-執行時常見的問題)。

## 文件與限制

[TRIAL_GUIDE.md](TRIAL_GUIDE.md) 新手教學、[docs/USER_GUIDE.md](docs/USER_GUIDE.md) 完整選項與排錯、
[docs/ADMIN.md](docs/ADMIN.md) 管理員設定、[docs/kjob-tutorial.pptx](docs/kjob-tutorial.pptx) 簡報。

* 不能另外指定 image；只在 Notebook 系統目錄 `pip install` 的套件，工作看不到（用 `--user` 或 `/home/jovyan` 下的 venv）。
* 一個工作在一台節點上，多卡用 `torchrun`；不支援跨節點、job array、`--dependency`。
* 排隊由 Kubernetes 排程器處理，不是嚴格先到先得；GPU 配額包含 Notebook 本身占用的 GPU。

# kjob 管理員說明

給叢集管理員：讓使用者能在 Notebook 裡使用 kjob 需要什麼，以及選用的叢集狀態服務。kjob 是使用者自行安裝的
Python 套件，**不需要**修改 Notebook image、training operator 或節點。

## 1. 叢集需求

| 項目 | 需求 |
|---|---|
| Kubeflow Training Operator | 已安裝 PyTorchJob CRD（`kubectl get crd pytorchjobs.kubeflow.org`）。已在 `kubeflow/training-operator:v1-855e096` 驗證 |
| Notebook 的 service account | 在自己 namespace 內：讀自己的 Pod、PodDefault、ResourceQuota、Event、Pod log；建立／讀取／刪除 PyTorchJob；`kexec` 另需 `pods/exec`。Kubeflow profile 預設的 `default-editor` 已具備 |
| 工作磁碟 | 使用者的 workspace 與資料要在 NFS 或 RWX 儲存上（工作可能跑在別的節點）。底層是 NFS 的 RWO PVC 由 kjob 自動改成 NFS 掛載；一般 RWO 區塊儲存不適用 |
| GPU 標籤 | GPU 節點最好有 NVIDIA GPU Feature Discovery 的 `nvidia.com/gpu.product` 標籤，使用者才能用 `-g` 選型號 |
| 網路 | 使用者安裝時需要能連 GitHub 與 PyPI；離線環境請改發 wheel（見第 4 節） |

kjob 以 Notebook Pod 的 service account 呼叫 API（不用使用者本人的登入帳號，也不需要 root）。從 Kubeflow 介面建立的
Notebook 都用 `default-editor`；用 `default` 的 Notebook 沒有任何權限，kjob 會在第一個指令就出現 403。

各指令需要的權限（都在自己的 namespace 內，kjob 不用 watch）：

| 指令 | 需要 |
|---|---|
| 全部 | `get pods`（讀自己的 Notebook Pod） |
| `kinfo` | `list poddefaults.kubeflow.org`、`list resourcequotas` |
| `ksub`、`ksrun` | 上一列，加上 `create pytorchjobs.kubeflow.org`；`ksrun` 另需下面 `kwait` 的權限 |
| `kqueue`、`kacct` | `list pytorchjobs.kubeflow.org`、`list pods` |
| `kshow` | `get pytorchjobs.kubeflow.org`、`list events` |
| `klogs` | `get pods/log`、`get pytorchjobs.kubeflow.org` |
| `kwait` | `get pytorchjobs.kubeflow.org`、`get pods/log`、`delete pytorchjobs.kubeflow.org` |
| `kexec` | `create pods/exec` |
| `kcancel`、`kclean` | `list`、`delete pytorchjobs.kubeflow.org` |
| `ksinfo`、`kqueue -A` | `get configmaps`（`kjob-system/kjob-cluster-status`，由第 2 節的服務開放） |

檢查某個 namespace 的權限（全部是 `yes` 才完整）：

```bash
NS=<namespace>; SA=system:serviceaccount:$NS:default-editor
for r in "get pods" "list pods" "get pods/log" "create pods/exec" "list events" "list resourcequotas" \
         "list poddefaults.kubeflow.org" "create pytorchjobs.kubeflow.org" "get pytorchjobs.kubeflow.org" \
         "list pytorchjobs.kubeflow.org" "delete pytorchjobs.kubeflow.org"; do
  echo "$r: $(kubectl auth can-i $r -n $NS --as=$SA)"
done
echo "status ConfigMap: $(kubectl auth can-i get configmaps/kjob-cluster-status -n kjob-system --as=$SA)"
kubectl get notebooks -n $NS -o custom-columns=NAME:.metadata.name,SA:.spec.template.spec.serviceAccountName
```

受限 GPU 節點不是 RBAC 權限，而是 taint／toleration（見第 3 節）；沒開通時 `ksub` 會在提交前直接報錯。

## 2. 叢集狀態服務（選用，讓使用者看到整個叢集）

`ksinfo` 與 `kqueue -A` 需要列出所有節點與所有 namespace 的 Pod，一般 Notebook 沒有這個權限。這個服務用一個小
Deployment 每 30 秒把摘要寫進 ConfigMap `kjob-system/kjob-cluster-status`，並只開放所有 service account（Notebook）讀取這一個 ConfigMap。
沒有安裝時，提交工作不受影響，只是這兩個指令會提示無法使用。

```bash
kubectl apply -f deploy/cluster-status.yaml
kubectl -n kjob-system logs deploy/kjob-status        # 看到 "published N GPU nodes, M workloads" 即可
kubectl delete -f deploy/cluster-status.yaml           # 完整移除
```

* 新增的物件：namespace `kjob-system`、ServiceAccount、只讀 nodes/pods 的 ClusterRole、只能寫該 ConfigMap 的 Role、
  讓 `system:serviceaccounts` 讀該 ConfigMap 的 Role、該 ConfigMap 本身，以及一個 Deployment（`python:3.12-slim`，
  requests 20m CPU / 96Mi，UID 65534、不可提權、drop ALL capabilities、RuntimeDefault seccomp）。不修改任何既有物件。
* 摘要內容：每個 GPU 節點的型號、GPU/CPU/記憶體總量與已申請量、taint；每個 GPU 工作的 namespace、名稱、GPU 數、
  節點、狀態、等待原因。不含環境變數、指令、磁碟等內容。
* 隱私：namespace 通常是學號／員工編號，所有 Notebook 都看得到。在 Deployment 的指令加上 `--hide-names`，
  就不公開工作／Notebook 名稱，排隊原因中引號內的名稱（例如其他人的 PVC 名稱）也會遮掉。
* 安全：收集者可以讀全叢集的 Pod spec（含環境變數），所以 `KJOB_SOURCE` 固定在一個審過的 commit
  （`archive/<sha>.tar.gz`），升級時刻意改 SHA；也可以改用預裝 kjob 的 image。該 commit 必須已推上 GitHub 且
  不會消失：PR 若以 squash／rebase 合併，合併後要改成 main 上的新 SHA。依賴套件固定為 `kubernetes==32.0.1` wheel。
* 管理員在主機上直接執行 `ksinfo`、`kqueue -A`（有 cluster 讀取權限的 kubeconfig）會讀即時資料，不經過這個服務。

## 3. GPU 型號與受限節點

* `-g` 依 `nvidia.com/gpu.product` 標籤選節點。沒有此標籤的 GPU 節點，在 `ksinfo` 以節點名稱顯示，使用者可用
  `-g 節點名稱` 或 `-w 節點名稱`。建議在該節點部署 GPU Feature Discovery，讓型號正確顯示。
* 受限節點用 taint 保護（例如 `example.com/restricted=true:NoSchedule`），再以 namespace 的 PodDefault 加上 toleration 開通。
  kjob 會把 Notebook 符合的 PodDefault selector 標籤帶到工作上，selector 為空的 PodDefault 也會套用到工作，
  所以已開通的 namespace 不需另外設定。Notebook 本身設定的 toleration 也會帶到工作；kjob 不會自行加 toleration。
  沒開通的 namespace 指定受限型號時，`ksub` 會在提交前報錯（不會留下一個永遠排隊的工作）。
* 簡稱表在 `kjob/core.py` 的 `GPU_PRODUCTS`；新增型號時可加一行，或讓使用者直接用 `ksinfo` 顯示的名稱。
* MPS 共享 GPU：產品標籤以 `-SHARED` 結尾且 `nvidia.com/gpu.sharing-strategy=mps` 的節點。整張 GPU 的工作會自動
  避開這類節點；共享工作不帶 `/dev/shm` volume（device plugin 會掛）。MPS server 的 UID 必須與工作相容，
  目前 kjob 只允許 UID 0 的 Notebook 提交共享工作。

## 4. 發布方式

| 方式 | 做法 |
|---|---|
| GitHub（預設） | 使用者執行 README 的安裝指令，裝在自己的 `~/.local` |
| wheel | `python3 -m pip wheel --no-deps -w dist .`，把 `dist/kjob-0.5.0-py3-none-any.whl` 給使用者，`pip install --user` 安裝 |
| 預裝在 Notebook image | `deploy/Dockerfile`：`docker build --build-arg BASE_IMAGE=<image> --build-arg NOTEBOOK_USER=<原本的 USER> -f deploy/Dockerfile dist/` |

## 5. 會建立的 Kubernetes 物件

每個工作是一個 PyTorchJob（一個 Master replica、`restartPolicy: Never`），標籤
`app.kubernetes.io/managed-by=kjob`、`kjob/notebook=<Notebook Pod>`，annotation 記錄指令、工作目錄、日誌路徑。
`runPolicy`：`cleanPodPolicy: None`、`ttlSecondsAfterFinished`（預設 3600）。`-t` 設在 Pod 的
`activeDeadlineSeconds`，由 kubelet 從 Pod 開始執行起計時（排隊時間不算）；`runPolicy.activeDeadlineSeconds`
只在 operator 重新協調時才檢查，實測不會準時停止，所以不使用。不設定 `backoffLimit`：舊版 operator 搭配 `backoffLimit: 0` 會在記錄失敗前刪除並重建 Pod，
導致腳本跑兩次。工作 Pod 不掛 service account token、不注入 Istio sidecar。

列出／清除所有 kjob 工作：

```bash
kubectl get pytorchjobs -A -l app.kubernetes.io/managed-by=kjob
kubectl delete pytorchjobs -n <namespace> -l app.kubernetes.io/managed-by=kjob
```

## 6. 尚未處理的平台事項

kjob 的提交前檢查只是給使用者的提示，不能取代伺服器端的控制。以下需要在平台層處理：依 GPU 型號分配額、
公平排隊（例如 Kueue）、限制工作可掛載的 NFS 路徑的 admission policy。

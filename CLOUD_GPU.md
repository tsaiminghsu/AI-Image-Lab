# 雲端 GPU 參考（Replicate / RunPod）

本地這張 RTX 2070 8GB 目前撐得住文字生圖、圖生圖（見 README），撐不住的是：
影片生成（`video` / `video-animatediff`，VRAM 常態吃緊）、以及 kohya_ss LoRA
訓練（跟 ComfyUI 搶同一張卡，見 README「疑難排解」章節）。這份文件記錄
**什麼時候該往雲端搬、搬去哪裡、怎麼搬**——不管是本地顯卡先天不夠力，還是
以後升級了顯卡但某個工作量還是划不來自己養一張卡，都可以回來對照這份文件
重新估算。

## 先看結論：什麼情況用哪個

| 情境 | 建議 | 為什麼 |
|---|---|---|
| 文字生圖、圖生圖 | **繼續用本地** | RTX 2070 8GB 已經跑得動，成本 $0，沒有理由搬走 |
| 影片生成，零星測試（一週幾次、驗證效果用） | **Replicate** | 按次/按秒計費，不用管開關機、不會有「忘記關閉導致空燒錢」的風險 |
| 影片生成，量變大、變成常態批次產出 | **RunPod** | 整台 GPU 用小時計費，量大時比 Replicate 一次次疊加的按次費用便宜 |
| kohya_ss LoRA 訓練 | **RunPod**（A100 等大 VRAM 選項） | 訓練需要接近滿版 VRAM，本地會跟 ComfyUI 搶卡；Replicate 是「跑別人包好的模型」，不適合你自己的訓練腳本 |
| 想把現有這整套 ComfyUI + custom node（IPAdapter FaceID/ControlNet/Impact-Pack/AnimateDiff-Evolved）原封不動搬上雲 | **RunPod** | RunPod 給的是原始 GPU 主機，可以照 README 的安裝流程整套搬過去；Replicate 是單一模型的 API 平台，你的多 workflow 自訂 pipeline 沒辦法直接塞進去，得重新包裝 |

> 兩邊都是「用多少付多少」，沒有月費/訂閱綁定——但計費模型完全不同：
> Replicate 是「這次呼叫跑了幾秒 GPU 時間」，RunPod 是「這台 pod 開了幾小時」。
> 這也是上面「零星測試選 Replicate、常態批次選 RunPod」判斷的核心依據：
> pod 開著不用一樣在燒錢，呼叫次數少時反而是浪費。

## Replicate

### 是什麼

跑別人（或自己）包裝好的模型，用 API 呼叫，按實際運算時間計費——不用自己
管伺服器、不用自己下載模型權重。適合「用一下試效果」，不適合「跑我自己這套
用一堆 custom node 拼出來的 ComfyUI workflow」。

### 計費（2026 查證）

- 大部分公開模型走**兩種計費模式**其中一種：
  - **按 GPU 秒數**：依硬體分級，從 CPU 的 $0.000025/秒 到 8x H100 的
    $0.012200/秒，模型頁面會標明用的是哪一級硬體
  - **按輸出計價**（熱門模型常見）：例如圖片模型直接寫 $0.025/張，不用自己
    算秒數
- **沒有月費**，帳號儲值後照實際用量扣款
- 影片模型通常跑比較久（10 秒以上），送出去之後用 polling 或 webhook 拿
  結果，不要傻等在原地——官方文件建議超過幾秒的工作都走非同步模式

### 怎麼用

1. 註冊帳號，到 <https://replicate.com/account/api-tokens> 產生 API token
2. 裝 client、設定 token：

```powershell
pip install replicate
$env:REPLICATE_API_TOKEN = "r8_你的token"
```

3. 最小範例（官方文件的圖片生成範例，影片模型用法相同，差別只在
   model 名稱和 `input` 參數）：

```python
import replicate

output = replicate.run(
    "owner/model-name",   # 去 replicate.com/explore 找適合的影片模型，
                           # 版本會一直更新，不要寫死在筆記裡，用的時候上網站確認
    input={"prompt": "..."},
)

with open("output.mp4", "wb") as f:
    f.write(output[0].read())
```

> **找影片模型**：上 <https://replicate.com/explore> 搜尋 "image to video" 或
> "text to video"，每個模型頁面會列清楚的 `input` 參數、單價、範例程式碼。
> 這裡刻意不寫死特定 model slug——Replicate 上的影片模型汰換速度快，寫死了
> 半年後可能就不是最好的選擇，用的當下上網站選就好。

### 跟這個專案的關係

Replicate 上的模型是別人包好的固定 pipeline，跟 `comfyui_client.py` 那套
「送 workflow JSON 給自己的 ComfyUI server」完全是两條路——**不會共用同一套
程式碼**，是獨立的呼叫方式。真的要接的話，會是新寫一個類似
`image_api.py` 的薄 wrapper，而不是把 `comfyui_client.py` 的邏輯搬過去用。

## RunPod

### 是什麼

租一整台掛了 GPU 的 Linux 主機（Pod），要跑什麼自己裝——概念上跟本地那張
RTX 2070 一樣，只是換一台機器、用小時計費。RunPod 官方就有現成的 ComfyUI
template，可以照 README 現有的安裝流程幾乎原封不動搬過去。

### 計費（2026 查證）

| 項目 | Community Cloud | Secure Cloud |
|---|---|---|
| RTX 4090 | ~$0.34/hr | ~$0.69/hr |
| L40 | ~$0.86/hr | 較高 |
| A100 PCIe 80GB | ~$1.39–1.64/hr | 較高 |
| H100 PCIe | ~$1.99/hr | ~$2.89/hr |

- **Community Cloud**：便宜，但沒有 uptime 保證（別人的機器，可能被收回）——
  適合「開機跑一批、跑完就關」這種批次工作，不適合長期常駐服務
- **Secure Cloud**：貴一截，換穩定的資料中心硬體——要長時間掛著跑（例如常態
  對外提供服務）才需要
- **Serverless**：按 worker 使用量計費（約 $0.58/hr 起，依 GPU 等級），適合
  「有請求才啟動、沒請求不計費」的場景，但有冷啟動延遲
- **儲存**：Network Volume 約 $0.05–0.07/GB/月，Volume Disk 開著用 $0.10/GB、
  **閒置也要 $0.20/GB**——不用的網路硬碟記得清掉

> **最容易被坑的地方**：pod 沒手動停止/刪除，就算你人已經離開、沒在用，
> 一樣照小時扣錢。每次工作階段結束記得回控制台確認 pod 狀態，不要只是關掉
> 瀏覽器分頁。

### 怎麼用

1. 註冊帳號並儲值（沒有訂閱制，儲值餘額扣款）
2. 進 RunPod 控制台，搜尋官方 **ComfyUI** template（Blackwell 架構的
   RTX 5090/B200 要選 "ComfyUI Blackwell Edition" 版本）
3. 選 GPU：SDXL 這個量級建議 RTX 4090 或 L40；要跑 kohya_ss 訓練或大量
   IP-Adapter+ControlNet+FaceDetailer 疊在一起的重 workflow，選 A100 80GB
4. **強烈建議掛一個 Network Volume**——模型、custom node、workflow 輸出都會
   持久化保存，不然每次重開 pod 都要重新下載 6.6GB 的 Juggernaut checkpoint
   之類的大檔案
5. Port 8188（ComfyUI 網頁/API）官方 template 預設就會開好，部署完成後（第
   一次啟動可能要等到 30 分鐘）在控制台點 Pod 的「Connect」→ 選 port 8188 的
   HTTP service，網址格式是：

   ```
   https://<POD_ID>-8188.proxy.runpod.net
   ```

   看到 "Not Ready" 或 Bad Gateway 是還在啟動，等 2-3 分鐘刷新即可
6. custom node（IPAdapter/ControlNet-aux/Impact-Pack/AnimateDiff-Evolved）可
   以用內建的 ComfyUI Manager 裝，或直接照 README「安裝」章節的 `git clone`
   步驟手動裝——跟本地流程一致

### 跟這個專案的關係

`comfyui_client.py` 目前把 ComfyUI 位址寫死在檔案開頭：

```python
COMFYUI_URL = "http://127.0.0.1:8188"
```

要接 RunPod 的 pod，把這行改成上面那個 `https://<POD_ID>-8188.proxy.runpod.net`
即可——`comfyui_client.py`／`generate_character.py`／`gui.py`／`image_api.py`
其他部分完全不用動，因為整條 pipeline本來就是透過 HTTP 呼叫 ComfyUI，不管
ComfyUI 是跑在同一台機器還是雲端的 pod 上，呼叫方式沒有差別。要注意兩點：

- 網路延遲比本地高很多，`POLL_TIMEOUT_SECONDS` 系列的 timeout 常數可能要
  調高（尤其是第一次跑、要下載 InsightFace/OpenPose 模型那幾次）
- 是 `https`，本地是 `http`——改網址時這個前綴容易漏改

## 之後重新評估的心法

不管是本地換了更好的顯卡、還是 Replicate/RunPod 的價格變了，回來重算的
方法都一樣：

1. **估這段時間實際跑了多少次、每次多久**（`benchmark.py` 已經有本地的
   單張生成秒數基準，可以拿來換算）
2. **本地顯卡是一次性成本**（買卡的錢 + 電費），跟用量無關；**雲端是持續
   成本**，用多少付多少
3. 把「雲端估計花費」除以「本地卡的購入價」，抓一個大概幾個月打平的數字——
   使用量穩定、長期用的工作適合買/用本地卡；用量零星、一次性、或需求會變
   （例如換更大的模型）的工作，繼續留在雲端更划算

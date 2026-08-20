# 啟動 AI Image Lab GUI

## 背景：為什麼會搞混

`D:\AI-Image-Lab\training\gui.py`（AI Image Lab 自訂生圖）跟 `kohya_ss` 自己的
訓練 GUI（`kohya_gui.py`）預設都會用 Gradio 的預設 port **7860**。
`kohya_gui.py` 似乎會在開機時自動啟動，搶先佔掉 7860——所以只要用舊網址或瀏覽器
自動完成連到 `127.0.0.1:7860`，很容易打開成 kohya_ss 的訓練介面，而不是這裡要用
的圖片生成工具。

為了不再互撞，`gui.py` 已經改成固定監聽 **7861**（見檔案最底下
`demo.launch(..., server_port=7861, ...)`）。

**請把常用網址/書籤改成 `http://127.0.0.1:7861`，不要再用 7860。**

## 正確啟動步驟

1. 確認 ComfyUI server 已經在跑（GUI 靠它做實際生成，沒開會連不到）：
   ```powershell
   D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe D:\AI-Image-Lab\ComfyUI\main.py --listen 127.0.0.1 --port 8188
   ```
2. 另開一個終端機視窗，啟動 GUI：
   ```powershell
   cd D:\AI-Image-Lab\training
   D:\AI-Image-Lab\ComfyUI\.venv\Scripts\python.exe gui.py
   ```
3. 瀏覽器開啟 **`http://127.0.0.1:7861`**（不是 7860）。

## 怎麼確認自己開對頁面

- 正確的頁面：分頁標題/頁面最上方是「**AI Image Lab - 自訂生圖**」，網址是
  `:7861`，畫面上看得到 Checkpoint 下拉選單、Anchor 圖上傳、Prompt 輸入框。
- 如果看到的是 LoRA 訓練設定（dataset、learning rate、epoch 這類欄位），那是
  kohya_ss 的訓練 GUI，網址通常是 `:7860`——不是這份文件要用的工具，關掉分頁、
  改連 `:7861`。

## 疑難排解

- **`:7861` 打不開**：GUI 可能還沒啟動或已經當掉。回到終端機執行上面步驟 2 的
  指令重新啟動。
- **畫面卡住、生成完了但看不到結果**：通常是瀏覽器分頁還連著重啟前的舊
  session。按 **Ctrl+F5**（強制重新整理）而不是一般 F5。
- **確認目前哪個 port 有服務在跑**（PowerShell）：
  ```powershell
  netstat -ano | Select-String ":7860|:7861" | Select-String "LISTENING"
  ```
- **如果不想 kohya_ss GUI 每次開機自動啟動**：檢查 Windows 啟動項目
  （工作管理員 → 啟動應用程式，或 `shell:startup` 資料夾）裡有沒有指向
  `gui.bat`/`kohya_gui.py` 的捷徑，不需要的話可以移除。

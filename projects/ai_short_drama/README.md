# AI 短劇概念片：Z-Image 候選圖

五個 60 秒故事（《23:47》《不要回答》《第七個人》《這次，我不救你》《另一個我》），每個 7 個 Scene。
目前只做到 **Approved Master Image**，不進 H3、不做影片。挑圖是人的事，工具不會宣布哪張最好。

```
劇本 → Scene → Z-Image 候選 A/B → HUMAN_REVIEW → APPROVED（master.png）或 REGENERATE（再補 C/D）
```

所有指令用 `ComfyUI\.venv\Scripts\python.exe training\concept_stills.py ...`（`init`／`prompts`／`status`／`index`／
`sheet`／`review`／`generate --dry-run` 不碰 GPU）。

| 檔案 | 內容 |
| --- | --- |
| `visual_bible.json` | 統一風格、9:16 720×1280、steps／cfg、A/B/C/D 構圖規則、審核清單、seed 規則 |
| `story_*/story.json` | Character Bible ＋ 7 個 Scene（劇情、台詞、字幕、畫面文字、光線、A/B 構圖） |
| `story_*/prompts.json` | 展開後的最終 prompt／negative／seed（改 story.json 後跑 `prompts` 重生） |
| `story_*/scene_NN/candidate_NN.png/.json/.workflow.json` | 圖、metadata、填好的節點圖（可原樣重生） |
| `story_*/scene_NN/review.json`、`master.png` | 審核狀態與歷史、核准的母圖 |
| `candidate_index.json`、`review_sheet.html` | 全部候選的索引、人工審核用的並排頁面（都是產生物，不進 git） |

## 流程

1. 讀 `prompts.json`，不滿意就改 `story.json` 再 `prompts`（還沒花任何 GPU）。
2. 啟動 ComfyUI 後 `generate`（預設每個 Scene 兩張 A/B，共 70 張）；可用 `--story 1 --scene 3` 縮小範圍。
3. `sheet` 產生 `review_sheet.html`，逐 Scene 看：劇情資訊、角色一致、場景、構圖、光線看得清細節、
   適不適合 Image→Video（人物會不會變形、手、背景複雜度、動作好不好描述、鏡頭好不好動）、有沒有明顯瑕疵。
   有畫面文字的 Scene（手機「ME」、時鐘、牆上規則）要逐字核對。
4. `review --story N --scene M --approve 1|2|3|4`（複製成 `master.png`）或 `--regenerate --notes "原因"`。
5. 只有 `REGENERATE` 的 Scene 才能 `generate --candidates C,D`。

## 注意

- 劇本沒給外觀的角色是補的設定，story.json 的 `label` 標了「請確認」。
- 《另一個我》與《23:47》月台雙人、《不要回答》窗戶倒影、《第七個人》玻璃七人，對模型都是高難度，預期要補抽。
- 年齡安全負面詞與 cfg ≥ 1.5 由共用的 prompt 組裝保證，關不掉。
- 圖片與 `review_sheet.html` 不進 git（`.gitignore`），規格、prompt 與審核紀錄進。

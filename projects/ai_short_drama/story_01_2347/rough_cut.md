# 《23:47》粗剪（2026-10-03）

> 2026-10-04：檔案搬進這個故事的 `videos/`（片段在 `videos/scene_NN/clip_vN.mp4`，成片在 `videos/cut/`）。
> 下面的路徑都是相對於 `projects/ai_short_drama/story_01_2347/`。當時實際執行的指令用的是舊檔名，內容相同。
> 各片段的 `clip_vN.job.json` 是 H3 當時的記錄，裡面的 `output` 欄位仍是搬動前的路徑。

`videos/cut/story_01_2347_cut_v1.mp4`：7 支 H3 片段直接硬切接起來，沒有轉場、字幕、配樂，音量沒有調整。
768×1376、24 fps、1276 幀、53.17 秒、H.264（crf 18）＋AAC 32 kHz 雙聲道。

| 順序 | 片段 | 秒 | 整體響度 |
| --- | --- | ---: | ---: |
| 1 | `videos/scene_01/clip_v1.mp4` | 5.17 | −20.1 LUFS |
| 2 | `videos/scene_02/clip_v1.mp4` | 8.00 | −14.6 LUFS |
| 3 | `videos/scene_03/clip_v1.mp4` | 8.00 | −17.7 LUFS |
| 4 | `videos/scene_04/clip_v1.mp4` | 8.00 | −15.1 LUFS |
| 5 | `videos/scene_05/clip_v2.mp4` | 8.00 | −24.2 LUFS |
| 6 | `videos/scene_06/clip_v1.mp4` | 8.00 | −10.6 LUFS |
| 7 | `videos/scene_07/clip_v2.mp4` | 8.00 | −35.1 LUFS |

首幀全部是候選 B，都還沒有 `review --approve`；Scene 5、7 用的是修正版（v2）。重做（ffmpeg 7.1，從 repo 根目錄）：

```
ffmpeg -i <1> -i <2> -i <3> -i <4> -i <5> -i <6> -i <7> ^
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a][6:v][6:a]concat=n=7:v=1:a=1[v][a]" ^
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 24 ^
  -c:a aac -b:a 192k -ar 32000 -movflags +faststart videos/cut/story_01_2347_cut_v1.mp4
```

## v2：音量調平＋結尾字幕（2026-10-03）

`videos/cut/story_01_2347_cut_v2.mp4`：同樣 7 支、硬切、53.17 秒、1276 幀。畫面沒有重新調色或裁切。

- **音量**（照 `training/drama_compose.py` 的做法：先量、再用固定增益，不用單趟 `loudnorm`）：每支先用固定增益對齊到
  −23 LUFS，接起來量到 −24.1 LUFS，再整體 `volume=10.10dB,alimiter=limit=0.84:level=false`。
  成片 **−14.2 LUFS**、LRA 10.0 LU、峰值 −1.2 dBFS。
- **例外**：Scene 7 是空月台的環境聲（原本 −35.1 LUFS），只對齊到 −31，比有台詞的片段低 8 LU；拉到同樣大聲會把底噪放大。
- 各段增益（dB）：−2.9／−8.4／−5.3／−7.9／＋1.2／−12.4／＋4.1。成片裡各段實測：−12.9／−13.6／−13.0／−14.0／−12.9／−13.0／−20.9 LUFS。
- **字幕**：「第十八次，開始。」，微軟正黑體粗體 44 px、白字黑邊、置中、y=988，從 51.9 秒（時鐘剛變成 23:47）到結尾，約 1.27 秒。
- 完整的 `filter_complex` 存在 `videos/cut/_work_v2/filter_complex.txt`。

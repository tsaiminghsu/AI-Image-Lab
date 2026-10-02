# 《23:47》粗剪（2026-10-03）

`output/story01/story01_2347_rough_cut.mp4`：7 支 H3 片段直接硬切接起來，沒有轉場、字幕、配樂，音量沒有調整。
768×1376、24 fps、1276 幀、53.17 秒、H.264（crf 18）＋AAC 32 kHz 雙聲道。

| 順序 | 片段 | 秒 | 整體響度 |
| --- | --- | ---: | ---: |
| 1 | `output/s01_scene01_B_h3.mp4` | 5.17 | −20.1 LUFS |
| 2 | `output/story01/s01_scene02_B_h3.mp4` | 8.00 | −14.6 LUFS |
| 3 | `output/story01/s01_scene03_B_h3.mp4` | 8.00 | −17.7 LUFS |
| 4 | `output/story01/s01_scene04_B_h3.mp4` | 8.00 | −15.1 LUFS |
| 5 | `output/story01/s01_scene05_B_h3_v2.mp4` | 8.00 | −24.2 LUFS |
| 6 | `output/story01/s01_scene06_B_h3.mp4` | 8.00 | −10.6 LUFS |
| 7 | `output/story01/s01_scene07_B_h3_v2.mp4` | 8.00 | −35.1 LUFS |

首幀全部是候選 B，都還沒有 `review --approve`；Scene 5、7 用的是修正版（v2）。重做（ffmpeg 7.1，從 repo 根目錄）：

```
ffmpeg -i <1> -i <2> -i <3> -i <4> -i <5> -i <6> -i <7> ^
  -filter_complex "[0:v][0:a][1:v][1:a][2:v][2:a][3:v][3:a][4:v][4:a][5:v][5:a][6:v][6:a]concat=n=7:v=1:a=1[v][a]" ^
  -map "[v]" -map "[a]" -c:v libx264 -crf 18 -preset medium -pix_fmt yuv420p -r 24 ^
  -c:a aac -b:a 192k -ar 32000 -movflags +faststart output/story01/story01_2347_rough_cut.mp4
```

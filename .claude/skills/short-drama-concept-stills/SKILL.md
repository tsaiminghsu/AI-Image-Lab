---
name: short-drama-concept-stills
description: Make concept stills for the AI short-drama pitches in projects/ai_short_drama (Z-Image candidates A/B per scene, human review, approved master), then optionally turn masters into H3 clips and a joined cut. Use when the user asks for 短劇概念片 / 分鏡候選圖 / Scene 候選 / 《23:47》《不要回答》《第七個人》《這次，我不救你》《另一個我》, or to review, approve, regenerate or assemble those scenes. Not for the JV Tutor Corner episodes (training/drama.py).
---

# Concept stills -> master image -> (H3 clip) -> cut

Specs are data in `projects/ai_short_drama/` (visual bible, five `story_*/story.json`). The tool is
`training/concept_stills.py`; python is `ComfyUI\.venv\Scripts\python.exe`, run from the repo root.
The project's rule: **people choose, the tool does not.** Never call a candidate "the best"; give technical
observations (legibility, identity, artifacts) and let the user decide. No H3/video before a scene has a
chosen first frame, and a first frame the user picked only for a trial is not `APPROVED`.

## Steps

1. Free, no GPU: `concept_stills.py init|prompts|status|generate --dry-run|sheet`. Read `story_*/prompts.json`
   before spending GPU; edit `story.json` and re-run `prompts` to change anything.
2. **Ask before GPU work**: scope and time. Measured on the RTX 2070 at 720x1280: 100-308 s per image, mean
   141 s (every new prompt reloads the model), 83-84 C after ~10 min, VRAM peak 7.8 GB. 14 images = 33 min;
   all 70 = ~2.5 h. A yes covers that batch only.
3. Start ComfyUI yourself with `--disable-smart-memory` (cwd `ComfyUI`, `.venv\Scripts\python.exe main.py
   --listen 127.0.0.1 --port 8188`), check nothing else holds the card, run `generate --story N`, stop it with
   `training\stop_comfyui.ps1`. `generate` refuses when ComfyUI is down.
4. `sheet` writes `review_sheet.html`; the user decides: `review --story N --scene M --approve 1|2|3|4` (copies
   `master.png`) or `--regenerate --notes "..."`. Candidates C/D only after `REGENERATE`; `--force` on an
   `APPROVED` scene is refused.
5. Video: use the `minimax-h3-colab` skill with the chosen candidate as first frame; batch all clips in one
   session and read its "Lessons" section first. This user wants `H3_MODEL_SOURCE=download`.
6. Cut: ffmpeg `concat` of the clips (all 768x1376, 24 fps, AAC 32 kHz, so they join without re-scaling).
   Level first: each clip to a fixed reference (-23 LUFS; ambience-only clips lower), measure the joined audio,
   then one fixed gain plus `alimiter=limit=0.84:level=false` to -14 LUFS. Do not use single-pass `loudnorm`
   (it pumps). A subtitle is a `drawtext` with `textfile` (UTF-8) and `msjhbd.ttc`; reuse
   `drama_compose._drawtext`. The worked example is `projects/ai_short_drama/story_01_2347/rough_cut.md`.
   Copying a result to Google Drive needs Drive for desktop (not installed on this machine as of 2026-10-03)
   or a Colab session; ask the user which.

## Problems seen (2026-10-02/03) and what to do

- **Text in the frame** ("ME" on a phone, wall rules) is often illegible in the Z-Image candidate and H3 will not
  fix it. Check the "畫面文字要核對" line in the sheet, regenerate, or overlay the text in post.
- **Doubles, reflections, faceless crowds** (story 1 scene 5, story 2 scene 6, story 3 scene 6, story 5) are the
  hard cases: expect REGENERATE, and judge whether the second person is really identical.
- **Style words in the positive prompt get painted** by Z-Image ("no anime" lives in the negative on purpose);
  never put the story title or a character id in a prompt - it appears as lettering.
- **Group / crowd scenes** need `solo: false`; scenes with in-image text need `on_screen_text` (it drops the
  `text` negative). An empty-platform scene needs `negative_extra: "people, person, crowd"`.
- **The first frame must contain what the script needs to happen**: if a person must vanish, the first frame needs
  that person. Scene 6 of story 1 had no double in it, so the vanishing was not made.
- **Characters the brief did not describe** (Mr. Chen, the six strangers and the seventh, the killer, Another Me's
  clothing) are my additions, labelled "請確認" in `story.json`; do not present them as the author's.
- A bug the tests caught: `"" in "ABCD"` is true, so a bad `--approve` value looked valid. Check letters with
  `len(x) == 1 and x in LETTERS`.
- Other Claude sessions share this machine, the Colab account and :8188; check before starting anything.

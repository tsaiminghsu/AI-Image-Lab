# Local model / prompt-syntax test report

Base prompt (identical across all checkpoints):

> a 25 year old woman with long brown wavy hair, sitting at a wooden cafe table, holding a ceramic coffee cup, soft window light, cozy interior background, shot on DSLR, natural skin texture, candid photograph

| Checkpoint | Family | Prompt style | Resolution | Time | Result |
|---|---|---|---|---|---|
| juggernaut | sdxl | plain natural language | 1024x1024 | 40.3s | OK |
| pony | pony | plain natural language (no score_9 tags) | 1024x1024 | 40.3s | OK |
| pony | pony | 'score_9, score_8_up, score_7_up' prefix + natural language (Pony convention) | 1024x1024 | 30.3s | OK |
| cyberrealistic_pony | pony | plain natural language (no score_9 tags) | 1024x1024 | 38.3s | OK |
| cyberrealistic_pony | pony | 'score_9, score_8_up, score_7_up' prefix + natural language (Pony convention) | 1024x1024 | 28.2s | OK |
| pony_realism | pony | plain natural language (no score_9 tags) | 1024x1024 | 42.3s | OK |
| pony_realism | pony | 'score_9, score_8_up, score_7_up' prefix + natural language (Pony convention) | 1024x1024 | 36.2s | OK |
| realistic_vision | sd15 | plain natural language | 512x768 | 14.1s | OK |
| cyberrealistic | sd15 | plain natural language | 512x768 | 14.2s | OK |

## Notes

- SDXL (`juggernaut`) and SD1.5 (`realistic_vision`, `cyberrealistic`) checkpoints take plain natural-language prompts directly.
- Pony-family checkpoints (`pony`, `cyberrealistic_pony`, `pony_realism`) were trained on a `score_9, score_8_up, score_7_up` quality-tag convention; compare the `no_score_tags` vs `with_score_tags` rows/images for the same seed to see the effect.
- `generate_character.py`'s `gen_custom()` auto-prepends the score tags for Pony checkpoints, so normal pipeline callers never need to type them manually.

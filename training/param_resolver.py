"""Normalise generation parameters that came from our own UI, LLM or HTTP caller.

The rule is clamp, don't trust: a bad suggestion is normalised, never turned into an error.
A Gradio slider, a chat model proposing 4000x4000, or the ai-companion backend echoing a stale
width should all produce a picture rather than a stack trace.

This is the exact opposite of what worker/jobs.py does, and the difference is not stylistic.
worker/jobs.py is a trust boundary receiving untrusted JSON off the network, so it raises
(worker/jobs.py:71 `_get_int` and friends); clamping there would silently accept a hostile
payload and run it on a GPU somebody is paying for. This module normalises a *suggestion from
our own front end*, where refusing to draw anything is the worse outcome. Getting these two
backwards would be a security regression, not a UX change.

Picking a random seed when none was given stays the caller's decision - image_api.py:191 and
worker/jobs.py:129 both do it with `random.randint(0, 2**31 - 1)`. This module only normalises
a seed it was actually handed; it has no sentinel for "surprise me".

generate_character is imported lazily rather than at module scope, because the import graph runs
generate_character -> capability_catalog -> param_resolver and a top-level import here would
close that into a cycle. Deferring it costs nothing: by the time resolve_prompt is called,
generate_character is fully loaded. The repo already does this in the same shape for
cloud_workflow inside _submit_and_wait and for PIL inside gen_gif.
"""

STEP = 32
MIN_SIDE = 256
MAX_SIDE = 1536

# 1536x1024. The ceiling is the 8 GB card, not a taste judgement: above roughly this the HQ
# two-pass path starts spilling into the pagefile on a 32 GB box (see CLAUDE.md's hardware
# notes), and a request that pages for ten minutes is worse than one that is quietly smaller.
PIXEL_BUDGET = MAX_SIDE * 1024

SEED_MIN = 0
SEED_MAX = 0xFFFFFFFF


def _finite(value):
    """float(value) when that is a real number, else None. Catches NaN and both infinities,
    which float() accepts happily and which would survive clamping as garbage."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def snap_side(value, fallback=MIN_SIDE):
    """Snap to the nearest multiple of 32 (ties up) and clamp into [256, 1536]."""
    number = _finite(value)
    if number is None:
        number = _finite(fallback)
    if number is None:
        number = MIN_SIDE
    snapped = int((number + STEP / 2) // STEP) * STEP
    return max(MIN_SIDE, min(MAX_SIDE, snapped))


def fit_pixel_budget(width, height, budget=PIXEL_BUDGET):
    """Shrink the longer side until width*height fits the budget. Returns snapped sides.

    The longer side is chosen because shrinking it moves the aspect ratio toward square, which
    distorts a framing less than squashing the short side would. A square is a tie; height
    loses, arbitrarily but deterministically, so the same input always gives the same output.
    """
    if width * height <= budget:
        return width, height
    if width > height:
        width = max(MIN_SIDE, int(budget / height / STEP) * STEP)
    else:
        height = max(MIN_SIDE, int(budget / width / STEP) * STEP)
    return width, height


def resolve_seed(value):
    """Clamp into uint32. A negative seed clamps to 0 rather than meaning "random" - this repo
    has no -1 sentinel (verified: image_api.py:191 and worker/jobs.py:129 both randomise by
    *absence*), and inventing one here would make an explicit seed non-reproducible."""
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return SEED_MIN
    return max(SEED_MIN, min(SEED_MAX, seed))


def resolve_prompt(value, limit=None):
    """Strip, then truncate to the limit on a word boundary when one is close enough.

    Cutting mid-token turns a tag like "photorealistic" into "photoreal", which CLIP happily
    encodes as something else entirely. Backing up to whitespace is only worth it if the
    boundary is near the end; otherwise (a single enormous token) a hard cut is the honest
    answer.
    """
    if limit is None:
        import generate_character as gc  # deferred: see the module docstring's import-cycle note

        limit = gc.MAX_PROMPT_CHARS
    text = "" if value is None else str(value).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = cut.rfind(" ")
    if boundary >= limit - 40:
        cut = cut[:boundary]
    return cut.rstrip().rstrip(",")


def resolve(*, prompt, width=None, height=None, seed=None, prompt_limit=None,
            default_width=None, default_height=None):
    """Normalise one generation request. Returns a dict with exactly the keys it resolved, so a
    caller can splat it into params without carrying Nones for things it never asked about."""
    default_width = 1024 if default_width is None else default_width
    default_height = 1024 if default_height is None else default_height
    resolved_width = snap_side(default_width if width is None else width, default_width)
    resolved_height = snap_side(default_height if height is None else height, default_height)
    resolved_width, resolved_height = fit_pixel_budget(resolved_width, resolved_height)
    return {
        "prompt": resolve_prompt(prompt, prompt_limit),
        "width": resolved_width,
        "height": resolved_height,
        "seed": resolve_seed(seed),
    }

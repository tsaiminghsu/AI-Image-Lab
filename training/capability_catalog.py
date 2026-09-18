"""What each checkpoint can actually be asked to do, declared in one place instead of discovered
by hitting a raise.

Today the same knowledge lives in three places that can drift apart: generate_character.
_plan_custom:706-758 raises seven UsageErrors, gui.py hand-writes its own warnings, and
plan_picker:817 decides "is this family text-only?" with its own membership test against
client.ZIMAGE_MODELS / client.SD15_CHECKPOINTS. This module is the single source, and
tests/test_capability_catalog.py re-runs _plan_custom against every row so the two cannot
disagree - without that test a capability table is just documentation, and documentation rots.

A disabled capability is a first-class row with a reason, not an absent one. The GUI is expected
to render it visible-but-unselectable with the reason attached: someone who cannot find a control
assumes the tool is broken, and a bare greyed-out checkbox says nothing about what to do instead.

Two reasons per row, on purpose. `reason` is Traditional Chinese for the GUI and the CLI's
capability listing. `reason_en` carries the exact English text _plan_custom already raises, so
turning the catalog into the source of that message does not change one byte of what a library
caller or an existing script sees.

What is NOT here, so nobody reads this as complete:

  * Flag-combination rules. On the legacy (hq=False) path, FaceDetailer needs an anchor and
    cannot be combined with a pose reference. Those are constraints between arguments, not
    statements about what a model can do, and they stay in _plan_custom.
  * Video modes. AnimateDiff, Wan i2v and SVD have their own checkpoint sets and parameter
    ranges, and the GUI tabs that drive them are not migrated yet.
"""

import comfyui_client as client
import param_resolver as pr
import workflow_contracts as wc

FAMILY_SDXL = "sdxl"
FAMILY_SD15 = "sd15"
FAMILY_ZIMAGE = "zimage"

MODE_TXT2IMG = "txt2img"
MODE_TXT2IMG_HQ = "txt2img_hq"
FEATURE_FACEID_ANCHOR = "faceid_anchor"
FEATURE_POSE_CONTROLNET = "pose_controlnet"
FEATURE_POSE_SKELETON = "pose_skeleton"
FEATURE_FACEDETAILER = "facedetailer"

MODES = (MODE_TXT2IMG, MODE_TXT2IMG_HQ)
FEATURES = (FEATURE_FACEID_ANCHOR, FEATURE_POSE_CONTROLNET, FEATURE_POSE_SKELETON, FEATURE_FACEDETAILER)
CAPABILITIES = MODES + FEATURES

# Which templates a family can reach at all. A capability is available to a family when some
# template it can reach structurally contains the nodes for it - so this list plus
# workflow_contracts.CONTRACTS is the derivation, and a template that loses its ControlNet nodes
# takes the matching rows down with it rather than failing at generation time.
FAMILY_TEMPLATES = {
    FAMILY_SDXL: (
        "workflow_template.json",
        "workflow_template_txt2img.json",
        "workflow_template_hq.json",
        "workflow_template_controlnet.json",
        "workflow_template_facedetailer.json",
        "workflow_template_mediapipe_facedetailer.json",
        "workflow_template_img2img.json",
    ),
    FAMILY_SD15: ("workflow_template_txt2img_sd15.json",),
    FAMILY_ZIMAGE: ("workflow_template_txt2img_zimage.json",),
}

# The node class that has to be present for a capability to exist. Any-of, because the same
# capability is spelled with different classes in different templates (the mediapipe FaceDetailer
# variant re-samples through DetailerForEach rather than FaceDetailer).
CAPABILITY_MARKERS = {
    MODE_TXT2IMG: frozenset({"KSampler"}),
    MODE_TXT2IMG_HQ: frozenset({"UpscaleModelLoader"}),
    FEATURE_FACEID_ANCHOR: frozenset({"IPAdapterFaceID"}),
    FEATURE_POSE_CONTROLNET: frozenset({"ControlNetApplyAdvanced"}),
    FEATURE_POSE_SKELETON: frozenset({"ControlNetApplyAdvanced"}),
    FEATURE_FACEDETAILER: frozenset({"FaceDetailer", "DetailerForEach"}),
}

# A library skeleton is fed to ControlNet directly, and only the HQ template wires that line up -
# so the row is disabled wherever txt2img_hq is, independently of whether ControlNet exists.
CAPABILITY_REQUIRES = {FEATURE_POSE_SKELETON: MODE_TXT2IMG_HQ}

LABELS = {
    MODE_TXT2IMG: "純文字生圖",
    MODE_TXT2IMG_HQ: "HQ 兩段式高解析",
    FEATURE_FACEID_ANCHOR: "anchor 鎖臉（IP-Adapter FaceID）",
    FEATURE_POSE_CONTROLNET: "姿勢參考照（ControlNet OpenPose）",
    FEATURE_POSE_SKELETON: "姿勢骨架庫（ControlNet）",
    FEATURE_FACEDETAILER: "臉部精修（FaceDetailer）",
}

# Traditional Chinese for humans. {label} is the checkpoint key.
_REASON_ZIMAGE = (
    "{label} 是 Z-Image，只接了純文字生圖：FaceID／ControlNet／FaceDetailer 都需要 SDXL 或 "
    "SD1.5 專用的 adapter 檔案。角色仍可以當文字描述用，但不會鎖臉。"
)
_REASON_SD15 = (
    "{label} 是 SD1.5，UNet／CLIP 形狀跟 SDXL 系的 adapter 檔不合，只接了純文字生圖。"
)
_REASON_NO_HQ = "{label} 沒有 HQ 兩段式模板（那是 SDXL 系專用的 upscale + 二次取樣鏈）。"
_REASON_SKELETON_NEEDS_HQ = (
    "骨架姿勢是直接餵給 ControlNet 的，只有 HQ 模板接了那條線，而 {label} 沒有 HQ 路徑。"
)

# The exact English text _plan_custom already raises, so making the catalog the source of that
# message changes nothing a library caller or an existing script sees. {label} is the checkpoint.
_REASON_EN_ZIMAGE = (
    "checkpoint {label!r} is Z-Image - only plain txt2img is wired up (anchor/FaceID, "
    "pose reference / skeleton ControlNet and FaceDetailer all need SDXL- or SD1.5-specific "
    "adapter files; --character still works as a text description)"
)
_REASON_EN_SD15 = (
    "checkpoint {label!r} is SD1.5 - only plain txt2img is wired up (anchor/IP-Adapter and "
    "pose_reference/ControlNet need the SDXL-family adapter files, which don't match SD1.5's "
    "UNet/CLIP shape)"
)
_REASON_EN_NO_HQ = "checkpoint {label!r} has no HQ path (the two-pass upscale chain is SDXL-family only)"

# Public, because _plan_custom raises it for a *combination* (pose_name with --no-hq on a family
# that does have an HQ path) rather than for a disabled row. That case is not a statement about
# what the model can do, so it has no row - but the sentence still needs exactly one home.
SKELETON_NEEDS_HQ_EN = (
    "pose_name (library skeleton) needs the HQ path - it feeds ControlNet directly, which only "
    "the HQ template wires up (drop --no-hq)"
)
_REASON_EN_SKELETON_NEEDS_HQ = SKELETON_NEEDS_HQ_EN

# The catch-all for a capability whose nodes are not in any template this family can reach. On a
# healthy tree the family-specific reasons above cover every disabled row, so seeing this text
# means a template lost the nodes that used to back a capability - which is exactly the case the
# derivation exists to notice, and it must not be allowed to surface as a reasonless row.
_REASON_NO_TEMPLATE = "{label} 目前沒有任何工作流模板接上「{capability}」。"
_REASON_EN_NO_TEMPLATE = "no workflow template reachable by checkpoint {label!r} wires up {capability}"


class Row:
    """One declared capability for one checkpoint."""

    __slots__ = ("catalog_id", "kind", "name", "label", "checkpoint", "family", "enabled",
                 "reason", "reason_en", "templates", "params")

    def __init__(self, checkpoint, family, name, enabled, reason, reason_en, templates, params):
        self.catalog_id = f"{checkpoint}:{name}"
        self.kind = "mode" if name in MODES else "feature"
        self.name = name
        self.label = LABELS[name]
        self.checkpoint = checkpoint
        self.family = family
        self.enabled = enabled
        self.reason = reason
        self.reason_en = reason_en
        self.templates = templates
        self.params = params
        if not enabled and not (reason and reason_en):
            raise AssertionError(f"{self.catalog_id} is disabled with no reason")

    def __repr__(self):
        return f"Row({self.catalog_id}, enabled={self.enabled})"


def checkpoints():
    """Every checkpoint key any entry point accepts, in a stable order."""
    return tuple(sorted(client.CHECKPOINTS)) + tuple(sorted(client.ZIMAGE_MODELS))


def family_of(checkpoint):
    if checkpoint in client.ZIMAGE_MODELS:
        return FAMILY_ZIMAGE
    if checkpoint in client.SD15_CHECKPOINTS:
        return FAMILY_SD15
    if checkpoint in client.CHECKPOINTS:
        return FAMILY_SDXL
    raise KeyError(f"unknown checkpoint {checkpoint!r}")


def _classes_for(family):
    """Every node class reachable by this family, straight out of workflow_contracts.CONTRACTS."""
    classes = set()
    for template in FAMILY_TEMPLATES[family]:
        classes.update(wc.CONTRACTS[template].values())
    return classes


def _structurally_supported(family, name):
    return bool(CAPABILITY_MARKERS[name] & _classes_for(family))


def _params_for(family, name):
    """Parameter bounds and defaults, taken from the constants that already govern them. The GUI
    reads these instead of keeping its own copies, which is what stops a slider's maximum drifting
    away from what the workflow will accept."""
    if family == FAMILY_SD15:
        width, height = client.SD15_WIDTH, client.SD15_HEIGHT
    elif family == FAMILY_ZIMAGE:
        width, height = client.ZIMAGE_WIDTH, client.ZIMAGE_HEIGHT
    else:
        width, height = client.WIDTH, client.HEIGHT
    if name not in MODES:
        return {}
    params = {
        "width": {"minimum": pr.MIN_SIDE, "maximum": pr.MAX_SIDE, "step": pr.STEP, "default": width},
        "height": {"minimum": pr.MIN_SIDE, "maximum": pr.MAX_SIDE, "step": pr.STEP, "default": height},
        "seed": {"minimum": pr.SEED_MIN, "maximum": pr.SEED_MAX, "default": 0},
    }
    if family == FAMILY_ZIMAGE:
        params["steps"] = {"minimum": 1, "maximum": 50, "default": client.ZIMAGE_STEPS}
        # The floor, not the official template's 1.0: at cfg 1.0 ComfyUI skips the negative
        # conditioning entirely and the age-safety negatives silently stop applying.
        params["cfg"] = {"minimum": client.SAFETY_MIN_CFG, "maximum": 10.0, "default": client.ZIMAGE_CFG}
    return params


def _reasons(family, name, checkpoint, hq_enabled):
    """Why this row is off. Returns (reason, reason_en), or (None, None) when it is on."""
    # Checked before the family branches: a missing HQ path is about the two-pass upscale chain,
    # not about adapter files, so Z-Image must not be told its hires pass needs an IP-Adapter.
    if name == MODE_TXT2IMG_HQ:
        return _REASON_NO_HQ.format(label=checkpoint), _REASON_EN_NO_HQ.format(label=checkpoint)
    if CAPABILITY_REQUIRES.get(name) == MODE_TXT2IMG_HQ and not hq_enabled:
        if family == FAMILY_ZIMAGE:
            return (_REASON_ZIMAGE.format(label=checkpoint),
                    _REASON_EN_ZIMAGE.format(label=checkpoint))
        return (_REASON_SKELETON_NEEDS_HQ.format(label=checkpoint),
                _REASON_EN_SKELETON_NEEDS_HQ)
    if family == FAMILY_ZIMAGE:
        return _REASON_ZIMAGE.format(label=checkpoint), _REASON_EN_ZIMAGE.format(label=checkpoint)
    if family == FAMILY_SD15:
        return _REASON_SD15.format(label=checkpoint), _REASON_EN_SD15.format(label=checkpoint)
    return (_REASON_NO_TEMPLATE.format(label=checkpoint, capability=LABELS[name]),
            _REASON_EN_NO_TEMPLATE.format(label=checkpoint, capability=name))


def rows_for(checkpoint):
    """Every capability row for one checkpoint, modes first then features, in a stable order."""
    family = family_of(checkpoint)
    hq_enabled = _structurally_supported(family, MODE_TXT2IMG_HQ)
    rows = []
    for name in CAPABILITIES:
        enabled = _structurally_supported(family, name)
        if CAPABILITY_REQUIRES.get(name) == MODE_TXT2IMG_HQ and not hq_enabled:
            enabled = False
        reason, reason_en = (None, None) if enabled else _reasons(family, name, checkpoint, hq_enabled)
        rows.append(Row(checkpoint, family, name, enabled, reason, reason_en,
                        FAMILY_TEMPLATES[family], _params_for(family, name)))
    return rows


def all_rows():
    return [row for checkpoint in checkpoints() for row in rows_for(checkpoint)]


def row(checkpoint, name):
    for candidate in rows_for(checkpoint):
        if candidate.name == name:
            return candidate
    raise KeyError(f"unknown capability {name!r}")


def is_enabled(checkpoint, name):
    return row(checkpoint, name).enabled


def reason_for(checkpoint, name):
    """The Chinese reason a capability is unavailable, or None when it is available."""
    return row(checkpoint, name).reason


def reason_en_for(checkpoint, name):
    """The English reason, for a UsageError a library caller or script will read."""
    return row(checkpoint, name).reason_en


def disabled_rows(checkpoint):
    return [candidate for candidate in rows_for(checkpoint) if not candidate.enabled]


def defaults_for(checkpoint, name=MODE_TXT2IMG_HQ):
    """The parameter defaults a UI should start from. Falls back to the plain txt2img row when the
    HQ row is disabled, so a caller does not have to know which families have an HQ path."""
    candidate = row(checkpoint, name)
    if not candidate.enabled:
        candidate = row(checkpoint, MODE_TXT2IMG)
    return {key: bounds["default"] for key, bounds in candidate.params.items()}

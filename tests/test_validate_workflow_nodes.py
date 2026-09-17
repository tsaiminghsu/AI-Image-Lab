"""The /object_info checker behind the Phase 0 structural gate - offline, against a small fixture.

The live run (training/validate_workflow_nodes.py with ComfyUI up) is what validates the real
templates; this pins that the checker itself flags what it claims to, so a clean live report can't
be a vacuous one.
"""

import validate_workflow_nodes as v

OBJECT_INFO = {
    "KSampler": {"input": {"required": {"model": [], "seed": [], "cfg": []}, "optional": {}}},
    "SaveVideo": {
        "input": {"required": {"video": [], "filename_prefix": [], "format": [], "codec": []}, "hidden": {"prompt": []}}
    },
    "LoadImage": {"input": {"required": {"image": []}}},
}


def test_known_classes_and_inputs_pass():
    wf = {
        "3": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 1, "cfg": 5}},
        "9": {"class_type": "SaveVideo", "inputs": {"video": ["8", 0], "codec": "h264", "codec.encoding.crf": 20}},
    }
    assert v.check_against_object_info(wf, OBJECT_INFO) == []


def test_unknown_class_is_reported():
    problems = v.check_against_object_info({"5": {"class_type": "Nope", "inputs": {}}}, OBJECT_INFO)
    assert problems and "Nope" in problems[0]


def test_misspelled_input_is_reported_with_the_known_names():
    wf = {"3": {"class_type": "KSampler", "inputs": {"seeds": 1}}}
    (problem,) = v.check_against_object_info(wf, OBJECT_INFO)
    assert "'seeds'" in problem and "seed" in problem


def test_dotted_input_needs_a_declared_first_segment():
    wf = {"9": {"class_type": "SaveVideo", "inputs": {"codecs.encoding": "x"}}}
    assert v.check_against_object_info(wf, OBJECT_INFO)


def test_hidden_inputs_count_as_declared():
    wf = {"9": {"class_type": "SaveVideo", "inputs": {"prompt": {}}}}
    assert v.check_against_object_info(wf, OBJECT_INFO) == []

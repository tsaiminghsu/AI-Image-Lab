"""The workflow registry: workflows/registry.json names every workflow, the JSON graph it uses, and how job
parameters map onto that graph. Adding a model is a new graph file plus a new entry here; nothing in the
controller names a workflow (the one exception is safety.SCREENED_PROMPT_WORKFLOWS, on purpose).

An entry carries what the core must know but must not hard-code:

  parameters   name -> type, range, default, and `bind` (which node input receives it)
  constants    node inputs the workflow always sets (model file names, a fixed sampler)
  compose      a text template built from several parameters, bound like a parameter
  derived      values computed on the worker from an input image (a video's width and height)
  inputs       files a job needs (a first frame) and where they may come from
  output       the save node, the kind of file it writes and what a valid one looks like
  models       the files the graph loads, with their pinned source, size and sha256
  gpu          the minimum the graph needs; the worker leaves a job pending on a smaller runtime
"""

import copy

from . import WorkflowError, safety

REGISTRY_FILE = "workflows/registry.json"
PARAM_TYPES = ("string", "int", "float", "bool", "choice", "seed", "negative", "aspect")
OUTPUT_KINDS = {"image": "outputs/images", "video": "outputs/videos"}
INPUT_SOURCES = ("generate", "job", "file")


def _fail(name, message):
    raise WorkflowError("INVALID_REGISTRY", "workflow %r: %s" % (name, message))


def _binds(name, where, binds, graph):
    if not isinstance(binds, list):
        _fail(name, "%s: bind must be a list of {node, input}" % where)
    for b in binds:
        if not isinstance(b, dict) or str(b.get("node")) not in graph or not isinstance(b.get("input"), str):
            _fail(name, "%s: bind %r does not point at a node of the graph" % (where, b))


class Workflow:
    def __init__(self, name, spec, graph):
        self.name = name
        self.spec = spec
        self.graph = graph

    def __getitem__(self, key):
        return self.spec[key]

    def get(self, key, default=None):
        return self.spec.get(key, default)

    @property
    def type(self):
        return self.spec["type"]

    @property
    def profile(self):
        return self.spec["safety"]

    @property
    def parameters(self):
        return self.spec.get("parameters", {})

    @property
    def inputs(self):
        return self.spec.get("inputs", {})

    @property
    def models(self):
        return self.spec.get("models", [])

    def template(self):
        return copy.deepcopy(self.graph)

    def describe(self):
        """What a caller needs to fill a job in. No node ids, no file paths."""
        params = []
        for pname, p in self.parameters.items():
            item = {"name": pname, "type": p["type"], "in": p.get("in", "parameters")}
            for key in ("required", "default", "min", "max", "multiple_of", "max_chars", "label", "simple"):
                if key in p:
                    item[key] = p[key]
            if p["type"] == "choice":
                item["choices"] = list(p["choices"])
            if p["type"] == "aspect":
                item["choices"] = list(p["sizes"])
            params.append(item)
        inputs = [
            {
                "name": iname,
                "kind": i["kind"],
                "sources": list(i["sources"]),
                "default_source": i.get("default_source"),
                "attestation_sources": list(i.get("attestation_sources", [])),
                "label": i.get("label"),
            }
            for iname, i in self.inputs.items()
        ]
        return {
            "name": self.name,
            "type": self.type,
            "task_types": list(self.spec.get("task_types", [])),
            "title": self.spec.get("title", self.name),
            "enabled": bool(self.spec.get("enabled", True)),
            "parameters": params,
            "inputs": inputs,
            "output": self.spec["output"]["kind"],
            "min_vram_gib": self.spec.get("gpu", {}).get("min_vram_gib", 0),
        }


def validate_entry(name, spec, graph):
    if not isinstance(spec, dict):
        _fail(name, "the entry is not an object")
    for key in ("type", "workflow_file", "safety", "output"):
        if key not in spec:
            _fail(name, "missing %r" % key)
    if spec["type"] not in OUTPUT_KINDS:
        _fail(name, "type must be one of %s" % sorted(OUTPUT_KINDS))
    safety.check_profile_name(name, spec["safety"])
    if not isinstance(graph, dict) or not graph:
        _fail(name, "%s is not a ComfyUI API-format graph" % spec["workflow_file"])
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node or not isinstance(node.get("inputs"), dict):
            _fail(name, "node %s of %s is not an API-format node" % (node_id, spec["workflow_file"]))

    out = spec["output"]
    if out.get("kind") != spec["type"] or str(out.get("node")) not in graph or not out.get("ext"):
        _fail(name, "output must name a node of the graph, a kind equal to the type, and an ext")

    for pname, p in spec.get("parameters", {}).items():
        where = "parameter %r" % pname
        if not isinstance(p, dict) or p.get("type") not in PARAM_TYPES:
            _fail(name, "%s: type must be one of %s" % (where, list(PARAM_TYPES)))
        if p.get("in", "parameters") not in ("input", "parameters"):
            _fail(name, "%s: in must be 'input' or 'parameters'" % where)
        if p["type"] == "choice" and not (isinstance(p.get("choices"), list) and p["choices"]):
            _fail(name, "%s: a choice needs choices" % where)
        if p["type"] == "aspect":
            sizes = p.get("sizes")
            if not isinstance(sizes, dict) or not sizes or p.get("default") not in sizes:
                _fail(name, "%s: an aspect needs sizes and a default among them" % where)
            for other in ("width", "height"):
                if spec["parameters"].get(other, {}).get("type") != "int":
                    _fail(name, "%s: an aspect needs int parameters width and height" % where)
        if "bind" in p:
            _binds(name, where, p["bind"], graph)
    for i, c in enumerate(spec.get("constants", [])):
        _binds(name, "constant %d" % i, [c], graph)
        if "value" not in c:
            _fail(name, "constant %d has no value" % i)
    for i, c in enumerate(spec.get("compose", [])):
        if not isinstance(c.get("template"), str):
            _fail(name, "compose %d has no template" % i)
        _binds(name, "compose %d" % i, c.get("bind"), graph)
    for iname, inp in spec.get("inputs", {}).items():
        where = "input %r" % iname
        if inp.get("kind") != "image":
            _fail(name, "%s: only image inputs are supported" % where)
        sources = inp.get("sources")
        if not isinstance(sources, list) or not sources or any(s not in INPUT_SOURCES for s in sources):
            _fail(name, "%s: sources must be among %s" % (where, list(INPUT_SOURCES)))
        if "generate" in sources and not isinstance(inp.get("generate"), dict):
            _fail(name, "%s: source 'generate' needs a generate section" % where)
        _binds(name, where, inp.get("bind"), graph)
    for i, d in enumerate(spec.get("derived", [])):
        if d.get("rule") != "auto_resolution" or d.get("from_input") not in spec.get("inputs", {}):
            _fail(name, "derived %d: only auto_resolution from a declared input is supported" % i)
        _binds(name, "derived %d width" % i, d.get("bind_width"), graph)
        _binds(name, "derived %d height" % i, d.get("bind_height"), graph)
    for m in spec.get("models", []):
        for key in ("name", "folder", "store", "repo", "revision", "path_in_repo", "size", "sha256"):
            if not m.get(key):
                _fail(
                    name, "model %r: missing %r (models are pinned: revision, size and sha256)" % (m.get("name"), key)
                )


class Registry:
    def __init__(self, workflows, problems=None):
        self.workflows = workflows
        self.problems = problems or {}

    @classmethod
    def load(cls, storage):
        """One broken entry does not take the others down: it is left out and reported in `problems`
        (list_workflows shows it), and asking for it by name raises its own error."""
        data = storage.read_json(REGISTRY_FILE)
        if not isinstance(data, dict) or not isinstance(data.get("workflows"), dict):
            raise WorkflowError("INVALID_REGISTRY", '%s is missing or is not {"workflows": {...}}' % REGISTRY_FILE)
        workflows, problems = {}, {}
        for name, spec in data["workflows"].items():
            try:
                graph = None
                if isinstance(spec, dict) and spec.get("workflow_file"):
                    graph = storage.read_json("workflows/%s" % spec["workflow_file"])
                validate_entry(name, spec, graph)
            except WorkflowError as exc:
                problems[name] = exc
                continue
            workflows[name] = Workflow(name, spec, graph)
        return cls(workflows, problems)

    def names(self, include_disabled=False):
        return [n for n, w in self.workflows.items() if include_disabled or w.spec.get("enabled", True)]

    def get(self, name):
        if name in self.problems:
            raise self.problems[name]
        wf = self.workflows.get(name)
        if wf is None or not wf.spec.get("enabled", True):
            raise WorkflowError(
                "UNKNOWN_WORKFLOW",
                "No enabled workflow named %r." % name,
                hint="Available: %s" % ", ".join(self.names()),
            )
        return wf

    def custom_nodes(self):
        """Every custom node any enabled workflow needs: installed once, so one ComfyUI serves them all."""
        names = []
        for name in self.names():
            for node in self.workflows[name].spec.get("custom_nodes", []):
                if node not in names:
                    names.append(node)
        return names

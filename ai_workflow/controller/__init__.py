"""Workflow core: the same package runs on the local machine (tool interface) and on the compute runtime
(notebooks and the worker), loaded straight from the workspace folder. Standard library only.

Nothing in this package knows which assistant, if any, is driving it. The entry points are
controller.tools (six functions) and controller.notebook (what the notebooks call).
"""

VERSION = 1


class WorkflowError(Exception):
    """A failure with a machine-readable code. Library code raises this, never SystemExit: the notebook
    kernel and any caller's `except Exception` must be able to catch it."""

    def __init__(self, code, message, hint=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint

    def as_dict(self):
        out = {"code": self.code, "message": self.message}
        if self.hint:
            out["hint"] = self.hint
        return out

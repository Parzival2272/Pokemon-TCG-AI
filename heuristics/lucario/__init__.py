"""Package shim so training/ can import the Lucario agent.

Unlike the other heuristics, this one was written as a standalone Kaggle
submission: it imports the engine bindings as a top-level `cg` package
(`from cg.api import ...`) and ships its own copy of them in
heuristics/lucario/cg/. That copy is byte-identical to the repo's ptcg/ --
sim.py, game.py and utils.py match exactly, and api.py differs only in
formatting (black-reflowed; the parse trees are the same apart from trailing
whitespace inside one docstring) -- EXCEPT that it ships libcg.so only, with
no cg.dll, so on Windows it cannot load at all.

So rather than let `cg` resolve to that directory, alias it onto ptcg before
importing the agent. Two reasons this is the alias and not an edit to
agent.py's imports:

  * agent.py stays runnable standalone (run_local.py / Dockerfile /
    kaggle_smoketest.py all run from inside this directory, where `cg`
    must keep resolving to the vendored copy).
  * ptcg/sim.py is import-time stateful -- it LoadLibrary's the native
    engine and calls GameInitialize(), and its Battle.battle_ptr global is
    the one live battle CabtEnv starts and finishes each episode
    (training/cabt_env.py reset()). A second copy of that module would mean
    a second engine handle in-process.

The submodules must be aliased individually. Binding only `sys.modules["cg"]`
is not enough: `from cg.api import X` would then miss the cache, walk
ptcg.__path__, and load ptcg/api.py a SECOND time under the name "cg.api" --
duplicate IntEnum classes that fail cross-module identity checks, and a
second import of sim.py underneath it.
"""

import sys

import ptcg
import ptcg.api
import ptcg.game
import ptcg.sim
import ptcg.utils

for _name, _module in (
    ("cg", ptcg),
    ("cg.api", ptcg.api),
    ("cg.game", ptcg.game),
    ("cg.sim", ptcg.sim),
    ("cg.utils", ptcg.utils),
):
    sys.modules.setdefault(_name, _module)

from .agent import agent  # noqa: E402  (must follow the aliases above)

__all__ = ["agent"]

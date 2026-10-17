from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from typing_extensions import Self

import coraplex

# %% the demo script of coraplex


@dataclass(frozen=True)
class DemoScript:
    """
    A ``demo.py`` of ``coraplex/demos``, which is a script rather than a package,
    imported as a module so its classes can be used.
    """

    path: Path
    """
    The script file.
    """

    module_name: str
    """
    The name the script is imported under.
    """

    @classmethod
    def daisy_cable_demo(cls) -> Self:
        """
        :return: The script of the DAiSy cable demo.
        """
        demos = Path(coraplex.__file__).resolve().parents[2] / "demos"
        return cls(
            path=demos / "daisy_cable_demo" / "demo.py", module_name="daisy_cable_demo"
        )

    def load(self) -> ModuleType:
        """
        Import the script once; its ``if __name__ == "__main__"`` block does not run.

        :return: The script as a module.
        """
        if self.module_name in sys.modules:
            return sys.modules[self.module_name]
        # The script imports the modules next to it by their bare names.
        if str(self.path.parent) not in sys.path:
            sys.path.append(str(self.path.parent))
        specification = importlib.util.spec_from_file_location(
            self.module_name, self.path
        )
        module = importlib.util.module_from_spec(specification)
        # Registered before running it, so its dataclasses can resolve their module.
        sys.modules[self.module_name] = module
        specification.loader.exec_module(module)
        return module

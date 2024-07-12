from __future__ import annotations
from typing import Optional, Self
from dataclasses import dataclass

import lekkersim as lk


@dataclass(frozen=True)
class Pin:
    """Class representing a pin in the circuit.

    Applicable to models, solvers, and structures.

    It stores a base name and a mode neame. The mode name is optional and is used to distinguish between different modes of the same pin.
    """

    basename: str
    """Base name of the pin"""

    mode_name: Optional[str] = None
    """Mode name of the pin"""

    @property
    def name(self):
        """Full name of the pin.

        - {base name}_{mode name} if mode name is not None,
        - base name otherwise
        """
        if self.mode_name is None:
            return self.basename
        return f"{self.basename}_{self.mode_name}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.basename!r}, {self.mode_name!r})"

    def put(self, pin: Optional[lk.StructurePin] = None):
        """Creates an pin in the solver and maps it to the given pin.

        Args:
            pin (StructurePin) : Structure pin to be mapped.
        """
        if pin is not None:
            lk.sol_list[-1].map_pins({self: pin})


if __name__ == "__main__":
    p = Pin("A")

    a = {p: "A"}

    print(a[p])
    print(a[Pin("A")])

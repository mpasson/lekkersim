from __future__ import annotations
from typing import Optional, Self
from dataclasses import dataclass

import lekkersim as lk


@dataclass(frozen=True)
class Pin:
    basename: str
    mode_name: Optional[str] = None

    @property
    def name(self):
        if self.mode_name is None:
            return self.basename
        return f"{self.basename}_{self.mode_name}"

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.basename!r}, {self.mode_name!r})"

    def put(self, pin: Optional[tuple[lk.Structure, Self]] = None):
        """Maps the pins in the tuple to self.name

        Args:
            pin (tuple) : tuple of (structure (Structure), pin (str)) containing the data to the pin to be mapped
        """
        if pin is not None:
            lk.sol_list[-1].map_pins({self: pin})


if __name__ == "__main__":
    p = Pin("A")

    a = {p: "A"}

    print(a[p])
    print(a[Pin("A")])

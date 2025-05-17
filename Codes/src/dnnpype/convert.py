"""DNNPype/convert.py Convert units"""

from __future__ import annotations

import argparse

import rich as r
from pint import UnitRegistry

_ureg = UnitRegistry()
_ureg.define("millimiters_water_column = 9806.38 millipascals = mmH2O")

quantity = _ureg.Quantity


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert units")
    parser.add_argument(
        "--value",
        default=None,
        type=float,
        help="Magnitude of the value to convert",
        required=True,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Convert from this unit, e.g. 'm', 'bar', 'mmH2O'",
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Convert to this unit, e.g. 'm', 'bar', 'mmH2O'",
        required=True,
    )
    return parser.parse_args()


def main() -> None:
    """Main function to convert units."""
    args = parse_args()
    value = args.value
    from_unit = args.input
    to_unit = args.output

    try:
        parsec_value = quantity(value, from_unit)
    except Exception as e:
        r.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    try:
        converted_value = parsec_value.to(to_unit)
    except Exception as e:
        r.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    r.print(
        f"From {value} {from_unit:<10} to "
        f"{converted_value.magnitude} {to_unit:<10}"
    )


if __name__ == "__main__":
    main()

"""Decimal half-up display formatting; stored measurements remain unchanged."""
from decimal import Decimal, ROUND_HALF_UP

def display(value, places=2, signed=False):
    rounded = Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    if not rounded:
        rounded = abs(rounded)
    return format(rounded, ("+" if signed else "") + f".{places}f")

def percent(value):
    return display(Decimal(str(value)) * 100, places=0)

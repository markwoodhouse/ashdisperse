class PhysicalParameters:
    Kappa_h: float
    Kappa_v: float
    g: float
    mu: float

    def __init__(self, Kappa_h: float=100, Kappa_v: float=10, g: float=9.81, mu: float=18.5e-6): ...

    def validate(self) -> int: ...
        
    def describe(self): ...

    
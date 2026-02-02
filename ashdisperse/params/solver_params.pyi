class SolverParameters:
    meps: float
    domX: float
    domY: float
    minN_log2: float
    maxN_log2: float
    Nx_log2: float
    Ny_log2: float
    epsilon: float
    plateau_factor: float
    fft_tol: float

    def __init__(
        self,
        domX: float=1.5,
        domY: float=1.5,
        minN_log2: float=4,
        maxN_log2: float=8,
        Nx_log2: float=8,
        Ny_log2: float=8,
        epsilon: float=1e-8,
        plateau_factor: float=10.0,
        fft_tol: float=1e-10,
    ): ...

    def validate(self) -> int: ...

    @property
    def Nx(self) -> int: ...
    
    @property
    def Ny(self) -> int: ...

    @property
    def minN(self) -> int: ...
    
    @property
    def maxN(self) -> int: ...
    
    @property
    def chebIts(self) -> int: ...

    def describe(self): ...
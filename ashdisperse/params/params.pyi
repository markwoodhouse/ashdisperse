from emission_params import EmissionParameters
from grain_params import GrainParameters
from met_params import MetParameters
from model_params import ModelParameters
from output_params import OutputParameters
from physical_params import PhysicalParameters
from solver_params import SolverParameters
from source_params import SourceParameters

class Parameters:
    source: SourceParameters
    grains: GrainParameters
    emission: EmissionParameters
    solver: SolverParameters
    physical: PhysicalParameters
    met: MetParameters
    model: ModelParameters
    output: OutputParameters

    def __init__(self): ...

    def update(self, with_utm: bool=True): ...

    def describe(self): ...


def copy_parameters(A: Parameters) -> Parameters: ...

def update_parameters(
    A: Parameters,
    name: str | None = None,
    domX: int | None = None,
    domY: int | None = None,
    minN_log2: int | None = None,
    maxN_log2: int | None = None,
    epsilon: float | None = None,
    plateau_factor: float | None = None,
    fft_tol: float | None = None,
    Nx_log2: int | None = None,
    Ny_log2: int | None = None,
    grains:  list[dict[str, float]] | None = None,
    emissions: list[dict[str,int | float]] | str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    radius: float | None =None,
    PlumeHeight: float | None =None,
    MER: float | None =None,
    duration: float | None =None,
    Kappa_h: float | None = None,
    Kappa_v: float | None = None,
    g: float | None = None,
    mu: float | None = None,
    start: float | None = None,
    stop: float | None = None,
    step: float | None = None,
) -> Parameters: ...

def save_parameters(params: Parameters, file: str="parameters.toml") -> None: ...

def load_parameters(file: str) -> Parameters: ...

def parameters_from_dict(paramset: dict) -> Parameters: ...


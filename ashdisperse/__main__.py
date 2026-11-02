import argparse
import pathlib
from datetime import datetime, timezone

from ashdisperse.ashdisperse import solve
from ashdisperse.config import set_default_threads
from ashdisperse.config.config import set_threading_layer
from ashdisperse.interface import set_met, set_parameters
from ashdisperse.params import load_parameters
from ashdisperse.utilities.datetime_strings import is_valid_datetime


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ashdisperse")
    parser.add_argument("--parameters", "-p", type=str, help="Parameters file name.")
    parser.add_argument(
        "--met_source",
        "-m",
        type=str,
        choices=["interface", "gfs", "ecmwf", "local"],
        help="Source of met data.",
    )
    parser.add_argument(
        "-d",
        "--met_datetime",
        type=str,
        help="Date-time string for met data, in format yyyy-mm-dd hh:mm",
    )
    parser.add_argument(
        "-f",
        "--met_file",
        type=str,
        help=(
            "Met data file. Required when met_source is local; "
            "download location when met_source is ecmwf."
        ),
    )

    args = parser.parse_args(argv)

    if args.met_source in {"gfs", "ecmwf", "local"} and args.met_datetime is None:
        parser.error(f"--met_datetime is required when met_source is {args.met_source}")
    if args.met_datetime is not None and not is_valid_datetime(args.met_datetime):
        parser.error(
            f"--met_datetime '{args.met_datetime}' is not valid; "
            "required format is yyyy-mm-dd hh:mm"
        )
    if args.met_source == "local" and args.met_file is None:
        parser.error("--met_file is required when met_source is local")

    return args


def main(argv=None):
    args = _parse_args(argv)

    print("Running AshDisperse")
    run_start = datetime.now(timezone.utc).strftime("%Y%m%d%H%M")

    if args.parameters is not None:
        params_file = pathlib.Path(args.parameters)
        if not params_file.is_file():
            raise FileNotFoundError(f"Parameters file {params_file} does not exist")
        params = load_parameters(params_file)
    else:
        params = set_parameters()

    if args.met_source is not None:
        met_kwargs = {}
        if args.met_datetime is not None:
            met_kwargs["datetime"] = args.met_datetime
        if args.met_file is not None:
            met_kwargs["met_filename"] = args.met_file

        met = set_met(params, source=args.met_source, **met_kwargs)

        save_name = f"AshDisperse_{params.source.name}_{args.met_source}"
        if args.met_datetime is not None:
            met_datetime_str = (
                args.met_datetime.replace("-", "").replace(":", "").replace(" ", "")
            )
            save_name += f"_{met_datetime_str}"
        save_name += f"_{run_start}"
    else:
        met = set_met(params)
        save_name = f"AshDisperse_{params.source.name}_{run_start}"

    set_threading_layer(thread_layer="tbb")
    set_default_threads()
    result = solve(params, met, timer=True)

    for grain_i in range(params.grains.bins):
        result.plot_ashload_for_grain_class(
            grain_i,
            vmin=1e-2,
            show=False,
            save=True,
            save_name=f"{save_name}_grain{grain_i}.png",
        )


if __name__ == "__main__":
    main()

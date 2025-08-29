import argparse
import pathlib
from datetime import datetime

from ashdisperse.ashdisperse import solve
from ashdisperse.config import set_default_threads
from ashdisperse.config.config import set_threading_layer
from ashdisperse.interface import set_met, set_parameters
from ashdisperse.params import load_parameters

print("Running AshDisperse")
now = datetime.utcnow()
run_start = now.strftime("%Y%m%d%H%M")

parser = argparse.ArgumentParser()
parser.add_argument("--parameters", "-p", type=str, help="Parameters file name.")
parser.add_argument("--met_source", "-m", type=str,
                    choices=['interface', 'gfs', 'ecmwf', 'local'], help="Source of met data.")
parser.add_argument("-d", "--met_datetime", type=str, help="Date-time string for met data, in format yyyy-mm-dd hh:mm")

args = parser.parse_args()

if args.met_source in {'gfs', 'ecmwf'} and args.met_datetime is None:
    parser.error("met_datetime is required when met_source is {}".format(args.met_source))

if args.parameters is not None:
    params_file = pathlib.Path(args.parameters)
    if not params_file.exists():
        raise FileExistsError('Parameters file {} does not exist'.format(params_file))
    if not params_file.is_file():
        raise FileNotFoundError('Input parameters {} is not a file'.format(args.parameters))

    params = load_parameters(params_file)
else:
    params = set_parameters()

if args.met_source is not None:
    met_source = args.met_source
    met_datetime = args.met_datetime
    met_datetime_str = met_datetime.replace('-','').replace(':','').replace(' ','')

    met = set_met(params, source=met_source, datetime=met_datetime)

    save_name = 'AshDisperse_{name}_{met_source}_{met_date}_{run_start}'.format(name=params.source.name, 
                                                                            met_source=met_source,
                                                                            met_date=met_datetime_str,
                                                                            run_start=run_start)
else:
    met = set_met(params)
    save_name = 'AshDisperse_{name}_{run_start}'.format(name=params.source.name, run_start=run_start)

set_threading_layer(thread_layer='tbb')
set_default_threads()
result = solve(params, met, timer=True)

for grain_i in range(0, params.grains.bins):
    result.plot_ashload_for_grain_class(0, vmin=1e-2, save_name=save_name)

# for grain_i in range(0, params.grains.bins):
#     result.plotSettlingFlux(grain_i)
#     result.plotConc(grain_i)
#     result.plotIsoConc(grain_i, 1e-4)

# _ = result.getAshLoad(resolution=500., vmin=1e-4, nodata=-1,
#                       export_gtiff=True, export_name='AshLoad.tif')

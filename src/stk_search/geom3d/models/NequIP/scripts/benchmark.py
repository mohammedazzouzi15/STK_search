import argparse
import itertools
import tempfile
import textwrap
import time

import torch
from e3nn.util.jit import script
from NequIP.data import AtomicData, AtomicDataDict, dataset_from_config
from NequIP.model import model_from_config
from NequIP.scripts.deploy import _compile_for_deploy
from NequIP.scripts.train import check_code_version, default_config
from NequIP.utils import Config
from NequIP.utils._global_options import _set_global_options
from torch.utils.benchmark import Measurement, Timer
from torch.utils.benchmark.utils.common import select_unit, trim_sigfig


def main(args=None):
    parser = argparse.ArgumentParser(
        description=textwrap.dedent(
            """Benchmark the approximate MD performance of a given model configuration / dataset pair."""
        )
    )
    parser.add_argument("config", help="configuration file")
    parser.add_argument(
        "--profile",
        help="Profile instead of timing, creating and outputing a Chrome trace JSON to the given path.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--device",
        help="Device to run the model on. If not provided, defaults to CUDA if available and CPU otherwise.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-n",
        help="Number of trials.",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--n-data",
        help="Number of frames to use.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--timestep",
        help="MD timestep for ns/day esimation, in fs. Defauts to 1fs.",
        type=float,
        default=1,
    )

    # TODO: option to show memory use

    # Parse the args
    args = parser.parse_args(args=args)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)


    config = Config.from_file(args.config, defaults=default_config)
    _set_global_options(config)
    check_code_version(config)

    # Load dataset to get something to benchmark on
    dataset_time = time.time()
    dataset = dataset_from_config(config)
    dataset_time = time.time() - dataset_time
    dataset_rng = torch.Generator()
    dataset_rng.manual_seed(config.get("dataset_seed", config.get("seed", 12345)))
    datas = [
        AtomicData.to_AtomicDataDict(dataset[i].to(device))
        for i in torch.randperm(len(dataset), generator=dataset_rng)[: args.n_data]
    ]
    n_atom: int = len(datas[0]["pos"])
    if not all(len(d["pos"]) == n_atom for d in datas):
        msg = "NequIP-benchmark does not currently handle benchmarking on data frames with variable number of atoms"
        raise NotImplementedError(
            msg
        )
    # print some dataset information
    torch.mean(
        torch.cat(
            [
                torch.bincount(
                    d[AtomicDataDict.EDGE_INDEX_KEY][0],
                    minlength=d[AtomicDataDict.POSITIONS_KEY].shape[0],
                ).float()
                for d in datas
            ]
        )
    ).item()

    # cycle over the datas we loaded
    datas = itertools.cycle(datas)

    # short circut
    if args.n == 0:
        return

    # Load model:
    model_time = time.time()
    model = model_from_config(config, initialize=True, dataset=dataset)
    model_time = time.time() - model_time
    # "Deploy" it
    model.eval()
    compile_time = time.time()
    model = script(model)
    model = _compile_for_deploy(model)
    compile_time = time.time() - compile_time

    # save and reload to avoid bugs
    with tempfile.NamedTemporaryFile() as f:
        torch.jit.save(model, f.name)
        model = torch.jit.load(f.name, map_location=device)
        # freeze like in the LAMMPS plugin
        model = torch.jit.freeze(model)
        # and reload again just to avoid bugs
        torch.jit.save(model, f.name)
        model = torch.jit.load(f.name, map_location=device)

    # Make sure we're warm past compilation
    warmup = config["_jit_bailout_depth"] + 4  # just to be safe...

    if args.profile is not None:

        def trace_handler(p) -> None:
            p.export_chrome_trace(args.profile)

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ]
            + ([torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else []),
            schedule=torch.profiler.schedule(
                wait=1, warmup=warmup, active=args.n, repeat=1
            ),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(1 + warmup + args.n):
                model(next(datas).copy())
                p.step()
    else:
        warmup_time = time.time()
        for _ in range(warmup):
            model(next(datas).copy())
        warmup_time = time.time() - warmup_time

        # just time
        t = Timer(
            stmt="model(next(datas).copy())", globals={"model": model, "datas": datas}
        )
        perloop: Measurement = t.timeit(args.n)

        trim_time = trim_sigfig(perloop.times[0], perloop.significant_figures)
        time_unit, time_scale = select_unit(trim_time)
        ("{:.%dg}" % perloop.significant_figures).format(
            trim_time / time_scale
        )
        per_atom_time = trim_time / n_atom
        time_unit_per, time_scale_per = select_unit(per_atom_time)
        (86400.0 / trim_time) * args.timestep * 1e-6
        #     day in s^   s/step^         ^ fs / step      ^ ns / fs


if __name__ == "__main__":
    main()

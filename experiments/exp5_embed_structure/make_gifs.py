from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eval import save_side_by_side_gif, save_time_series_plot
from simulate_springs import save_spring_gif


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", type=str, required=True, help="Path to .npy trajectory or .npz dataset")
    parser.add_argument("--generated", type=str, default=None, help="Optional generated .npy trajectory")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.real.endswith(".npz"):
        arr = np.load(args.real)
        real = arr["test"][args.index]
    else:
        real = np.load(args.real)
        if real.ndim == 3:
            real = real[args.index]

    save_spring_gif(real, outdir / "real.gif")

    if args.generated is not None:
        fake = np.load(args.generated)
        if fake.ndim == 3:
            fake = fake[args.index]
        save_spring_gif(fake, outdir / "generated.gif")
        save_side_by_side_gif(real, fake, outdir / "real_vs_generated.gif")
        save_time_series_plot(real, fake, outdir / "real_vs_generated.png")


if __name__ == "__main__":
    main()

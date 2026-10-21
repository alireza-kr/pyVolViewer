#!/usr/bin/env python3
# coding=utf-8
"""
Command-line entry point for the Interactive Brain Viewer.

Quickest way to try it, no data required::

    python run.py --demo

With your own files::

    python run.py --anat sub-01_T1w.nii.gz --func sub-01_bold.nii.gz \\
                   --model sub-01_glm.pkl --events sub-01_events.tsv \\
                   --roi left_ffa.nii.gz --roi right_ffa.nii.gz \\
                   --contrast "Face - House"

Every option has a matching argument here, but if you'd rather configure
things in Python directly (e.g. to pass an atlas through
:func:`interactive_glm.data.loading.split_atlas`, or to load a model object
you already have in memory), skip this script and call
:func:`interactive_glm.data.loading.load_brain_data` /
:func:`interactive_glm.app.create_app` yourself - see ``examples/run_demo.py``
for the minimal version of what this script does.
"""

from __future__ import annotations

import argparse
import sys

from interactive_glm.app import create_app
from interactive_glm.config import RoiSource, ViewerSettings
from interactive_glm.data.loading import load_brain_data


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Interactive viewer for brain imaging data: one anatomical/functional "
        "background with any number of ROI and GLM statistical-map overlays.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data_group = parser.add_argument_group("data")
    data_group.add_argument("--demo", action="store_true", help="Use a fully offline synthetic demo dataset.")
    data_group.add_argument("--anat", metavar="PATH", help="Anatomical (e.g. T1w) NIfTI file.")
    data_group.add_argument("--func", metavar="PATH", help="4D functional NIfTI file.")
    data_group.add_argument("--model", metavar="PATH", help="Fitted FirstLevelModel, saved with joblib.dump.")
    data_group.add_argument("--events", metavar="PATH", help="Events .tsv/.csv (onset, duration, trial_type).")
    data_group.add_argument(
        "--roi", metavar="PATH", action="append", default=[], dest="rois",
        help="ROI mask NIfTI file. Repeat for multiple ROIs, e.g. --roi a.nii.gz --roi b.nii.gz.",
    )

    display_group = parser.add_argument_group("display")
    display_group.add_argument("--contrast", default=None, help="Contrast expression to show on startup.")
    display_group.add_argument("--threshold", type=float, default=3.1, help="Initial statistical threshold.")
    display_group.add_argument("--title", default="Interactive Brain Viewer", help="Browser tab / header title.")

    server_group = parser.add_argument_group("server")
    server_group.add_argument("--host", default="127.0.0.1")
    server_group.add_argument("--port", type=int, default=8050)
    server_group.add_argument("--debug", action="store_true", help="Enable Dash's debug/reloader mode.")

    return parser


def main(argv: list = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if not args.demo and not (args.anat or args.func):
        build_arg_parser().error("Provide --demo, or at least one of --anat / --func.")

    settings = ViewerSettings(
        title=args.title,
        default_contrast=args.contrast,
        default_threshold=args.threshold,
        host=args.host,
        port=args.port,
        debug=args.debug,
    )

    if args.demo:
        from interactive_glm.data.demo import build_demo_dataset

        print("Building the offline synthetic demo dataset (a few seconds)...")
        data = build_demo_dataset()
    else:
        rois = [RoiSource(img=path, label=_label_from_path(path)) for path in args.rois]
        data = load_brain_data(
            anat=args.anat, func=args.func, glm_model=args.model, events=args.events, rois=rois or None,
        )

    app = create_app(data, settings)
    print(f"Serving on http://{settings.host}:{settings.port}  (Ctrl+C to stop)")
    app.run(host=settings.host, port=settings.port, debug=settings.debug)


def _label_from_path(path: str) -> str:
    stem = path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for suffix in (".nii.gz", ".nii"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


if __name__ == "__main__":
    main(sys.argv[1:])

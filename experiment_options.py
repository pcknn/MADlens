"""Shared command-line options; CNN execution stays in each original script."""

import argparse


def dispatch_tree_method(workflow):
    parser = argparse.ArgumentParser(
        description=f"Run {workflow} CNN or tree experiments on MURA."
    )
    parser.add_argument(
        "--method", default="cnn",
        choices=("cnn", "regression_tree", "random_forest", "trees"),
        help="Default: cnn. Use trees to compare both tree methods.",
    )
    args = parser.parse_args()
    if args.method == "cnn":
        return
    from main_anatomy_tree import main

    methods = None if args.method == "trees" else (args.method,)
    main(workflow=workflow, methods=methods)
    raise SystemExit(0)

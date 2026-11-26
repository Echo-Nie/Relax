# Copyright (c) 2026 Relax Authors. All Rights Reserved.

"""Keep the isolated experiment environment across native initialization."""

import json
import os


def run():
    frozen = json.loads(os.environ["RELAX_COUNTDOWN_RUNTIME_ENV"])
    os.environ.update(frozen)
    from relax.entrypoints.train import main, parse_args

    args = parse_args()
    os.environ.update(frozen)
    main(args)


if __name__ == "__main__":
    run()

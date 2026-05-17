#!/bin/bash

export PYTHONPATH=/workspaces/phd_baikal/StrangRTE:/workspaces/phd_baikal/StrangRTE/lib:$PYTHONPATH

# To run the automated pytests:
# PYTHONPATH=. pytest StrangRTE/test/

# For manual verification profiling scripts that output GIFs and 3D plotting:
# PYTHONPATH=. python StrangRTE/test/collision/test_collision_profiler.py
# PYTHONPATH=. python StrangRTE/test/streaming/test_streaming_profiler.py

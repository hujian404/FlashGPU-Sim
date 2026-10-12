# Accel-Sim trace parser import

These parser and `.tracez` streaming files are copied from
`accel-sim/accel-sim-framework` dev commit `d930ad6d02c09bb56867132583735aba0389cff4`.
The original project's BSD 2-Clause license is preserved in
`ACCEL_SIM_LICENSE`. The only source adjustment is adding the missing
`<stdexcept>` include to `warp_trace_stream.h` for standalone compilation.

The parser is built by `tools/sass_trace/Makefile` and has been tested on
SM120 traces from RTX 5090. It is an input library, not a timing frontend.

#include "trace_parser.h"
#include "warp_trace_stream.h"

#include <cstdint>
#include <iostream>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: tracez_probe <kernelslist.g>\n";
    return 2;
  }

  trace_parser parser(argv[1]);
  const auto commands = parser.parse_commandlist_file();
  for (const auto &command : commands) {
    if (command.m_type != kernel_launch) continue;
    auto *kernel = parser.parse_kernel_info(command.command_string);
    if (kernel->binary_verion != 120 || !kernel->is_tracez ||
        kernel->trace_verion != 6) {
      std::cerr << "unsupported trace architecture or format: "
                << command.command_string << "\n";
      parser.kernel_finalizer(kernel);
      return 3;
    }

    uint64_t records = 0;
    uint64_t warps = 0;
    uint64_t ctas = 0;
    while (auto *tb = kernel->tracez_reader->get_next_tb_index()) {
      ++ctas;
      for (const auto &warp : tb->warps) {
        WarpTraceStream stream;
        stream.init(kernel->tracez_reader->fd(), warp,
                    kernel->tracez_reader->ddict(), kernel->trace_verion,
                    kernel->enable_lineinfo,
                    dim3(tb->tb_id_x, tb->tb_id_y, tb->tb_id_z),
                    dim3(tb->cluster_cta_id_x, tb->cluster_cta_id_y,
                         tb->cluster_cta_id_z),
                    dim3(tb->cluster_id_x, tb->cluster_id_y, tb->cluster_id_z),
                    tb->cluster_rank);
        for (uint32_t i = 0; i < stream.total_inst_count(); ++i) {
          if (stream.get(i).opcode.empty()) {
            std::cerr << "empty opcode in " << command.command_string << "\n";
            parser.kernel_finalizer(kernel);
            return 4;
          }
          ++records;
        }
        ++warps;
      }
    }

    std::cout << "kernel=" << kernel->kernel_name << " ctas=" << ctas
              << " warps=" << warps << " records=" << records << "\n";
    parser.kernel_finalizer(kernel);
  }
}

#include "trace_parser.h"
#include "warp_trace_stream.h"

#include <cstdint>
#include <iostream>
#include <map>
#include <string>

int main(int argc, char **argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--details")) {
    std::cerr << "usage: tracez_probe <kernelslist.g> [--details]\n";
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

    std::map<std::string, uint64_t> opcodes;
    uint64_t tma_records = 0;
    uint64_t tma_active = 0;
    uint64_t tma_missing_payload = 0;
    uint64_t tma_with_addrs = 0;
    uint64_t tma_with_bytes = 0;
    uint64_t syncs_records = 0;
    uint64_t syncs_with_memadd = 0;
    uint64_t syncs_with_src_vals = 0;
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
          const auto &inst = stream.get(i);
          ++opcodes[inst.get_opcode_tokens().front()];
          const auto base = inst.get_opcode_tokens().front();
          if (base == "UTMALDG" || base == "UTMASTG" ||
              base == "UTMAREDG" || base == "UBLKCP") {
            ++tma_records;
            if (inst.mask != 0) {
              ++tma_active;
              if (!inst.tma_memadd_info ||
                  inst.tma_memadd_info->addrs.empty() ||
                  inst.tma_byte_count == 0) {
                ++tma_missing_payload;
              }
            }
            tma_with_addrs += inst.tma_memadd_info &&
                              !inst.tma_memadd_info->addrs.empty();
            tma_with_bytes += inst.tma_byte_count > 0;
            if (argc == 3) {
              std::cout << "tma pc=0x" << std::hex << inst.m_pc << std::dec
                        << " opcode=" << inst.opcode
                        << " mask=0x" << std::hex << inst.mask << std::dec
                        << " addrs=" << (inst.tma_memadd_info ?
                            inst.tma_memadd_info->addrs.size() : 0)
                        << " width=" << (inst.tma_memadd_info ?
                            inst.tma_memadd_info->width : 0)
                        << " bytes=" << inst.tma_byte_count
                        << " oob=" << inst.tma_oob_byte_count
                        << " mbar=" << inst.tma_mbar_addr << "\n";
            }
          }
          if (base == "SYNCS") {
            ++syncs_records;
            syncs_with_memadd += inst.memadd_info != nullptr;
            syncs_with_src_vals += !inst.reg_src_vals.empty();
            if (argc == 3) {
              std::cout << "syncs pc=0x" << std::hex << inst.m_pc << std::dec
                        << " opcode=" << inst.opcode
                        << " memadd=" << (inst.memadd_info != nullptr)
                        << " srcvals=" << inst.reg_src_vals.size()
                        << " dstvals=" << inst.reg_dest_vals.size()
                        << " imm=" << inst.imm << "\n";
            }
          }
          ++records;
        }
        ++warps;
      }
    }

    std::cout << "kernel=" << kernel->kernel_name << " ctas=" << ctas
              << " warps=" << warps << " records=" << records << "\n";
    std::cout << "tma records=" << tma_records
              << " active=" << tma_active
              << " missing_payload=" << tma_missing_payload
              << " with_addrs=" << tma_with_addrs
              << " with_bytes=" << tma_with_bytes << "\n";
    std::cout << "syncs records=" << syncs_records
              << " with_memadd=" << syncs_with_memadd
              << " with_src_vals=" << syncs_with_src_vals << "\n";
    if (argc == 3) {
      for (const auto &[name, count] : opcodes)
        std::cout << "opcode " << name << "=" << count << "\n";
    }
    parser.kernel_finalizer(kernel);
    if (tma_missing_payload) {
      std::cerr << "active TMA instructions lack replay payload\n";
      return 5;
    }
  }
}

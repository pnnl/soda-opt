//===- SyscBackend.cpp - Shared code of the SystemC backends --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SyscBackend.h"

#include <cstdlib>

/// libsystemc references sc_main. The simulation is driven by the runtime
/// calls instead, so it is never called.
int sc_main(int, char *[]) { return 0; }

double soda::axi::getSyscBudgetNs() {
  static const double budget = [] {
    if (const char *value = std::getenv("SODA_AXI_SYSC_MAX_NS")) {
      double parsed = std::strtod(value, nullptr);
      if (parsed > 0)
        return parsed;
    }
    return 1e9;
  }();
  return budget;
}

void soda::axi::configureSyscReports() {
  using namespace sc_core;
  sc_report_handler::set_actions("/IEEE_Std_1666/deprecated", SC_DO_NOTHING);
  sc_report_handler::set_actions(SC_ID_LOGIC_X_TO_BOOL_, SC_LOG);
  sc_report_handler::set_actions(SC_ID_VECTOR_CONTAINS_LOGIC_VALUE_, SC_LOG);
}

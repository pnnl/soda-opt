# Finds the SystemC library.
#
# Search order:
#   1. A SystemC CMake package (SystemCLanguageConfig.cmake), e.g. a SystemC
#      installed with CMake. Point CMAKE_PREFIX_PATH or SystemCLanguage_DIR at it.
#   2. pkg-config (systemc.pc), as installed by distribution packages.
#   3. The headers and library under SYSTEMC_HOME (CMake or environment
#      variable), in include/ and lib/, lib64/ or lib-linux64/.
#
# Defines:
#   SystemC_FOUND        - True if SystemC was found.
#   SystemC_VERSION      - The version, when known.
#   SystemC::SystemC     - Imported target to link against.

if(TARGET SystemC::SystemC)
  set(SystemC_FOUND TRUE)
  return()
endif()

find_package(SystemCLanguage CONFIG QUIET)
if(SystemCLanguage_FOUND AND TARGET SystemC::systemc)
  add_library(SystemC::SystemC INTERFACE IMPORTED)
  target_link_libraries(SystemC::SystemC INTERFACE SystemC::systemc)
  set(SystemC_VERSION "${SystemCLanguage_VERSION}")
  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(SystemC
    REQUIRED_VARS SystemCLanguage_DIR
    VERSION_VAR SystemC_VERSION)
  return()
endif()

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  pkg_check_modules(PC_SYSTEMC QUIET systemc)
endif()

set(_systemc_hints)
if(SYSTEMC_HOME)
  list(APPEND _systemc_hints "${SYSTEMC_HOME}")
endif()
if(DEFINED ENV{SYSTEMC_HOME})
  list(APPEND _systemc_hints "$ENV{SYSTEMC_HOME}")
endif()

find_path(SystemC_INCLUDE_DIR systemc.h
  HINTS ${_systemc_hints} ${PC_SYSTEMC_INCLUDEDIR} ${PC_SYSTEMC_INCLUDE_DIRS}
  PATH_SUFFIXES include)
find_library(SystemC_LIBRARY NAMES systemc
  HINTS ${_systemc_hints} ${PC_SYSTEMC_LIBDIR} ${PC_SYSTEMC_LIBRARY_DIRS}
  PATH_SUFFIXES lib lib64 lib-linux64)

if(PC_SYSTEMC_VERSION)
  set(SystemC_VERSION "${PC_SYSTEMC_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SystemC
  REQUIRED_VARS SystemC_LIBRARY SystemC_INCLUDE_DIR
  VERSION_VAR SystemC_VERSION)

if(SystemC_FOUND)
  add_library(SystemC::SystemC UNKNOWN IMPORTED)
  set_target_properties(SystemC::SystemC PROPERTIES
    IMPORTED_LOCATION "${SystemC_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SystemC_INCLUDE_DIR}")
  find_package(Threads QUIET)
  if(Threads_FOUND)
    set_property(TARGET SystemC::SystemC APPEND PROPERTY
      INTERFACE_LINK_LIBRARIES Threads::Threads)
  endif()
endif()

mark_as_advanced(SystemC_INCLUDE_DIR SystemC_LIBRARY)

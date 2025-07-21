add_subdirectory(TestLib)

configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TPP_OPT_TEST_DEPENDS
        FileCheck count not
        mlir-gen
        tpp-opt
        tpp-run
        tpp-sched
        fpcmp
	check-cpuid
        )

add_lit_testsuite(check-tpp "Running the regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${TPP_OPT_TEST_DEPENDS}
        )
set_target_properties(check-tpp PROPERTIES FOLDER "Tests")

add_lit_testsuites(TPP_OPT ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TPP_OPT_TEST_DEPENDS})

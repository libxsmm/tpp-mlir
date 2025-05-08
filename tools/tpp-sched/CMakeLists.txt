file(MAKE_DIRECTORY
     ${CMAKE_BINARY_DIR}/bin)
file(CREATE_LINK
     ${CMAKE_CURRENT_SOURCE_DIR}/tpp-sched.py
     ${CMAKE_BINARY_DIR}/bin/tpp-sched
     SYMBOLIC)


add_custom_target(tpp-sched DEPENDS ${CMAKE_BINARY_DIR}/bin/tpp-sched)

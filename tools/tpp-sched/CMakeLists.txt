#configure_file(
#        ${CMAKE_CURRENT_SOURCE_DIR}/tpp-sched.py
#        ${CMAKE_BINARY_DIR}/bin/tpp-sched.py
#        COPYONLY)

file(MAKE_DIRECTORY
     ${CMAKE_BINARY_DIR}/bin)
file(CREATE_LINK
     ${CMAKE_CURRENT_SOURCE_DIR}/tpp-sched.py
     ${CMAKE_BINARY_DIR}/bin/tpp-sched.py
     SYMBOLIC)

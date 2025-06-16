file(MAKE_DIRECTORY
     ${CMAKE_BINARY_DIR}/bin)
file(CREATE_LINK
     ${CMAKE_CURRENT_SOURCE_DIR}/tpp-tune.py
     ${CMAKE_BINARY_DIR}/bin/tpp-tune
     SYMBOLIC)


add_custom_target(tpp-tune DEPENDS ${CMAKE_BINARY_DIR}/bin/tpp-tune TPPPythonModules)

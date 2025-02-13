function (add_exec source output)
  add_executable(${output} ${source})
  target_link_libraries(${output} PRIVATE ui_project_options ui_project_warnings ui_core)
endfunction()

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -save-temps -fno-verbose-asm -ffunction-sections")
add_exec("example_1.cpp" example_1)

add_library(fl_core STATIC
    AllocationCounter.cpp
    Assert.cpp
    Log.cpp
    LinearAllocator.cpp
    Math.cpp)
target_include_directories(fl_core PUBLIC "${CMAKE_SOURCE_DIR}/engine")
target_link_libraries(fl_core PUBLIC glm::glm)
target_compile_definitions(fl_core PUBLIC
    FL_DEV=$<BOOL:${FL_DEV}>
    FL_TRACY=$<BOOL:${FL_TRACY}>)
if(FL_TRACY)
    target_link_libraries(fl_core PUBLIC Tracy::TracyClient)
endif()
fl_target_defaults(fl_core)

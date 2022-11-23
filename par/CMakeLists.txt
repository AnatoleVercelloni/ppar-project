cmake_minimum_required(VERSION 3.10)

# set the project name
project(model LANGUAGES C)
project(validate LANGUAGES C)

set(APP_EXE modelslice)
set(APP_EXE2 validate)
set(APP_EXE3 mdoelcycle)


# add the executable
add_executable(${APP_EXE} 
    model_par3.c
    harmonics.c
)
add_executable(${APP_EXE2} 
    validate.c
    harmonics.c
)

add_executable(${APP_EXE3} 
    model_par4.c
    harmonics.c
)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(${APP_EXE} PUBLIC MPI::MPI_C)

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
target_link_libraries(${APP_EXE3} PUBLIC MPI::MPI_C)


target_link_libraries(${APP_EXE} PRIVATE m)

target_link_libraries(${APP_EXE2} PRIVATE m)

target_link_libraries(${APP_EXE3} PRIVATE m)

target_compile_options(${APP_EXE}
    PRIVATE -Wall
)


target_compile_options( ${APP_EXE2}
    PRIVATE -Wall 
)

target_compile_options( ${APP_EXE2}
    PRIVATE -Wall 
)
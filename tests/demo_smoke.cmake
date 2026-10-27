file(MAKE_DIRECTORY "${SMOKE_DIR}/captures")
set(capture "${SMOKE_DIR}/captures/water.png")
file(REMOVE "${capture}")
execute_process(
    COMMAND "${CMAKE_COMMAND}" -E env
        WATER_VALIDATE_SECONDS=1 WATER_VALIDATE_WARMUP=0
        "WATER_VALIDATE_CAPTURE=${capture}" "${DEMO}"
    WORKING_DIRECTORY "${SMOKE_DIR}"
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors
    TIMEOUT 20)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Desktop demo failed (${result}):\n${output}\n${errors}")
endif()
if(NOT output MATCHES "VALIDATION 512x512")
    message(FATAL_ERROR "Desktop demo did not finish its frame samples:\n${output}")
endif()
if(NOT EXISTS "${capture}")
    message(FATAL_ERROR "Desktop demo did not save the requested capture: ${capture}\n${output}")
endif()

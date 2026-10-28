file(MAKE_DIRECTORY "${SMOKE_DIR}/captures")
get_filename_component(demo_dir "${DEMO}" DIRECTORY)
if(NOT EXISTS "${demo_dir}/assets/grass.png")
    message(FATAL_ERROR "Desktop demo is missing its grass texture")
endif()
execute_process(COMMAND "${DEMO}" --help OUTPUT_VARIABLE help RESULT_VARIABLE result)
if(NOT result EQUAL 0 OR NOT help MATCHES "--reflection" OR NOT help MATCHES "reflection.*refraction"
   OR NOT help MATCHES "--reflection 0.7"
   OR NOT help MATCHES "[+] / - reflection/refraction"
   OR NOT help MATCHES "W / S forward/reverse" OR NOT help MATCHES "A / D rudder")
    message(FATAL_ERROR "Desktop demo help does not explain appearance and boat controls:\n${help}")
endif()
foreach(mix IN ITEMS 0 1)
    set(capture "${SMOKE_DIR}/captures/water-${mix}.png")
    file(REMOVE "${capture}")
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env
            WATER_VALIDATE_SECONDS=1 WATER_VALIDATE_WARMUP=0
            "WATER_VALIDATE_CAPTURE=${capture}" "${DEMO}" --reflection "${mix}"
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
endforeach()

project('vision-core', 'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++17',
        'warning_level=3',
        'werror=true'
    ]
)

# Include directory
inc_dir = include_directories('include')

# Core dependencies
gtest_dep = dependency('gtest', required: true)
gtest_main_dep = dependency('gtest_main', required: true)

spdlog_dep = dependency('spdlog', required: true)
json_dep = dependency('nlohmann_json', required: true)
opencv_dep = dependency('opencv4', modules: ['core'], required: true)

# Header-only library dependency that will be exported
vision_core_dep = declare_dependency(
    include_directories: inc_dir,
    dependencies: [
        spdlog_dep,
        json_dep,
        opencv_dep
    ]
)

# Test executables
test_sources = [
    'tests/frame_test.cpp',
    'tests/detection_test.cpp',
    'tests/vector_utils_test.cpp',
    'tests/geometry_utils_test.cpp',
    'tests/detection_utils_test.cpp'
]

test_exe = executable('vision_core_tests', 
    test_sources,
    include_directories: inc_dir,
    dependencies: [
        vision_core_dep,
        gtest_dep,
        gtest_main_dep
    ]
)

test('vision_core_tests', test_exe)
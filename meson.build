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
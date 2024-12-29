# Vision Core

Core utilities and data structures for computer vision applications.

## Overview
This header-only library provides common building blocks for computer vision applications, particularly focused on object detection, tracking, and classification pipelines. It includes standardized data structures and utility functions to ensure consistency across projects.

## Features
- **Type Definitions**: Standard data structures for computer vision applications
  - Detection and tracking primitives (bounding boxes, tracks)
  - Frame and image metadata
  - Common geometry types

- **Utility Functions**: 
  - JSON serialization/deserialization
  - Vector operations and manipulations
  - Geometry calculations (IoU, distances)
  - Common preprocessing and validation functions

## Usage
This library serves as a foundation for vision applications, ensuring consistent data handling and reducing code duplication across projects. It is designed to be lightweight, efficient, and easy to integrate into existing codebases.

## Requirements
- C++17 or higher
- Meson build system
- Dependencies:
  - nlohmann_json
  - spdlog (for logging utilities)

## Integration
Add as a subproject in your Meson project:
```ini
[wrap-git]
directory = vision-core
url = https://github.com/tensorworksio/vision-core.git
revision = main
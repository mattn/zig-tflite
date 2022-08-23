const builtin = @import("builtin");
const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    var tflitePkg = std.build.Pkg{
        .name = "zig-tflite",
        .source = std.build.FileSource{ .path = "src/main.zig" },
    };

    const lib = b.addStaticLibrary("zig-tflite", "src/main.zig");
    lib.setBuildMode(mode);
    if (builtin.os.tag == .windows) {
        lib.include_dirs.append(.{ .raw_path = "c:/msys64/mingw64/include" }) catch unreachable;
        lib.lib_paths.append("c:/msys64/mingw64/lib") catch unreachable;
    }
    lib.install();

    const exe = b.addExecutable("fizzbuzz", "example/fizzbuzz.zig");
    if (builtin.os.tag == .windows) {
        exe.include_dirs.append(.{ .raw_path = "c:/msys64/mingw64/include" }) catch unreachable;
        exe.lib_paths.append("c:/msys64/mingw64/lib") catch unreachable;
    }
    exe.setBuildMode(mode);
    exe.addPackage(tflitePkg);
    exe.linkLibrary(lib);
    exe.linkSystemLibrary("tensorflowlite-delegate_xnnpack");
    exe.linkSystemLibrary("tensorflowlite_c");
    exe.linkSystemLibrary("c");
    b.default_step.dependOn(&exe.step);
    exe.install();

    const main_tests = b.addTest("src/main.zig");
    main_tests.setBuildMode(mode);
    if (builtin.os.tag == .windows) {
        main_tests.include_dirs.append(.{ .raw_path = "c:/msys64/mingw64/include" }) catch unreachable;
        main_tests.lib_paths.append("c:/msys64/mingw64/lib") catch unreachable;
    }
    main_tests.linkSystemLibrary("tensorflowlite-delegate_xnnpack");
    main_tests.linkSystemLibrary("tensorflowlite_c");
    main_tests.linkSystemLibrary("c");

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);
}

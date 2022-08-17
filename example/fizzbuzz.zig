const std = @import("std");
const tflite = @import("zig-tflite");

fn bin(n: u64, f: []f32) void {
    var i: u6 = 0;
    while (i < f.len) : (i += 1) {
        var a: u8 = @truncate(u8, n >> i);
        f[i] = @intToFloat(f32, @intCast(i32, a) & 1);
    }
}

fn dec(f: []f32) usize {
    var i: usize = 0;
    while (i < f.len) : (i += 1) {
        if (f[i] > 0.4) {
            return i;
        }
    }
    @panic("Sorry, I'm wrong");
}

pub fn main() anyerror!void {
    var m = try tflite.modelFromFile("testdata/fizzbuzz_model.tflite");
    defer m.deinit();

    var o = try tflite.interpreterOptions();
    defer o.deinit();

    var i = try tflite.interpreter(m, o);
    defer i.deinit();

    try i.allocateTensors();

    var inputTensor = i.inputTensor(0);
    var outputTensor = i.outputTensor(0);

    var input = inputTensor.data(f32);
    var output = outputTensor.data(f32);

    var writer = std.io.getStdOut().writer();

    var n: u64 = 1;
    while (n < 100) : (n += 1) {
        bin(n, input);
        try i.invoke();
        var r = dec(output);
        switch (r) {
            0 => {
                try writer.print("{}\n", .{n});
            },
            1 => {
                try writer.print("{s}\n", .{"Fizz"});
            },
            2 => {
                try writer.print("{s}\n", .{"Buzz"});
            },
            3 => {
                try writer.print("{s}\n", .{"FizzBuzz"});
            },
            else => {},
        }
    }
}

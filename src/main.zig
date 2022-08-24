const std = @import("std");
const c = @cImport({
    @cDefine("WIN32_LEAN_AND_MEAN", "1");
    @cInclude("tensorflow/lite/c/c_api.h");
    @cInclude("string.h");
});

const Status = enum(u32) {
    OK = 0,
    Error = 1,
};

pub const RuntimeError = error{
    IllegalArgumentError,
};

const TensorType = enum(u32) {
    NoType = 0,
    Float32 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Complex64 = 8,
    Int8 = 9,
};

const Model = struct {
    m: *c.TfLiteModel,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        c.TfLiteModelDelete(self.m);
    }
};

pub fn modelFromData(data: []const u8) !Model {
    var m = c.TfLiteModelCreate(@ptrCast([*]const u8, data), data.len);
    if (m == null) {
        return error.AllocationError;
    }
    return Model{ .m = m.? };
}

pub fn modelFromFile(path: []const u8) !Model {
    var m = c.TfLiteModelCreateFromFile(@ptrCast([*]const u8, path));
    if (m == null) {
        return error.AllocationError;
    }
    return Model{ .m = m.? };
}

const InterpreterOptions = struct {
    o: *c.TfLiteInterpreterOptions,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        c.TfLiteInterpreterOptionsDelete(self.o);
    }

    pub fn setNumThreads(self: *Self, num_threads: i32) void {
        c.TfLiteInterpreterOptionsSetNumThreads(self.o, num_threads);
    }

    pub fn addDelegate(self: *Self, d: anytype) void {
        c.TfLiteInterpreterOptionsAddDelegate(self.o, @ptrCast(*c.TfLiteDelegate, d));
    }
};

pub fn interpreterOptions() !InterpreterOptions {
    var o = c.TfLiteInterpreterOptionsCreate();
    if (o == null) {
        return error.AllocationError;
    }
    return InterpreterOptions{ .o = o.? };
}

const Interpreter = struct {
    i: *c.TfLiteInterpreter,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        c.TfLiteInterpreterDelete(self.i);
    }

    pub fn allocateTensors(self: *Self) !void {
        if (@intToEnum(Status, c.TfLiteInterpreterAllocateTensors(self.i)) != Status.OK) {
            return error.AllocationError;
        }
    }

    pub fn invoke(self: *Self) !void {
        if (@intToEnum(Status, c.TfLiteInterpreterInvoke(self.i)) != Status.OK) {
            return error.RuntimeError;
        }
    }

    pub fn inputTensorCount(self: *Self) i32 {
        return c.TfLiteInterpreterGetInputTensorCount(self.i);
    }

    pub fn inputTensor(self: *Self, index: i32) Tensor {
        return Tensor{
            .t = c.TfLiteInterpreterGetInputTensor(self.i, index).?,
        };
    }

    pub fn outputTensorCount(self: *Self) i32 {
        return c.TfLiteInterpreterGetOutputTensorCount(self.i);
    }

    pub fn outputTensor(self: *Self, index: i32) Tensor {
        return Tensor{
            .t = c.TfLiteInterpreterGetOutputTensor(self.i, index).?,
        };
    }
};

pub fn interpreter(model: Model, options: InterpreterOptions) !Interpreter {
    var i = c.TfLiteInterpreterCreate(model.m, options.o);
    if (i == null) {
        return error.AllocationError;
    }
    return Interpreter{ .i = i.? };
}

const Tensor = struct {
    t: *const c.TfLiteTensor,

    const Self = @This();

    pub fn tensorType(self: *Self) TensorType {
        return @intToEnum(TensorType, c.TfLiteTensorType(self.t));
    }

    pub fn numDims(self: *Self) i32 {
        return c.TfLiteTensorNumDims(self.t);
    }

    pub fn dim(self: *Self, index: i32) i32 {
        return c.TfLiteTensorDim(self.t, index);
    }

    pub fn shape(self: *Self, allocator: std.mem.Allocator) !std.ArrayList(i32) {
        var s = std.ArrayList(i32).init(allocator);
        var i: i32 = 0;
        while (i < self.numDims()) : (i += 1) {
            try s.append(self.dim(i));
        }
        return s;
    }

    pub fn byteSize(self: *Self) usize {
        return c.TfLiteTensorByteSize(self.t);
    }

    pub fn data(self: *Self, comptime T: type) []T {
        var d = c.TfLiteTensorData(self.t);
        var a = c.TfLiteTensorByteSize(self.t) / @sizeOf(T);
        return @ptrCast([*]T, @alignCast(@alignOf(T), d.?))[0..a];
    }

    pub fn name(self: *Self) []const u8 {
        var n = c.TfLiteTensorName(self.t);
        var len = c.strlen(n);
        return n[0..len];
    }
};

test "basic test" {
    var allocator = std.testing.allocator;

    var m = try modelFromFile("testdata/xor_model.tflite");
    defer m.deinit();

    var o = try interpreterOptions();
    defer o.deinit();

    o.setNumThreads(4);

    var i = try interpreter(m, o);
    defer i.deinit();

    try i.allocateTensors();

    try std.testing.expectEqual(@as(i32, 1), i.inputTensorCount());
    try std.testing.expectEqual(@as(i32, 1), i.outputTensorCount());

    var inputTensor = i.inputTensor(0);
    var outputTensor = i.outputTensor(0);

    try std.testing.expectEqual(TensorType.Float32, inputTensor.tensorType());
    try std.testing.expectEqual(TensorType.Float32, outputTensor.tensorType());
    try std.testing.expectEqual(@as(i32, 2), inputTensor.numDims());
    try std.testing.expectEqual(@as(i32, 2), outputTensor.numDims());
    try std.testing.expectEqual(@as(i32, 1), inputTensor.dim(0));
    try std.testing.expectEqual(@as(i32, 2), inputTensor.dim(1));
    try std.testing.expectEqual(@as(i32, 1), outputTensor.dim(0));
    try std.testing.expectEqual(@as(i32, 1), outputTensor.dim(1));
    try std.testing.expectEqual(@as(usize, 8), inputTensor.byteSize());
    try std.testing.expectEqual(@as(usize, 4), outputTensor.byteSize());
    try std.testing.expectEqualStrings("serving_default_dense_input:0", inputTensor.name());
    try std.testing.expectEqualStrings("StatefulPartitionedCall:0", outputTensor.name());

    var shape = try inputTensor.shape(allocator);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 2 }, shape.items);
    shape.deinit();
    shape = try outputTensor.shape(allocator);
    try std.testing.expectEqualSlices(i32, &[_]i32{ 1, 1 }, shape.items);
    shape.deinit();

    var input = inputTensor.data(f32);
    var output = outputTensor.data(f32);

    try std.testing.expectEqual(@as(usize, 2), input.len);
    try std.testing.expectEqual(@as(usize, 1), output.len);

    const T = struct { input: []const f32, want: f32 };
    var tests = [_]T{
        .{ .input = &.{ 0, 0 }, .want = 0 },
        .{ .input = &.{ 1, 0 }, .want = 1 },
        .{ .input = &.{ 0, 1 }, .want = 1 },
        .{ .input = &.{ 1, 1 }, .want = 0 },
    };
    for (tests) |item| {
        input[0] = item.input[0];
        input[1] = item.input[1];
        try i.invoke();
        var result: f32 = if (output[0] > 0.5) 1 else 0;
        try std.testing.expectEqual(item.want, result);
    }
}

test "test modelFromData" {
    var allocator = std.testing.allocator;

    const model = try std.fs.cwd().readFileAlloc(allocator, "testdata/xor_model.tflite", 1024 * 1024);
    defer allocator.free(model);
    var m = try modelFromData(model);
    defer m.deinit();
    _ = m;
}

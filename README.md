# zig-tflite

Zi binding for TensorFlow Lite

## Usage

```zig
const std = @import("std");
const tflite = @import("zig-tflite");

pub fn main() anyerror!void {
    var m = try tflite.modelFromFile("testdata/xor_model.tflite");
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

    input[0] = 0;
    input[1] = 1;

    try i.invoke();

    var result: f32 = if (output[0] > 0.5) 1 else 0;
    std.log.warn("0 xor 1 = {}", .{result});
}
```

## Requirements

* TensorFlow Lite - This release requires 2.2.0-rc3

## Tensorflow Installation

You must install Tensorflow Lite C API. Assuming the source is under /source/directory/tensorflow

```
$ cd /source/directory/tensorflow
$ bazel build --config opt --config monolithic tensorflow:libtensorflow_c.so
```

Or to just compile the tensorflow lite libraries:
```
$ cd /some/path/tensorflow
$ bazel build --config opt --config monolithic //tensorflow/lite:libtensorflowlite.so
$ bazel build --config opt --config monolithic //tensorflow/lite/c:libtensorflowlite_c.so
```

## Installation

```
$ zig build
```

## License

MIT

## Author

Yasuhiro Matsumoto (a.k.a. mattn)

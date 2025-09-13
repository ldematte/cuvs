/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.cuvs;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Random;
import java.util.Set;

import static java.nio.file.StandardOpenOption.READ;

@Fork(value = 1, warmups = 0)
@State(Scope.Benchmark)
public class DatasetToDeviceBenchmarks {

  @Param({"2048"})
  private int dims;

  @Param({"16384"}) // 524288
  private int size;

  private static final Random random = new Random();

  private float[][] data;

  private Path preloadedVectorDataFile;

  private CuVSResources resources;

  private float[][] createRandomData() {
    var array = new float[size][dims];

    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < dims; ++j) {
        array[i][j] = random.nextFloat();
      }
    }
    return array;
  }

  @Setup
  public void initialize() throws Throwable {
    data = createRandomData();
    resources = CuVSResources.create();
    // Serialize to tmp file
    preloadedVectorDataFile = Files.createTempFile("vec_", "_preloaded");
    writeToFile(preloadedVectorDataFile);
  }

  @TearDown
  public void cleanUp() throws IOException {
    if (resources != null) {
      resources.close();
    }
    if (preloadedVectorDataFile != null) {
      Files.deleteIfExists(preloadedVectorDataFile);
    }
  }

  private void writeToFile(Path vectorDataFile) throws IOException {
    try (var output = Files.newOutputStream(vectorDataFile)) {
      var buffer = ByteBuffer.allocate(dims * Float.BYTES);
      for (int i = 0; i < data.length; ++i) {
        buffer.asFloatBuffer().put(data[i]);
        output.write(buffer.array());
      }
    }
  }

  @Benchmark
  public void serializePlusHeapToHostToDevice(Blackhole bh) throws IOException {
    // serialize to file
    var vectorDataFile = Files.createTempFile("vec_", "");
    writeToFile(vectorDataFile);

    // transfer heap to host matrix
    var builder = CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder.addVector(array);
    }
    try (var hostMatrix = builder.build()) {
      // transfer host to device
      hostMatrix.toDevice(resources).close();
    }

    Files.deleteIfExists(vectorDataFile);
  }

  @Benchmark
  public void serializePlusTmpFileMMapToDevice(Blackhole bh) throws IOException {
    // serialize to file
    var vectorDataFile = Files.createTempFile("vec_", "");
    writeToFile(vectorDataFile);

    // Also to tmp file
    var tmpVectorDataFile = Files.createTempFile("vec_", "");
    writeToFile(tmpVectorDataFile);

    // Map the tmp file to memory
    try (var fc = FileChannel.open(tmpVectorDataFile, Set.of(READ));
         Arena arena = Arena.ofConfined()) {

      MemorySegment mapped = fc.map(FileChannel.MapMode.READ_ONLY, 0L, size * dims * Float.BYTES, arena);

      try (var hostMatrix = DatasetHelper.fromMemorySegment(mapped, size, dims, CuVSMatrix.DataType.FLOAT)) {
        // transfer memory-mapped file to device
        hostMatrix.toDevice(resources).close();
      }
    }

    Files.deleteIfExists(vectorDataFile);
    Files.deleteIfExists(tmpVectorDataFile);
  }

  @Benchmark
  public void serializePlusHeapToDevice(Blackhole bh) throws IOException {
    // serialize to file
    var vectorDataFile = Files.createTempFile("vec_", "");
    writeToFile(vectorDataFile);

    // Direct heap -> device
    var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder.addVector(array);
    }
    CuVSDeviceMatrix matrix = builder.build();
    matrix.close();

    Files.deleteIfExists(vectorDataFile);
  }

  @Benchmark
  public void transferHeapToHostToDevice(Blackhole bh) throws IOException {
    // transfer heap to host matrix
    var builder = CuVSMatrix.hostBuilder(size, dims, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder.addVector(array);
    }
    try (var hostMatrix = builder.build()) {
      // transfer host to device
      hostMatrix.toDevice(resources).close();
    }
  }

  @Benchmark
  public void transferTmpFileMMapToDevice(Blackhole bh) throws IOException {
    // Serialize to tmp file
    var tmpVectorDataFile = Files.createTempFile("vec_", "");
    writeToFile(tmpVectorDataFile);

    // Map the tmp file to memory
    try (var fc = FileChannel.open(tmpVectorDataFile, Set.of(READ));
         Arena arena = Arena.ofConfined()) {

      MemorySegment mapped = fc.map(FileChannel.MapMode.READ_ONLY, 0L, size * dims * Float.BYTES, arena);

      try (var hostMatrix = DatasetHelper.fromMemorySegment(mapped, size, dims, CuVSMatrix.DataType.FLOAT)) {
        // transfer memory-mapped file to device
        hostMatrix.toDevice(resources).close();
      }
    }

    Files.deleteIfExists(tmpVectorDataFile);
  }

  @Benchmark
  public void transferPreloadedMMapFileToDevice(Blackhole bh) throws IOException {
    // Map the tmp file to memory
    try (var fc = FileChannel.open(preloadedVectorDataFile, Set.of(READ));
         Arena arena = Arena.ofConfined()) {

      MemorySegment mapped = fc.map(FileChannel.MapMode.READ_ONLY, 0L, size * dims * Float.BYTES, arena);

      try (var hostMatrix = DatasetHelper.fromMemorySegment(mapped, size, dims, CuVSMatrix.DataType.FLOAT)) {
        // transfer memory-mapped file to device
        hostMatrix.toDevice(resources).close();
      }
    }
  }

  @Benchmark
  public void transferHeapToDevice(Blackhole bh) throws IOException {
    // Direct heap -> device
    var builder = CuVSMatrix.deviceBuilder(resources, size, dims, CuVSMatrix.DataType.FLOAT);
    for (int i = 0; i < size; ++i) {
      var array = data[i];
      builder.addVector(array);
    }
    CuVSDeviceMatrix matrix = builder.build();
    matrix.close();
  }
}

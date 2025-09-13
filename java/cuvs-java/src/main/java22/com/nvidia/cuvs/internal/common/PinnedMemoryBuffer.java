package com.nvidia.cuvs.internal.common;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static com.nvidia.cuvs.internal.common.LinkerHelper.C_POINTER;
import static com.nvidia.cuvs.internal.common.Util.checkCudaError;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaFreeHost;
import static com.nvidia.cuvs.internal.panama.headers_h_1.cudaMallocHost;

public class PinnedMemoryBuffer implements AutoCloseable {

  private static final int CHUNK_BYTES =
          8 * 1024 * 1024; // Based on benchmarks, 8MB seems the minimum size to optimize PCIe bandwidth
  private final long hostBufferBytes;

  private MemorySegment hostBuffer = MemorySegment.NULL;

  public PinnedMemoryBuffer(
          long size,
          long columns,
          ValueLayout valueLayout) {

    long rowBytes = columns * valueLayout.byteSize();
    long matrixBytes = size * rowBytes;
    if (matrixBytes < CHUNK_BYTES) {
      this.hostBufferBytes = matrixBytes;
    } else if (rowBytes > CHUNK_BYTES) {
      // We need to buffer at least one row at time
      this.hostBufferBytes = rowBytes;
    } else {
      var rowCount = (CHUNK_BYTES / rowBytes);
      this.hostBufferBytes = rowBytes * rowCount;
    }
  }


  private static MemorySegment createPinnedBuffer(long bufferBytes) {
    try (var localArena = Arena.ofConfined()) {
      MemorySegment pointer = localArena.allocate(C_POINTER);
      checkCudaError(cudaMallocHost(pointer, bufferBytes), "cudaMallocHost");
      return pointer.get(C_POINTER, 0);
    }
  }

  private static void destroyPinnedBuffer(MemorySegment bufferSegment) {
    checkCudaError(cudaFreeHost(bufferSegment), "cudaFreeHost");
  }

  public MemorySegment address() {
    if (hostBuffer == MemorySegment.NULL) {
      //      System.out.println("Creating a buffer of size " + hostBufferBytes);
      hostBuffer = createPinnedBuffer(hostBufferBytes);
    }
    return hostBuffer;
  }

  public long size() {
    return hostBufferBytes;
  }

  @Override
  public void close() {
    if (hostBuffer != MemorySegment.NULL) {
      destroyPinnedBuffer(hostBuffer);
      hostBuffer = MemorySegment.NULL;
    }
  }
}

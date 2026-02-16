"""Benchmarking framework for pose estimation performance profiling."""

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import psutil
import torch


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    num_runs: int
    latencies_ms: list[float]
    memory_mb: float
    
    @property
    def avg_latency_ms(self) -> float:
        return float(np.mean(self.latencies_ms))
    
    @property
    def median_latency_ms(self) -> float:
        return float(np.median(self.latencies_ms))
    
    @property
    def p95_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 95))
    
    @property
    def p99_latency_ms(self) -> float:
        return float(np.percentile(self.latencies_ms, 99))
    
    @property
    def min_latency_ms(self) -> float:
        return float(np.min(self.latencies_ms))
    
    @property
    def max_latency_ms(self) -> float:
        return float(np.max(self.latencies_ms))
    
    @property
    def std_latency_ms(self) -> float:
        return float(np.std(self.latencies_ms))
    
    @property
    def fps(self) -> float:
        return 1000.0 / self.avg_latency_ms if self.avg_latency_ms > 0 else 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "num_runs": self.num_runs,
            "avg_latency_ms": self.avg_latency_ms,
            "median_latency_ms": self.median_latency_ms,
            "p95_latency_ms": self.p95_latency_ms,
            "p99_latency_ms": self.p99_latency_ms,
            "min_latency_ms": self.min_latency_ms,
            "max_latency_ms": self.max_latency_ms,
            "std_latency_ms": self.std_latency_ms,
            "fps": self.fps,
            "memory_mb": self.memory_mb,
        }


class Benchmarker:
    """Benchmark inference functions."""
    
    def __init__(self, warmup_runs: int = 10, num_runs: int = 100) -> None:
        """Initialize benchmarker.
        
        Args:
            warmup_runs: Number of warmup runs before measuring.
            num_runs: Number of measurement runs.
        """
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs
    
    def benchmark(
        self,
        inference_fn: Callable,
        test_image: np.ndarray,
        name: str,
        device: str = "cuda",
    ) -> BenchmarkResult:
        """Benchmark an inference function.
        
        Args:
            inference_fn: Function that takes an image and returns predictions.
            test_image: Test image to use for benchmarking.
            name: Name for this benchmark.
            device: Device type ('cuda' or 'cpu').
            
        Returns:
            BenchmarkResult with performance metrics.
        """
        print(f"\nBenchmarking: {name}")
        print(f"  Warmup runs: {self.warmup_runs}")
        print(f"  Measurement runs: {self.num_runs}")
        
        # Warmup
        print("  Running warmup...", end="", flush=True)
        for _ in range(self.warmup_runs):
            _ = inference_fn(test_image)
        
        if device == "cuda":
            torch.cuda.synchronize()
        print(" done")
        
        # Benchmark
        print("  Measuring performance...", end="", flush=True)
        latencies = []
        
        # Get initial memory
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        for i in range(self.num_runs):
            start = time.perf_counter()
            _ = inference_fn(test_image)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = (time.perf_counter() - start) * 1000  # ms
            latencies.append(elapsed)
            
            if (i + 1) % 20 == 0:
                print(f" {i+1}/{self.num_runs}", end="", flush=True)
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = mem_after - mem_before
        
        print(" done")
        
        result = BenchmarkResult(
            name=name,
            num_runs=self.num_runs,
            latencies_ms=latencies,
            memory_mb=memory_used,
        )
        
        print(f"  Avg latency: {result.avg_latency_ms:.2f} ms")
        print(f"  FPS: {result.fps:.2f}")
        print(f"  P95 latency: {result.p95_latency_ms:.2f} ms")
        
        return result

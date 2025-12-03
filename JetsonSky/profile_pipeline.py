"""
JetsonSky Filter Pipeline Profiler

This script profiles individual filter stages in the image processing pipeline
to identify performance bottlenecks.

Usage:
    python profile_pipeline.py [--resolution WxH] [--iterations N]

Example:
    python profile_pipeline.py --resolution 1920x1080 --iterations 50
"""

import time
import argparse
import sys
import os

# Ensure we can import from current directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_image(cp, width, height, is_color=True):
    """Create a synthetic test image for profiling."""
    if is_color:
        # Create 3 separate channel arrays (simulating pipeline format)
        r = cp.random.randint(0, 256, (height, width), dtype=cp.uint8)
        g = cp.random.randint(0, 256, (height, width), dtype=cp.uint8)
        b = cp.random.randint(0, 256, (height, width), dtype=cp.uint8)
        return r, g, b
    else:
        return cp.random.randint(0, 256, (height, width), dtype=cp.uint8)


class PipelineProfiler:
    """Profiles individual filter stages in the JetsonSky pipeline."""

    def __init__(self, width=1920, height=1080, iterations=50):
        self.width = width
        self.height = height
        self.iterations = iterations
        self.results = {}

        # Import dependencies
        import cupy as cp
        import cv2
        import numpy as np

        self.cp = cp
        self.cv2 = cv2
        self.np = np

        # Initialize CUDA stream
        self.stream = cp.cuda.Stream(non_blocking=True)

        # Import kernels
        import cuda_kernels as ck
        self.ck = ck

        print(f"Profiler initialized: {width}x{height}, {iterations} iterations")

    def _sync_gpu(self):
        """Force GPU synchronization for accurate timing."""
        self.cp.cuda.Stream.null.synchronize()

    def time_operation(self, name, func, warmup=5):
        """
        Time a single operation with warmup and multiple iterations.
        """
        # Warmup
        for _ in range(warmup):
            func()
            self._sync_gpu()

        # Timed iterations
        times = []
        for _ in range(self.iterations):
            self._sync_gpu()
            start = time.perf_counter()
            func()
            self._sync_gpu()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        mean_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        self.results[name] = {
            'mean': mean_time,
            'min': min_time,
            'max': max_time,
            'std': (sum((t - mean_time)**2 for t in times) / len(times)) ** 0.5
        }

        return mean_time

    def profile_nlm2_color(self):
        """Profile NLM2 (Non-Local Means) denoising - color."""
        print("Profiling NLM2 Color...")

        r, g, b = create_test_image(self.cp, self.width, self.height)
        dest_r = r.copy()
        dest_g = g.copy()
        dest_b = b.copy()

        # NLM2 uses 8x8 thread blocks
        nb_blocksX = (self.width // 8) + 1
        nb_blocksY = (self.height // 8) + 1

        def run_nlm2():
            self.ck.NLM2_Colour_GPU(
                (nb_blocksX, nb_blocksY), (8, 8),
                (dest_r, dest_g, dest_b, r, g, b,
                 self.np.int32(self.width), self.np.int32(self.height),
                 self.np.float32(10.0), self.np.float32(0.5))
            )

        return self.time_operation("NLM2 Color", run_nlm2)

    def profile_nlm2_mono(self):
        """Profile NLM2 (Non-Local Means) denoising - mono."""
        print("Profiling NLM2 Mono...")

        img = create_test_image(self.cp, self.width, self.height, is_color=False)
        dest = img.copy()

        nb_blocksX = (self.width // 8) + 1
        nb_blocksY = (self.height // 8) + 1

        def run_nlm2():
            self.ck.NLM2_Mono_GPU(
                (nb_blocksX, nb_blocksY), (8, 8),
                (dest, img,
                 self.np.int32(self.width), self.np.int32(self.height),
                 self.np.float32(10.0), self.np.float32(0.5))
            )

        return self.time_operation("NLM2 Mono", run_nlm2)

    def profile_knn_color(self):
        """Profile KNN (K-Nearest Neighbors) denoising - color."""
        print("Profiling KNN Color...")

        r, g, b = create_test_image(self.cp, self.width, self.height)
        dest_r = r.copy()
        dest_g = g.copy()
        dest_b = b.copy()

        nb_blocksX = (self.width // 8) + 1
        nb_blocksY = (self.height // 8) + 1

        def run_knn():
            self.ck.KNN_Colour_GPU(
                (nb_blocksX, nb_blocksY), (8, 8),
                (dest_r, dest_g, dest_b, r, g, b,
                 self.np.int32(self.width), self.np.int32(self.height),
                 self.np.float32(10.0), self.np.float32(0.5))
            )

        return self.time_operation("KNN Color", run_knn)

    def profile_knn_mono(self):
        """Profile KNN (K-Nearest Neighbors) denoising - mono."""
        print("Profiling KNN Mono...")

        img = create_test_image(self.cp, self.width, self.height, is_color=False)
        dest = img.copy()

        nb_blocksX = (self.width // 8) + 1
        nb_blocksY = (self.height // 8) + 1

        def run_knn():
            self.ck.KNN_Mono_GPU(
                (nb_blocksX, nb_blocksY), (8, 8),
                (dest, img,
                 self.np.int32(self.width), self.np.int32(self.height),
                 self.np.float32(10.0), self.np.float32(0.5))
            )

        return self.time_operation("KNN Mono", run_knn)

    def profile_fnr_color(self):
        """Profile 3FNR (Frame Noise Reduction) - color."""
        print("Profiling 3FNR Color...")

        r1, g1, b1 = create_test_image(self.cp, self.width, self.height)
        r2, g2, b2 = create_test_image(self.cp, self.width, self.height)
        r3, g3, b3 = create_test_image(self.cp, self.width, self.height)
        dest_r, dest_g, dest_b = create_test_image(self.cp, self.width, self.height)

        nb_blocksX = (self.width // 32) + 1
        nb_blocksY = (self.height // 32) + 1

        def run_fnr():
            self.ck.FNR_Color(
                (nb_blocksX, nb_blocksY), (32, 32),
                (dest_r, dest_g, dest_b,
                 r1, g1, b1, r2, g2, b2, r3, g3, b3,
                 self.np.int_(self.width), self.np.int_(self.height),
                 self.np.float32(1.0))
            )

        return self.time_operation("3FNR Color", run_fnr)

    def profile_fnr_mono(self):
        """Profile 3FNR (Frame Noise Reduction) - mono."""
        print("Profiling 3FNR Mono...")

        img1 = create_test_image(self.cp, self.width, self.height, is_color=False)
        img2 = create_test_image(self.cp, self.width, self.height, is_color=False)
        img3 = create_test_image(self.cp, self.width, self.height, is_color=False)
        dest = img1.copy()

        nb_blocksX = (self.width // 32) + 1
        nb_blocksY = (self.height // 32) + 1

        def run_fnr():
            self.ck.FNR_Mono(
                (nb_blocksX, nb_blocksY), (32, 32),
                (dest, img1, img2, img3,
                 self.np.int_(self.width), self.np.int_(self.height),
                 self.np.float32(1.0))
            )

        return self.time_operation("3FNR Mono", run_fnr)

    def profile_clahe_color_cpu(self):
        """Profile CLAHE contrast enhancement - color (CPU path)."""
        print("Profiling CLAHE Color (CPU)...")

        r, g, b = create_test_image(self.cp, self.width, self.height)
        clahe = self.cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        def run_clahe():
            b_np = b.get()
            g_np = g.get()
            r_np = r.get()
            clahe.apply(b_np)
            clahe.apply(g_np)
            clahe.apply(r_np)

        return self.time_operation("CLAHE Color (CPU)", run_clahe)

    def profile_clahe_color_cuda(self):
        """Profile CLAHE with OpenCV CUDA if available."""
        print("Profiling CLAHE Color (CUDA)...")

        try:
            r, g, b = create_test_image(self.cp, self.width, self.height)
            clahe = self.cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

            src_gpu = self.cv2.cuda_GpuMat()

            def run_clahe_cuda():
                src_gpu.upload(b.get())
                clahe.apply(src_gpu, self.cv2.cuda_Stream.Null())
                src_gpu.upload(g.get())
                clahe.apply(src_gpu, self.cv2.cuda_Stream.Null())
                src_gpu.upload(r.get())
                clahe.apply(src_gpu, self.cv2.cuda_Stream.Null())

            return self.time_operation("CLAHE Color (CUDA)", run_clahe_cuda)
        except Exception as e:
            print(f"  CUDA CLAHE not available: {e}")
            self.results["CLAHE Color (CUDA)"] = None
            return None

    def profile_saturation_color(self):
        """Profile saturation enhancement - color."""
        print("Profiling Saturation Color...")

        r, g, b = create_test_image(self.cp, self.width, self.height)
        dest_r, dest_g, dest_b = r.copy(), g.copy(), b.copy()

        nb_blocksX = (self.width // 32) + 1
        nb_blocksY = (self.height // 32) + 1

        def run_saturation():
            # Gaussian blur (part of saturation)
            r_np = self.cv2.GaussianBlur(r.get(), (3, 3), 0)
            g_np = self.cv2.GaussianBlur(g.get(), (3, 3), 0)
            b_np = self.cv2.GaussianBlur(b.get(), (3, 3), 0)
            coul_r = self.cp.asarray(r_np)
            coul_g = self.cp.asarray(g_np)
            coul_b = self.cp.asarray(b_np)

            self.ck.Saturation_Colour(
                (nb_blocksX, nb_blocksY), (32, 32),
                (dest_r, dest_g, dest_b, coul_r, coul_g, coul_b,
                 self.np.int_(self.width), self.np.int_(self.height),
                 self.np.float32(1.5), self.np.int_(0))
            )

        return self.time_operation("Saturation Color", run_saturation)

    def profile_gaussian_blur(self):
        """Profile Gaussian blur (used by multiple filters)."""
        print("Profiling Gaussian Blur 5x5...")

        r, g, b = create_test_image(self.cp, self.width, self.height)

        def run_blur():
            self.cv2.GaussianBlur(r.get(), (5, 5), 0)
            self.cv2.GaussianBlur(g.get(), (5, 5), 0)
            self.cv2.GaussianBlur(b.get(), (5, 5), 0)

        return self.time_operation("Gaussian Blur 5x5", run_blur)

    def profile_gpu_transfer(self):
        """Profile GPU memory transfer overhead."""
        print("Profiling GPU Memory Transfer...")

        np_img = self.np.random.randint(0, 256, (self.height, self.width, 3), dtype=self.np.uint8)

        def run_transfer():
            gpu_img = self.cp.asarray(np_img)
            _ = gpu_img.get()

        return self.time_operation("GPU Transfer (upload+download)", run_transfer)

    def profile_rgb_split_merge(self):
        """Profile RGB channel split and merge operations."""
        print("Profiling RGB Split/Merge...")

        rgb_np = self.np.random.randint(0, 256, (self.height, self.width, 3), dtype=self.np.uint8)
        r, g, b = create_test_image(self.cp, self.width, self.height)

        def run_split():
            gpu_img = self.cp.asarray(rgb_np)
            r_ch = gpu_img[:, :, 2].copy()
            g_ch = gpu_img[:, :, 1].copy()
            b_ch = gpu_img[:, :, 0].copy()
            return r_ch, g_ch, b_ch

        def run_merge():
            result = self.cp.zeros((self.height, self.width, 3), dtype=self.cp.uint8)
            result[:, :, 2] = r
            result[:, :, 1] = g
            result[:, :, 0] = b
            return result

        self.time_operation("RGB Split", run_split)
        return self.time_operation("RGB Merge", run_merge)

    def profile_aanr_color(self):
        """Profile AANR (Adaptive Absorber Noise Reduction) - color."""
        print("Profiling AANR Color...")

        try:
            r1, g1, b1 = create_test_image(self.cp, self.width, self.height)
            r2, g2, b2 = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r1.copy(), g1.copy(), b1.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            def run_aanr():
                self.ck.adaptative_absorber_denoise_Color(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r1, g1, b1, r2, g2, b2,
                     self.np.int_(self.width), self.np.int_(self.height),
                     self.np.int_(0), self.np.int_(0), self.np.int_(0))
                )

            return self.time_operation("AANR Color", run_aanr)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["AANR Color"] = None
            return None

    def profile_denoise_paillou(self):
        """Profile Denoise Paillou filter - color."""
        print("Profiling Denoise Paillou Color...")

        try:
            r, g, b = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r.copy(), g.copy(), b.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            def run_denoise_paillou():
                self.ck.Denoise_Paillou_Colour(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r, g, b,
                     self.np.int_(self.width), self.np.int_(self.height))
                )

            return self.time_operation("Denoise Paillou Color", run_denoise_paillou)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["Denoise Paillou Color"] = None
            return None

    def profile_reduce_variation_color(self):
        """Profile reduce variation (turbulence reduction) - color."""
        print("Profiling Reduce Variation Color...")

        try:
            r1, g1, b1 = create_test_image(self.cp, self.width, self.height)
            r2, g2, b2 = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r1.copy(), g1.copy(), b1.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            def run_reduce_var():
                self.ck.reduce_variation_Color(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r1, g1, b1, r2, g2, b2,
                     self.np.int_(self.width), self.np.int_(self.height),
                     self.np.int_(128))
                )

            return self.time_operation("Reduce Variation Color", run_reduce_var)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["Reduce Variation Color"] = None
            return None

    def profile_histo_color(self):
        """Profile histogram equalization - color."""
        print("Profiling Histogram Color...")
        try:
            r, g, b = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r.copy(), g.copy(), b.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            def run_histo():
                self.ck.Histo_Color(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r, g, b,
                     self.np.int_(self.width), self.np.int_(self.height),
                     self.np.float32(0.0), self.np.float32(255.0))
                )

            return self.time_operation("Histogram Color", run_histo)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["Histogram Color"] = None
            return None

    def profile_amplification_color(self):
        """Profile amplification - color."""
        print("Profiling Amplification Color...")
        try:
            r, g, b = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r.copy(), g.copy(), b.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            # Create correction curve
            Corr_GS = self.cp.linspace(0, 255, 256, dtype=self.cp.float32)

            def run_amp():
                self.ck.Colour_ampsoft_GPU(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r, g, b, Corr_GS,
                     self.np.int_(self.width), self.np.int_(self.height))
                )

            return self.time_operation("Amplification Color", run_amp)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["Amplification Color"] = None
            return None

    def profile_cll_color(self):
        """Profile Contrast Low Light - color."""
        print("Profiling CLL Color...")
        try:
            r, g, b = create_test_image(self.cp, self.width, self.height)
            dest_r, dest_g, dest_b = r.copy(), g.copy(), b.copy()

            nb_blocksX = (self.width // 32) + 1
            nb_blocksY = (self.height // 32) + 1

            # Create correction curve
            Corr_CLL = self.cp.linspace(0, 255, 256, dtype=self.cp.float32)

            def run_cll():
                self.ck.Contrast_Low_Light_Colour_GPU(
                    (nb_blocksX, nb_blocksY), (32, 32),
                    (dest_r, dest_g, dest_b, r, g, b, Corr_CLL,
                     self.np.int_(self.width), self.np.int_(self.height))
                )

            return self.time_operation("CLL Color", run_cll)
        except Exception as e:
            print(f"  Error: {e}")
            self.results["CLL Color"] = None
            return None

    def profile_sharpen_cupy(self):
        """Profile sharpening using CuPy operations (actual implementation)."""
        print("Profiling Sharpen Color (CuPy)...")

        r, g, b = create_test_image(self.cp, self.width, self.height)

        def run_sharpen():
            # This is the actual sharpening implementation
            r_blur = self.cv2.GaussianBlur(r.get(), (3, 3), 0)
            g_blur = self.cv2.GaussianBlur(g.get(), (3, 3), 0)
            b_blur = self.cv2.GaussianBlur(b.get(), (3, 3), 0)

            tmp_r = self.cp.asarray(r).astype(self.cp.int16)
            tmp_g = self.cp.asarray(g).astype(self.cp.int16)
            tmp_b = self.cp.asarray(b).astype(self.cp.int16)

            sharpen_val = 1.5
            tmp_r = tmp_r + sharpen_val * (tmp_r - self.cp.asarray(r_blur))
            tmp_g = tmp_g + sharpen_val * (tmp_g - self.cp.asarray(g_blur))
            tmp_b = tmp_b + sharpen_val * (tmp_b - self.cp.asarray(b_blur))

            res_r = self.cp.clip(tmp_r, 0, 255).astype(self.cp.uint8)
            res_g = self.cp.clip(tmp_g, 0, 255).astype(self.cp.uint8)
            res_b = self.cp.clip(tmp_b, 0, 255).astype(self.cp.uint8)

        return self.time_operation("Sharpen Color", run_sharpen)

    def run_all_profiles(self):
        """Run all profiling tests and generate report."""
        print("\n" + "="*60)
        print(f"JetsonSky Filter Pipeline Profiler")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Iterations: {self.iterations}")
        print("="*60 + "\n")

        # Profile infrastructure operations
        print("--- Infrastructure ---")
        self.profile_gpu_transfer()
        self.profile_rgb_split_merge()
        self.profile_gaussian_blur()

        print("\n--- Noise Reduction Filters ---")
        self.profile_nlm2_color()
        self.profile_nlm2_mono()
        self.profile_knn_color()
        self.profile_knn_mono()
        self.profile_fnr_color()
        self.profile_fnr_mono()
        self.profile_aanr_color()
        self.profile_denoise_paillou()
        self.profile_reduce_variation_color()

        print("\n--- Enhancement Filters ---")
        self.profile_clahe_color_cpu()
        self.profile_clahe_color_cuda()
        self.profile_saturation_color()
        self.profile_histo_color()
        self.profile_amplification_color()
        self.profile_cll_color()
        self.profile_sharpen_cupy()

        return self.generate_report()

    def generate_report(self):
        """Generate a formatted profiling report."""
        print("\n" + "="*70)
        print("PROFILING RESULTS")
        print("="*70)

        # Sort by mean time (slowest first)
        sorted_results = sorted(
            [(k, v) for k, v in self.results.items() if v is not None],
            key=lambda x: x[1]['mean'],
            reverse=True
        )

        print(f"\n{'Filter':<35} {'Mean (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
        print("-"*71)

        total_time = 0
        for name, stats in sorted_results:
            print(f"{name:<35} {stats['mean']:>10.3f}   {stats['min']:>10.3f}   {stats['max']:>10.3f}")
            total_time += stats['mean']

        print("-"*71)
        print(f"{'TOTAL (if all enabled)':<35} {total_time:>10.3f} ms")
        if total_time > 0:
            print(f"{'Theoretical max FPS':<35} {1000/total_time:>10.1f} fps")

        # Identify bottlenecks
        print("\n" + "="*70)
        print("BOTTLENECK ANALYSIS")
        print("="*70)

        threshold_ms = 2.0
        bottlenecks = [(n, s) for n, s in sorted_results if s['mean'] > threshold_ms]

        if bottlenecks:
            print(f"\nFilters taking > {threshold_ms}ms (significant bottlenecks):")
            for name, stats in bottlenecks:
                pct = (stats['mean'] / total_time) * 100 if total_time > 0 else 0
                print(f"  - {name}: {stats['mean']:.2f}ms ({pct:.1f}% of pipeline)")
        else:
            print(f"\nNo individual filter exceeds {threshold_ms}ms")

        # Recommendations
        print("\n" + "="*70)
        print("KNOWN BOTTLENECKS & OPTIMIZATION RECOMMENDATIONS")
        print("="*70)

        print("""
Based on the pipeline architecture analysis:

1. NLM2 (Non-Local Means 2):
   - O(n*m) complexity where n=pixels, m=window_radius^2
   - Uses 3x3 block radius + 3x3 window radius = 49 comparisons/pixel
   - RECOMMENDATION: Use KNN instead for real-time (faster, similar quality)

2. KNN (K-Nearest Neighbors):
   - Similar to NLM2 but simpler weight calculation
   - Still O(n*m) but faster constants
   - RECOMMENDATION: Reduce window radius for higher FPS

3. CLAHE (without CUDA):
   - CPU-based histogram computation
   - RECOMMENDATION: Enable flag_OpenCvCuda if OpenCV CUDA available

4. Gaussian Blur (used by Saturation, Sharpen, Gradient):
   - Called multiple times per frame
   - RECOMMENDATION: Cache blur results when same kernel size used

5. GPU<->CPU Transfers:
   - Every .get() call transfers data from GPU to CPU
   - Used by CLAHE (CPU), GaussianBlur, final output
   - RECOMMENDATION: Minimize transfers, keep data on GPU

6. Saturation (2-pass mode):
   - Calls Gaussian blur 4x per frame (2 passes, 2 sizes)
   - RECOMMENDATION: Disable 2-pass mode for real-time

7. Frame-Temporal Filters (3FNR, AANR):
   - Require storing 2-3 previous frames
   - RECOMMENDATION: Consider disabling during high-motion scenes
""")

        print("="*70)

        return self.results


def main():
    parser = argparse.ArgumentParser(description='Profile JetsonSky filter pipeline')
    parser.add_argument('--resolution', default='1920x1080',
                        help='Image resolution WxH (default: 1920x1080)')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of profiling iterations (default: 50)')

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}")
        print("Use format: WIDTHxHEIGHT (e.g., 1920x1080)")
        sys.exit(1)

    # Run profiler
    profiler = PipelineProfiler(width, height, args.iterations)
    profiler.run_all_profiles()


if __name__ == '__main__':
    main()

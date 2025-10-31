"""
Filter pipeline for sequential image processing.

Replaces the monolithic application_filtrage_color() and application_filtrage_mono()
functions with a modular, configurable pipeline.
"""

from typing import List, Optional, Dict, Any

# Optional numpy import
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

from .base import Filter


class FilterPipeline:
    """
    Manages sequential application of multiple filters.

    Replaces the monolithic filter application functions with a clean,
    testable pipeline that can be dynamically configured.

    Example:
        >>> pipeline = FilterPipeline()
        >>> pipeline.add_filter(DenoiseFilter())
        >>> pipeline.add_filter(SharpenFilter())
        >>> pipeline.add_filter(CLAHEFilter())
        >>> result = pipeline.apply(image)
    """

    def __init__(self, filters: Optional[List[Filter]] = None):
        """
        Initialize pipeline with optional filter list.

        Args:
            filters: List of Filter instances to apply in order
        """
        self.filters: List[Filter] = filters if filters is not None else []
        self.stats: Dict[str, Any] = {}

    def add_filter(self, filter_instance: Filter) -> 'FilterPipeline':
        """
        Add a filter to the end of the pipeline.

        Args:
            filter_instance: Filter to add

        Returns:
            Self for method chaining
        """
        self.filters.append(filter_instance)
        return self

    def remove_filter(self, filter_name: str) -> bool:
        """
        Remove a filter by name.

        Args:
            filter_name: Name of filter to remove

        Returns:
            True if filter was found and removed
        """
        for i, f in enumerate(self.filters):
            if f.get_name() == filter_name:
                self.filters.pop(i)
                return True
        return False

    def get_filter(self, filter_name: str) -> Optional[Filter]:
        """
        Get a filter by name.

        Args:
            filter_name: Name of filter to find

        Returns:
            Filter instance if found, None otherwise
        """
        for f in self.filters:
            if f.get_name() == filter_name:
                return f
        return None

    def enable_filter(self, filter_name: str) -> bool:
        """
        Enable a filter by name.

        Args:
            filter_name: Name of filter to enable

        Returns:
            True if filter was found
        """
        filter_instance = self.get_filter(filter_name)
        if filter_instance:
            filter_instance.enable()
            return True
        return False

    def disable_filter(self, filter_name: str) -> bool:
        """
        Disable a filter by name.

        Args:
            filter_name: Name of filter to disable

        Returns:
            True if filter was found
        """
        filter_instance = self.get_filter(filter_name)
        if filter_instance:
            filter_instance.disable()
            return True
        return False

    def clear(self) -> None:
        """Remove all filters from pipeline."""
        self.filters.clear()
        self.stats.clear()

    def apply(self, image, collect_stats: bool = False, **kwargs):
        """
        Apply all enabled filters in sequence.

        Args:
            image: Input image array
            collect_stats: Whether to collect per-filter timing stats
            **kwargs: Additional parameters passed to each filter

        Returns:
            Processed image after all filters applied
        """
        if collect_stats:
            import time
            self.stats = {}

        # Copy image if numpy available, otherwise work with original
        if HAS_NUMPY and hasattr(image, 'copy'):
            result = image.copy()
        else:
            result = image

        for filter_instance in self.filters:
            # Skip disabled filters
            if not filter_instance.is_enabled():
                continue

            # Apply filter with timing if requested
            if collect_stats:
                import time
                start_time = time.time()
                result = filter_instance.process(result, **kwargs)
                elapsed = time.time() - start_time
                self.stats[filter_instance.get_name()] = {
                    'time_ms': elapsed * 1000,
                    'enabled': True
                }
            else:
                result = filter_instance.process(result, **kwargs)

        return result

    def get_enabled_filters(self) -> List[Filter]:
        """
        Get list of currently enabled filters.

        Returns:
            List of enabled Filter instances
        """
        return [f for f in self.filters if f.is_enabled()]

    def get_filter_count(self) -> int:
        """Get total number of filters in pipeline."""
        return len(self.filters)

    def get_enabled_count(self) -> int:
        """Get number of enabled filters."""
        return len(self.get_enabled_filters())

    def get_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics from last apply() with collect_stats=True.

        Returns:
            Dictionary mapping filter names to stats
        """
        return self.stats

    def print_stats(self) -> None:
        """Print performance statistics in human-readable format."""
        if not self.stats:
            print("No statistics available. Run apply() with collect_stats=True")
            return

        print("=" * 60)
        print("FILTER PIPELINE STATISTICS")
        print("=" * 60)

        total_time = sum(s['time_ms'] for s in self.stats.values())

        for filter_name, stats in self.stats.items():
            time_ms = stats['time_ms']
            percent = (time_ms / total_time * 100) if total_time > 0 else 0
            print(f"{filter_name:30s} {time_ms:8.2f}ms ({percent:5.1f}%)")

        print("-" * 60)
        print(f"{'TOTAL':30s} {total_time:8.2f}ms (100.0%)")
        print("=" * 60)

    def __len__(self) -> int:
        """Return number of filters in pipeline."""
        return len(self.filters)

    def __repr__(self) -> str:
        """String representation of pipeline."""
        enabled = self.get_enabled_count()
        total = self.get_filter_count()
        return f"FilterPipeline({enabled}/{total} filters enabled)"

    def __str__(self) -> str:
        """Detailed string representation."""
        lines = [f"FilterPipeline with {len(self.filters)} filters:"]
        for i, f in enumerate(self.filters, 1):
            status = "✓" if f.is_enabled() else "✗"
            lines.append(f"  {i}. [{status}] {f.get_name()}")
        return "\n".join(lines)


class ColorPipeline(FilterPipeline):
    """
    Pipeline specifically for color image processing.

    Replaces the application_filtrage_color() function.
    """

    def apply(self, image, **kwargs):
        """Apply with color image validation."""
        if not hasattr(image, 'shape'):
            raise ValueError("ColorPipeline requires image with 'shape' attribute")
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("ColorPipeline requires 3-channel color image")
        return super().apply(image, **kwargs)


class MonoPipeline(FilterPipeline):
    """
    Pipeline specifically for monochrome image processing.

    Replaces the application_filtrage_mono() function.
    """

    def apply(self, image, **kwargs):
        """Apply with mono image validation."""
        if not hasattr(image, 'shape'):
            raise ValueError("MonoPipeline requires image with 'shape' attribute")
        if len(image.shape) != 2:
            raise ValueError("MonoPipeline requires 2D monochrome image")
        return super().apply(image, **kwargs)


def create_default_color_pipeline() -> ColorPipeline:
    """
    Create default color processing pipeline.

    This replicates the default filter order from the original
    application_filtrage_color() function.

    Returns:
        Configured ColorPipeline instance
    """
    pipeline = ColorPipeline()

    # Import filters as they are created
    # TODO: Add default filters once implemented
    # pipeline.add_filter(HotPixelFilter())
    # pipeline.add_filter(DenoiseKNNFilter())
    # pipeline.add_filter(DenoisePaillouFilter())
    # pipeline.add_filter(SharpenFilter())
    # pipeline.add_filter(CLAHEFilter())
    # pipeline.add_filter(SaturationFilter())

    return pipeline


def create_default_mono_pipeline() -> MonoPipeline:
    """
    Create default monochrome processing pipeline.

    This replicates the default filter order from the original
    application_filtrage_mono() function.

    Returns:
        Configured MonoPipeline instance
    """
    pipeline = MonoPipeline()

    # Import filters as they are created
    # TODO: Add default filters once implemented
    # pipeline.add_filter(HotPixelFilter())
    # pipeline.add_filter(DenoiseKNNFilter())
    # pipeline.add_filter(DenoisePaillouFilter())
    # pipeline.add_filter(SharpenFilter())
    # pipeline.add_filter(CLAHEFilter())

    return pipeline

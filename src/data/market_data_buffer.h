// market_data_buffer.h — Structure-of-Arrays (SoA), contiguous, fixed-capacity
// OHLCV store for one asset.
//
// Cache-efficiency rationale (report Section 9): columns (timestamps, prices,
// volumes) are stored in SEPARATE contiguous std::vector<double> buffers (SoA)
// rather than an array of {t,p,v} structs (AoS). The hot per-bar / per-scan loop
// touches only the column it needs, fetching one contiguous, prefetch-friendly
// cache line per column instead of striding through interleaved structs. This is
// benchmarked against the AoS layout in bench/microbench.cpp.
//
// The buffer is fixed-capacity and contiguous. Memory is reserved ONCE in the
// constructor; push() only appends up to capacity and performs no dynamic
// allocation on the hot path (verified by an allocation-counting test). In this
// engine the full history of an asset is loaded once into the buffer and then
// streamed by the backtest loop, so elements stay physically contiguous and any
// window slice is a contiguous view — ideal for SIMD-friendly access.
#ifndef STATARB_MARKET_DATA_BUFFER_H
#define STATARB_MARKET_DATA_BUFFER_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace statarb {

class MarketDataBuffer {
public:
    explicit MarketDataBuffer(size_t capacity) : capacity_(capacity > 0 ? capacity : 1) {
        // Pre-allocate once so no reallocation occurs during the simulation.
        timestamps_.reserve(capacity_);
        prices_.reserve(capacity_);
        volumes_.reserve(capacity_);
    }

    // O(1), no allocation while size() < capacity_.
    void push(std::int64_t ts, double price, double volume) {
        if (size_ >= capacity_) {
            throw std::overflow_error("MarketDataBuffer capacity exceeded");
        }
        timestamps_.push_back(ts);
        prices_.push_back(price);
        volumes_.push_back(volume);
        ++size_;
    }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    std::int64_t timestampAt(size_t i) const noexcept { return timestamps_[i]; }
    double priceAt(size_t i) const noexcept { return prices_[i]; }
    double volumeAt(size_t i) const noexcept { return volumes_[i]; }

    // Contiguous, copy-free raw pointers for the hot loops.
    const std::int64_t* timestampsData() const noexcept { return timestamps_.data(); }
    const double* pricesData() const noexcept { return prices_.data(); }
    const double* volumesData() const noexcept { return volumes_.data(); }

private:
    std::vector<std::int64_t> timestamps_;
    std::vector<double> prices_;
    std::vector<double> volumes_;
    size_t capacity_;
    size_t size_ = 0;
};

}  // namespace statarb
#endif  // STATARB_MARKET_DATA_BUFFER_H

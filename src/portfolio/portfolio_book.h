// portfolio_book.h — flat, cache-friendly position table for one asset book.
//
// Positions are a flat std::vector<double> indexed by asset_id, and per-bar PnL
// is appended to a contiguous vector. Full portfolio-level risk controls (risk
// parity across concurrent baskets, VaR/CVaR, gross/net limits, sector neutrality)
// are layered on in Part 7; the flat-table design is fixed now so that extension
// does not require restructuring.
#ifndef STATARB_PORTFOLIO_BOOK_H
#define STATARB_PORTFOLIO_BOOK_H

#include <cmath>
#include <cstddef>
#include <vector>

namespace statarb {

class PortfolioBook {
public:
    explicit PortfolioBook(size_t n_assets) : positions_(n_assets, 0.0), px_(n_assets, 0.0) {}

    void setPrice(size_t id, double price) { px_[id] = price; }
    double position(size_t id) const { return positions_[id]; }
    void setPosition(size_t id, double pos) { positions_[id] = pos; }

    double grossExposure() const {
        double g = 0.0;
        for (double p : positions_) g += std::abs(p);
        return g;
    }
    double netExposure() const {
        double n = 0.0;
        for (double p : positions_) n += p;
        return n;
    }
    // Mark-to-market notional value of a position in dollars (pos in shares).
    double notional(size_t id) const { return positions_[id] * px_[id]; }

    void recordBar(double equity) { equity_history_.push_back(equity); }
    size_t bars() const noexcept { return equity_history_.size(); }
    double equityAt(size_t i) const noexcept { return equity_history_[i]; }

private:
    std::vector<double> positions_;       // shares, indexed by asset_id
    std::vector<double> px_;              // last mark price per asset
    std::vector<double> equity_history_;  // contiguous per-bar equity
};

}  // namespace statarb
#endif  // STATARB_PORTFOLIO_BOOK_H

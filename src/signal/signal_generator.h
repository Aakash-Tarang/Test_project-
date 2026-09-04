// signal_generator.h — turns the residual z-score into a trade signal.
//
// A deviation is "statistically justified" here by the entry band on the
// standardized residual. The Ornstein-Uhlenbeck half-life (estimated on real data
// in Part 6) is a freshness cap: when a position has been open longer than a few
// half-lives we force it flat so we do not ride stale, no-longer-reverting spreads
// (the spec requires the half-life to be compared to the holding period). This is
// a thin, allocation-free decision unit so it can live on the per-bar hot path.
#ifndef STATARB_SIGNAL_GENERATOR_H
#define STATARB_SIGNAL_GENERATOR_H
#include <cmath>

namespace statarb {

enum class SignalAction { FLAT, LONG_TARGET, SHORT_TARGET };

struct TradeSignal {
    SignalAction action = SignalAction::FLAT;
    double score = 0.0;  // signed z-score of the residual
};

class SignalGenerator {
public:
    SignalGenerator(double entry_z, double exit_z, double half_life_cap)
        : entry_z_(entry_z), exit_z_(exit_z), half_life_cap_(half_life_cap) {}

    // residual_z: standardized residual. >0 => target is rich (short target).
    // bars_open: how many bars the current position has been held (for the cap).
    // returns an action to take from the current state via evaluate(state).
    TradeSignal evaluate(double residual_z, double bars_open) const {
        TradeSignal s;
        s.score = residual_z;
        bool stale = half_life_cap_ > 0.0 && bars_open > half_life_cap_;
        if (stale) { s.action = SignalAction::FLAT; return s; }
        if (residual_z >= entry_z_) s.action = SignalAction::SHORT_TARGET;
        else if (residual_z <= -entry_z_) s.action = SignalAction::LONG_TARGET;
        else if (std::abs(residual_z) <= exit_z_) s.action = SignalAction::FLAT;
        else s.action = SignalAction::FLAT;  // between exit and entry => maintain (book decides)
        return s;
    }

    double entryZ() const noexcept { return entry_z_; }
    double exitZ() const noexcept { return exit_z_; }

private:
    double entry_z_, exit_z_, half_life_cap_;
};

}  // namespace statarb
#endif  // STATARB_SIGNAL_GENERATOR_H

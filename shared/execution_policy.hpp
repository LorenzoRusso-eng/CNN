#pragma once

// Questo file definisce le policy di esecuzione usate per scegliere
// il livello principale di parallelismo del training e dell'inferenza.

enum class ExecutionMode {
    Serial,
    IntraExample,
    BatchSamples,
    FoldLevel
};

struct ExecutionPolicy {
    ExecutionMode mode = ExecutionMode::IntraExample;

    constexpr bool allows_intra_example_parallelism() const noexcept{
        return mode == ExecutionMode::IntraExample;
    }

    constexpr bool allows_batch_sample_parallelism() const noexcept{
        return mode == ExecutionMode::BatchSamples;
    }

    constexpr bool allows_fold_parallelism() const noexcept{
        return mode == ExecutionMode::FoldLevel;
    }

    constexpr const char *name() const noexcept{
        switch(mode){
            case ExecutionMode::Serial:
                return "Serial";
            case ExecutionMode::IntraExample:
                return "IntraExample";
            case ExecutionMode::BatchSamples:
                return "BatchSamples";
            case ExecutionMode::FoldLevel:
                return "FoldLevel";
        }

        return "Unknown";
    }
};

namespace execution_policy {

constexpr ExecutionPolicy serial() noexcept{
    return ExecutionPolicy{ExecutionMode::Serial};
}

constexpr ExecutionPolicy intra_example() noexcept{
    return ExecutionPolicy{ExecutionMode::IntraExample};
}

constexpr ExecutionPolicy batch_samples() noexcept{
    return ExecutionPolicy{ExecutionMode::BatchSamples};
}

constexpr ExecutionPolicy fold_level() noexcept{
    return ExecutionPolicy{ExecutionMode::FoldLevel};
}

} // namespace execution_policy

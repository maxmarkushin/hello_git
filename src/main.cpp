#include <algorithm>
#include <cmath>
#include <exception>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

struct ProgramOptions {
    std::size_t count = 1;
    std::string distribution = "uniform";
    bool normalize = false;
    std::optional<unsigned long long> seed;
    std::optional<double> min;
    std::optional<double> max;
    std::optional<double> mean;
    std::optional<double> stddev;
    std::optional<double> lambda;
};

class ArgumentParser {
  public:
    ArgumentParser(int argc, char **argv) : argc_(argc), argv_(argv) {}

    ProgramOptions parse() {
        ProgramOptions options;
        for (int i = 1; i < argc_; ++i) {
            std::string_view arg = argv_[i];
            if (arg == "--help" || arg == "-h") {
                print_help();
                std::exit(EXIT_SUCCESS);
            } else if (arg == "--count" && i + 1 < argc_) {
                options.count = parse_unsigned(argv_[++i]);
            } else if (arg == "--distribution" && i + 1 < argc_) {
                options.distribution = argv_[++i];
            } else if (arg == "--normalize") {
                options.normalize = true;
            } else if (arg == "--seed" && i + 1 < argc_) {
                options.seed = parse_unsigned(argv_[++i]);
            } else if (arg == "--min" && i + 1 < argc_) {
                options.min = parse_double(argv_[++i]);
            } else if (arg == "--max" && i + 1 < argc_) {
                options.max = parse_double(argv_[++i]);
            } else if (arg == "--mean" && i + 1 < argc_) {
                options.mean = parse_double(argv_[++i]);
            } else if (arg == "--stddev" && i + 1 < argc_) {
                options.stddev = parse_double(argv_[++i]);
            } else if (arg == "--lambda" && i + 1 < argc_) {
                options.lambda = parse_double(argv_[++i]);
            } else {
                std::cerr << "Unknown or incomplete argument: " << arg << "\n";
                print_help();
                std::exit(EXIT_FAILURE);
            }
        }

        validate(options);
        return options;
    }

  private:
    static std::size_t parse_unsigned(const std::string &value) {
        try {
            std::size_t pos;
            unsigned long long parsed = std::stoull(value, &pos);
            if (pos != value.size()) {
                throw std::invalid_argument("Trailing characters");
            }
            return static_cast<std::size_t>(parsed);
        } catch (const std::exception &ex) {
            throw std::invalid_argument("Failed to parse unsigned integer: " + value + " (" + ex.what() + ")");
        }
    }

    static double parse_double(const std::string &value) {
        try {
            std::size_t pos;
            double parsed = std::stod(value, &pos);
            if (pos != value.size()) {
                throw std::invalid_argument("Trailing characters");
            }
            return parsed;
        } catch (const std::exception &ex) {
            throw std::invalid_argument("Failed to parse floating-point number: " + value + " (" + ex.what() + ")");
        }
    }

    static void validate(const ProgramOptions &options) {
        if (options.count == 0) {
            throw std::invalid_argument("Count must be greater than zero");
        }

        const std::vector<std::string> known_distributions = {
            "uniform", "normal", "exponential", "bernoulli"
        };
        if (std::find(known_distributions.begin(), known_distributions.end(), options.distribution) == known_distributions.end()) {
            throw std::invalid_argument("Unsupported distribution: " + options.distribution);
        }

        if (options.distribution == "uniform") {
            if (options.min && options.max && *options.min >= *options.max) {
                throw std::invalid_argument("--min must be less than --max for uniform distribution");
            }
        } else if (options.distribution == "normal") {
            if (options.stddev && *options.stddev <= 0.0) {
                throw std::invalid_argument("--stddev must be positive for normal distribution");
            }
        } else if (options.distribution == "exponential") {
            if (options.lambda && *options.lambda <= 0.0) {
                throw std::invalid_argument("--lambda must be positive for exponential distribution");
            }
        }

        if ((options.min.has_value() || options.max.has_value()) && options.distribution != "uniform") {
            throw std::invalid_argument("--min/--max are only valid with uniform distribution");
        }

        if ((options.mean.has_value() || options.stddev.has_value()) && options.distribution != "normal") {
            throw std::invalid_argument("--mean/--stddev are only valid with normal distribution");
        }

        if (options.lambda.has_value() && options.distribution != "exponential") {
            throw std::invalid_argument("--lambda is only valid with exponential distribution");
        }
    }

    static void print_help() {
        std::cout << "True Random Number Generator\n"
                  << "Usage: rng-cli [options]\n\n"
                  << "Options:\n"
                  << "  -h, --help             Show this help message\n"
                  << "  --count <n>           Number of values to generate (default: 1)\n"
                  << "  --distribution <name> Distribution: uniform, normal, exponential, bernoulli (default: uniform)\n"
                  << "  --normalize           Normalize the output so that values sum to 1 (probability vector)\n"
                  << "  --seed <value>        Override the entropy seed\n"
                  << "  --min/--max           Bounds for uniform distribution (default: 0.0 to 1.0)\n"
                  << "  --mean <value>        Mean for normal distribution (default: 0.0)\n"
                  << "  --stddev <value>      Standard deviation for normal distribution (default: 1.0)\n"
                  << "  --lambda <value>      Rate parameter for exponential distribution (default: 1.0)\n";
    }

    int argc_;
    char **argv_;
};

std::mt19937_64 create_engine(const ProgramOptions &options) {
    if (options.seed.has_value()) {
        return std::mt19937_64(*options.seed);
    }

    std::random_device rd;
    std::seed_seq seed{rd(), rd(), rd(), rd(), rd(), rd(), rd(), rd()};
    return std::mt19937_64(seed);
}

std::vector<double> generate_uniform(std::mt19937_64 &engine, const ProgramOptions &options) {
    const double min = options.min.value_or(0.0);
    const double max = options.max.value_or(1.0);
    std::uniform_real_distribution<double> dist(min, max);
    std::vector<double> values(options.count);
    std::generate(values.begin(), values.end(), [&]() { return dist(engine); });
    return values;
}

std::vector<double> generate_normal(std::mt19937_64 &engine, const ProgramOptions &options) {
    const double mean = options.mean.value_or(0.0);
    const double stddev = options.stddev.value_or(1.0);
    std::normal_distribution<double> dist(mean, stddev);
    std::vector<double> values(options.count);
    std::generate(values.begin(), values.end(), [&]() { return dist(engine); });
    return values;
}

std::vector<double> generate_exponential(std::mt19937_64 &engine, const ProgramOptions &options) {
    const double lambda = options.lambda.value_or(1.0);
    std::exponential_distribution<double> dist(lambda);
    std::vector<double> values(options.count);
    std::generate(values.begin(), values.end(), [&]() { return dist(engine); });
    return values;
}

std::vector<double> generate_bernoulli(std::mt19937_64 &engine, const ProgramOptions &options) {
    const double probability = options.mean.value_or(0.5);
    if (probability < 0.0 || probability > 1.0) {
        throw std::invalid_argument("Bernoulli probability must be in [0, 1]");
    }
    std::bernoulli_distribution dist(probability);
    std::vector<double> values(options.count);
    std::generate(values.begin(), values.end(), [&]() { return dist(engine) ? 1.0 : 0.0; });
    return values;
}

std::vector<double> generate_values(std::mt19937_64 &engine, const ProgramOptions &options) {
    static const std::map<std::string, std::vector<double> (*)(std::mt19937_64 &, const ProgramOptions &)> generators = {
        {"uniform", generate_uniform},
        {"normal", generate_normal},
        {"exponential", generate_exponential},
        {"bernoulli", generate_bernoulli},
    };

    auto it = generators.find(options.distribution);
    if (it == generators.end()) {
        throw std::invalid_argument("Unsupported distribution: " + options.distribution);
    }

    return it->second(engine, options);
}

void normalize(std::vector<double> &values) {
    if (values.empty()) {
        return;
    }

    double min_value = *std::min_element(values.begin(), values.end());
    if (min_value < 0.0) {
        for (double &value : values) {
            value -= min_value;
        }
    }

    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    if (sum <= 0.0) {
        double uniform_value = 1.0 / static_cast<double>(values.size());
        std::fill(values.begin(), values.end(), uniform_value);
        return;
    }

    for (double &value : values) {
        value /= sum;
    }
}

void print_values(const std::vector<double> &values, bool normalized) {
    std::cout << std::fixed << std::setprecision(normalized ? 6 : 10);
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::cout << values[i];
        if (i + 1 != values.size()) {
            std::cout << '\n';
        }
    }
    std::cout << '\n';
}

int main(int argc, char **argv) {
    try {
        ArgumentParser parser(argc, argv);
        ProgramOptions options = parser.parse();
        std::mt19937_64 engine = create_engine(options);
        std::vector<double> values = generate_values(engine, options);

        if (options.normalize) {
            normalize(values);
        }

        print_values(values, options.normalize);
    } catch (const std::exception &ex) {
        std::cerr << "Error: " << ex.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

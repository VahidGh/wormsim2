// connectome_trace — load a c302 NeuroML2 connectome and run NeuralIntegrator
//
// Usage:
//   connectome_trace <nml_file> <T_ms> <dt_ms> <neuron_names>
//
//   <neuron_names>  comma-separated list of population names to trace,
//                   e.g. "AVAL[0],AVAR[0],AVBL[0],AVBR[0]"
//
// Output (stdout): CSV  t,V_<name1>,V_<name2>,...
// Stats   (stderr): neurons, gap junctions, chemical synapses, NMJs
//
// Units: mV · ms · pA · pF · nS

#include "io/NetworkInputParser.h"
#include "io/NetworkConfig.h"
#include "neural/NeuralIntegrator.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>

using namespace wormsim2;

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::fputs(
            "Usage: connectome_trace <nml_file> <T_ms> <dt_ms> <neuron_names>\n"
            "  neuron_names: comma-separated e.g. AVAL[0],AVAR[0],AVBL[0]\n",
            stderr);
        return 1;
    }

    const std::string nml_path = argv[1];
    const float T_ms  = std::atof(argv[2]);
    const float dt_ms = std::atof(argv[3]);
    const std::string names_csv = argv[4];

    // ---------------------------------------------------------------------------
    // Load connectome
    // ---------------------------------------------------------------------------
    NetworkConfig cfg = NetworkInputParser::load(nml_path);

    std::fprintf(stderr,
        "Loaded: %d cells, %d gap junctions, %d chem synapses, %d NMJs\n",
        cfg.neuron_count(), cfg.gap_junction_count(), cfg.synapse_count(),
        static_cast<int>(cfg.nmj_connections.size()));

    // ---------------------------------------------------------------------------
    // Resolve requested neuron names → indices
    // ---------------------------------------------------------------------------
    std::vector<std::string> names;
    std::vector<int>         indices;
    {
        std::istringstream ss(names_csv);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (tok.empty()) continue;
            names.push_back(tok);
            const NeuronDef* nd = cfg.find_neuron(tok);
            if (!nd) {
                std::fprintf(stderr, "Warning: neuron '%s' not found — skipped\n",
                             tok.c_str());
                names.pop_back();
            } else {
                indices.push_back(nd->id);
            }
        }
    }
    if (indices.empty()) {
        std::fputs("No valid neuron names; aborting\n", stderr);
        return 1;
    }

    // ---------------------------------------------------------------------------
    // Run NeuralIntegrator
    // ---------------------------------------------------------------------------
    NeuralIntegrator ni(cfg);

    // CSV header
    std::printf("t");
    for (const auto& n : names) std::printf(",V_%s", n.c_str());
    std::printf("\n");

    // t=0 initial state
    {
        const auto vv = ni.voltages();
        std::printf("%.4f", 0.0f);
        for (int idx : indices)
            std::printf(",%.6f", vv[static_cast<std::size_t>(idx)]);
        std::printf("\n");
    }

    const int steps = static_cast<int>(T_ms / dt_ms);
    for (int i = 0; i < steps; ++i) {
        ni.step(dt_ms);
        const float t_now = (i + 1) * dt_ms;
        const auto  vv    = ni.voltages();
        std::printf("%.4f", t_now);
        for (int idx : indices)
            std::printf(",%.6f", vv[static_cast<std::size_t>(idx)]);
        std::printf("\n");
    }

    return 0;
}

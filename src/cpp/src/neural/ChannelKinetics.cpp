#include "neural/ChannelKinetics.h"

namespace wormsim2 {

namespace {

// ---------------------------------------------------------------------------
// C. elegans c302 channel catalog — parameters from Wormbook / Jospin 2002 /
// Bhatt 2012, fitted for single-compartment lumped-neuron models.
// Units: g_bar (nS), e_rev_mV (mV), tau (ms), v_half (mV).
// ---------------------------------------------------------------------------

// NCA — NALCN-family background Na⁺ leak; no gate (pure ohmic)
const ChannelSpec kNCA{
    .g_bar    = 0.8f,
    .e_rev_mV = -20.0f,
    .gates    = {}
};

// KD — delayed-rectifier K⁺ (EXP-2 / Kv11 equivalent); 1 gate n², Gaussian tau
const ChannelSpec kKD{
    .g_bar    = 5.0f,
    .e_rev_mV = -80.0f,
    .gates    = { GateKinetics{
        .v_half  = -15.0f, .k     = 12.0f,
        .tau_0   = 20.0f,  .tau_a = 50.0f, .tau_v = -15.0f, .tau_w = 25.0f,
        .initial = 0.05f,  .power = 2
    }}
};

// KA — fast transient A-type K⁺ (SHL-1 equivalent); activation m × inactivation h
const ChannelSpec kKA{
    .g_bar    = 3.0f,
    .e_rev_mV = -80.0f,
    .gates    = {
        GateKinetics{ .v_half=-20.0f, .k= 8.0f, .tau_0= 2.0f,
                      .initial=0.10f, .power=1 },
        GateKinetics{ .v_half=-55.0f, .k=-8.0f, .tau_0=15.0f,
                      .initial=0.90f, .power=1 }
    }
};

// KQS — quasi-ohmic voltage-sensitive K⁺; 1 gate s
const ChannelSpec kKQS{
    .g_bar    = 2.0f,
    .e_rev_mV = -80.0f,
    .gates    = { GateKinetics{
        .v_half=-10.0f, .k=7.0f, .tau_0=10.0f, .initial=0.15f, .power=1
    }}
};

// KVS — voltage-sensitive K⁺ (SHK-1 / Shaw-type equivalent); m × h
const ChannelSpec kKVS{
    .g_bar    = 2.0f,
    .e_rev_mV = -80.0f,
    .gates    = {
        GateKinetics{ .v_half=-15.0f, .k= 10.0f, .tau_0= 5.0f,
                      .initial=0.10f, .power=1 },
        GateKinetics{ .v_half=-45.0f, .k=-10.0f, .tau_0=40.0f,
                      .initial=0.85f, .power=1 }
    }
};

// IR — inward-rectifier K⁺ (activated at hyperpolarisation); 1 gate q
const ChannelSpec kIR{
    .g_bar    = 1.0f,
    .e_rev_mV = -90.0f,
    .gates    = { GateKinetics{
        .v_half=-80.0f, .k=-15.0f, .tau_0=30.0f, .initial=0.50f, .power=1
    }}
};

// ---------------------------------------------------------------------------
// Boyle & Cohen 2008 channels — parameters from NEURON .mod files published
// in openworm/CElegansNeuroML (pythonScripts/c302/neuron_interaction/).
// x_inf uses the same Boltzmann form; tau is constant (tau_a = 0).
// ---------------------------------------------------------------------------

// k_slow_bc — slow-inactivating K⁺ (Boyle-Cohen 2008); 1 activation gate n
const ChannelSpec kKSlowBC{
    .g_bar    = 1.0f,
    .e_rev_mV = -60.0f,
    .gates    = { GateKinetics{
        .v_half  = 19.8741f,   .k      = 15.8512f,
        .tau_0   = 25.0007f,   .tau_a  = 0.0f, .tau_v = 0.0f, .tau_w = 1.0f,
        .initial = 0.012f,     .power  = 1
    }}
};

// k_fast_bc — fast transient K⁺ (Boyle-Cohen 2008); activation p^4, inactivation q^1
const ChannelSpec kKFastBC{
    .g_bar    = 0.5f,
    .e_rev_mV = -60.0f,
    .gates    = {
        GateKinetics{                               // gate p (activation)
            .v_half = -8.0523205f, .k      = 7.42636f,
            .tau_0  = 2.25518f,    .tau_a  = 0.0f, .tau_v = 0.0f, .tau_w = 1.0f,
            .initial = 0.004f,     .power  = 4
        },
        GateKinetics{                               // gate q (inactivation, k<0)
            .v_half = -15.645601f, .k      = -9.97468f,
            .tau_0  = 149.96301f,  .tau_a  = 0.0f, .tau_v = 0.0f, .tau_w = 1.0f,
            .initial = 0.97f,      .power  = 1
        }
    }
};

// leak_bc — passive leak used in Boyle-Cohen / c302-D models
const ChannelSpec kLeakBC{
    .g_bar    = 0.1f,
    .e_rev_mV = -50.0f,
    .gates    = {}   // pure ohmic
};

// ca_boyle — voltage-gated Ca²⁺ channel (Boyle-Cohen 2008 / c302 C2)
// Parameters from ca_boyle.mod (openworm/CElegansNeuroML).
// Gate e² (activation, τ=0.10 ms) × gate f (inactivation, τ=150.9 ms).
// Ca²⁺-dependent h gate is omitted (treated as 1; negligible at resting [Ca²⁺]ᵢ).
// E_rev set to eca = +40 mV (Nernst Ca²⁺ at c302 default concentrations).
const ChannelSpec kCABoyle{
    .g_bar    = 1.0f,
    .e_rev_mV = 40.0f,
    .gates    = {
        GateKinetics{ .v_half = -3.3568f,  .k =  6.74821f, .tau_0 = 0.100027f,
                      .tau_a  = 0.0f,      .initial = 0.00023f, .power = 2 },
        GateKinetics{ .v_half = 25.1815f,  .k = -5.03176f, .tau_0 = 150.88f,
                      .tau_a  = 0.0f,      .initial = 1.0f,    .power = 1 }
    }
};

const ChannelSpec kUnknown{};

} // anonymous namespace

const ChannelSpec& channel_spec(std::string_view id) noexcept {
    if (id == "NCA")     return kNCA;
    if (id == "KD")      return kKD;
    if (id == "KA")      return kKA;
    if (id == "KQS")     return kKQS;
    if (id == "KVS")     return kKVS;
    if (id == "IR")      return kIR;
    if (id == "KSLOW_BC") return kKSlowBC;
    if (id == "KFAST_BC") return kKFastBC;
    if (id == "LEAK_BC")  return kLeakBC;
    if (id == "CA_BOYLE") return kCABoyle;
    // c302 C2 channel aliases (ionChannel IDs used in NeuroML2 network files)
    if (id == "k_slow" || id == "k_slow_muscle") return kKSlowBC;
    if (id == "k_fast" || id == "k_fast_muscle") return kKFastBC;
    if (id == "Leak")                             return kLeakBC;
    if (id == "ca_boyle" || id == "ca_boyle_muscle") return kCABoyle;
    return kUnknown;
}

} // namespace wormsim2

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

const ChannelSpec kUnknown{};

} // anonymous namespace

const ChannelSpec& channel_spec(std::string_view id) noexcept {
    if (id == "NCA") return kNCA;
    if (id == "KD")  return kKD;
    if (id == "KA")  return kKA;
    if (id == "KQS") return kKQS;
    if (id == "KVS") return kKVS;
    if (id == "IR")  return kIR;
    return kUnknown;
}

} // namespace wormsim2

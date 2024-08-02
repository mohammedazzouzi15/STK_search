import logging
from typing import Optional

from e3nn import o3
from stk_search.geom3d.models.Allegro._keys import EDGE_ENERGY, EDGE_FEATURES
from stk_search.geom3d.models.Allegro.nn import (
    Allegro_Module,
    EdgewiseEnergySum,
    NormalizedBasis,
    ScalarMLP,
)
from stk_search.geom3d.models.NequIP.data import AtomicDataDict, AtomicDataset
from stk_search.geom3d.models.NequIP.nn import (
    AtomwiseReduce,
    SequentialGraphNetwork,
)
from stk_search.geom3d.models.NequIP.nn.embedding import (
    OneHotAtomEncoding,
    RadialBasisEdgeEncoding,
    SphericalHarmonicEdgeAttrs,
)
from stk_search.geom3d.models.NequIP.nn.radial_basis import BesselBasis


def Allegro(config, initialize: bool, dataset: Optional[AtomicDataset] = None):
    logging.debug("Building Allegro model...")

    # # Handle avg num neighbors auto
    # builder_utils.add_avg_num_neighbors(
    #     config=config, initialize=initialize, dataset=dataset
    # )

    # Handle simple irreps
    if "l_max" in config:
        l_max = int(config["l_max"])
        parity_setting = config["parity"]
        assert parity_setting in ("o3_full", "o3_restricted", "so3")
        irreps_edge_sh = repr(
            o3.Irreps.spherical_harmonics(
                l_max, p=(1 if parity_setting == "so3" else -1)
            )
        )
        nonscalars_include_parity = parity_setting == "o3_full"
        # check consistant
        assert config.get("irreps_edge_sh", irreps_edge_sh) == irreps_edge_sh
        assert (
            config.get("nonscalars_include_parity", nonscalars_include_parity)
            == nonscalars_include_parity
        )
        config["irreps_edge_sh"] = irreps_edge_sh
        config["nonscalars_include_parity"] = nonscalars_include_parity

    layers = {
        # -- Encode --
        # Get various edge invariants
        "one_hot": OneHotAtomEncoding,
        "radial_basis": (
            RadialBasisEdgeEncoding,
            {
                "basis": (
                    NormalizedBasis
                    if config.get("normalize_basis", True)
                    else BesselBasis
                ),
                "out_field": AtomicDataDict.EDGE_EMBEDDING_KEY,
            },
        ),
        # Get edge nonscalars
        "spharm": SphericalHarmonicEdgeAttrs,
        # The core Allegro model:
        "Allegro": (
            Allegro_Module,
            {
                "field": AtomicDataDict.EDGE_ATTRS_KEY,  # initial input is the edge SH
                "edge_invariant_field": AtomicDataDict.EDGE_EMBEDDING_KEY,
                "node_invariant_field": AtomicDataDict.NODE_ATTRS_KEY,
            },
        ),
        "edge_eng": (
            ScalarMLP,
            {"field": EDGE_FEATURES, "out_field": EDGE_ENERGY, "mlp_output_dimension": 1},
        ),
        # Sum edgewise energies -> per-atom energies:
        "edge_eng_sum": EdgewiseEnergySum,
        # Sum system energy:
        "total_energy_sum": (
            AtomwiseReduce,
            {
                "reduce": "sum",
                "field": AtomicDataDict.PER_ATOM_ENERGY_KEY,
                "out_field": AtomicDataDict.TOTAL_ENERGY_KEY,
            },
        ),
    }

    return SequentialGraphNetwork.from_parameters(shared_params=config, layers=layers)


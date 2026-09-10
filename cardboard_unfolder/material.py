"""Material and machine parameters.

Everything here is empirical and machine-dependent. Measure it on the board you
actually own before trusting a default. Ship a calibration cut.
"""
from dataclasses import dataclass, field


# Caliper / flute pitch by grade, mm. Measured-caliper figures (C&T Matrix),
# not the "nominal fraction" numbers US box brokers quote.
FLUTES = {
    "F":  dict(thickness=1.0, pitch=2.4, liner=0.20),
    "E":  dict(thickness=1.5, pitch=3.2, liner=0.25),
    "B":  dict(thickness=3.0, pitch=6.5, liner=0.30),
    "C":  dict(thickness=4.0, pitch=8.0, liner=0.30),
    "BC": dict(thickness=6.5, pitch=6.5, liner=0.30),
}


@dataclass
class Material:
    """Board and laser parameters.

    thickness / liner / pitch describe the board; kerf and the fold settings
    describe how your machine cuts it.
    """
    thickness: float = 3.0          # caliper, mm. Calipers squeeze corrugated -
                                    # they read 0.2-0.5mm low. Prefer the grade table.
    liner: float = 0.30             # single liner caliper, mm
    pitch: float = 6.5              # flute pitch, mm. Only used for perf dashing.
    kerf: float = 0.30              # mm. Wider than wood: the char halo exceeds the beam.

    # --- fold geometry ---
    crush: float = 0.75             # Corrugated crushes, so the theoretical relief
                                    # width is conservative. 0.6 soft/recycled,
                                    # 1.0 stiff or double-wall. EMPIRICAL.
    relief_threshold: float = 1.2   # mm. Below this computed width, just perforate;
                                    # don't remove material.
    max_relief: float = 12.0        # mm. Above this, warn - the fold is too sharp
                                    # for single-score and wants a double score.

    # --- perforation ---
    perf_cut: float = 1.0           # mm of cut per dash
    perf_tie: float = 0.6           # mm of uncut tie per dash
    # Period must stay well under the flute pitch or consecutive flutes get very
    # different treatment. Enforced in folds.perf_dash().

    # --- relief slot bridges ---
    bridge_len: float = 6.0         # mm of uncut material holding the hinge
    bridge_every: float = 45.0      # mm between bridge centres

    # --- joinery ---
    tab_depth: float = 12.0         # mm
    tab_inset_deg: float = 35.0     # trapezoid side angle
    tab_min_edge: float = 14.0      # don't tab an edge shorter than this

    # --- machine ---
    bed: tuple = (600.0, 400.0)     # mm
    sheet_margin: float = 8.0       # mm from bed edge
    part_gap: float = 3.0           # mm between parts. Char halo sets this, not kerf.
                                    # 1mm will scorch the web between neighbours.

    # --- nesting ---
    lock_grain: bool = False        # Flute-direction constraint. Off for costume work
                                    # where +/- a few mm doesn't matter. Turn on if a
                                    # part visibly racks, then creases must cross flutes.

    label_height: float = 5.0       # mm. Below ~3.5mm the character burns closed.

    @classmethod
    def from_flute(cls, grade: str, **overrides):
        g = FLUTES[grade.upper()]
        return cls(thickness=g["thickness"], pitch=g["pitch"],
                   liner=g["liner"], **overrides)

    @property
    def pivot_depth(self) -> float:
        """Distance from the grooved (inner) face to the effective hinge axis.

        With the far liner left intact the board pivots about that liner's
        mid-plane, not the outer surface - about 5% less than naive thickness.
        """
        return self.thickness - self.liner / 2.0

    @property
    def usable_bed(self) -> tuple:
        m = 2 * self.sheet_margin
        return (self.bed[0] - m, self.bed[1] - m)

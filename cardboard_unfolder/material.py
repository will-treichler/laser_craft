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

    # --- relief width ---
    relief_adaptive: bool = True    # width from thickness and bend angle, as derived
                                    # above. Off uses relief_fixed for every fold,
                                    # which is what you want when cutting to a
                                    # known blade width rather than to the model.
    relief_fixed: float = 3.0       # mm, used when relief_adaptive is off

    # --- perforation ---
    perf_cut: float = 1.0           # mm of cut per dash
    connect_ratio: float = 0.5      # fraction of the fold line left CONNECTED.
                                    # 0.5 cuts half the material away; 0.8 leaves a
                                    # stiff fold, 0.2 an easy one.
    min_connect: float = 10.0       # mm of material that must survive a fold, total.
                                    # A short fold given a blanket ratio ends up with
                                    # almost nothing holding it; this floor is what
                                    # stops small joints tearing off. EMPIRICAL.
    round_cut_ends: bool = True     # radius slot ends instead of leaving them square.

    # Growing the part by kerf/2 cannot reach into a notch: the cut stops
    # short of the apex by (kerf/2)/sin(theta/2). corner_relief runs a slit
    # down the bisector to reach it. Empirical only in the sense that how
    # far PAST the apex to go depends on the board - 0 is enough for B
    # flute, a thicker board may want a little more.
    # How far the cut sits off the part's own edge, in mm. Positive is kerf
    # compensation; NEGATIVE cuts the part back, which is what thick board
    # wants where two panels meet. 0 puts the beam on the line, which is right
    # for cardboard at costume tolerance and keeps sharp notches honest.
    edge_offset: float = 0.0

    corner_relief: bool = True
    corner_overcut: float = 0.0     # mm past the apex
                                    # A square internal corner is a crack starter, and
                                    # the beam dwells there anyway.
    # Period must stay well under the flute pitch or consecutive flutes get very
    # different treatment. Enforced in folds.perf_segments().

    # Relief slots used to carry their own bridge_len / bridge_every pair.
    # They are gone: a slot and a score are the same decision about how much of
    # the bend line survives, so perf_cut / connect_ratio / min_connect govern
    # both. Two sets of controls meant the exposed ones appeared to do nothing
    # on any model whose folds were wide enough to become slots.

    # --- joinery ---
    use_tabs: bool = False          # Most cardboard goes together with tape or a
                                    # glued backing strip, and tabs cost ~20% of the
                                    # sheet in flap area. Off by default for that
                                    # reason; turn on for a glued assembly.
    tab_depth: float = 12.0         # mm
    tab_inset_deg: float = 35.0     # trapezoid side angle
    tab_min_edge: float = 14.0      # don't tab an edge shorter than this

    # Interlocking fingers along every cut seam: a box joint, so two panels
    # drop together and cannot slide while the tape or glue goes on. Width is
    # a judgement about handling (EMPIRICAL); depth is the board thickness,
    # because a finger has to pass through the board it mates with - so it
    # follows `thickness` unless finger_depth_override says otherwise.
    use_fingers: bool = False
    finger_width: float = 15.0      # mm per finger, stretched to fill the edge
    finger_depth_override: bool = False
    finger_depth: float = 3.0       # mm, used only when overridden

    # --- machine ---
    bed: tuple = (600.0, 400.0)     # mm
    sheet_margin: float = 8.0       # mm from bed edge
    part_gap: float = 3.0           # mm between parts. Char halo sets this, not kerf.
                                    # 1mm will scorch the web between neighbours.

    # --- nesting ---
    lock_grain: bool = False        # Flute-direction constraint. Off for costume work
                                    # where +/- a few mm doesn't matter. Turn on if a
                                    # part visibly racks, then creases must cross flutes.

    # Engraved annotation. 4mm reads easily on board; 2mm is the floor - the
    # stroke width of a defocused beam is a real fraction of the cell below
    # that and the character starts to burn closed. A label that cannot fit
    # its space at the floor is left off rather than shrunk into a smudge.
    label_height: float = 4.0
    label_min_height: float = 2.0
    # How much of an edge its own label may span. A number as long as the
    # edge it names reads as a label for the whole part; at a quarter it reads
    # as belonging to that edge. It is also the dominant lever on how many
    # edges get numbered at all - on a pikachu net the median cut edge is
    # 10.5 mm and a two-digit number at the 2 mm floor is 2.88 mm wide, so a
    # quarter needs an 11.5 mm edge and only 41% of edges have one.
    # The smallest piece worth handing to a human. Measured as the largest
    # circle that fits inside the part (layout.inscribed_width), not area: a
    # 2 x 40 mm splinter and a 9 x 9 mm square have the same area and only one
    # of them can be held while the glue sets. Empirical - it is a statement
    # about fingers, not geometry.
    splinter_width: float = 15.0

    label_edge_frac: float = 0.25
    label_folds: bool = True        # M / V beside every fold
    model_id: str = ""              # stamped on each island with its number

    @property
    def perf_tie(self) -> float:
        """Uncut length per dash, from the connect ratio.

        Expressed as a ratio rather than a length because the useful question
        is "how much of this fold survives", which is what governs whether it
        hinges or tears - not how long one tie happens to be.
        """
        r = min(max(self.connect_ratio, 0.0), 0.99)
        return self.perf_cut * r / (1.0 - r) if r > 0.0 else 0.0

    @classmethod
    def from_flute(cls, grade: str, **overrides):
        g = FLUTES[grade.upper()]
        return cls(thickness=g["thickness"], pitch=g["pitch"],
                   liner=g["liner"], **overrides)

    @property
    def finger_depth_used(self) -> float:
        """How far a finger stands proud of its edge, and a notch cuts in."""
        return self.finger_depth if self.finger_depth_override else self.thickness

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

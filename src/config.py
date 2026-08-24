"""
config.py
=========
Every parameter a student would want to tweak lives in this ONE file.
Nothing below should require touching any other module.

All time units are in "simulation steps", and for this project we treat
1 step = 1 second, so a rate like INCOMING_W = 5 means "5 vehicles try
to arrive at the west approach every second".
"""

# ---------------------------------------------------------------------------
# 1. SIMULATION LENGTH
# ---------------------------------------------------------------------------
SIMULATION_TIME = 300  # total seconds simulated per run

# ---------------------------------------------------------------------------
# 2. TRAFFIC DEMAND (vehicles arriving per second, per approach)
#    Kept IDENTICAL across designs in Experiment A so that the ONLY thing
#    that changes is the graph structure of the intersection.
# ---------------------------------------------------------------------------
INCOMING_N = 2
INCOMING_S = 1
INCOMING_W = 5
INCOMING_E = 2
INCOMING_NE = 1  # only used by the five-way design

DEMAND = {
    "N": INCOMING_N,
    "S": INCOMING_S,
    "W": INCOMING_W,
    "E": INCOMING_E,
    "NE": INCOMING_NE,
}

# ---------------------------------------------------------------------------
# 3. SIGNAL TIMING (signalised designs only)
# ---------------------------------------------------------------------------
GREEN_LIGHT_TIME = 40  # seconds of green given to each phase by default

# ---------------------------------------------------------------------------
# 4. ROAD (EDGE) DEFAULTS
#    Individual designs may override these, but everything starts here.
# ---------------------------------------------------------------------------
DEFAULT_APPROACH_CAPACITY = 12  # max vehicles physically "on" an approach road
DEFAULT_APPROACH_TRAVEL_TIME = 2  # seconds to cross an approach road segment

DEFAULT_DEPARTURE_CAPACITY = 12
DEFAULT_DEPARTURE_TRAVEL_TIME = 1

DEFAULT_LINK_CAPACITY = 6  # short connector edges (staggered link, ring segments)
DEFAULT_LINK_TRAVEL_TIME = 2

# ---------------------------------------------------------------------------
# 5. ROUNDABOUT-SPECIFIC
# ---------------------------------------------------------------------------
ROUNDABOUT_MAX_MERGE_PER_STEP = 1  # gap-acceptance: at most 1 car merges
# from a given approach per second,
# modelling "yield to circulating traffic"

# ---------------------------------------------------------------------------
# 6. REPEATED SIMULATION (Section 14)
# ---------------------------------------------------------------------------
NUMBER_OF_RUNS = 10
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 7. EFFICIENCY METRIC REFERENCE CONSTANTS (Section 12)
#    These convert unbounded quantities (seconds of waiting, cars queued)
#    into bounded [0,1] "scores" so they can be combined fairly.
# ---------------------------------------------------------------------------
EFFICIENCY_REFERENCE_WAIT = 30.0  # seconds; "typical tolerable wait"
EFFICIENCY_REFERENCE_QUEUE = 8.0  # vehicles per approach; "typical tolerable queue"

EFFICIENCY_WEIGHTS = {
    "completion_rate": 0.35,
    "throughput": 0.25,
    "wait_score": 0.25,
    "queue_score": 0.15,
}

# ---------------------------------------------------------------------------
# 8. OUTPUT
# ---------------------------------------------------------------------------
OUTPUT_DIR = "output"

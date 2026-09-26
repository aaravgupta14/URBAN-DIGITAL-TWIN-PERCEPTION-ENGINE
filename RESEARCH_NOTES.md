# Research Notes — Next Phase of TalkToMyTwin

Scope: collision detection, traffic management algorithms, EDA on the trajectory log,
and how to evaluate a digital twin. Written as a decision guide, not as code.

---

## 1. Collision detection / risk — Surrogate Safety Measures (SSM)

Pixel proximity (what the twin does today) is a presence measure. The traffic-safety
field uses **surrogate safety measures**, which fall into two families:

* **Proximity-based** — how close in space *and time* two road users come.
* **Evasive-action-based** — did someone brake hard or swerve.

### The four to implement, in order

**TTC — Time to Collision** (Hayward, 1972)
For a leader/follower pair on the same path:

    TTC = (x_lead - x_follow - L) / (v_follow - v_lead),   defined only when v_follow > v_lead

`L` = leader length. Thresholds in the literature: TTC < 1.5 s is commonly treated as a
serious conflict, 1.5-3 s as a moderate one. This is the primary alert signal.

**MTTC — Modified TTC**
Plain TTC assumes constant speed. MTTC solves the quadratic that includes relative
acceleration, so it stays valid while a vehicle is actively braking or accelerating —
exactly the case that matters. Use once the Kalman filter provides acceleration.

**PET — Post-Encroachment Time**
Time between vehicle A leaving a conflict area and vehicle B entering it. Retrospective,
not predictive, so it cannot drive a live alert — but it is the cleanest *offline*
severity statistic for crossing/merging conflicts, and it needs no speed estimate,
only arrival timestamps at a polygon. Cheap to add, high analytical value.

**DRAC — Deceleration Rate to Avoid a Crash**

    DRAC = (v_follow - v_lead)^2 / (2 * (x_lead - x_follow - L))

Compare against a Maximum Available Deceleration (MADR, typically ~3.4 m/s^2 for a
passenger car on dry asphalt). DRAC > MADR = crash is unavoidable by braking alone.

### Sequencing advice

1. None of these can be computed without **speed**, so speed estimation is the blocking
   prerequisite (Section 3).
2. Compute TTC only for pairs whose paths actually conflict. Two vehicles 2 m apart in
   adjacent lanes travelling the same direction at the same speed have infinite TTC and
   should never alert. Gate on lane assignment + heading difference first.
3. Log every SSM value per pair per frame, not just threshold crossings. The
   distribution is the deliverable; the alert is a threshold on it.
4. Report conflicts *rated* by severity, and validate against manual video review of a
   held-out clip. An SSM engine nobody has spot-checked is an opinion generator.

### Known pitfalls for a monocular setup

* TTC is a ratio with a small denominator — relative-speed noise blows up TTC variance.
  Smooth speeds before differencing, never after.
* The log already flags 90% of detections as far-field/low-confidence. Do not raise
  alerts from that band until depth-dependent uncertainty is modelled — the error bars
  on TTC there are wider than the thresholds.

---

## 2. Traffic management algorithms

Ordered by how much infrastructure they assume.

**Fixed-time / Webster** — analytical cycle-length optimisation from historical volumes.
Worth implementing once as a baseline to beat.

**SCOOT / SCATS / TUC** — the deployed, industrial adaptive systems. Study them for
their *interfaces* (what a real controller expects as input: queue length, degree of
saturation, occupancy) rather than to reimplement.

**Max-Pressure — recommended next step.** Define the pressure of a phase as
(vehicles on incoming lanes) − (vehicles on outgoing lanes). After each minimum green,
serve the phase with maximum pressure. It is decentralised (needs no neighbour
communication), has a provable maximum-stability / throughput guarantee, and its input
is *exactly* what the twin already produces — per-lane vehicle counts.
**BackPressure** is the cyclic variant: fixed cycle length, green time split
proportionally to phase pressure. Use it if the target controller cannot run acyclic.

**Reinforcement learning (DQN / DDQN / pressure-based rewards)** — reported gains in the
recent literature are around 9% travel time and ~5% delay against SCATS-derived data.
Real, but modest, and it needs a simulator. Treat it as phase 3, and use Max-Pressure as
the benchmark it must beat — much of the RL literature quietly loses to Max-Pressure.

**Simulation** — SUMO is the standard open-source microsimulator and the practical way
to close the loop: replay extracted trajectories, apply a control policy, measure delay.
Calibrate SUMO against real counts *before* trusting any policy comparison.

---

## 3. Speed estimation (prerequisite for everything above)

Two options, both defensible:

* **Savitzky-Golay filter** over the world-coordinate time series per track (window ~9-15
  frames, polynomial order 2-3), then differentiate. This is the standard treatment for
  NGSIM/highD trajectory noise and the easier one to justify in a report.
* **Constant-acceleration Kalman filter** in world coordinates, state = [x, y, vx, vy,
  ax, ay]. Gives acceleration for free (needed for MTTC and hard-braking detection) and
  handles missing frames naturally.

Either way: filter in **world metres**, not image pixels, and resample to a fixed time
base using frame index / FPS. Validate on a vehicle that can be timed manually across two
known road markings — if speed is off by 20%, every SSM downstream is fiction.

---

## 4. EDA on tracking_log.csv

Current snapshot (483 frames, 3,151 rows, 60 tracks) for reference:
~6.5 vehicles/frame, median track length 30 frames, 60% car / 20% truck / 20% two-wheeler,
94% stable, **67% in-bounds, 90% flagged low-confidence**.

### Data-quality pass (do this first — it decides what the rest is worth)

1. **Track-length histogram.** Tracks under ~5 frames are usually fragments. Count them,
   and check whether one physical vehicle owns several IDs (ID fragmentation) by looking
   for a new track starting where an old one died, within a few frames and a small radius.
2. **In-bounds rate by image region.** 33% out-of-bounds is a calibration-coverage
   problem, not a tracking problem. Map where those detections land to decide how to
   extend the calibration polygon.
3. **Confidence vs. depth.** Bin detections by world Y and plot mean confidence, box
   height, and position jitter per bin. This gives the empirical error-vs-distance curve —
   what is needed to justify the low-confidence cutoff instead of guessing it.
4. **Frame-to-frame displacement distribution per track.** The tail is the anomaly
   population; check that the jump filter's median-multiple rule is cutting the right tail.

### Traffic-science pass

5. **Lane assignment** by clustering world X (lateral coordinate) — a 1D histogram of X
   should show one mode per lane. If it does not, the homography is off.
6. **Speed distribution** per class and per lane; free-flow speed as the 85th percentile.
7. **Headway / gap distributions** — time headway between consecutive vehicles at a
   virtual detector line. Log-normal-ish is expected; deviations are worth explaining.
8. **Fundamental diagrams** — flow-density, speed-density, speed-flow. Standard method:
   lay rectangular **time-space cells** over the trajectory field (Edie's definitions:
   flow = total distance travelled in the cell / cell area; density = total time spent /
   cell area). This is the single most credible analytics artefact producible from this
   data, because it lets a transport engineer sanity-check the twin instantly.
9. **Spatial heatmaps** of occupancy and of conflict locations — the visual that makes the
   safety case legible to a non-technical audience.
10. **Temporal profiles** — counts and mean speed in 30 s bins, with class breakdown.

---

## 5. Evaluating the digital twin

Split evaluation into three layers. Most projects only do the first, which is why they
stay demos.

### Layer 1 — Perception accuracy

* **HOTA** as the headline tracking metric. It decomposes into **DetA** (detection) and
  **AssA** (association) and averages over localisation thresholds, so it says *which
  half* is broken. MOTA over-weights detection errors and hides ID problems.
* Report **IDF1** and **ID switches** alongside it, plus **MOTP** for localisation.
* This needs annotated ground truth: hand-label ~200-500 frames of one clip, or a sampled
  subset across the clip. Painful, non-negotiable, and the single highest-value thing
  available to make the project defensible.

### Layer 2 — Geometric / twin fidelity

* **Reprojection error**: pick road landmarks with known relative positions (lane-marking
  ends, pole bases) that were *not* used to fit the homography, project them, measure the
  metric error. Report it as a function of distance from the camera.
* **Scale validation**: a known length in the scene (dashed lane-marking segments have a
  standard length in most road codes) should come out correct in the twin.
* **Speed MAE** against a manually timed sample of vehicles.
* **Coverage**: fraction of detections falling inside the valid calibrated region — the
  67% figure is already measured, which is more than most projects report.

### Layer 3 — Behavioural / functional fidelity

Compare aggregate twin outputs against reality: vehicle counts vs. manual count, mean
speed, class distribution, queue lengths. Published traffic digital twins report this as
percentage agreement (e.g. ~93% on average speed, ~97% on trip length) — that is the bar
and the format to match.

### Robustness and honesty

* Evaluate per condition: near vs. far field, occluded vs. clear, per class.
  A single aggregate number hides exactly the failure that matters.
* Report a **latency/throughput** figure (FPS end-to-end on named hardware) — a twin that
  cannot keep up with the camera is not real-time, and this is easy to measure honestly.
* State the failure modes explicitly. The far-field 90% flag is a finding, not an
  embarrassment.

---

## Sources

* Surrogate safety measures / TTC / PET / DRAC:
  https://www.sciencedirect.com/science/article/abs/pii/S0001457522001919 ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10943440/ ·
  https://arxiv.org/pdf/2312.07019 · https://doi.org/10.3390/su13169278
* Digital twin for traffic safety analysis: https://arxiv.org/pdf/2502.09561
* Max-Pressure / BackPressure / RL signal control:
  https://arxiv.org/pdf/2406.19305 · https://arxiv.org/pdf/2210.02612 ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC13035168/ ·
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9002556/
* Fundamental diagrams from trajectory data: https://arxiv.org/html/2507.09648v1 ·
  https://www.sciencedirect.com/science/article/pii/S0191261518303527
* Trajectory smoothing (Savitzky-Golay, NGSIM):
  https://www.researchgate.net/publication/269853999_Making_NGSIM_Data_Usable_for_Studies_on_Traffic_Flow_Theory
* HOTA / MOT evaluation: https://arxiv.org/pdf/2405.11536 · https://arxiv.org/pdf/2502.04478
* Digital twin fidelity + validation numbers:
  https://thesai.org/Downloads/Volume16No5/Paper_42-Digital_Twin_Based_Predictive_Analytics.pdf ·
  https://www.mdpi.com/2075-1702/13/9/750
* TrackTrack (CVPR 2025):
  https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html
* Industry context (NVIDIA Omniverse smart-city blueprint, VivaCity/Dublin):
  https://blogs.nvidia.com/blog/smart-city-ai-blueprint-europe/

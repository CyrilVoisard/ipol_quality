# Indicators of the recording quality score

Reference definition of the score implemented in `package/`. The default values of the
parameters are those of `package/params.py`.

## 1. Data

The test is a standing period (6 s requested), a 10 m straight walk, a U-turn, a 10 m walk back
and a standing period (2 s requested). Sensors: head HE (optional), lower back LB, left foot LF,
right foot RF; set $\mathcal{S}$, feet $\mathcal{F}=\{LF,RF\}$. For a sensor $s$ and a sample $i$:
acceleration $\mathbf{a}_s(i)\in\mathbb{R}^3$ (m/s², gravity included), angular velocity
$\boldsymbol{\omega}_s(i)$ (rad/s), packet counter $c_s(i)$; frequency $F$; $g=9.81$ m/s².

Only the **raw files** are used. No filter and no offset correction are applied to the signals of
the indicators: the usual pre-processing (gyroscope offset removed and gravity estimated on the
first seconds) would make several indicators ideal by construction.

**Packet counters** (`timebase.py`). The 16-bit counter of each file is unwrapped. Sensor $s$
covers $[f_s, l_s]$ (first and last counter), span $N_s=l_s-f_s+1$; $m_s$ samples are lost in its
gaps, the longest gap has $L_s$ samples. The signals are linearly interpolated on the integer
counters common to the sensors provided.

**Segmentation** (`segmentation.py`). The U-turn and the gait events come from the result files
of the two IPOL segmentation demos. U-turn: boundaries $[u_0,u_1]$, number of U-turns $n_u$ and the
model of Salarian et al. fitted on the heading of the lower back,
$\theta(x)=a\,\varsigma\!\left(\frac{x-b}{c}\right)+C\,(x-x_0)$, with
$\varsigma(u)=\tfrac12\sin(\pi u)$ on $[-\tfrac12,\tfrac12]$ and $\pm\tfrac12$ outside: amplitude $a$ (deg),
centre $b$ and duration $c$ (samples), residual drift $C$ (deg/sample), coefficient of
determination $R^2$. Indices
count the samples from the first packet counter of each file, lost samples included; they are
converted to the common grid.

## 2. From a deviation to a quality

Each indicator $k$ is a deviation $\delta_k\ge0$ from its ideal value ($\delta_k=0$), mapped to

$$Q(\delta)=\begin{cases}100 & \delta\le\delta_{100}\\[2pt]
100\,\max\left(0,\ 1-\dfrac{\varphi(\delta)-\varphi(\delta_{100})}{2\,\big(\varphi(\delta_{50})-\varphi(\delta_{100})\big)}\right) & \delta>\delta_{100}\end{cases}$$

with $\varphi(\delta)=\delta$ (linear scale) or $\varphi(\delta)=\ln\delta$ (logarithmic scale). The
quality is 100 up to the tolerance $\delta_{100}$ (0 unless stated), 50 at the half-quality
deviation $\delta_{50}$, and 0 from $2\delta_{50}-\delta_{100}$ (linear) or $\delta_{50}^2/\delta_{100}$
(logarithmic). Without tolerance, $Q=100\,\max(0,1-\delta/(2\delta_{50}))$. A tolerance is used
where a small deviation is harmless: an offset of the gyroscope within the specifications of the
sensors, part of the energy of the walk out of its plane. The logarithmic scale is used for the
drift of the heading, which spans orders of magnitude: each doubling costs 25 points (100 up to
0.025, 75 at 0.07, 50 at 0.2). Parameters: `params.DELTA50`,
`params.DELTA100`, `params.SCALE`. The number of U-turns has its own table of qualities
(`settings["n_uturns_quality"]`).

**Choice of the thresholds.** The distribution of each deviation on a reference cohort of healthy
subjects (AMISIA, 85 trials) describes a normal recording; $\delta_{50}$ starts at twice its 95th
percentile, so that 95 % of the reference keeps at least 75. It is then adjusted to what the
processing needs: a defect that the usual pre-processing corrects without effect on the
algorithms is tolerated (stable gyroscope offset, scale error of the accelerometer, isolated lost
samples, sensor turned in its plane), a defect that changes what the algorithms see is scored.
The tolerance of the defects without consequence (movement and offset at rest, norm of the
acceleration, tilt, plane of the walk) follows the judgement of an expert of instrumented gait
analysis, from a blinded rating of recordings of the public data set (companion study).

**Classes.** A quality is *good* above 75, *acceptable* above 50 up to 75, *critical* up to 50: a
value on a boundary falls in the more severe class (`scoring.quality_class`, `settings["good"]`,
`settings["critical"]`). Two U-turns counted (75) or a walk whose end is not recorded (still_time 75
at best) are thus acceptable, not good.

**An indicator always has a quality** (`indicators/base.py`):

| situation | quality | note |
|---|---|---|
| computed | $g\,Q(\delta)$, $g\in\{0,1\}$ a gate (orient_rest only) | explanation when low |
| not computable because of the recording (U-turn not segmented, no gait events, no walk detected, invalid value) | 0 | `not computable: <reason>` |
| not assessed because its precondition is a protocol step already scored by another indicator: orient_stability without standing period before or after the walk; the indicators at rest (still_acc, rest_acc, rest_gyr, gyro_bias, drift_ratio) without any standing period, since their values would then be taken while walking. Both are scored by still_time: the same deviation is not penalised twice | none: left out of the scores, listed in the report and the figure | `not assessed: <reason>` |
| sensor not provided (head) | none: left out of the scores | `sensor not provided` |

## 3. Aggregation

Rule $\mathrm{agg}$ (`scoring.combine`): the minimum of the qualities. A recording is as good as
its weakest point: one failing element (a sensor upside down, a long gap on one sensor) cannot be
hidden by the others, and one indicator at 75 or below makes its group, and the recording, at
best acceptable. This is also how the expert rates a recording: its class is the worst class of
its groups.

- per sensor, for the groups raw, sensor and orientation: $\mathrm{agg}$ of the qualities of the
  indicators of the sensor (cards of `score.png`);
- per group: $\mathrm{agg}$ of the per-sensor scores and of the trial-level indicators of the group;
- global score: $\mathrm{agg}$ of the available group scores (six groups: raw, sensor, orientation,
  protocol, U-turn, segmentation): the lowest of the six groups.

The report lists the indicators at 75 or below with their class and their note, and the sensors
turned in their plane with the axes corrected.

## 4. Phases of the trial

**Activity** (`stillness.py`). With moving standard deviations $\sigma$ over $w=0.5$ s and
$\mathcal{S}_0=\{LB,LF,RF\}$ (the head may move while the subject stands still):

$$A_a(i)=\max_{s\in\mathcal{S}_0}\sigma\big(\lVert\mathbf{a}_s\rVert\big)(i),\qquad
A_\omega(i)=\max_{s\in\mathcal{S}_0}\Big(\sum_{k}\sigma(\omega_{s,k})^2(i)\Big)^{1/2}.$$

Thresholds $\tau_a=\max(0.05,\ 0.1\,\mathrm{med}\,A_a)$ and $\tau_\omega=\max(0.05,\ 0.1\,\mathrm{med}\,A_\omega)$,
medians on the central third of the recording (walk). A sample moves if $A_a\ge\tau_a$ or
$A_\omega\ge\tau_\omega$.

**Walk**: the movement periods separated by less than 1 s are merged; the walk $[i_0,i_1)$ is the
longest one. Standing durations $t_b=i_0/F$ before and $t_a=(N-i_1)/F$ after.

**Standing windows**: longest still run of at least 0.3 s in $[0,\ i_0-0.3F)$ (before) and in
$[i_1+0.3F,\ N)$ (after). The rest window $W$ is the window before, else the window after; without
any standing period, the indicators at rest are not assessed (see §2).

**Quietest second of a sensor**: $W_s\subset W$ is the 1 s sub-window of minimal mean
$\big(\sum_k\sigma(\omega_{s,k})^2\big)^{1/2}$. All rest values of sensor $s$ are taken on $W_s$,
so that a subject who moves the head while standing is not mistaken for a faulty head sensor.
$\bar{\mathbf{a}}_s$ and $\bar{\boldsymbol\omega}_s$ denote means on $W_s$, $A=[i_0,i_1)$ the walk.

**Placement of the sensors** (`placement.py`, `settings["placement"]="detect"`). Classic frames:
head and lower back with gravity on $+X$ at rest ($X$ cranio-caudal, $Y$ medio-lateral, $Z$
antero-posterior); feet on the dorsum with gravity shared between $+X$ and $+Z$, $Y$ medio-lateral.
The variants met in practice are rotations of the sensor in its own plane, about $Z$, by
$\theta\in\{0,90,180,270\}$ degrees (strap put another way). Head and lower back:
$\varphi=\operatorname{atan2}(\bar a_{s,Y},\bar a_{s,X})$ is snapped to the nearest quarter turn $k$
and $\theta=-90k$. Feet: the medio-lateral axis is the in-plane axis with the larger energy of
$\boldsymbol\omega_s$ on $A$ ($Y$: $\theta\in\{0,180\}$, $X$: $\theta\in\{90,270\}$); the sign of
$\bar a_{s,X}$ (resp. $\bar a_{s,Y}$) chooses between the two. The signals are then rotated,
$\mathbf{a}_s\leftarrow R_z(\theta)\,\mathbf{a}_s$ and
$\boldsymbol\omega_s\leftarrow R_z(\theta)\,\boldsymbol\omega_s$, before the indicators.
A variant is reported (report and figure), not scored: it is corrected without loss. What a
rotation in the plane cannot fix (a sensor on another face, tilted, swapped axes) is measured by
the orientation group. With `settings["placement"]="fixed"` the classic frames are assumed.
On the 2025 data set, the detection agrees with the axis changes made by hand in the published
processed files for 100 % of the head and lower-back sensors and 99.3 to 99.6 % of the foot sensors.

## 5. Indicators

### Raw data — was the stream received entirely? (per sensor)

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| missing | $100\,m_s/N_s$ (%): isolated lost samples are interpolated without effect, long gaps are scored by max_gap | 6 % |
| max_gap | $L_s/F$ (s): a gap of half a stride can hide a gait event | 0.5 s |
| disconnection | $100\,\big(N-N_s\big)/N$ (%), $N$ the span of all the sensors together: the sensor started late or stopped early (4 % is about 1 s of a 25 s trial) | 4 % |

### Sensors — does each sensor behave as expected at rest? (per sensor)

rest_acc, rest_gyr and gyro_bias are not computed for the head: the subject may turn or tilt the
head while standing still, and a slow turn of the head cannot be told from an offset of its
gyroscope (a faulty head gyroscope still shows in head_trunk).

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| still_acc | $\big\lvert\mathrm{mean}_{W_s}\lVert\mathbf{a}_s\rVert-g\big\rvert$ (m/s²): the norm of the acceleration at rest is $g$; a scale error of a few per cent is corrected by calibration (75 at 5 % of $g$, 50 at 10 %) | 1 m/s² |
| rest_acc | $100\ \mathrm{mean}_{W_s}\lVert\mathbf{a}_s-\bar{\mathbf{a}}_s\rVert\ /\ \mathrm{mean}_{A}\lVert\mathbf{a}_s-\bar{\mathbf{a}}_s\rVert$ (%): movement or noise at rest (75 at a fifth of the walking level) | 40 % |
| rest_gyr | $100\ \mathrm{mean}_{W_s}\lVert\boldsymbol\omega_s-\bar{\boldsymbol\omega}_s\rVert\ /\ \mathrm{mean}_{A}\lVert\boldsymbol\omega_s-\bar{\boldsymbol\omega}_{s,A}\rVert$ (%): movement or noise at rest, without the offset (scored by gyro_bias) | 40 % |
| gyro_bias | $\lVert\bar{\boldsymbol\omega}_s\rVert$ (deg/s): offset of the gyroscope, removed by the usual pre-processing when it is stable: 100 up to 3, 75 at 6, 50 at 9 deg/s | $\delta_{100}=3$, $\delta_{50}=9$ deg/s |

### Protocol — was the protocol respected? (trial level)

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| n_uturns | $\lvert n_u-1\rvert$: 1 U-turn gives 100, 2 give 75, 3 give 50, more give 0 | table |
| still_time | $\max\big(s_b,\ (s_b+s_a)/2\big)$ (s), with the shortfalls $s_b=4-\min(t_b,4)$ before and $s_a=2-\min(t_a,2)$ after the walk | 2 s |

still_time: 4 s of standing before the walk and 2 s after give 100; 3 s and 1 s give 75 (as 4 s and
nothing after); less than 2 s before the walk gives less than 50 whatever comes after, since the
standing period before the walk is needed to calibrate the sensors and as the baseline of the
segmentation codes. The deviation is linear in each duration: the shortfall before the walk alone
sets a floor, the mean shortfall spreads the shortfall after the walk.

**End of the walk.** When the sensors still move at the end of the recording ($t_a=0$), the last
steps and the stop are not recorded; the report says so. For the analyses that need them (shape of
the last steps, gait termination), `settings["end_standing_required"] = True` sets still_time to 0
for such a recording, which makes it critical. By default it keeps its still_time (75 at best):
the recording remains usable for the spatio-temporal parameters and the U-turn.

### Orientation — once the placement corrected, is each sensor aligned with its segment? (lower back, feet)

Only gross errors are scored: a tilt or a lateral sway explained by the gait is not a defect of the
recording. The head is not scored (its tilt and its sway are behaviour); its placement is still
detected and corrected. With $E_s$ the expected axes of the gravity at rest ($\{X\}$ for LB,
$\{X,Z\}$ for the feet):

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| orient_rest | $1-\rho_s$, $\rho_s=\sum_{k\in E_s}\bar a_{s,k}^2/\lVert\bar{\mathbf{a}}_s\rVert^2$; $\sin^2$ of the tilt (75 at 39°, 50 at 63°: a tilt is corrected by aligning the sensor on the gravity); gate $g=0$ if $\bar a_{s,k}\le0$ for some $k\in E_s$ (sensor on another face, axes swapped) | 0.8 |
| walk_plane_acc | share of the energy of $\mathbf{a}_s-\bar{\mathbf{a}}_s$ on $A$ along $Y$ (off the sagittal plane): 100 while the sagittal plane holds at least half of the energy, 75 at 0.7, 50 at 0.9; the lateral sway of the trunk also depends on the gait | $\delta_{100}=0.5$, $\delta_{50}=0.9$ |
| walk_plane_gyr (feet) | $1-$ share of the energy of $\boldsymbol\omega_s$ on $A$ about $Y$ (medio-lateral axis, used by the segmentation of the steps), same scale | $\delta_{100}=0.5$, $\delta_{50}=0.9$ |
| orient_stability | angle (deg) between the mean accelerations of the quietest seconds of the standing periods before and after the walk: the sensor did not move on the body. Not computed for the head (its tilt changes with the gaze); not assessed without a standing period before or after the walk (that deviation is scored by still_time: 4 s before and nothing after gives still_time 75) | 15 deg |

### U-turn — is the U-turn segmentation consistent? (trial level)

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| uturn_fit | $1-R^2$: the model is a simplification of a real turn, so the tolerance is wide; $R^2=0.95$ gives 75, $R^2=0.9$ gives 50 | 0.1 |
| uturn_amplitude | $\big\lvert180-\lvert a\rvert\big\rvert$ (deg): a turn of 135° or 225° gives 50 | 45 deg |
| drift_ratio | $\lvert\bar{\boldsymbol\omega}_{LB}\cdot\bar{\mathbf{a}}_{LB}\rvert/\lVert\bar{\mathbf{a}}_{LB}\rVert\ /\ \big(\lvert a\rvert F/c\big)$: offset of the trunk gyroscope about the vertical at rest (the raw drift of the heading that the U-turn detection integrates) relative to the mean turning rate of the U-turn, i.e. the share of the U-turn amplitude that the drift adds over the duration of the turn; logarithmic scale, 100 up to 0.025 (4.5° on 180°), 75 at 0.07 (13°), 0 from 1.6 | $\delta_{100}=0.025$, $\delta_{50}=0.2$ |
| head_trunk | $1-\min(\Delta_{HE},\Delta_{LB})/\max(\Delta_{HE},\Delta_{LB})$, $\Delta_s$ the range of the heading of $s$ over $[u_0-1\,\mathrm{s},\ u_1+1\,\mathrm{s}]$ | 0.3 |

The heading of sensor $s$ is the integral of its angular velocity about the vertical,
$\boldsymbol\omega_s\cdot\bar{\mathbf{a}}_s/\lVert\bar{\mathbf{a}}_s\rVert$ (low-pass filtered,
Butterworth order 8, 14 Hz), which does not depend on how the sensor is mounted. All four are
0 when the U-turn is not segmented; head_trunk is left out without head sensor, drift_ratio without
any standing period (scored by still_time).

### Segmentation — are the gait events consistent? (trial level)

| indicator | deviation $\delta$ | $\delta_{50}$ |
|---|---|---|
| alternation | mean over the two straight walks (before and after the U-turn; the whole walk when the U-turn is not segmented) of $100-\alpha$, $\alpha=100\,(\text{number of foot changes})/(L-1)$ for the $L$ heel strikes of both feet of that walk in chronological order (%); a walk with fewer than 2 events counts 100. The alternation is not expected during the turn. A degraded alternation can come from the segmentation or from an altered gait (double steps) | 10 % |

## 6. Figures

`score.png`: on the left, the four sensors on a schematic body
seen from the back, with the axes of each sensor as detected ($Z$ towards the reader) and a card
per sensor: placement (classic, or variant corrected) and the scores of the sensor for the raw
data, the rest and the orientation (dotted: sensor not provided). In the centre, the global score
($\mathrm{agg}$ of the six groups) between the three sensor-level groups (left) and the three trial-level
groups (right). Colours: terracotta below 50, honey at 62.5, sage above 75. Below: the critical
indicators and the indicators not assessed, with their reason.

`construction.png`: the signals behind every indicator, with the same layout. Sensors (left):
stream of each sensor, lost samples in black; $\lVert\mathbf{a}\rVert-g$ on the quietest second;
integrated angular velocity at rest, whose slope is the bias; share of the gravity per axis at
rest (expected axes outlined) with the placement detected and the change of tilt; energy of the
walk per axis. Trial (right): activity, thresholds, standing periods and walk; heading of the
lower back and of the head with the model of the U-turn; angular velocity of the feet about $Y$
with the swing phases.

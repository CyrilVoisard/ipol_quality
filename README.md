# Recording Quality Score of a Walk with U-Turn Measured with Four Inertial Sensors

## General information

IPOL article: Voisard et al., *A recording quality score for instrumented gait tests* (in preparation).
IPOL demo: (link given once the article is published)
Contact: cyril.voisard@gmail.com
License: GNU Affero General Public License v3 or later (see `LICENSE`).

## What the code does

The test is a standing period, a 10 m straight walk, a U-turn, a 10 m walk back and a standing
period, recorded with four inertial sensors: head (optional), lower back, left foot and right foot.
In the spirit of the signal quality indices of the electrocardiogram, the code tells whether the
recording can be trusted before any gait parameter is computed from it.

The placement of each sensor is first detected: a sensor strapped in another way (turned by 90,
180 or 270 degrees in its own plane) is reported, its axes are brought back to the classic frame
before the indicators are computed, and its corrected file is written (`corrected_XX.txt`) so that
the U-turn and gait-event segmentations can be computed again on the right axes. Eighteen indicators are then computed **on the raw
signals** (no filter, no offset correction), in six groups, three per sensor and three for the
trial:

| group | question | indicators |
|---|---|---|
| Raw data | was the data stream received entirely? | missing, max_gap, disconnection (per sensor) |
| Sensors | does each sensor behave as expected at rest? | still_acc, rest_acc, rest_gyr, gyro_bias (per sensor) |
| Orientation | once the placement corrected, is each sensor aligned with the body segment, and did it stay so? | orient_rest, walk_plane_acc, walk_plane_gyr, orient_stability (lower back, feet) |
| Protocol | was the protocol respected? | n_uturns, still_time |
| U-turn | is the U-turn segmentation consistent? | uturn_fit, uturn_amplitude, drift_ratio, head_trunk |
| Segmentation | are the gait events physiologically consistent? | alternation |

Each indicator is a deviation from its ideal value, mapped to a quality between 0 and 100 (100 up
to a tolerance, 50 at a half-quality deviation). Defects that the usual pre-processing corrects
without effect on the gait algorithms are tolerated, and the orientation only scores gross errors;
the head, which may move while the subject stands still, is not scored on movement or gyroscope
offset at rest nor on its orientation (its placement is still detected and corrected).
An indicator that cannot be computed because of the recording (U-turn not segmented, no gait
events...) gets the quality 0 **and the reason**. Two cases are left out of the scores, with the
reason reported: the indicators of a sensor that was not provided (head), and the indicators that
need a standing period when it is missing (orientation stability, and the values at rest when the
recording has no standing period at all), since that deviation from the protocol is already scored
by the standing time (it is not penalised twice). A recording that ends while the subject still
walks is reported and is at best acceptable; with `-end_required 1` (analyses of the last steps) it
becomes critical. A recording is as good as its weakest point: the per-sensor, group and global
scores are the minimum of what they contain, and the global score is the lowest of the six groups.
Three classes are reported: good above 75, acceptable above 50 up to 75, critical up to 50 (a
value on a boundary falls in the more severe class). Every definition is given in
[`indicators.md`](indicators.md).

## Inputs

| input | content | required |
|---|---|---|
| `input_0` | raw file of the head sensor | no |
| `input_1` | raw file of the lower-back sensor | yes |
| `input_2` | raw file of the left foot sensor | yes |
| `input_3` | raw file of the right foot sensor | yes |
| `input_4` | segmentation file (JSON) | yes |

**Raw files**: text tables (tab, space or comma separated) with the columns `PacketCounter`,
`Acc_X`, `Acc_Y`, `Acc_Z`, `Gyr_X`, `Gyr_Y`, `Gyr_Z` (other columns are ignored, header lines
before `PacketCounter` are skipped). The packet counter is used to detect lost samples and to
align the sensors (16-bit wrap-around handled). Sensor frames of the public data sets: X along the
cranio-caudal axis for the head and the lower back, foot sensors on the dorsum with Y as the
medio-lateral axis. Sensors turned in their own plane are detected and corrected (see
`indicators.md`, section 4).

**Segmentation file**: the results of the two IPOL segmentation demos, gathered in one JSON object

```json
{
  "uturn":       { ... content of uturn.json given by the U-turn segmentation demo ... },
  "gait_events": { ... content of gait_events.json given by the MP-DTW demo ... }
}
```

- U-turn: Voisard et al., *U-Turn Segmentation with a Lower-Back Inertial Sensor: an
  Implementation of the Model of Salarian et al.*, IPOL (code: `ipol_uturn`);
- gait events: Voisard et al., *Automatic Gait Events Detection with IMU: An in-Depth Look at the
  MP-DTW Algorithm*, IPOL (code: `ipol_gaitevents`).

Both demos count the samples from the first packet counter of each file, lost packets included:
the score uses the same convention. The result file of one demo alone is also accepted (the other
part is then scored 0, with the reason). A failed segmentation can be reported as
`{"uturn_error": "...", "gait_events_error": "..."}`.

**Parameters**: acquisition frequency (Hz), unit of the acceleration (m/s², g or automatic) and of
the angular velocity (rad/s, deg/s or automatic).

## Outputs

| file | content |
|---|---|
| `score.png` | on the left the four sensors on a schematic body (placement detected, scores of the raw data, of the rest and of the orientation of each sensor); in the centre the three sensor-level groups, the global score and the three trial-level groups (protocol, U-turn, segmentation); below, the placement corrections and the indicators at 75 or below with their reason |
| `construction.png` | the signals behind each group, with the same layout: on the left the sensors (stream, rest, gravity per axis with the placement detected, energy of the walk per axis), on the right the trial (standing periods and walk, U-turn, gait events) |
| `quality.txt` | scores and classes, per-sensor scores, placement of each sensor, indicators at 75 or below with their reason |
| `quality.json` | every value, deviation, quality and parameter |
| `indicators.csv` | one row per indicator and sensor |
| `corrected_XX.txt` | raw file of each sensor with its axes in the classic frame (same columns and samples) |

## Running the code

Python 3.9 or later with the packages of `requirements.txt`:

```bash
pip install -r requirements.txt
python main.py -i0 head.txt -i1 lower_back.txt -i2 left_foot.txt -i3 right_foot.txt \
               -i4 segmentation.json -freq 100 -acc_unit m/s2 -gyr_unit rad/s
```

`-i0` may be omitted. `-end_required 1` requires the end of the walk to be recorded (sensors still
at the end of the recording), as needed to study the last steps. The outputs are written in the current folder; if the trial cannot be scored
(required file missing, unreadable file), the reason is written in `demo_failure.txt`.

From Python:

```python
from package import assess_files
from package.report import save_figures, write_reports

a = assess_files({"HE": "head.txt", "LB": "lower_back.txt", "LF": "left_foot.txt", "RF": "right_foot.txt"},
                 "segmentation.json", freq=100)
print(a.table())
write_reports(a, "out")
save_figures(a, "out")
```

## Example data

`example_datasets/` holds five trials of the public data sets (raw files and segmentation given by
the two IPOL demos):

- `HS_1`, `HS_2`, `PD`, `CVA` (trials HS_1_1, HS_19_1, PD_15_11 and CVA_1_1): Voisard et al., *A
  dataset of clinical gait signals with wearable sensors from healthy, neurological, and
  orthopedic cohorts*, Scientific Data, 2025;
- `HS_3` (trial HS_1_1): AMISIA cohort (Limoges).

## Code organisation

| file | role |
|---|---|
| `main.py` | entry point of the IPOL demo |
| `package/io.py` | reading of the raw files and units |
| `package/timebase.py` | packet counters, lost samples, alignment of the sensors |
| `package/stillness.py` | standing periods and walk |
| `package/segmentation.py` | reading of the segmentation results, headings |
| `package/placement.py` | placement of the sensors: detection and correction |
| `package/indicators/` | the six groups of indicators |
| `package/params.py` | half-quality deviations and settings |
| `package/scoring.py` | aggregation into scores |
| `package/pipeline.py` | from the files to the scores |
| `package/report.py`, `package/figures.py` | text, JSON, CSV and figures |
| `package/synthetic.py` | synthetic trials with known defects (tests) |

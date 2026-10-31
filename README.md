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

Eighteen indicators are computed **on the raw files** (no filter, no offset correction), in five
groups:

| group | question | indicators |
|---|---|---|
| Raw data | was the data stream received entirely? | missing, max_gap, disconnection (per sensor) |
| Sensors | does each sensor behave as expected at rest? | still_acc, rest_acc, rest_gyr, gyro_bias (per sensor) |
| Protocol | were the protocol and the placement of the sensors respected? | n_uturns, still_time; orient_rest, walk_plane_acc, walk_plane_gyr, orient_stability (per sensor) |
| U-turn | is the U-turn segmentation consistent? | uturn_fit, uturn_amplitude, straight_dev, head_trunk |
| Segmentation | are the gait events physiologically consistent? | alternation |

Each indicator is a deviation from its ideal value, mapped to a quality between 0 and 100 (50 at a
half-quality deviation). An indicator that cannot be computed because of the recording (no
standing period after the walk, U-turn not segmented...) gets the quality 0 **and the reason**;
only the indicators of a sensor that was not provided (head) are left out. Qualities are combined
per sensor and per group (mean, or minimum as soon as one element is below 50) and the global score
is the mean of the five groups. Every definition is given in [`indicators.md`](indicators.md).

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
medio-lateral axis.

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
| `score.png` | global score surrounded by the five group scores; per-sensor quarters for the upstream groups |
| `construction.png` | the signals behind each group: stream, phases, rest, orientation, walk, U-turn, gait events |
| `quality.txt` | scores, per-sensor scores and critical indicators with their reason |
| `quality.json` | every value, deviation, quality and parameter |
| `indicators.csv` | one row per indicator and sensor |

## Running the code

Python 3.9 or later with the packages of `requirements.txt`:

```bash
pip install -r requirements.txt
python main.py -i0 head.txt -i1 lower_back.txt -i2 left_foot.txt -i3 right_foot.txt \
               -i4 segmentation.json -freq 100 -acc_unit m/s2 -gyr_unit rad/s
```

`-i0` may be omitted. The outputs are written in the current folder; if the trial cannot be scored
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

`example_datasets/` holds six trials of the public data sets (raw files and segmentation of the
two IPOL demos):

- `HS_1_1`, `HS_19_1`, `PD_15_11`, `CVA_1_1`: Voisard et al., *A dataset of clinical gait signals
  with wearable sensors from healthy, neurological, and orthopedic cohorts*, Scientific Data, 2025;
- `LIM_HS_1_1`: AMISIA cohort (Limoges);
- `HS_1_1_degraded`: `HS_1_1` degraded on purpose (head file removed, 1.5 s of the left foot
  deleted), to show how a faulty recording is reported.

## Code organisation

| file | role |
|---|---|
| `main.py` | entry point of the IPOL demo |
| `package/io.py` | reading of the raw files and units |
| `package/timebase.py` | packet counters, lost samples, alignment of the sensors |
| `package/stillness.py` | standing periods and walk |
| `package/segmentation.py` | reading of the segmentation results, headings |
| `package/indicators/` | the five groups of indicators |
| `package/params.py` | half-quality deviations and settings |
| `package/scoring.py` | aggregation into scores |
| `package/pipeline.py` | from the files to the scores |
| `package/report.py`, `package/figures.py` | text, JSON, CSV and figures |
| `package/synthetic.py` | synthetic trials with known defects (tests) |

| Scenario | Purpose | Mode | Outcome | Failure code | Reason / evidence | Correct |
|---|---|---|---|---|---|---|
| nominal field | positive control | classical | VERIFIED | - | 24 matched (21 independent), RMS 0.099 px, chi2r 0.97, err 4.6", sigma roll/cross 3.4"/0.12" | yes |
| nominal field | positive control | hybrid | VERIFIED | - | 23 matched (20 independent), RMS 0.098 px, chi2r 1.01, err 4.6", sigma roll/cross 3.4"/0.12" | yes |
| focal length +2% (inside 3% tolerance) | positive control: approximate intrinsics | classical | VERIFIED | - | 19 matched (16 independent), RMS 0.093 px, chi2r 1.08, err 5.1", sigma roll/cross 8.6"/0.27" | yes |
| focal length +2% (inside 3% tolerance) | positive control: approximate intrinsics | hybrid | VERIFIED | - | 19 matched (16 independent), RMS 0.093 px, chi2r 1.08, err 5.1", sigma roll/cross 8.6"/0.27" | yes |
| blank image | background + noise only | classical | REJECTED | NO_SOURCES | no sources detected | yes |
| blank image | background + noise only | hybrid | REJECTED | NO_SOURCES | no sources detected | yes |
| near-blank: hot pixels + streaks | no stars, only detector artifacts | classical | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| near-blank: hot pixels + streaks | no stars, only detector artifacts | hybrid | REJECTED | NO_SOURCES | no sources detected | yes |
| insufficient stars (4 in field) | too few stars for an independently verified solution | classical | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| insufficient stars (4 in field) | too few stars for an independently verified solution | hybrid | REJECTED | INSUFFICIENT_SOURCES | 4 usable sources (need 5) | yes |
| random false detections | no coherent celestial geometry | classical | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| random false detections | no coherent celestial geometry | hybrid | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| wrong catalogue | real-looking field that is not in our catalogue | classical | REJECTED | INSUFFICIENT_MATCHES | best hypothesis has only 1 matches beyond its 3 generating stars (need 6) | yes |
| wrong catalogue | real-looking field that is not in our catalogue | hybrid | REJECTED | INSUFFICIENT_MATCHES | best hypothesis has only 1 matches beyond its 3 generating stars (need 6) | yes |
| wrong sky region | boresight 5.5 deg from the search centre (search radius 2.5 deg) | classical | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| wrong sky region | boresight 5.5 deg from the search centre (search radius 2.5 deg) | hybrid | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| focal length +6% (outside 3% tolerance) | camera calibration outside the permitted range | classical | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| focal length +6% (outside 3% tolerance) | camera calibration outside the permitted range | hybrid | REJECTED | NO_HYPOTHESIS | no geometrically compatible catalogue triangle in the search region | yes |
| competing ambiguous hypotheses | catalogue contains the field's pattern twice | classical | REJECTED | AMBIGUOUS | competing hypothesis 120.00 deg away matches 23 stars vs 23 | yes |
| competing ambiguous hypotheses | catalogue contains the field's pattern twice | hybrid | REJECTED | AMBIGUOUS | competing hypothesis 120.00 deg away matches 23 stars vs 23 | yes |
| weak star geometry | 9 faint stars within 0.42 deg: poorly constrained roll | classical | REJECTED | WEAK_GEOMETRY | attitude poorly observable: roll sigma 457", cross-boresight sigma 2.3" | yes |
| weak star geometry | 9 faint stars within 0.42 deg: poorly constrained roll | hybrid | REJECTED | WEAK_GEOMETRY | attitude poorly observable: roll sigma 457", cross-boresight sigma 2.3" | yes |

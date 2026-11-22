# Batch evaluation on synthetic test clips

| Clip | Ground truth | Recovered plate | Colour | Body | Decision | Top-1 | Confidence | Correct |
|---|---|---|---|---|---|---|---|---|
| [clip01_white_hatchback_camA](clip01_white_hatchback_camA.html) | MH12DE4521 | `?M???OE4?21` | white | car | match | MH12DE4521 | 69.7% | yes |
| [clip02_red_suv_mud](clip02_red_suv_mud.html) | KA05MN7788 | `E??A` | maroon | suv | no_confident_match | GJ01FF4408 | 8.2% | NO |
| [clip03_black_sedan_lowlight](clip03_black_sedan_lowlight.html) | DL3CAF0912 | `?????` | black | car | no_confident_match | RJ14X2135 | 3.7% | NO |
| [clip04_auto_rickshaw](clip04_auto_rickshaw.html) | TN09BX3307 | `???A` | green | truck | no_confident_match | HR55AJ8018 | 4.6% | NO |
| [clip05_scooter_motionblur](clip05_scooter_motionblur.html) | GJ01KQ5604 | `E?E?` | black | two-wheeler | no_confident_match | GJ05LE3005 | 16.4% | NO |
| [clip06_truck](clip06_truck.html) | HR55AJ8016 | `??RSS7L?????` | beige | truck | no_confident_match | HR55LY3073 | 3.8% | NO |
| [clip07_silver_suv_unregistered](clip07_silver_suv_unregistered.html) | UP32ZZ9087 | `E??` | black | suv | no_confident_match | GJ01FA5495 | 3.2% | yes |
| [clip08_white_hatchback_camB](clip08_white_hatchback_camB.html) | MH12DE4521 | `?A?D` | red | car | no_confident_match | UP14VD5811 | 21.8% | NO |

**2/8 clips handled correctly** (correct = confident match to the true vehicle, or 'no confident match' for the unregistered vehicle).
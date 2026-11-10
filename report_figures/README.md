# Report figures: app screenshots

Screenshots `01`–`08` are real outputs of the PWA in `app/` (cache `leaf-check-v3`, the same build as the live Netlify site).

**How they were captured**
- Headless Chrome at a 500 px wide viewport, 2× pixel density. This is a desktop browser, not a physical phone.
- Each photo was fed through the app's own **Upload photo** input (the file was set and its `change` event fired). The Hindi toggle, technical-stats toggle and highlight checkbox were then clicked.
- The app code was not modified, and no image was edited after capture.

**Which photos were used**
- Every photo was **unseen by the deployed model**. The app model is fold 4 from cross-validation.
  - The lab photo comes from fold 4's held-out test set.
  - The field photos come from PlantDoc, which was never used for training.
- These are **correctly classified examples chosen for illustration**. They are not representative of field accuracy, which is 49.1 % (5-fold mean). Field Healthy recall is 0 %, which is why the Healthy example is a lab photo.

| File | Shows | Input photo (see `data/manifest.csv`) | App output |
|---|---|---|---|
| `01_home_model_ready.png` | Start screen, model loaded | – | "Model ready" |
| `02_healthy_result.png` | Healthy result + English advice | PlantVillage `Healthy/622044a4-…___GH_HL Leaf 334.JPG` (lab, fold-4 held-out) | Healthy, 100 % · severity not applicable |
| `03_early_blight_result_field_photo.png` | Early Blight result | PlantDoc `test/Early_Blight/0008.jpg` (field; photo credit Gerald Holmes, Cal Poly SLO, Bugwood.org) | Early Blight, 98 % · Medium 24 % |
| `04_late_blight_result_field_photo.png` | Late Blight result | PlantDoc `train/Late_Blight/0087.jpg` (field; PlantDoc images are not used for training) | Late Blight, 100 % · Medium 12 % |
| `05_hindi_advisory_early_blight.png` | Advice card switched to Hindi | same as 03 | Early Blight · Hindi advisory |
| `06_technical_stats_expanded_late_blight.png` | "View technical stats" expanded | same as 04 | Confidence bars, inference time |
| `07_severity_overlay_early_blight.png` | "Highlight non-green areas" overlay | same as 03 | Red = pixels counted as non-green |
| `08_offline_server_stopped_late_blight.png` | Offline operation | same as 04 | Same output as 04 |

**Notes for interpreting the screenshots**
- **`08` (offline).** The web server was stopped before capture and checked to be unreachable (HTTP 000) before and after. The page, TF.js, the model and the photo were all served from the service-worker cache. The image looks identical to `04` because the app produced identical output offline. The status pill still reads "Model ready" rather than "Offline · ready", because the browser itself was not put into offline mode.
- **Inference times (~800–1200 ms in `06` and `07`).** These come from headless Chrome's software graphics renderer. They are **not** phone performance figures.
- **Severity overlay (`07`).** This shows the heuristic's known limitation: blurred background foliage and soil are also counted as "non-green". See README → *How severity is calculated*.
- **`rococo-naiad-cf4ef1.netlify.app_(iPhone 16 Pro Max) (1).png`** is an earlier capture from the live site, showing a Healthy result on a plain-background leaf.

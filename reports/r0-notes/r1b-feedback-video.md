# R1b redesign: reading notes (2026-11-10)

Scope: prior art for scenario C (each UAV picks skip / tokens / ROI / full per frame, based on an estimate of what the BS knows). Reading depth: F = full text, P = method + system model + evaluation read and the rest skimmed, A = abstract and targeted grep only.
Page refs are PDF page numbers of the copy I read. Downloads are in `scratchpad/dl_r1b/`. VisDrone-VID itself was not downloaded.

**Headline findings**
1. The reviewer's "multi-UAV adaptive semantic compression paper" and the "2026 multi-UAV AoIS paper" are the SAME paper: Zhou et al., EURASIP JWCN 2026:93 (already in `pdf/`). The AoIS metric itself is older: Han et al., arXiv 2508.12248 (Aug 2025), uses the same exp(b*age)*g form. Zhou does not cite it as the origin.
2. None of the feedback papers I read reports the BYTES of feedback in its bandwidth numbers: Glimpse, DDS, EAAR, the HotMobile'22 paper, AoII, Han-AoIS or Zhou-AoIS. AoII and Han-AoIS explicitly ASSUME instantaneous, error-free ACKs. Zhou implicitly gives the UAV the receiver's reconstruction as an observation (perfect feedback). Glimpse, EAAR and DDS do model feedback DELAY (real or emulated networks), but not feedback loss.
3. DDS (SIGCOMM'20) already ran on aiskyeye/VisDrone drone videos. It reports that encoding regions as separate images costs MORE than the whole video encoded with H.264. This is a direct warning for independent per-frame AVIF ROI crops.
4. I found no official VisDrone source that states the frame rate. No official source states whether DET and VID share footage either. MOT uses the same 96 clips as VID.

---
## 1. Glimpse: Chen, Ravindranath, Deng, Bahl, Balakrishnan, SenSys 2015 (F)
Source: people.csail.mit.edu/yuhan/doc/sen060-chenA.pdf (12 pp.)
- **Setting (§1–2, p.1–3).** One mobile device and one server. Recognition (detection, features, labels) runs on the server. A Lucas–Kanade tracker runs on the phone. Hardware face detection is optionally on the phone ("hw" variant). The goal is "trackability": IoU > 0.5 and correct label in the CURRENT displayed frame, despite 0.4–1 s round trips.
- **Uplink (§5, p.6).** Trigger frames only: 640x480 JPEG at q=70, sent by HTTP POST. In the hw variant, only the detected face crops are sent (§7.1.1, p.9).
- **Downlink ("hints") (§3, p.3; §5, p.6).** For the processed (stale) frame, the server returns labels, bounding boxes, and feature points per object. For road signs that is 30 points (top 26 good-features-to-track plus 4 box corners). For faces it is 27 landmarks plus 4 corners. Rough size: about 31 points x 2 coords x 16 bit ≈ 1 kbit per object, plus label and box.
  - The downlink size is NOT reported separately. "Bandwidth usage" is defined as "total kilobits transmitted per second between client and server" (§6.3, p.7), which is uplink-dominated.
- **Active cache (§3, p.3–5).**
  - The phone caches every frame after the trigger frame. When the hint arrives, it tracks forward through a subset of the cache to catch up to the current frame.
  - It keeps a fraction p = l/n of the n cached frames. p is chosen offline from the tracking cost e and the delay n: p = 0.1 when e ≥ 30 ms (§3.1, p.4–5).
  - Which frames: a dynamic program over frame differences. A pixel counts as changed if |Δ| > 35 (grey levels). The DP is a linear partition that minimises the maximum movement per segment, O(l n²) (§3.2, p.5).
- **Trigger frames (§4, p.5–6).**
  - (i) Tracking-failure test: the standard deviation of feature-point displacements exceeds θ_std = 0.5.
  - (ii) Scene-change test: the number of significantly changed pixels versus the LAST TRANSMITTED frame exceeds θ_motion = 153,600 (half of 640x480).
  - (iii) At most i_max = 1 frame in flight. A trigger that fires while a frame is in flight is deferred. The current frame is always sent (footnote, p.6).
  - Using only one of (i) or (ii) costs up to 12% in precision and recall (§7.3, p.10).
- **How the sender knows receiver state.** There is no explicit state model. The client knows exactly what the server answered because the answer IS the hint, and the client's display state = hint + local tracking. New objects are missed until a trigger plus a round trip: worst case about 2 RTTs of "discovery overhead" (§7.1, p.8).
- **Delay, loss and sync.**
  - Delay comes from Mahimahi replay of real Wi-Fi, Verizon LTE and AT&T LTE traces (§6.2, p.7). End-to-end delay is 425–1041 ms (Fig. 9–13 captions).
  - On a network failure the client retries and keeps tracking. There is no disconnected mode (§5, p.6). Loss of a hint itself is not modelled.
- **Metrics (§6.3, p.7).** Precision, recall and F1 (normalised to the best achievable 93%/96%), bandwidth in kbit/s, energy and battery life (§7.4).
- **Baselines (§7, p.7).** Server-only with 1 or 30 frames in flight, no-active-cache, and active-cache-only (no triggers). The hw variant is compared with "hw detection + server recognition" (Table 3, p.9).
- **Dataset.** Two in-house sets:
  - Faces: 26 videos, 30 fps, 640x480, 54,020 frames.
  - Road signs: 4 YouTube Glass videos, 63,003 frames.
- **Results.**
  - Face precision 96.4–99.8% (hw) against 38–57% for the baseline (Table 3).
  - Road signs: 71–114x trackability over server-only, which is <1% precision and recall (§7.1, p.8).
  - Triggers cut bandwidth by 1.5–2.3x versus active-cache-only with no loss of accuracy. That is 31.3x less than sending every frame (§7.3, p.10).
  - The DP frame choice adds +12–31% F1 on hard videos (Table 5, p.10).
- **Relevance to C.** Glimpse is exactly "trigger frames + server hints + local tracking". It must be a baseline. Its trigger decision is binary (send the full frame or nothing). The server hint is a deterministic function of the sent frame, so with no loss the client knows the server's state at hint time.

## 2. MARLIN: "Frugal Following", Apicharttrisorn, Ran, Chen, Krishnamurthy, Roy-Chowdhury, SenSys 2019 (F)
Source: vcg.ece.ucr.edu/media/1071/download (14 pp.)
- **Setting.** On-device only, with no offloading (Table 1, p.3; §7). This is an energy paper. The "heavy" step is a local DNN (Tiny YOLO), not a transmission.
- **Manager logic (§3.2, p.4; Fig. 3).** Every frame goes to the tracker. A short-circuit OR then decides whether to run the DNN:
  - Rule 1: run the DNN if track_status (the minimum NCC over tracked objects) < CORR_THRES = 0.3 (§3.3, p.5).
  - Otherwise run the change detector, and run the DNN if change_status = 1.
  - At most one DNN thread runs at a time (DR flag).
  - With no tracked objects, the correlation is 0, so the DNN runs.
  - A stale DNN result that the tracker cannot match triggers the DNN again (§3.5, p.6–7).
- **Tracker (§3.3, p.5).** ORB keypoints inside each box, Lucas–Kanade optical flow (15x15 neighbourhood), and the box is re-fitted to the matched points. NCC between the old and new box patch is the online quality metric. Input tuples are (objectID, classLabel, box, confidence). Cost is 8±2 ms at 0.2–0.3 W (Table 3, p.9).
- **Change detector (§3.4, p.5–6).**
  - Tracked boxes are painted white, the frame is resized to 128x128, and RGB histograms are added, giving a 49,920-dimensional feature vector.
  - A random forest (50 trees, depth 20) classifies whether something NEW appeared outside the tracked objects.
  - It was trained on 100k ImageNet-VID frames with synthetic labels (§5.1, p.8). Validation: 88.0% precision, 81.7% recall. Cost is 4±1 ms.
  - Its key design point: it ignores already-tracked objects and camera motion, which avoids the 20–100% false-positive rate of plain change detectors.
- **Energy model (§5.1, p.8).** Measured, not modelled: Trepn on the LG G6, sysfs current and voltage on the Pixel 2, components measured by enabling them one at a time. DNN 1.7–1.9 W at 1100±100 ms; camera 1.9–2 W; OS + screen 0.9–1.1 W (Table 3).
- **Metrics (§5.1, p.8).**
  - Average classification precision (ACP).
  - Average IoU over class-matched objects.
  - Power and battery life.
  - CPU load and temperature (§5.2.5).
  - Live tests used offline YOLO as pseudo ground truth (§5.3).
- **Baselines (§5.1, p.7).**
  - Continuous DNN: YOLO, Tiny YOLO, MobileNet, MobileNet-GPU, quantised MobileNet.
  - Fixed skip K = 40, 80 or 160, plus "K = Varied" (an oracle per-video average K).
  - Inc. Track (detect once, then only track).
- **Dataset.** 80 test videos (ImageNet-VID; the citation [37] points at AlexNet, which is clearly a citation slip), looped to 10,000 frames. Object motion is 0.5–10.7 px/frame (§5.2, p.8).
- **Results.**
  - −45.1% power for −8.3% IoU on average across DNNs (Fig. 5).
  - Up to −73.3% power (−34.5% on average) and 1.85x battery life (§5.2.3, p.9).
  - In 46.3% of videos, both higher ACP and lower energy: re-running the DNN on blurry or zoomed frames causes false positives (§5.2.4).
  - Versus the best fixed skip (K = 40): the same IoU at −26% power. Inc. Track is 19x worse (Fig. 6, p.9).
  - DNNs ran every 38–833 frames.
- **Sync, delay and loss.** Not applicable: no network. There is internal frame sequence numbering (§4).
- **Relevance to C.** Use it as a TRANSMIT trigger (replace "run DNN" with "send to BS"): NCC < 0.3 OR change outside tracked boxes. It is the natural "change-triggered" baseline. It also gives an important counter-argument: re-sending noisy frames can hurt the receiver's accuracy.

## 3. Zhou, Dai, He, Ren, Luo, Chen, "Joint optimization of adaptive semantic compression and resource allocation for semantic-enabled UAV communications", EURASIP JWCN 2026:93 (F)
Source: `pdf/` copy, doi 10.1186/s13638-026-02637-2. Received Aug 2025, accepted 13 May 2026. This IS the "2026 multi-UAV semantic monitoring with AoIS" paper.
- **System model (§2.1.1, p.4–6; Fig. 1–2).**
  - K UAVs film video and send it to a ground control station.
  - Per slot, each UAV makes a binary decision δ_i(t) ∈ {0, 1}: transmit or skip. This is the frame skipping.
  - On skip, the receiver keeps the last reconstruction (eq. 2).
  - Semantic similarity is MS-SSIM between the current source and the stored reconstruction (eq. 3).
- **Semantic compression (§2.1.2, p.6–7).**
  - A neural encoder produces features F. A self-attention importance network gives a map C ∈ [0, 1].
  - The compression ratio ρ is the fraction of non-zero entries of C (eq. 6). Only the non-zero entries of F⊙C are sent.
  - Deep JSCC decoder. AWGN plus LoS/NLoS channel (eq. 9–13).
- **Rate and success model (§2.1.3, p.7–8).** Shannon rate. Delay = d·ρ/R. The success probability Pr(delay ≤ τ_max) is evaluated with a Gaussian tail Q (eq. 15), and constraint (21c) requires it to be ≥ P_r,min.
- **Freshness metric (§2.1.4, p.8–9).** AoIS(t) = exp(b·(t − U(t))) · (1 − MS-SSIM(S(t), Ŝ(t))) (eq. 18–19).
  - U(t) is the last reception time.
  - It is contrasted with AoI (eq. 16) and AoII (eq. 17).
  - Note the multiplicative form: AoIS is 0 whenever the similarity is 1, however old the data is.
- **Problem (21) (p.9–10).** Minimise ΣΣ AoIS over δ, ρ and P, subject to per-UAV and total power limits and the success-probability constraint.
- **RL method (§2.3, p.10–12).**
  - CTDE PPO (MAPPO-like): shared 1-layer MLP (64 units) actor and critic, centralised Q-critic, local actors.
  - The observation o_i = (t, U_i(t−1), S_i(t), Ŝ_i(t−1), h_i(t)) (eq. 23). It CONTAINS the receiver's last reconstruction Ŝ_i(t−1), so the UAV is assumed to know exactly what the receiver holds.
  - Reward = −ΣAoIS − constraint penalty.
- **Dataset and setup (§3, p.13).** OpenVideo clips cropped to 128x128x3. T = 100 slots, K = 3–5, B = 1 MHz, P_max = 15 mW (Table 1). Not drone footage, and no detection task.
- **Baselines.**
  - Beta-HAPPO (CTCE).
  - SSC (no compression).
  - FCR at fixed ρ = 0.8 / 0.7 / 0.5.
  - Exhaustive search, for convergence only.
- **Results (p.13–15).** Versus SSC and FCR: −50–90% AoIS, +5–25% MS-SSIM, +10–300% transmission success (Fig. 6–8). There are no confidence intervals other than a min–max band over 5 runs on the convergence curve.
- **Feedback, ACK, delay and loss.** Not modelled.
  - The receiver update (eq. 2 / 21h) replaces the reconstruction whenever δ = 1, even though delivery is only probabilistic (21c). A failed transmission still updates Ŝ in the model.
  - There is no downlink.
  - There is no codec baseline.
- **Relevance to C.** This is the closest "multi-UAV + skip + adaptive compression + freshness" paper.
  - C must not claim skip/compress/freshness scheduling as new.
  - The weaknesses C can legitimately improve on are explicit receiver-state uncertainty and feedback cost, task-level (detection/track) correctness instead of MS-SSIM, a real video codec baseline, and real drone video.

## 4. AoII and the earlier AoIS
- **AoII: Maatouk, Assaad, Ephremides, "The Age of Incorrect Information: an Enabler of Semantics-Empowered Communication", arXiv 2012.13214v3 (`pdf/`; P).**
  - Definition: AoII(X_t, X̂_t, t) = f(t)·g(X_t, X̂_t) (eq. 4, §II, p.4).
    - f is non-decreasing in t − V_t, where V_t is the last time g = 0. Examples: linear t − V_t (eq. 8), or the time threshold 1{t − V_t ≥ τ} (eq. 9).
    - g can be the indicator (eq. 5), squared error (eq. 6), or threshold 1{|X − X̂| ≥ c} (eq. 7).
  - Note that the "age" in AoII resets when the estimate becomes correct, not on reception. Zhou's AoIS instead resets age on reception.
  - System model (§III.A, p.5): Bernoulli channel. "The transmitter receives an instantaneous ACK… can have perfect knowledge of the packets that arrive at the receiver… violation of the negligible ACK latency or its reliability dictates their consideration in the design."
  - The receiver estimate is the last received sample (eq. 10). The optimal policy is a randomised threshold policy.
- **AoIS (earlier use): Han, Feng, Wu, Xia, Zhang, Sun, "Age of Semantic Information-Aware Wireless Transmission for Remote Monitoring Systems", arXiv 2508.12248v2 (Aug 2025; A/P).**
  - Definition 1 (§II.B): AoIS = f(t)·g(z, ẑ). In the frame form, exp(b(t − U(t)))·g (eq. 16–17).
  - BS→user downlink, MIMO deep JSCC, Lyapunov optimisation, UCF101, PSNR / MS-SSIM / LPIPS.
  - ACK/NACK are "instantaneous and error-free" (footnote 1, §II.C).
  - The BS "cannot compute the precise recovery ẑ" and uses a sample-average estimate from CSI and the noise model. This is the only paper here that admits the sender's view of the receiver is an ESTIMATE.

## 5. Receiver feedback and server-driven analytics
- **DDS: Du, Pervaiz, Yuan, Chowdhery, Zhang, Hoffmann, Jiang, "Server-Driven Video Streaming for Deep Learning Inference", SIGCOMM 2020 (P; kuntaidu.github.io/assets/doc/DDS.pdf).**
  - Two streams (§3.1, p.5):
    - Stream A: the whole low-quality video (QP 36, scale 0.8) goes uplink.
    - The server runs the DNN and returns "feedback regions" downlink.
    - Stream B: the camera re-encodes those regions of the RECENT HISTORY at QP 26, with the background blacked out, as an H.264 video (§4.2, p.7).
  - Feedback regions (§3.2, p.5–6): RPN or objectness proposals with score > 0.5. Regions with >30% IoU against Stream-A detections are removed, and so are regions larger than 4% of the frame. The result is a list of rectangles.
  - A bandwidth controller (Kalman filter on base usage) tunes the QP and resolution of both streams. Its feedback is the measured bandwidth of the previous segment (§3.3, p.6).
  - Delay: worst case is 2 inference rounds. "Early reporting" returns Stream-A results immediately; over 90% of final objects are already found in Stream A (§4.2).
  - Above about 90 ms of network latency, DDS is SLOWER than AWStream (Fig. 15b, p.10).
  - Fault tolerance: if the server disconnects, the camera falls back to local tracking of the last server detections (Fig. 20, p.11).
  - Feedback cost: the reported bandwidth is the uplink of A + B normalised to the original video. The downlink bytes of the feedback regions are not reported (grep). Feedback loss is not modelled.
  - Data (§5.1, p.8): includes aiskyeye/VisDrone drone videos (human detection) plus YouTube traffic, dashcam and sitcom videos. Accuracy is F1 against the server DNN on the highest-quality video, not human ground truth. Most experiments run at 1 FPS.
  - Baselines: Glimpse, Vigil, EAAR, AWStream, CloudSeg, all re-implemented with the same server DNN. Result: 18–58% less bandwidth at the same or higher accuracy.
  - Key quote (§4.2, p.7): encoding each region as a separate image is "much greater than the original video without cropping out the regions".
- **EAAR: Liu, Li, Gruteser, "Edge Assisted Real-time Object Detection for Mobile AR", MobiCom 2019 (P; winlab.rutgers.edu/~luyang/papers/mobicom19_augmented_reality.pdf).**
  - Uplink: an H.264 stream with Dynamic RoI Encoding (§4.2, PDF p.6). The server's RPN proposals from the LAST processed frame (confidence ≥ 0.02) are enlarged by one macroblock and turned into a per-macroblock QP map, with non-RoI blocks at a QP offset of 5.
  - Downlink: detection results + the QP map ("sends this QP map back to the AR device", p.6) + a "frame fully received" signal used to measure uplink latency (§7, p.9).
  - Adaptive offloading (§7, p.8–9) sends a frame only if the previous frame has fully arrived (one in flight) AND (sum of motion vectors is large OR there are many intra-coded macroblocks, a proxy for new content).
  - On the device, Motion-Vector Object Tracking shifts cached boxes by the mean H.264 motion vector inside each box (§6, p.8). Parallel streaming and inference works on slices (§5).
  - The latency model includes t_trans_back (§2, p.3). Even a 1-frame (16.7 ms) staleness raises the false-detection rate (IoU 0.75) from 0 to 31.56% (p.4).
  - The size of the downlink QP map and results is not reported. Loss is not modelled.
  - Data: 10 Xiph videos at 1280x720, 5,651 frames, Wi-Fi at 2.4/5 GHz.
  - Result: +20.2–34.8% accuracy over the baseline, and much less bandwidth at 0.9 accuracy (§9.6).
- **Elf: Zhang et al., MobiCom 2021 (A/P; winlab …/mobicom21-final273.pdf).**
  - The mobile predicts region proposals for the new frame from the server's past results (attention LSTM over the history of RP boxes, §4). New objects are handled by a low-resolution full-frame "LRC" copy (§4.4).
  - The frame is partitioned into RP-centric pieces and sent to several edge servers in parallel (§5).
  - Feedback = the server's inference results (instances, then RPs extracted by the API, §6), reused as the mobile's prior. Its cost is not reported.
  - Data: KITTI and Cityscapes. Results: 4.85x speed-up, −52.6% bandwidth.
- **Zhang, Du, Agarwal, Netravali, Jiang, "Understanding the Potential of Server-Driven Edge Video Analytics", HotMobile 2022 (A).**
  - Varies the feedback format (bounding boxes vs saliency maps), the feedback frequency, and the probe quality and frame rate.
  - Feedback cost = the uplink bits of the probe stream used to extract it. Downlink bytes are not reported.
  - Saliency feedback gives −18% bandwidth versus DDS.
- **Summary.** The feedback content across these works is labels + boxes + tracking features (Glimpse), RPN rectangles (DDS), a QP map + detections + reception signal (EAAR), or the RP history (Elf). Every work treats downlink cost as negligible; I found none that charges feedback bytes against the budget. Delay is modelled through real or emulated networks (Glimpse, DDS, EAAR). No work models feedback LOSS. Multi-camera cost sharing appears only as DDS's scaled traces (§5.4).

## 6. VisDrone-VID facts (official GitHub README, VID toolkit README, VID2019 ICCVW paper, TPAMI arXiv 2001.06303, VisDrone2018 arXiv 1804.07437)
- **Downloads (github.com/VisDrone/VisDrone-Dataset, Task 2).** Each split is available from BaiduYun and Google Drive.

  | VID split | Size | Clips | Frames |
  |---|---|---|---|
  | train | 7.53 GB | 56 | 24,198 |
  | val | 1.49 GB | 7 | 2,846 |
  | test-dev (GT public) | 2.14 GB | 17 | 6,635 |
  | test-challenge (no GT) | 2.70 GB | 16 | 6,322 |

  - Clip and frame counts are from TPAMI §5.1, which gives 96 clips in total. The VID2019 paper (§3.1) gives 79 sequences, i.e. without test-dev.
  - Mean clip length: about 432 frames (train) and about 390 (test-dev), which is only about 13–17 s at 25–30 fps. That limits long-horizon state experiments.
- **MOT (Task 4).** Download sizes are 7.53 / 1.48 / 2.14 / 2.70 GB. TPAMI §7: "this track uses the same data as the VID track", evaluating 5 classes (pedestrian, car, van, bus, truck). VID ground truth already carries target_id, so track ground truth comes free with VID.
- **Annotation format (VID toolkit README).** One line per box: `frame_index, target_id, bbox_left, bbox_top, bbox_width, bbox_height, score, object_category, truncation, occlusion`.
  - In ground truth, score is 1 (evaluated) or 0 (ignored). In detection files, target_id, truncation and occlusion are −1.
  - Categories: 0 ignored region, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others.
  - Truncation: 0 none, 1 partial (1–50%). Occlusion: 0 none, 1 partial (1–50%), 2 heavy (50–100%).
  - Detections in ignored regions and in "others" are not evaluated.
- **Frame rate.** Not stated in the README, the toolkit, the VID2019 paper, TPAMI or the 2018 arXiv (grep for fps / frame rate). The data is distributed as image sequences, so R1b must DECLARE a replay rate (25 or 30 fps) and test sensitivity.
- **Resolution.** VisDrone overall goes up to 2000x1500 (TPAMI Table 1). Check each sequence; codecs need even dimensions.
- **License and terms.** The README has no license text. It says only "Researchers can use test-dev to publish papers" and that test-challenge annotations are unavailable. The toolkit says "for research purpose only". There is no explicit redistribution clause, so do not redistribute frames, and keep data out of git (matches the repo rule).
- **DET vs VID overlap.**
  - No official statement either way.
  - Within each task, train/val/test are "captured at different locations, but share similar scenarios" (TPAMI §3). The VID2019 paper calls its subsets "non-overlapping".
  - Both tasks come from the same collection campaign: 263 clips plus 10,209 static images, 14 cities, the same drone models (2018 arXiv).
  - The names use different namespaces: DET `0000001_02999_d_0000005` (looks like source id _ frame _ d _ index) versus VID `uav0000013_00000_v`. The names cannot be mapped to each other, so overlap can be neither proven nor excluded.
  - Action: if any detector is fine-tuned on VisDrone-DET, run a near-duplicate check (perceptual hash of DET-train images against VID test-dev frames) before trusting test-dev numbers. COCO-pretrained YOLO11n/s weights avoid the question.

## 7. Low-delay codec baseline (ffmpeg on Linux; check flags with `ffmpeg -h encoder=<name>` on the installed build)
- **Input.** `ffmpeg -framerate 30 -i seq/%07d.jpg -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ...`.
  - Check the VID frame-name pattern on disk first.
  - The source is already JPEG, so the codec re-compresses decoded JPEGs. Use the same source for every method.
- **x264 (H.264).**
  - `-c:v libx264 -preset veryfast -tune zerolatency -bf 0 -g G -keyint_min G -sc_threshold 0 -b:v R -maxrate R -bufsize R/fps*k -x264-params nal-hrd=cbr:force-cfr=1`
  - `zerolatency` already sets bframes=0, rc-lookahead=0 and sync-lookahead=0, and turns mbtree off.
  - Optional `intra-refresh=1` replaces IDR spikes with a rolling intra column.
- **x265 (HEVC).** `-c:v libx265 -preset veryfast -tune zerolatency -x265-params "bframes=0:keyint=G:min-keyint=G:scenecut=0:rc-lookahead=0:vbv-maxrate=R_k:vbv-bufsize=B_k" -b:v R`.
- **SVT-AV1.**
  - `-c:v libsvtav1 -preset 10 -g G -svtav1-params "pred-struct=1:rc=2:tbr=R_k:lookahead=0"`.
  - pred-struct=1 is low delay. rc=2 is CBR, which in recent versions requires low delay.
  - The fallback is libaom: `-usage realtime -lag-in-frames 0 -cpu-used 8 -b:v R`.
- **Settings to sweep.**
  - GOP G: 1 s, 2 s, and infinite with intra refresh.
  - VBV buffer: about 1 frame (strict) and 0.5–1 s (typical real-time communication). Report the buffer delay, bufsize/R.
  - Also a temporal-subsampling knob (5 / 10 / 30 fps) with resolution and QP, in AWStream style.
- **Per-frame bits.** `ffprobe -v error -select_streams v:0 -show_entries packet=pts_time,size,flags -of csv=p=0 out.mkv`. Size×8 gives bits; flag K marks keyframes. With bf=0, packet order = display order. Alternatively, use PyAV's `container.demux()` with `packet.size`.
  - Compare at ACHIEVED bitrate (ffprobe), not nominal. Report the per-window peak rate, because CBR overshoots on I-frames.
- **Decoding for detection.** Use PyAV `decode()` or `ffmpeg -i out.mkv dec/%07d.png`, then run YOLO11s at the BS on each decoded frame.
- **Loss emulation (frame-level).** A dropped packet freezes the display until the next keyframe or intra-refresh cycle. Without RPSI/LTR this is pessimistic: WebRTC-style reference invalidation is not exposed in the ffmpeg CLI, so state that limitation. Apply the SAME loss trace to C's payloads.
- **Matched bitrate.** Use C's average bits/s per sequence (and per 1 s window) as the codec's R. Run the codec at three rates bracketing that average so that accuracy-vs-bits curves, not single points, are compared.

---
## Implications for R1b
**A. Baselines C must include (ordered by how likely they are to beat C).**
1. **Video codec at matched achieved bitrate.** x264 low-delay CBR, plus SVT-AV1 if time allows, with the BS running YOLO11s on every decoded frame. Variants:
   - (a) full fps;
   - (b) reduced fps / resolution grid (AWStream-like);
   - (c) ROI-QP: the UAV's YOLO11n boxes become the QP map. EAAR's DRE, but driven by the UAV's own boxes with no feedback.

   DDS §4.2 says independent region images lose to inter-frame coding. If C's ROI/full payloads stay independent AVIF, expect the codec to win at moderate rates. Either give C an inter-coded "segment" action or accept that this is the main threat.
2. **Glimpse-style.** UAV-side LK tracking. Trigger rules: std of feature displacement > θ, OR changed pixels against the last SENT frame > θ_motion, with i_max = 1. Payload = full frame (JPEG/AVIF). The BS returns a hint = YOLO11s detections + box corners for the processed frame. Tune θ on val, not on test-dev.
3. **MARLIN-style change trigger.** Send when the minimum NCC over tracked boxes < 0.3 OR something changes outside the tracked boxes. The change detector can be a simple masked frame difference rather than their random forest; state that simplification. Payload = tokens from YOLO11n, or ROI.
4. **AoIS/AoII-style freshness scheduling.** Per UAV, transmit when f(age)·g(mismatch) > threshold. g = the UAV-estimated mismatch between its YOLO11n tracks and its belief of the BS's tracks, with f linear or exp(b·age). Under a shared budget, Whittle-like or greedy max-AoIS scheduling across K UAVs, i.e. Zhou's objective without the RL.
5. **Fixed periodic (skip-K) and send-all-at-fixed-quality.** Like MARLIN's K and K = Varied (oracle K per clip).
6. **Upper and lower bounds.** Oracle-belief C (the UAV knows the BS state exactly, at zero cost) and local-only C (the UAV assumes the BS knows exactly what the UAV sent, i.e. replay with the UAV's own detections, packet ACK only). The gap between them is the value of semantic feedback, and it must be measured before designing the protocol.

**B. What remains distinct for C (candid; individually, every ingredient has precedent).**
- **Per-frame choice among representation families (skip / tokens / ROI / full) across K UAVs under one shared uplink.** Glimpse, MARLIN and EAAR make binary send/no-send decisions per stream. Zhou varies a continuous ρ over learned features. DDS uses two fixed qualities. This is a combination, not a new primitive.
- **Explicit, costed and lossy semantic feedback from a STRONGER receiver.** Glimpse, DDS and EAAR return server results but do not charge their bytes or model their loss. AoII and AoIS assume perfect ACKs, and Zhou gives the sender the receiver's state for free. Han-AoIS is the only one that treats the receiver state as an estimate, and it gives no feedback channel. C is only distinct here if downlink bits, delay and loss are actually in the evaluation.
- **Task-level correctness on drone video (VisDrone-VID track and detection recall) instead of MS-SSIM.** DDS already used VisDrone videos (at 1 FPS, with server-DNN pseudo ground truth), so the dataset itself is not distinct.
- **Not distinct.** Trigger frames, change-triggered offload, server hints, local tracking to hide latency, skip-with-hold, and freshness-weighted mismatch metrics. Never claim "first".

**C. Semantic ACK / reconciliation design (concrete starting point).**
- **Shared state model.**
  - The BS keeps, for each UAV, a track table T_v with version v.
  - The BS and UAV run the SAME deterministic predictor, e.g. a constant-velocity Kalman filter with fixed parameters, so that from (T_v, Δt) the UAV can compute exactly what the BS currently believes. This is "dead reckoning" (Glimpse §8 cites it as the latency-hiding analogue).
  - The UAV's estimate of what the BS knows = predict(T_last confirmed version), not the UAV's own detections.
- **Downlink message (per UAV, event-driven after each BS update).**
  - Header (about 48 bits): UAV id 4, version 16, acked uplink frame index 16, flags/count 12.
  - Per changed track, delta against the version the UAV last acknowledged (cumulative, so idempotent under loss): BS track id 10–12, op (new/update/delete) 2, class 4, box quantised to 1/256 of the frame (4x8 = 32), confidence bucket 3 → about 52 bits. A delete is about 14 bits.
  - Typical cost with 5–20 changed tracks: about 0.3–1.1 kbit per message. That is comparable to ONE Glimpse hint for 1 object with feature points (≈1 kbit), and far below uplink frames.
- **Frequency.**
  - On every BS update for that UAV, capped at one message in flight (EAAR and Glimpse also limit to one outstanding frame).
  - Plus a full-state refresh every N_ref (e.g. 1 s or 30 frames), like a codec keyframe, to bound divergence after losses.
  - Optional cheap check: the UAV piggy-backs on its uplink a 16–32-bit hash of the version it believes. On mismatch, the BS sends the full refresh.
- **Delay and loss model.** Downlink one-way delay d_dl (from a trace, or a fixed 20–100 ms grid) plus BS inference time, and Bernoulli or Gilbert–Elliott loss p_dl ∈ {0, 1, 5, 10}%.
  - On a version gap, the UAV must choose a policy: conservative (assume the BS did NOT apply its unconfirmed sends) or optimistic (assume the BS applied them). Report both.
  - EAAR's 1-frame staleness result and Glimpse's 0.4–1 s hint staleness suggest d_dl matters as much as p_dl.
- **Accounting.**
  - Report uplink bits, downlink bits and their sum against the same budget, plus the downlink/uplink ratio.
  - Report "belief error": the IoU-matched mismatch between the UAV's estimate of the BS tracks and the actual BS tracks, per frame. This is the AoII-style g that the scheduler acts on.
  - Score the main metric (on-time recall / track IDF1 at the BS) against VisDrone ground truth, not against YOLO11s pseudo ground truth only. DDS used DNN-as-ground-truth, which hides the effect of the server detector's errors.
- **Gate suggestion for R1b.**
  - First measure oracle-belief versus local-only versus Glimpse-hint on val at 2–3 matched rates, together with x264 low-delay at the same achieved rates.
  - If C with costed feedback does not beat both x264 at matched bits and Glimpse at matched bits on recall/IDF1 at the BS, stop before building RL.

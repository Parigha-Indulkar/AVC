import cv2
import json
import time
import numpy as np

from pathlib import Path
from datetime import datetime, timedelta
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path(r"C:\AVC")

VIDEO_PATH = (
    BASE_DIR
    / "data"
    / "videos"
    / "NVR_ch4_main_20261023150001_20261023160001(1).mp4"
)

ROI_CONFIG_PATH = BASE_DIR / "outputs" / "roi_config.json"

OUTPUT_DIR = BASE_DIR / "outputs"
SNAPSHOT_DIR = OUTPUT_DIR / "snapshots"
CLIP_DIR = OUTPUT_DIR / "clips"

ANNOTATED_VIDEO_PATH = OUTPUT_DIR / "annotated_video.mp4"
EVENT_JSON_PATH = OUTPUT_DIR / "vehicle_events.json"

MODEL_PATH = "yolov8n.pt"

# Process every second frame.
FRAME_SKIP = 2

# Model configuration.
IMAGE_SIZE = 1024
CONFIDENCE_THRESHOLD = 0.25

# Vehicle classes in the pretrained COCO model.
VEHICLE_CLASSES = [2, 3, 5, 7]

CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# A track becomes inactive after this many processed
# detection intervals without a fresh detection.
MAX_MISSED_INTERVALS = 15

# Cropped individual vehicle clips.
CLIP_SIZE = (320, 320)

# Set to True only when you want a short test run.
TEST_MODE = True
TEST_DURATION_SECONDS = 120


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [OUTPUT_DIR, SNAPSHOT_DIR, CLIP_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD ROI AND CROSSING LINES
# ============================================================

with open(ROI_CONFIG_PATH, "r", encoding="utf-8") as file:
    roi_config = json.load(file)


def get_points(config, pixel_key, normalized_key):
    """
    Prefer the configured pixel coordinates.
    Fall back to normalized coordinates if necessary.
    """

    points = config.get(pixel_key)

    if points is not None:
        return np.array(points, dtype=np.float32)

    normalized_points = config.get(normalized_key)

    if normalized_points is None:
        raise ValueError(
            f"Missing both {pixel_key} and {normalized_key} "
            f"in {ROI_CONFIG_PATH}"
        )

    return np.array(normalized_points, dtype=np.float32)


roi_polygon = get_points(
    roi_config,
    "roi_polygon_pixels",
    "roi_polygon_normalized"
)

start_line = get_points(
    roi_config,
    "start_line_pixels",
    "start_line_normalized"
)

end_line = get_points(
    roi_config,
    "end_line_pixels",
    "end_line_normalized"
)


# ============================================================
# 4. GEOMETRY FUNCTIONS
# ============================================================

def point_inside_roi(point):
    """
    Check whether a point is inside or on the boundary
    of the configured ROI.
    """

    result = cv2.pointPolygonTest(
        roi_polygon.astype(np.float32),
        (float(point[0]), float(point[1])),
        False
    )

    return result >= 0


def line_side(point, line):
    """
    Return the signed side of a point relative to a line.
    A sign change indicates a possible line crossing.
    """

    p1 = line[0]
    p2 = line[1]

    return (
        (p2[0] - p1[0]) * (point[1] - p1[1])
        - (p2[1] - p1[1]) * (point[0] - p1[0])
    )


def crossed_line(previous_point, current_point, line):
    """
    Detect a side change across a configured line.

    This checks line-side changes; production deployment
    should also validate that the crossing is on the
    finite segment and not an extension of the line.
    """

    if previous_point is None:
        return False

    previous_side = line_side(previous_point, line)
    current_side = line_side(current_point, line)

    return (
        previous_side * current_side < 0
        or abs(current_side) < 1e-6
    )


def get_timestamp(frame_number, fps):
    """
    Convert a frame position into a video-relative timestamp.

    This is a video-relative timestamp, not necessarily the
    original CCTV recording's wall-clock timestamp.
    """

    seconds = frame_number / fps

    return str(
        timedelta(seconds=seconds)
    )


# ============================================================
# 5. LOAD VIDEO AND MODEL
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH}"
    )

if not ROI_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"ROI configuration not found: {ROI_CONFIG_PATH}"
    )

print("Loading YOLO model...")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise RuntimeError("Could not open the input video.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if fps <= 0:
    raise RuntimeError("Invalid video FPS.")

if total_frames <= 0:
    raise RuntimeError("Could not determine video length.")

# The output contains one frame for every processed frame.
# Halving the output FPS preserves approximately the original
# video duration when FRAME_SKIP = 2.
output_fps = fps / FRAME_SKIP

video_writer = cv2.VideoWriter(
    str(ANNOTATED_VIDEO_PATH),
    cv2.VideoWriter_fourcc(*"mp4v"),
    output_fps,
    (width, height)
)

if not video_writer.isOpened():
    cap.release()
    raise RuntimeError("Could not create annotated video.")

print(f"Video resolution: {width} x {height}")
print(f"Original FPS: {fps}")
print(f"Frame sampling interval: every {FRAME_SKIP} frames")
print(f"Total source frames: {total_frames}")


# ============================================================
# 6. TRACKING STATE
# ============================================================

tracks = {}

completed_events = []

next_event_number = 1

frame_number = 0
processed_frames = 0

start_time = time.time()


# ============================================================
# 7. PROCESS VIDEO
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # Optional short test run.
    if (
        TEST_MODE
        and frame_number / fps > TEST_DURATION_SECONDS
    ):
        print("Test duration reached.")
        break

    # Process every Nth frame.
    if frame_number % FRAME_SKIP != 0:
        continue

    processed_frames += 1

    # --------------------------------------------------------
    # YOLO DETECTION + BYTETRACK
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=VEHICLE_CLASSES,
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    annotated_frame = frame.copy()

    # Save only the detections from this processed frame.
    current_track_ids = set()

    if result.boxes is not None and result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()

        for box, track_id, class_id, confidence in zip(
            boxes, ids, classes, confidences
        ):

            if class_id not in CLASS_NAMES:
                continue

            x1, y1, x2, y2 = map(int, box)

            # Clamp coordinates to the image boundaries.
            x1 = max(0, min(x1, width - 1))
            x2 = max(0, min(x2, width - 1))
            y1 = max(0, min(y1, height - 1))
            y2 = max(0, min(y2, height - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            # Bottom-center point is used for ROI membership.
            contact_point = (
                (x1 + x2) // 2,
                y2
            )

            # Lane 1 ROI filtering.
            if not point_inside_roi(contact_point):
                continue

            current_track_ids.add(track_id)

            vehicle_class = CLASS_NAMES[class_id]

            # ------------------------------------------------
            # INITIALIZE TRACK
            # ------------------------------------------------

            if track_id not in tracks:

                snapshot_path = (
                    SNAPSHOT_DIR
                    / f"vehicle_{track_id}.jpg"
                )

                clip_path = (
                    CLIP_DIR
                    / f"vehicle_{track_id}.mp4"
                )

                clip_writer = cv2.VideoWriter(
                    str(clip_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    output_fps,
                    CLIP_SIZE
                )

                tracks[track_id] = {
                    "track_id": int(track_id),
                    "vehicle_class": vehicle_class,
                    "first_frame": frame_number,
                    "last_frame": frame_number,
                    "last_point": contact_point,
                    "start_crossed": False,
                    "end_crossed": False,
                    "start_time": None,
                    "end_time": None,
                    "best_confidence": float(confidence),
                    "snapshot_path": str(snapshot_path),
                    "clip_path": str(clip_path),
                    "clip_writer": clip_writer,
                    "clip_frames": 0,
                    "missed_intervals": 0
                }

                # Save one representative snapshot initially.
                crop = frame[y1:y2, x1:x2]

                if crop.size > 0:
                    cv2.imwrite(
                        str(snapshot_path),
                        crop
                    )

            track = tracks[track_id]

            # ------------------------------------------------
            # UPDATE TRACK
            # ------------------------------------------------

            previous_point = track["last_point"]

            track["last_point"] = contact_point
            track["last_frame"] = frame_number
            track["missed_intervals"] = 0

            # Update class when a track changes classification.
            track["vehicle_class"] = vehicle_class

            # ------------------------------------------------
            # START-LINE CROSSING
            # ------------------------------------------------

            if (
                not track["start_crossed"]
                and crossed_line(
                    previous_point,
                    contact_point,
                    start_line
                )
            ):

                track["start_crossed"] = True

                track["start_time"] = get_timestamp(
                    frame_number,
                    fps
                )

            # ------------------------------------------------
            # END-LINE CROSSING
            # ------------------------------------------------

            if (
                track["start_crossed"]
                and not track["end_crossed"]
                and crossed_line(
                    previous_point,
                    contact_point,
                    end_line
                )
            ):

                track["end_crossed"] = True

                track["end_time"] = get_timestamp(
                    frame_number,
                    fps
                )

                duration_seconds = (
                    frame_number - track["first_frame"]
                ) / fps

                # Create one event record for this passage.
                event = {
                    "event_id": f"EVT_{next_event_number:06d}",
                    "track_id": int(track_id),
                    "lane_id": 1,
                    "vehicle_class": track["vehicle_class"],
                    "passage_start_time": track["start_time"],
                    "passage_end_time": track["end_time"],
                    "passage_duration_seconds": round(
                        duration_seconds, 3
                    ),
                    "start_line_crossed": True,
                    "end_line_crossed": True,
                    "snapshot_path": track["snapshot_path"],
                    "clip_path": track["clip_path"],
                    "status": "completed"
                }

                completed_events.append(event)

                next_event_number += 1

                print(
                    f"Completed passage: "
                    f"{event['event_id']} | "
                    f"{vehicle_class} | "
                    f"Track {track_id}"
                )

            # ------------------------------------------------
            # SAVE VEHICLE CLIP
            # ------------------------------------------------

            crop = frame[y1:y2, x1:x2]

            if crop.size > 0:

                crop_resized = cv2.resize(
                    crop,
                    CLIP_SIZE
                )

                if track["clip_writer"].isOpened():

                    track["clip_writer"].write(
                        crop_resized
                    )

                    track["clip_frames"] += 1

            # ------------------------------------------------
            # ANNOTATE OUTPUT VIDEO
            # ------------------------------------------------

            label = (
                f"ID {track_id} | "
                f"{vehicle_class} | "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

    # --------------------------------------------------------
    # HANDLE TEMPORARILY LOST TRACKS
    # --------------------------------------------------------

    tracks_to_remove = []

    for track_id, track in tracks.items():

        if track_id not in current_track_ids:

            track["missed_intervals"] += 1

        if (
            track["missed_intervals"]
            > MAX_MISSED_INTERVALS
        ):

            tracks_to_remove.append(track_id)

    for track_id in tracks_to_remove:

        track = tracks.pop(track_id)

        if track["clip_writer"] is not None:
            track["clip_writer"].release()

    # Write the annotated frame.
    # ROI polygon and crossing lines are deliberately not drawn.
    video_writer.write(annotated_frame)

    if processed_frames % 100 == 0:

        elapsed = time.time() - start_time

        print(
            f"Processed frames: {processed_frames} | "
            f"Source frame: {frame_number}/{total_frames} | "
            f"Events: {len(completed_events)} | "
            f"Elapsed: {elapsed:.1f}s"
        )


# ============================================================
# 8. RELEASE RESOURCES
# ============================================================

cap.release()
video_writer.release()

for track in tracks.values():

    if track["clip_writer"] is not None:
        track["clip_writer"].release()


# ============================================================
# 9. SAVE EVENT-LEVEL JSON
# ============================================================

output_data = {
    "camera_id": "CAM_LANE_01",
    "lane_id": 1,
    "source_video": VIDEO_PATH.name,
    "source_fps": fps,
    "frame_skip": FRAME_SKIP,
    "processed_frames": processed_frames,
    "total_source_frames_read": frame_number,
    "events_count": len(completed_events),
    "events": completed_events
}

with open(
    EVENT_JSON_PATH,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        output_data,
        file,
        indent=4
    )


# ============================================================
# 10. FINAL SUMMARY
# ============================================================

elapsed = time.time() - start_time

print("\n========== PROCESSING COMPLETE ==========")

print(f"Frames processed: {processed_frames}")
print(f"Completed vehicle passages: {len(completed_events)}")
print(f"Elapsed time: {elapsed:.2f} seconds")

print(f"\nAnnotated video: {ANNOTATED_VIDEO_PATH}")
print(f"JSON events: {EVENT_JSON_PATH}")
print(f"Snapshots: {SNAPSHOT_DIR}")
print(f"Vehicle clips: {CLIP_DIR}")
from ultralytics import YOLO
import cv2
import json
import sys
import matplotlib.pyplot as plt
from collections import defaultdict


# ==========================================
# CONFIGURATION
# ==========================================

if len(sys.argv) > 1:
    VIDEO_PATH = sys.argv[1]
else:
    VIDEO_PATH = r"runs\detect\track\Copy of Alif Store-Viewing-10.avi"

MODEL_PATH = "yolov8n.pt"


# ==========================================
# SHELF ZONES
# ==========================================

SHELVES = {
    "Shelf_A": (0, 0, 1280, 2160),
    "Shelf_B": (1280, 0, 2560, 2160),
    "Shelf_C": (2560, 0, 3840, 2160)
}


# ==========================================
# VIDEO FPS
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

cap.release()

if fps <= 0:
    fps = 30


# ==========================================
# LOAD MODEL
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# DATA STORAGE
# ==========================================

shelf_visitors = {
    shelf: set()
    for shelf in SHELVES
}

dwell_frames = defaultdict(
    lambda: defaultdict(int)
)

movement_points = []


# ==========================================
# PROCESS VIDEO
# ==========================================

results = model.track(
    source=VIDEO_PATH,
    persist=True,
    classes=[0],
    tracker="bytetrack.yaml",
    conf=0.3,
    iou=0.5,
    show=False,
    stream=True
)


frame_number = 0


for result in results:

    frame_number += 1

    if result.boxes is None:
        continue

    if result.boxes.id is None:
        continue

    boxes = result.boxes.xyxy.cpu().numpy()

    track_ids = result.boxes.id.cpu().numpy()


    for box, track_id in zip(
        boxes,
        track_ids
    ):

        x1, y1, x2, y2 = map(
            int,
            box
        )


        center_x = (
            x1 + x2
        ) // 2


        center_y = (
            y1 + y2
        ) // 2


        person_id = int(track_id)


        # Store movement position
        movement_points.append(
            (
                center_x,
                center_y
            )
        )


        print(
            f"Frame {frame_number}: "
            f"ID {person_id} "
            f"Center = "
            f"({center_x}, {center_y})"
        )


        # ==================================
        # CHECK SHELF ZONES
        # ==================================

        for shelf_name, zone in SHELVES.items():

            sx1, sy1, sx2, sy2 = zone


            if (
                sx1 <= center_x <= sx2
                and
                sy1 <= center_y <= sy2
            ):

                shelf_visitors[
                    shelf_name
                ].add(
                    person_id
                )


                dwell_frames[
                    person_id
                ][
                    shelf_name
                ] += 1


# ==========================================
# GENERATE ANALYTICS
# ==========================================

analytics = {}

all_visitors = set()


for shelf_name in SHELVES:

    visitors = shelf_visitors[
        shelf_name
    ]


    all_visitors.update(
        visitors
    )


    visitor_count = len(
        visitors
    )


    total_frames = 0


    for person_id in visitors:

        total_frames += (
            dwell_frames[
                person_id
            ][
                shelf_name
            ]
        )


    total_dwell_time = (
        total_frames / fps
    )


    if visitor_count > 0:

        average_dwell_time = (
            total_dwell_time
            / visitor_count
        )

    else:

        average_dwell_time = 0


    analytics[
        shelf_name
    ] = {

        "visitors": visitor_count,

        "total_dwell_time": round(
            total_dwell_time,
            2
        ),

        "average_dwell_time": round(
            average_dwell_time,
            2
        )
    }


# ==========================================
# POPULARITY ANALYSIS
# ==========================================

most_visited_shelf = max(
    analytics,
    key=lambda shelf:
    analytics[
        shelf
    ][
        "visitors"
    ]
)


least_visited_shelf = min(
    analytics,
    key=lambda shelf:
    analytics[
        shelf
    ][
        "visitors"
    ]
)


highest_engagement_shelf = max(
    analytics,
    key=lambda shelf:
    analytics[
        shelf
    ][
        "average_dwell_time"
    ]
)


# ==========================================
# SAVE ANALYTICS JSON
# ==========================================

final_data = {

    "total_unique_visitors": len(
        all_visitors
    ),

    "most_visited_shelf":
        most_visited_shelf,

    "least_visited_shelf":
        least_visited_shelf,

    "highest_engagement_shelf":
        highest_engagement_shelf,

    "shelves":
        analytics
}


with open(
    "analytics.json",
    "w"
) as file:

    json.dump(
        final_data,
        file,
        indent=4
    )


# ==========================================
# GENERATE CUSTOMER HEATMAP
# ==========================================

if len(movement_points) > 0:

    x_values = [
        point[0]
        for point in movement_points
    ]


    y_values = [
        point[1]
        for point in movement_points
    ]


    plt.figure(
        figsize=(12, 7)
    )


    plt.hist2d(
        x_values,
        y_values,
        bins=50
    )


    plt.colorbar(
        label="Customer Movement Density"
    )


    plt.title(
        "Customer Movement Heatmap"
    )


    plt.xlabel(
        "X Position"
    )


    plt.ylabel(
        "Y Position"
    )


    plt.gca().invert_yaxis()


    plt.savefig(
        "customer_heatmap.jpg",
        dpi=150,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "Customer heatmap saved to "
        "customer_heatmap.jpg"
    )


# ==========================================
# PRINT RESULTS
# ==========================================

print(
    "\n========== SHELF ANALYTICS ==========\n"
)


print(
    "Total Unique Visitors:",
    len(all_visitors)
)


for shelf_name, values in analytics.items():

    print(
        f"\n{shelf_name}"
    )


    print(
        "Visitors:",
        values[
            "visitors"
        ]
    )


    print(
        "Total Dwell Time:",
        values[
            "total_dwell_time"
        ],
        "seconds"
    )


    print(
        "Average Dwell Time:",
        values[
            "average_dwell_time"
        ],
        "seconds"
    )


    print(
        "--------------------------------------"
    )


print(
    "\n========== POPULARITY ANALYSIS =========="
)


print(
    "Most Visited Shelf:",
    most_visited_shelf
)


print(
    "Least Visited Shelf:",
    least_visited_shelf
)


print(
    "\n========== ENGAGEMENT RANKING =========="
)


ranking = sorted(
    analytics.items(),
    key=lambda item:
    item[1][
        "average_dwell_time"
    ],
    reverse=True
)


for rank, (
    shelf_name,
    values
) in enumerate(
    ranking,
    start=1
):

    print(
        f"{rank}. "
        f"{shelf_name} - "
        f"{values['average_dwell_time']} "
        f"seconds average dwell time"
    )


print(
    "\n======================================"
)


print(
    "\nAnalytics saved to analytics.json"
)
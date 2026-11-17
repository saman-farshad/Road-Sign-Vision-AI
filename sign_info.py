"""English names and short warning messages for 43 GTSRB traffic signs.

A normal 30 km/h speed limit is NOT a separate German Zone 30 sign.
"""

import re

GTSRB_LABELS = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No overtaking", "No overtaking for trucks",
    "Priority at next intersection", "Priority road", "Give way", "Stop",
    "No vehicles", "Trucks prohibited", "No entry", "General caution",
    "Dangerous curve left", "Dangerous curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows right", "Road works", "Traffic lights",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Ice or snow",
    "Wild animals", "End of restrictions", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Ahead or right", "Ahead or left", "Keep right", "Keep left",
    "Roundabout", "End no overtaking", "End no overtaking for trucks",
]

SPEED_IDS = {0: 20, 1: 30, 2: 50, 3: 60, 4: 70, 5: 80, 7: 100, 8: 120}

WARNINGS = {
    6: "80 km/h limit has ended",
    9: "Overtaking is prohibited",
    10: "Trucks must not overtake",
    11: "Give priority at the next junction",
    12: "You are on a priority road",
    13: "Yield to other traffic",
    14: "Come to a complete stop",
    15: "Vehicles are not allowed",
    16: "Trucks are not allowed",
    17: "Do not enter this road",
    18: "Watch out for hazards ahead",
    19: "Dangerous left curve ahead",
    20: "Dangerous right curve ahead",
    21: "Several curves ahead",
    22: "Uneven road ahead",
    23: "Road may be slippery",
    24: "Road narrows on the right",
    25: "Roadworks ahead",
    26: "Traffic signals ahead",
    27: "Watch for pedestrians",
    28: "Watch for children",
    29: "Watch for cyclists",
    30: "Risk of ice or snow",
    31: "Watch for wild animals",
    32: "Restrictions have ended",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Go straight ahead",
    36: "Go straight or turn right",
    37: "Go straight or turn left",
    38: "Keep to the right",
    39: "Keep to the left",
    40: "Roundabout ahead",
    41: "Overtaking ban has ended",
    42: "Truck overtaking ban has ended",
}


def get_sign_info(class_id: int, raw_name: str = "") -> tuple[str, str]:
    """Return (English sign title, English warning) from the GTSRB class ID."""
    if class_id in SPEED_IDS:
        speed = SPEED_IDS[class_id]
        return f"Speed Limit {speed} km/h", f"Maximum speed: {speed} km/h"

    if 0 <= class_id < len(GTSRB_LABELS):
        return GTSRB_LABELS[class_id], WARNINGS.get(class_id, "Follow this road sign")

    # Fallback for a custom model with different labels.
    match = re.search(r"(?:speed\s*limit|speedlimit|limit)[^0-9]*(\d{2,3})", raw_name, re.I)
    if match:
        speed = int(match.group(1))
        return f"Speed Limit {speed} km/h", f"Maximum speed: {speed} km/h"
    return raw_name or "Unknown Sign", "Check the road sign"

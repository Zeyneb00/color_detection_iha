import cv2
import numpy as np


class DroneTargetDetector:

  def __init__(self):
    # HSV Ranges for drone mission
    self.lower_blue = np.array([100, 100, 50])
    self.upper_blue = np.array([140, 255, 255])

    self.lower_red1 = np.array([0, 120, 50])
    self.upper_red1 = np.array([10, 255, 255])
    self.lower_red2 = np.array([170, 120, 50])
    self.upper_red2 = np.array([179, 255, 255])

    self.kernel = np.ones((5, 5), np.uint8)

  def compute_confidence(self, circularity, solidity, color_strength):
    """Calculates a weighted confidence score (0.0 to 1.0)

    combining circularity, fill ratio (solidity), and color purity.
    """
    # Normalize circularity: perfect circle = 1.0, max penalty if < 0.5 or > 1.3
    circ_score = max(0, 1.0 - abs(1.0 - circularity))

    # Solidity score is already between 0 and 1
    sol_score = solidity

    # Color strength score: normalize saturation (0-255 scale) to 0.0 - 1.0
    color_score = min(1.0, color_strength / 200.0)

    # Weighted blend (Weights sum to 1.0)
    # Geometry and circularity matter most for disarm targets, followed by color purity.
    confidence = (
        (0.40 * circ_score) + (0.35 * sol_score) + (0.25 * color_score)
    )
    return round(float(confidence), 3)

  def process_frame(self, image):
    # Resize standard for drone vision pipeline
    image = cv2.resize(image, (900, 600))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 1. Masking & Morphological Cleanup
    blue_mask = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
    blue_mask = cv2.morphologyEx(
        cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, self.kernel),
        cv2.MORPH_CLOSE,
        self.kernel,
    )

    red_mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
    red_mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    red_mask = cv2.morphologyEx(
        cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, self.kernel),
        cv2.MORPH_CLOSE,
        self.kernel,
    )

    detected_targets = []
    object_id_counter = 1

    # Internal helper to evaluate contours
    def evaluate_contours(mask, color_name, display_color):
      nonlocal object_id_counter
      contours, _ = cv2.findContours(
          mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
      )

      for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500:  # Ignore microscopic noise
          continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
          continue

        # Geometric metrics
        circularity = (4 * np.pi * area) / (perimeter**2)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        # Color strength (In-blob saturation mean)
        obj_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(hsv, mask=obj_mask)
        color_strength = mean_val[1]

        # Calculate Confidence Score
        confidence = self.compute_confidence(
            circularity, solidity, color_strength
        )

        # Bounding / Center geometry
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # Record structured candidate data
        target_record = {
            "id": object_id_counter,
            "color": color_name,
            "center": center,  # Image coordinates (u, v) -> Ready for World Projection
            "radius": radius,
            "confidence": confidence,
            "metrics": {
                "area": round(area, 2),
                "circularity": round(circularity, 2),
                "solidity": round(solidity, 2),
                "color_strength": round(color_strength, 2),
            },
        }

        detected_targets.append(target_record)

        # Visualization for operator view
        # Only highlight if confidence is above operational threshold (e.g., 0.60)
        if confidence >= 0.60:
          cv2.circle(image, center, radius, display_color, 2)
          cv2.circle(image, center, 3, (0, 0, 255), -1)
          label = f"ID:{object_id_counter} {color_name} | Conf:{confidence}"
          cv2.putText(
              image,
              label,
              (center[0] - 65, center[1] - radius - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,
              display_color,
              2,
          )
        else:
          # Low confidence objects (e.g., debris or jackets) marked in gray
          cv2.circle(image, center, radius, (100, 100, 100), 1)
          cv2.putText(
              image,
              f"Rejected ID:{object_id_counter} ({confidence})",
              (center[0] - 50, center[1] - radius - 10),
              cv2.FONT_HERSHEY_SIMPLEX,
              0.4,
              (100, 100, 100),
              1,
          )

        object_id_counter += 1

    # Evaluate both color channels
    evaluate_contours(blue_mask, "Blue", (0, 255, 0))
    evaluate_contours(red_mask, "Red", (255, 255, 0))

    return image, detected_targets


# --- Execution Example ---
if __name__ == "__main__":
  detector = DroneTargetDetector()

  # Test on example image
  test_image = cv2.imread("4thblue.jpeg")
  if test_image is not None:
    processed_frame, targets = detector.process_frame(test_image)

    print("\n--- DRONE TARGET MODULE OUTPUT REPORT ---")
    for t in targets:
      status = (
          "VALID TARGET (READY FOR WORLD MAPPING)"
          if t["confidence"] >= 0.60
          else "REJECTED (CLUTTER/DEBRIS)"
      )
      print(f"[{status}] -> {t}")
    print("------------------------------------------\n")

    cv2.imshow("Drone Target Acquisition", processed_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  else:
    print(
        "Error: 'example.png' not found. Place an image in the directory to"
        " test."
    )
import csv
import os
import cv2
import numpy as np

# --- CONFIGURATION ---
INPUT_DIR = "."  # Folder containing your test images (current directory by default)
OUTPUT_DIR = "output_results"  # Folder where processed images & reports will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Master list to store all data across all processed images
master_database = []

# Find all common image formats
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG")
image_files = []
for ext in image_extensions:
  image_files.extend(
      [f for f in os.listdir(INPUT_DIR) if f.endswith(ext[1:])]
  )

if not image_files:
  print(f"No images found in '{INPUT_DIR}'. Please add some test photos.")
  exit()

print(f"Found {len(image_files)} image(s) to process. Starting pipeline...\n")

for filename in image_files:
  img_path = os.path.join(INPUT_DIR, filename)
  image = cv2.imread(img_path)
  if image is None:
    print(f"Skipping {filename}: Could not read image.")
    continue

  # Resize for consistent processing and display
  image = cv2.resize(image, (900, 600))
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # --- 1. HSV Color Masking ---
  lower_blue = np.array([100, 100, 50])
  upper_blue = np.array([140, 255, 255])
  blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

  lower_red1, upper_red1 = np.array([0, 120, 50]), np.array([10, 255, 255])
  lower_red2, upper_red2 = np.array([170, 120, 50]), np.array([179, 255, 255])
  red_mask = cv2.bitwise_or(
      cv2.inRange(hsv, lower_red1, upper_red1),
      cv2.inRange(hsv, lower_red2, upper_red2),
  )

  # Morphological cleanup
  kernel = np.ones((5, 5), np.uint8)
  blue_mask = cv2.morphologyEx(
      cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel),
      cv2.MORPH_CLOSE,
      kernel,
  )
  red_mask = cv2.morphologyEx(
      cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel),
      cv2.MORPH_CLOSE,
      kernel,
  )

  image_objects = []
  local_object_id = 1

  def analyze_and_extract(
      mask, color_name, display_color, img, hsv_img, file_name
    ):
    global local_object_id
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for cnt in contours:
      area = cv2.contourArea(cnt)
      if area < 800:
        continue

      perimeter = cv2.arcLength(cnt, True)
      if perimeter == 0:
        continue

      # Geometry calculations
      circularity = (4 * np.pi * area) / (perimeter**2)
      hull = cv2.convexHull(cnt)
      hull_area = cv2.contourArea(hull)
      solidity = area / hull_area if hull_area > 0 else 0

      if 0.65 <= circularity <= 1.25 and solidity >= 0.80:
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius)

        # In-Blob Color Strength Analysis
        object_mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
        cv2.drawContours(object_mask, [cnt], 0, 255, -1)
        mean_val = cv2.mean(hsv_img, mask=object_mask)
        color_strength = mean_val[1]  # Saturation metric

        record = {
            "filename": file_name,
            "object_id": local_object_id,
            "color": color_name,
            "center_x": center[0],
            "center_y": center[1],
            "radius": radius,
            "area": round(area, 2),
            "circularity": round(circularity, 2),
            "solidity": round(solidity, 2),
            "color_strength": round(color_strength, 2),
        }

        image_objects.append(record)
        master_database.append(record)

        # Draw visuals
        cv2.circle(img, center, radius, display_color, 2)
        cv2.circle(img, center, 3, (0, 0, 255), -1)
        label = f"ID:{local_object_id} {color_name} | CS:{color_strength:.1f}"
        cv2.putText(
            img,
            label,
            (center[0] - 60, center[1] - radius - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            display_color,
            2,
        )

        local_object_id += 1

  # Run analysis on current image
  analyze_and_extract(blue_mask, "Blue", (0, 255, 0), image, hsv, filename)
  analyze_and_extract(red_mask, "Red", (255, 255, 0), image, hsv, filename)

  # --- 2. Summary Dashboard Overlay on Image ---
  blue_count = sum(1 for o in image_objects if o["color"] == "Blue")
  red_count = sum(1 for o in image_objects if o["color"] == "Red")

  # Draw background panel for dashboard
  cv2.rectangle(image, (15, 15), (320, 105), (50, 50, 50), -1)
  cv2.rectangle(image, (15, 15), (320, 105), (255, 255, 255), 1)

  cv2.putText(
      image,
      f"File: {filename}",
      (25, 38),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      image,
      f"Blue Objects Detected: {blue_count}",
      (25, 65),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      image,
      f"Red Objects Detected: {red_count}",
      (25, 90),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (255, 255, 0),
      2,
  )

  # Save processed image to output directory
  output_img_path = os.path.join(OUTPUT_DIR, f"processed_{filename}")
  cv2.imwrite(output_img_path, image)
  print(f"Processed & saved: {output_img_path}")

# --- 3. Export Master Database to CSV ---
csv_file_path = os.path.join(OUTPUT_DIR, "detected_objects_report.csv")
csv_headers = [
    "filename",
    "object_id",
    "color",
    "center_x",
    "center_y",
    "radius",
    "area",
    "circularity",
    "solidity",
    "color_strength",
]

with open(csv_file_path, mode="w", newline="") as csv_file:
  writer = csv.DictWriter(csv_file, fieldnames=csv_headers)
  writer.writeheader()
  for record in master_database:
    writer.writerow(record)

print(f"\nBatch processing complete!")
print(f"All object reports exported successfully to: {csv_file_path}")
print(f"Annotated output images are stored in: '{OUTPUT_DIR}/'")
# Image-Processing
Center detection &amp; Finding Distance


import cv2
import numpy as np

# Load the image
image_path = 'In.jpg'
image = cv2.imread(image_path, 0)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(image, (9, 9), 1.5)

# Use a high pass filter to enhance the bright spots (bold points)
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

# Convert to uint8
laplacian = np.uint8(np.absolute(laplacian))

# Threshold to isolate the bright spots
_, thresholded = cv2.threshold(laplacian, 15, 255, cv2.THRESH_BINARY)

# Define a structuring element for dilation
kernel = np.ones((2, 2), np.uint8)
dilated = cv2.dilate(thresholded, kernel, iterations=2)

# Find contours of the dilated spots
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Convert the original grayscale image to BGR for displaying colored points
display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

# Enlarge the image for display
scaling_factor = 1.5  # Change this factor to enlarge or reduce the image
new_width = int(display_image.shape[1] * scaling_factor)
new_height = int(display_image.shape[0] * scaling_factor)
display_image = cv2.resize(display_image, (new_width, new_height))

# Create a white canvas to show an extra white box at the right side of the image
extra_width = 300  # Width of the extra white box
canvas = np.ones((display_image.shape[0], display_image.shape[1] + extra_width, 3), dtype=np.uint8) * 255
canvas[:, :display_image.shape[1]] = display_image

# Draw the detected points (Mg symbol points) on the display image
for contour in contours:
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    center = (int(cx * scaling_factor), int(cy * scaling_factor))  # Adjust center based on scaling
    radius = int(radius * scaling_factor)  # Adjust radius based on scaling
    cv2.circle(canvas, center, radius, (0, 255, 0), 2)  # Draw a green circle for each detected point

# List to store clicked points
clicked_points = []
line_count = 0  # Counter for line numbers

# Function to calculate and display distances between two points
def display_distances(point1, point2):
    global line_count
    
    # Calculate horizontal, vertical, and Euclidean distances
    horizontal_distance = abs(point1[0] - point2[0])
    vertical_distance = abs(point1[1] - point2[1])
    actual_distance = int(np.sqrt(horizontal_distance ** 2 + vertical_distance ** 2))

    # Draw a line between the points
    cv2.line(canvas, point1, point2, (255, 0, 0), 2)
    
    # Increment line count and put the line number on the middle of the line
    line_count += 1
    mid_point = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
    cv2.putText(canvas, f'{line_count}', mid_point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Display distances on the white canvas box
    cv2.putText(canvas, f"Line {line_count}: H = {horizontal_distance}, V = {vertical_distance}, A = {actual_distance}", 
                (display_image.shape[1] + 10, 30 * line_count), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    print(f"Horizontal Distance: {horizontal_distance}, Vertical Distance: {vertical_distance}, Actual Distance: {actual_distance}")

# Mouse callback function to detect the center of each point
def detect_center(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Check for left mouse button click
        for contour in contours:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            center = (int(cx * scaling_factor), int(cy * scaling_factor))  # Adjust center based on scaling

            # Check if the clicked point is within the contour (adjusted for scaling)
            if cv2.pointPolygonTest(contour, (x / scaling_factor, y / scaling_factor), True) > -1:
                cv2.circle(canvas, center, 2, (0, 0, 255), -1)  # Draw a smaller red circle to show the selected center
                print(f"Center selected at: {center}")
                clicked_points.append(center)

                if len(clicked_points) == 2:  # If two points are clicked
                    # Calculate distances and draw line
                    display_distances(clicked_points[0], clicked_points[1])
                    clicked_points.clear()  # Clear points after calculating distances
                
                # Show updated image
                cv2.imshow('final_image', canvas)
                break

# Create a window and set the mouse callback function
cv2.namedWindow('final_image')
cv2.setMouseCallback('final_image', detect_center)

# Display the initial image with detected points
cv2.imshow('final_image', canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the final image to a file
output_path = 'final_image.jpg'
cv2.imwrite(output_path, canvas)
print(f"Final image saved as {output_path}")

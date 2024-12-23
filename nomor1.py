import cv2
import numpy as np

def calculate_vessel_fill_percentage(image_path):
    """
    Calculate the fill percentage of a vessel in an image.
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        float: Fill percentage of the vessel
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read the image")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 1: Detect vessel (object detection)
    # Using Otsu's thresholding to separate vessel from background
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 2: Detect edges
    edges = cv2.Canny(binary, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour (assumed to be the vessel)
    vessel_contour = max(contours, key=cv2.contourArea)
    
    # Step 3: Find center point
    M = cv2.moments(vessel_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        return 0
    
    # Step 4: Calculate total vessel area
    total_area = cv2.contourArea(vessel_contour)
    
    # Step 5: Calculate filled area
    # Create a mask for the filled region (everything below center point)
    height, width = binary.shape
    fill_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(fill_mask, [vessel_contour], -1, (255), -1)
    
    # Create a horizontal line at the center point
    fill_mask[cy:, :] = 0
    filled_area = cv2.countNonZero(fill_mask)
    
    # Step 6: Calculate percentage
    fill_percentage = (filled_area / total_area) * 100
    
    # Visualization (for debugging)
    debug_image = image.copy()
    cv2.drawContours(debug_image, [vessel_contour], -1, (0, 255, 0), 2)
    cv2.circle(debug_image, (cx, cy), 5, (0, 0, 255), -1)
    cv2.line(debug_image, (0, cy), (width, cy), (255, 0, 0), 2)
    
    # Display results
    cv2.imshow('Vessel Detection', debug_image)
    cv2.imshow('Fill Mask', fill_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return fill_percentage

def main():
    try:
        image_path = "image.png"
        percentage = calculate_vessel_fill_percentage(image_path)
        print(f"The vessel is {percentage:.2f}% filled")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
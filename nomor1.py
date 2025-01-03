import cv2
import numpy as np

def create_output_grid(images, titles):
    """
    Create a 2x3 grid of images with titles
    """
    row_height = images[0].shape[0]
    row_width = sum(img.shape[1] for img in images[:3])
    top_row = np.zeros((row_height, row_width, 3), dtype=np.uint8)
    bottom_row = np.zeros((row_height, row_width, 3), dtype=np.uint8)
    
    x_offset = 0
    
    for i in range(3):
        img_width = images[i].shape[1]
        if len(images[i].shape) == 2:  
            img_to_copy = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        else:
            img_to_copy = images[i].copy()
        top_row[:, x_offset:x_offset + img_width] = img_to_copy
        x_offset += img_width
    
    x_offset = 0
    

    for i in range(3, 6):
        img_width = images[i].shape[1]
        if len(images[i].shape) == 2:  
            img_to_copy = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        else:
            img_to_copy = images[i].copy()
        bottom_row[:, x_offset:x_offset + img_width] = img_to_copy
        x_offset += img_width

    final_output = np.vstack((top_row, bottom_row))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    y_offset = 30
    x_spacing = row_width // 3
    
    for i in range(3):
        x_pos = (i * x_spacing) + 10
        cv2.putText(final_output, titles[i], (x_pos, y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
    
    for i in range(3):
        x_pos = (i * x_spacing) + 10
        cv2.putText(final_output, titles[i+3], (x_pos, row_height + y_offset), 
                   font, font_scale, (255, 255, 255), thickness)
    
    return final_output

def analyze_vessel(image_path):

    original = cv2.imread(image_path)
    if original is None:
        raise ValueError("Could not read the image")
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    edges = cv2.Canny(binary, 50, 150)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vessel_contour = max(contours, key=cv2.contourArea)

    contour_image = original.copy()
    cv2.drawContours(contour_image, [vessel_contour], -1, (0, 255, 0), 2)
    
    M = cv2.moments(vessel_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    
    center_image = original.copy()
    cv2.circle(center_image, (cx, cy), 5, (0, 0, 255), -1)
    
    height, width = binary.shape
    fill_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(fill_mask, [vessel_contour], -1, (255), -1)
    fill_mask[cy:, :] = 0
    
    total_area = cv2.contourArea(vessel_contour)
    filled_area = cv2.countNonZero(fill_mask)
    fill_percentage = (filled_area / total_area) * 100
    
    binary_vis = binary.copy()
    
    images = [
        original,          # Original image
        contour_image,     # Contour detection
        edges,            # Edge detection
        center_image,     # Center point
        binary_vis,       # Binary threshold
        fill_mask         # Fill mask
    ]
    
    titles = [
        "Original",
        "Contour Detection",
        "Edge Detection",
        "Center Point",
        "Binary Threshold",
        "Fill Mask"
    ]
    
    final_output = create_output_grid(images, titles)
    
    return final_output, fill_percentage

def main():
    try:
        image_path = "image.png"
        final_output, percentage = analyze_vessel(image_path)
        
        # Display results
        cv2.imshow('Vessel Analysis', final_output)
        print(f"Vessel fill percentage: {percentage:.2f}%")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
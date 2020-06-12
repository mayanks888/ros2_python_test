
std_msgs/Header header                      # Header timestamp should be acquisition time of image detection published                       
sensor_msgs/Image raw_image                 # Raw image from camera
#sensor_msgs/Image  cropped_roi_image        # Croppped roi images
sensor_msgs/RegionOfInterest projection_roi # projected ROI
sensor_msgs/RegionOfInterest cropped_roi    # Roi bbox value after scale of 2.5
sensor_msgs/RegionOfInterest ref_roi        # Bbox info of projection roi
Detection2D[] detections                    # A list of 2D detections, for a multi-object 2D detector.
Detection2D selected_box                    # Final bbox selected
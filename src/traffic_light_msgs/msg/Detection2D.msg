# To store the infomation related to AI detection of traffic light

std_msgs/Header header # Header timestamp should be acquisition time of image

int64 id        # The unique numeric ID of object detected.

float64 score   # The probability or confidence value of the detected object. By convention,
                # This value should lie in the range [0-1].

uint32 x_offset  # Leftmost pixel of the ROI
                 # (0 if the ROI includes the left edge of the image)
uint32 y_offset  # Topmost pixel of the ROI
                   # (0 if the ROI includes the top edge of the image)

uint32 height   # detected bbox height,that is, number of rows

uint32 width    # detected bbox  width, that is, number of column

uint8  color      #color of the image in bbox 

sensor_msgs/Image bbox_crop_img    # Region proposal cropped out of the image 
                                # Not required for all use cases, so it may be empty.


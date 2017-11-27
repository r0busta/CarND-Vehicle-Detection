import glob
import cv2
import numpy as np
import matplotlib.image as mpimg
from utils import moving_avg


class Camera:

    NUM_CHESSBRD_CORNERS_X = 9
    NUM_CHESSBRD_CORNERS_Y = 6

    RES_W = 1280
    RES_H = 720

    def __init__(self):
        self.calibration_matr = None
        self.calibration_coeff = None

        self.transform_src_points = np.float32([[595, 450], [689, 450], [1104, 720], [215, 720]])
        self.transform_dst_points = np.float32([[250, 0], [self.RES_W - 250, 0], [self.RES_W - 250, self.RES_H], [250, self.RES_H]])
        self.M = self.get_transform_matr()
        self.M_INV = self.get_inv_transform_matr()

    def find_corners(self, img):
        """ Find chessboard corners
        """
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        return cv2.findChessboardCorners(gray, (self.NUM_CHESSBRD_CORNERS_X, self.NUM_CHESSBRD_CORNERS_Y), None)

    def calibrate_camera(self, objpoints, imgpoints):
        """ Calibrate camera fro the given set of object and image points.
            Return calibration matrix and distortion coefficients.
        """
        res, cmatr, coeff, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (self.RES_W, self.RES_H), None, None)
        return res, cmatr, coeff

    def calibrate(self, calibration_images):
        """ Find object and image points for each of the calibration images.
            Calibrate camera and return calibration matrix and distortion coefficients.
        """
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.NUM_CHESSBRD_CORNERS_X * self.NUM_CHESSBRD_CORNERS_Y, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.NUM_CHESSBRD_CORNERS_X, 0:self.NUM_CHESSBRD_CORNERS_Y].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Make a list of calibration images
        images = glob.glob(calibration_images)

        # Step through the list and search for chessboard corners
        for name in images:
            img = mpimg.imread(name)

            res, corners = self.find_corners(img)
            if res:
                objpoints.append(objp)
                imgpoints.append(corners)

        res, self.calibration_matr, self.calibration_coeff = self.calibrate_camera(objpoints, imgpoints)
        return res

    def undistort(self, img):
        """ Undistort a given image
        """
        assert(self.calibration_matr is not None and self.calibration_coeff is not None)
        return cv2.undistort(img, self.calibration_matr, self.calibration_coeff)

    def transform(self, img):
        """ Apply perspective transformation to a given image
        """
        return cv2.warpPerspective(img, self.M, (self.RES_W, self.RES_H))

    def unwarp(self, img):
        """ Apply backward perspective transformation to unwarp the image
        """
        return cv2.warpPerspective(img, self.M_INV, (self.RES_W, self.RES_H))

    def get_transform_matr(self):
        """ Get transformation matrix
        """
        # Given src and dst points, calculate the perspective transformation
        return cv2.getPerspectiveTransform(self.transform_src_points, self.transform_dst_points)

    def get_inv_transform_matr(self):
        """ Get an inverse transformation matrix
        """
        return cv2.getPerspectiveTransform(self.transform_dst_points, self.transform_src_points)


class LaneDetector:

    M_PER_PX_X = 3.7 / 700  # meters per pixel in x dimension
    M_PER_PX_Y = 30 / 720  # meters per pixel in y dimension

    SMOOTHING_WINDOW = 3

    def __init__(self, camera):
        self.camera = camera

        self.frames_count = 0
        self.frame_fit = []

    @staticmethod
    def get_binary(img, l_thresh=(0, 255), b_thresh=(0, 255)):
        """ Get a threshold binary image.
        """
        img = np.copy(img)

        # Using different color channels.
        # Convert to HLS color space and separate L channel
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hls[:, :, 1]
        # Convert to Lab color space and separate b channel
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        b_channel = lab[:, :, 2]

        # Threshold L channel
        l_binary = np.zeros_like(l_channel)
        l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1

        # Threshold b channel
        b_binary = np.zeros_like(b_channel)
        b_binary[(b_channel >= b_thresh[0]) & (b_channel <= b_thresh[1])] = 1

        combined_binary = np.zeros_like(l_binary)
        combined_binary[(l_binary == 1) | (b_binary == 1)] = 1

        return combined_binary

    def get_curvature_m(self, ploty, lefty, leftx, righty, rightx):
        """ Get curvature radius of left and right lines
        """
        y_eval = np.max(ploty)

        left_fit_cr = np.polyfit(lefty * self.M_PER_PX_Y, leftx * self.M_PER_PX_X, 2)
        right_fit_cr = np.polyfit(righty * self.M_PER_PX_Y, rightx * self.M_PER_PX_X, 2)
        # Calculate the new radii of curvature
        left_curverad = int(((1 + (2 * left_fit_cr[0] * y_eval * self.M_PER_PX_Y + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0]))
        right_curverad = int(((1 + (2 * right_fit_cr[0] * y_eval * self.M_PER_PX_Y + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0]))
        return left_curverad, right_curverad

    def get_offset_m(self, leftx, rightx):
        """ Get the car offset from the lane center axis
        """
        return ((self.camera.RES_W / 2) - (leftx[0] + rightx[0]) / 2) * self.M_PER_PX_X

    def calc_frames_count(self):
        if self.frames_count < self.SMOOTHING_WINDOW:
            self.frames_count += 1

    def draw_lane(self, img, binary_warped, ploty, left_fitx, right_fitx):
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        self.calc_frames_count()

        if self.frames_count == 1:
            self.frame_fit.append(np.copy(left_fitx))
            self.frame_fit.append(np.copy(right_fitx))
        else:
            self.frame_fit[0] = moving_avg(self.frame_fit[0], left_fitx, self.frames_count)
            self.frame_fit[1] = moving_avg(self.frame_fit[1], right_fitx, self.frames_count)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([self.frame_fit[0], ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([self.frame_fit[1], ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix
        newwarp = self.camera.unwarp(color_warp)
        # Combine the result with the original image
        return cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    def detect_lane(self, img, binary_warped):
        """ Detect lane lines and draw the lane polygon on the returned image
        """
        # Assuming a warped binary image is created
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[int(binary_warped.shape[0] / 2):, :], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 30
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = None
        right_fit = None
        if len(leftx) > 0 and len(lefty) > 0:
            left_fit = np.polyfit(lefty, leftx, 2)
        if len(rightx) > 0 and len(righty) > 0:
            right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        if left_fit is not None and right_fit is not None:
            ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            # Create an image and draw lane on it
            result = self.draw_lane(img, binary_warped, ploty, left_fitx, right_fitx)

            # Measure and print out curvature
            left_rad_m, right_rad_m = self.get_curvature_m(ploty, lefty, leftx, righty, rightx)
            cv2.putText(result,
                        "Left curvature: {} m Right curvature: {} m".format(left_rad_m, right_rad_m),
                        (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

            # Estimate and print out car offset
            offset_m = self.get_offset_m(leftx, rightx)
            cv2.putText(result,
                        "Car offset: {:.2f} m".format(offset_m),
                        (100, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)

        else:
            result = img

        return result

    def find_lane(self, img):
        """ Apply lane finding pipeline to a given image.
        """
        undistorted = self.camera.undistort(img)
        warped = self.camera.transform(undistorted)
        binary_warped = self.get_binary(warped, (225, 255), (155, 200))
        return self.detect_lane(undistorted, binary_warped)

import cv2
import numpy as np


class BlurDetector(object):

    def __init__(self):
        """Initialize a DCT based blur detector"""
        self.dct_threshold = 8.0
        self.max_hist = 0.1
        self.hist_weight = np.array([8, 7, 6, 5, 4, 3, 2, 1,
                                     7, 8, 7, 6, 5, 4, 3, 2,
                                     6, 7, 8, 7, 6, 5, 4, 3,
                                     5, 6, 7, 8, 7, 6, 5, 4,
                                     4, 5, 6, 7, 8, 7, 6, 5,
                                     3, 4, 5, 6, 7, 8, 7, 6,
                                     2, 3, 4, 5, 6, 7, 8, 7,
                                     1, 2, 3, 4, 5, 6, 7, 8
                                     ]).reshape(8, 8)
        self.weight_total = 344.0

    def check_image_size(self, image, block_size=8):
        """Make sure the image size is valid.
        Args:
            image: input image as a numpy array.
            block_size: the size of the minimal DCT block.
        Returns:
            result: boolean value indicating whether the image is valid.
            image: a modified valid image.
        """
        result = True
        height, width = image.shape[:2]
        _y = height % block_size
        _x = width % block_size

        pad_x = pad_y = 0

        if _y != 0:
            pad_y = block_size - _y
            result = False
        if _x != 0:
            pad_x = block_size - _x
            result = False

        image = cv2.copyMakeBorder(
            image, 0, pad_y, 0, pad_x, cv2.BORDER_REPLICATE)

        return result, image

    def get_blurness(self, image, block_size=8):
        """Estimate the blurness of an image.
        Args:
            image: image as a numpy array of shape [height, width, channels].
            block_size: the size of the minimal DCT block size.
        Returns:
            a float value represents the blurness.
        """
        # A 2D histogram.
        hist = np.zeros((block_size, block_size), dtype=int)

        # Only the illumination is considered in blur.
        if len(image.shape) > 2:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('result', image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # Split the image into patches and do DCT on the image patch.
        height, width = image.shape
        round_v = int(height / block_size)
        round_h = int(width / block_size)
        for v in range(round_v):
            for h in range(round_h):
                v_start = v * block_size
                v_end = v_start + block_size
                h_start = h * block_size
                h_end = h_start + block_size

                image_patch = image[v_start:v_end, h_start:h_end]
                image_patch = np.float32(image_patch)
                patch_spectrum = cv2.dct(image_patch)
                patch_none_zero = np.abs(patch_spectrum) > self.dct_threshold
                hist += patch_none_zero.astype(int)

        _blur = hist < self.max_hist * hist[0, 0]
        _blur = (np.multiply(_blur.astype(int), self.hist_weight)).sum()
        return _blur/self.weight_total


if __name__ == "__main__":
    bd = BlurDetector()
    image = cv2.imread('cat.jpg')
    if image is None:
        print('Image file is not exist!')
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        result, image = bd.check_image_size(gray)
        blur = bd.get_blurness(image)
        print("Blurness: {:.4f}".format(blur))
        
        blur33 = cv2.blur(image, (3, 3))
        blur = bd.get_blurness(blur33)
        print("Blurness: {:.4f}".format(blur))
        
        blur55 = cv2.blur(image, (5, 5))
        blur = bd.get_blurness(blur55)
        print("Blurness: {:.4f}".format(blur))
        
        blur77 = cv2.blur(image, (7, 7))
        blur = bd.get_blurness(blur77)
        print("Blurness: {:.4f}".format(blur))
        
        gaussian33 = cv2.GaussianBlur(image, (3, 3), 0)    
        blur = bd.get_blurness(gaussian33)
        print("Blurness: {:.4f}".format(blur))
        
        gaussian55 = cv2.GaussianBlur(image, (5, 5), 0)
        blur = bd.get_blurness(gaussian55)
        print("Blurness: {:.4f}".format(blur))
        
        gaussian77 = cv2.GaussianBlur(image, (7, 7), 0)
        blur = bd.get_blurness(gaussian77)
        print("Blurness: {:.4f}".format(blur))
        
        median33 = cv2.medianBlur(image, 3)
        blur = bd.get_blurness(median33)
        print("Blurness: {:.4f}".format(blur))
        
        median55 = cv2.medianBlur(image, 5)
        blur = bd.get_blurness(median55)
        print("Blurness: {:.4f}".format(blur))
        
        median77 = cv2.medianBlur(image, 7)
        blur = bd.get_blurness(median77)
        print("Blurness: {:.4f}".format(blur))
        
        bilateral33 = cv2.bilateralFilter(image, 5, 21, 21)
        blur = bd.get_blurness(bilateral33)
        print("Blurness: {:.4f}".format(blur))
        
        bilateral55 = cv2.bilateralFilter(image, 7, 31, 31)
        blur = bd.get_blurness(bilateral55)
        print("Blurness: {:.4f}".format(blur))
        
        bilateral77 = cv2.bilateralFilter(image, 9, 41, 41)
        blur = bd.get_blurness(bilateral77)
        print("Blurness: {:.4f}".format(blur))

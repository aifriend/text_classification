import os
from math import floor
from random import randrange

import cv2
import imutils
import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.transform import rotate

from common.commonsLib import loggerElk

logger = loggerElk(__name__)


class ClassImage:

    @staticmethod
    def load_image(file):
        img = cv2.imread(file, 1)
        if img is None:
            logger.Debug('Could not open or find the image: ', file)
        # show_img(img, 'loaded')
        return img
        # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def load_image_gs(file):
        img = cv2.imread(file, 1)
        if img is None:
            logger.Debug('Could not open or find the image: ', file)
        # show_img(img, 'loaded')
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def save_image(file, data):
        cv2.imwrite(file, data)

    @staticmethod
    def crop_image(file):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        crop_img = img[0:200, 0:200]  # y,x
        # show_img(crop_img, 'cropped')
        return crop_img

    @staticmethod
    def crop_image_loaded(img, y, x):
        crop_img = img[0:y, 0:x]  # y,x
        # show_img(crop_img, 'cropped')
        return crop_img

    @staticmethod
    def resize_image(file):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        new_img = cv2.resize(img, (600, 800))  # w,h
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def resize_image_loaded(img, width, height):
        new_img = cv2.resize(img, (width, height))
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def resize_factor_image_loaded(img, width_factor, height_factor):
        new_img = cv2.resize(img, None, width_factor, height_factor)
        # show_img(new_img, 'resized')
        return new_img

    @staticmethod
    def show_img(img, text=''):
        cv2.imshow(text, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def show_img_np(self, data):
        img = (data * 255).astype(np.uint8)
        self.show_img(img)

    @staticmethod
    def rotate(img, angle=0):
        if angle == 0:
            angle = randrange(-2, 2)

        rotated = imutils.rotate(img, angle)
        # show_img(self, self, rotated, 'rotated')
        return rotated

    @staticmethod
    def translate(img, x=0, y=0):
        if y == 0:
            y = randrange(-5, 5)
        if x == 0:
            x = randrange(-5, 5)
        translated = imutils.translate(img, x, y)
        # show_img(translated, 'translated')
        return translated

    @staticmethod
    def zoom(img):
        new_img = img
        height, width, channels = img.shape
        up_down = randrange(-1, 2)
        if up_down == 1:
            new_img = cv2.pyrUp(img, dstsize=(width * 2, height * 2))
        elif up_down == 0:
            new_img = cv2.pyrDown(img, dstsize=(width // 2, height // 2))
        # show_img(new_img, 'zoomed')
        # print(new_img.shape)
        return new_img

    @staticmethod
    def brightness(img, value=0):
        if value == 0:
            value = randrange(0, 10)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value

        final_hsv = cv2.merge((h, s, v))
        new_img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
        return new_img

    @staticmethod
    def generate_examples(img, num=10):
        examples = [
            ClassImage.invert(
                ClassImage.denoise(
                    ClassImage.gray_image(img)))]
        for i in range(num):
            example = ClassImage.translate(ClassImage.rotate(examples[0]))
            examples.append(example)
            # show_img(example, 'example ' + str(i))

        return examples

    @staticmethod
    def invert(img):
        return cv2.bitwise_not(img)

    def kill_hermits2(self, img):
        result = img.copy()
        kernel = np.ones((3, 3))
        kernel[1, 1] = 0
        mask = convolve2d(img, kernel, mode='same', fillvalue=1)
        # print(mask)
        result[np.logical_and(mask == 8, img == 0)] = 1
        self.show_img(result, 'cleaned')
        return result

    def kill_hermits3(self, img):
        result = img.copy()
        kernel = np.ones((4, 4))
        kernel[1, 1] = 0
        kernel[1, 2] = 0
        mask = convolve2d(img, kernel, mode='same', fillvalue=1)
        # print(mask)
        result[np.logical_and(mask == 12, img == 0)] = 1
        self.show_img(result, 'cleaned')
        return result

    @staticmethod
    def denoise_pil(image):
        img = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
        return Image.fromarray(ClassImage.denoise(img))

    @staticmethod
    def denoise(img):
        result = img.copy()
        cv2.fastNlMeansDenoising(img, result, 10.0, 7, 21)
        return result

    @staticmethod
    def no_mod(img):
        return img

    def kill_hermits(self, img):
        result = img.copy()
        height, width, channels = img.shape
        # print(img.shape)
        for y in range(height):
            up = y > 0
            down = y < height - 1
            for x in range(width):
                left = x > 0
                right = x < width - 1
                # print(y, x)
                if result[y, x][0] == 0:
                    has_neighbour = False
                    if up and result[y - 1, x][0] == 0.0:
                        has_neighbour = True
                    if down and result[y + 1, x][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if left and result[y, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if right and result[y, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if up and left and result[y - 1, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if up and right and result[y - 1, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if down and left and result[y + 1, x - 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True
                    if down and right and result[y + 1, x + 1][0] == 0.0 and not has_neighbour:
                        has_neighbour = True

                    if not has_neighbour:
                        result[y, x] = 255

        self.show_img(result, 'cleaned')
        return result

    @staticmethod
    def black_white_image(img):
        # (thresh, im_bw) = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        im_bw = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 7, 3)
        return im_bw

    @staticmethod
    def black_white_image2(img, limit=150):
        result = img.copy()
        height, width = img.shape
        # print(img.shape)
        for y in range(height):
            for x in range(width):
                if result[y, x] > limit:
                    result[y, x] = 255
                else:
                    result[y, x] = 0
        return result

    @staticmethod
    def gray_image(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def is_blank_page_from_path(imagePath):
        result = False
        try:
            img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            return ClassImage.is_blank_page_from_img(img)

        except Exception as ex:
            logger.Error(ex)

        return result

    @staticmethod
    def deskew_image(image):
        result = image
        im = np.asarray(image)
        grayscale = rgb2gray(im)
        angle = determine_skew(grayscale)
        if (angle > 10 or angle < -10) and (angle > -45 or angle < 45):
            rotated = rotate(im, angle, resize=True) * 255
            result = Image.fromarray(rotated)
        return result

    @staticmethod
    def crimmins_speckle_removal_pil(pil):
        return Image.fromarray(
            ClassImage.crimmins_speckle_removal(
                cv2.cvtColor(
                    np.asarray(pil), cv2.COLOR_BGR2GRAY)))

    @staticmethod
    def crimmins_speckle_removal(data):
        new_image = data.copy()
        nrow = len(data)
        ncol = len(data[0])

        # Dark pixel adjustment

        # First Step
        # N-S
        for i in range(1, nrow):
            for j in range(ncol):
                if data[i - 1, j] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(ncol - 1):
                if data[i, j + 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # NW-SE
        for i in range(1, nrow):
            for j in range(1, ncol):
                if data[i - 1, j - 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # NE-SW
        for i in range(1, nrow):
            for j in range(ncol - 1):
                if data[i - 1, j + 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # Second Step
        # N-S
        for i in range(1, nrow - 1):
            for j in range(ncol):
                if (data[i - 1, j] > data[i, j]) and (data[i, j] <= data[i + 1, j]):
                    new_image[i, j] += 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol - 1):
                if (data[i, j + 1] > data[i, j]) and (data[i, j] <= data[i, j - 1]):
                    new_image[i, j] += 1
        data = new_image
        # NW-SE
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i - 1, j - 1] > data[i, j]) and (data[i, j] <= data[i + 1, j + 1]):
                    new_image[i, j] += 1
        data = new_image
        # NE-SW
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i - 1, j + 1] > data[i, j]) and (data[i, j] <= data[i + 1, j - 1]):
                    new_image[i, j] += 1
        data = new_image
        # Third Step
        # N-S
        for i in range(1, nrow - 1):
            for j in range(ncol):
                if (data[i + 1, j] > data[i, j]) and (data[i, j] <= data[i - 1, j]):
                    new_image[i, j] += 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol - 1):
                if (data[i, j - 1] > data[i, j]) and (data[i, j] <= data[i, j + 1]):
                    new_image[i, j] += 1
        data = new_image
        # NW-SE
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i + 1, j + 1] > data[i, j]) and (data[i, j] <= data[i - 1, j - 1]):
                    new_image[i, j] += 1
        data = new_image
        # NE-SW
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i + 1, j - 1] > data[i, j]) and (data[i, j] <= data[i - 1, j + 1]):
                    new_image[i, j] += 1
        data = new_image
        # Fourth Step
        # N-S
        for i in range(nrow - 1):
            for j in range(ncol):
                if data[i + 1, j] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol):
                if data[i, j - 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # NW-SE
        for i in range(nrow - 1):
            for j in range(ncol - 1):
                if data[i + 1, j + 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image
        # NE-SW
        for i in range(nrow - 1):
            for j in range(1, ncol):
                if data[i + 1, j - 1] >= (data[i, j] + 2):
                    new_image[i, j] += 1
        data = new_image

        # Light pixel adjustment

        # First Step
        # N-S
        for i in range(1, nrow):
            for j in range(ncol):
                if data[i - 1, j] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(ncol - 1):
                if data[i, j + 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # NW-SE
        for i in range(1, nrow):
            for j in range(1, ncol):
                if data[i - 1, j - 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # NE-SW
        for i in range(1, nrow):
            for j in range(ncol - 1):
                if data[i - 1, j + 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # Second Step
        # N-S
        for i in range(1, nrow - 1):
            for j in range(ncol):
                if (data[i - 1, j] < data[i, j]) and (data[i, j] >= data[i + 1, j]):
                    new_image[i, j] -= 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol - 1):
                if (data[i, j + 1] < data[i, j]) and (data[i, j] >= data[i, j - 1]):
                    new_image[i, j] -= 1
        data = new_image
        # NW-SE
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i - 1, j - 1] < data[i, j]) and (data[i, j] >= data[i + 1, j + 1]):
                    new_image[i, j] -= 1
        data = new_image
        # NE-SW
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i - 1, j + 1] < data[i, j]) and (data[i, j] >= data[i + 1, j - 1]):
                    new_image[i, j] -= 1
        data = new_image
        # Third Step
        # N-S
        for i in range(1, nrow - 1):
            for j in range(ncol):
                if (data[i + 1, j] < data[i, j]) and (data[i, j] >= data[i - 1, j]):
                    new_image[i, j] -= 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol - 1):
                if (data[i, j - 1] < data[i, j]) and (data[i, j] >= data[i, j + 1]):
                    new_image[i, j] -= 1
        data = new_image
        # NW-SE
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i + 1, j + 1] < data[i, j]) and (data[i, j] >= data[i - 1, j - 1]):
                    new_image[i, j] -= 1
        data = new_image
        # NE-SW
        for i in range(1, nrow - 1):
            for j in range(1, ncol - 1):
                if (data[i + 1, j - 1] < data[i, j]) and (data[i, j] >= data[i - 1, j + 1]):
                    new_image[i, j] -= 1
        data = new_image
        # Fourth Step
        # N-S
        for i in range(nrow - 1):
            for j in range(ncol):
                if data[i + 1, j] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # E-W
        for i in range(nrow):
            for j in range(1, ncol):
                if data[i, j - 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # NW-SE
        for i in range(nrow - 1):
            for j in range(ncol - 1):
                if data[i + 1, j + 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image
        # NE-SW
        for i in range(nrow - 1):
            for j in range(1, ncol):
                if data[i + 1, j - 1] <= (data[i, j] - 2):
                    new_image[i, j] -= 1
        data = new_image

        cv2.imwrite('crimmins_speckle_removal.png', new_image)
        return new_image

    @staticmethod
    def halftoning_error_diffusion(path, name):
        image = cv2.imread(os.path.join(path, name))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(path, 'gray_' + name), gray_image)

        height = gray_image.shape[0]
        width = gray_image.shape[1]

        for y in range(0, height):
            for x in range(0, width):

                old_value = gray_image[y, x]
                new_value = 0
                if old_value > 128:
                    new_value = 255

                gray_image[y, x] = new_value

                error = old_value - new_value

                if x < width - 1:
                    new_number = gray_image[y, x + 1] + error * 7 / 16
                    if new_number > 255:
                        new_number = 255
                    elif new_number < 0:
                        new_number = 0
                    gray_image[y, x + 1] = new_number

                if x > 0 and y < height - 1:
                    new_number = gray_image[y + 1, x - 1] + error * 3 / 16
                    if new_number > 255:
                        new_number = 255
                    elif new_number < 0:
                        new_number = 0
                    gray_image[y + 1, x - 1] = new_number

                if y < height - 1:
                    new_number = gray_image[y + 1, x] + error * 5 / 16
                    if new_number > 255:
                        new_number = 255
                    elif new_number < 0:
                        new_number = 0
                    gray_image[y + 1, x] = new_number

                if y < height - 1 and x < width - 1:
                    new_number = gray_image[y + 1, x + 1] + error * 1 / 16
                    if new_number > 255:
                        new_number = 255
                    elif new_number < 0:
                        new_number = 0
                    gray_image[y + 1, x + 1] = new_number

        cv2.imwrite(os.path.join(path, 'dither_' + name), gray_image)

    @staticmethod
    def floyd_steinberg_dither(image_file):
        new_img = Image.open(image_file)

        new_img = new_img.convert('RGB')
        pixel = new_img.load()

        x_lim, y_lim = new_img.size

        for y in range(1, y_lim):
            for x in range(1, x_lim):
                red_oldpixel, green_oldpixel, blue_oldpixel = pixel[x, y]

                red_newpixel = 255 * floor(red_oldpixel / 128)
                green_newpixel = 255 * floor(green_oldpixel / 128)
                blue_newpixel = 255 * floor(blue_oldpixel / 128)

                pixel[x, y] = red_newpixel, green_newpixel, blue_newpixel

                red_error = red_oldpixel - red_newpixel
                blue_error = blue_oldpixel - blue_newpixel
                green_error = green_oldpixel - green_newpixel

                if x < x_lim - 1:
                    red = pixel[x + 1, y][0] + round(red_error * 7 / 16)
                    green = pixel[x + 1, y][1] + round(green_error * 7 / 16)
                    blue = pixel[x + 1, y][2] + round(blue_error * 7 / 16)

                    pixel[x + 1, y] = (red, green, blue)

                if x > 1 and y < y_lim - 1:
                    red = pixel[x - 1, y + 1][0] + round(red_error * 3 / 16)
                    green = pixel[x - 1, y + 1][1] + round(green_error * 3 / 16)
                    blue = pixel[x - 1, y + 1][2] + round(blue_error * 3 / 16)

                    pixel[x - 1, y + 1] = (red, green, blue)

                if y < y_lim - 1:
                    red = pixel[x, y + 1][0] + round(red_error * 5 / 16)
                    green = pixel[x, y + 1][1] + round(green_error * 5 / 16)
                    blue = pixel[x, y + 1][2] + round(blue_error * 5 / 16)

                    pixel[x, y + 1] = (red, green, blue)

                if x < x_lim - 1 and y < y_lim - 1:
                    red = pixel[x + 1, y + 1][0] + round(red_error * 1 / 16)
                    green = pixel[x + 1, y + 1][1] + round(green_error * 1 / 16)
                    blue = pixel[x + 1, y + 1][2] + round(blue_error * 1 / 16)

                    pixel[x + 1, y + 1] = (red, green, blue)

        new_img.show()

    @staticmethod
    def floyd_steinberg_dither_gs(_path, name):
        new_img = Image.open(os.path.join(_path, name)).convert('L')

        pixel = new_img.load()

        x_lim, y_lim = new_img.size

        for y in range(0, y_lim):
            for x in range(0, x_lim):
                oldpixel = pixel[x, y]
                newpixel = 255 * floor(oldpixel / 254)
                pixel[x, y] = newpixel
                error = oldpixel - newpixel

                if x < x_lim - 1:
                    pixel[x + 1, y] = pixel[x + 1, y] + round(error * 7 / 16)

                if x > 1 and y < y_lim - 1:
                    pixel[x - 1, y + 1] = pixel[x - 1, y + 1] + round(error * 3 / 16)

                if y < y_lim - 1:
                    pixel[x, y + 1] = pixel[x, y + 1] + round(error * 5 / 16)

                if x < x_lim - 1 and y < y_lim - 1:
                    pixel[x + 1, y + 1] = pixel[x + 1, y + 1] + round(error * 1 / 16)

        new_img.save(os.path.join(_path, 'fs_' + name))
        # new_img.show()

    @staticmethod
    def fusion_gs(path_fusion, path_doc, path_logo, filename_doc, filename_logo, size, dx=0, dy=0):
        img1 = Image.open(os.path.join(path_doc, filename_doc)).convert('L')
        img2 = Image.open(os.path.join(path_logo + size, filename_logo)).convert('L')

        pixel1 = img1.load()
        pixel2 = img2.load()

        x_lim1, y_lim1 = img1.size
        x_lim2, y_lim2 = img2.size

        sx = round((x_lim1 - x_lim2) / 2) + dx
        sy = round((y_lim1 - y_lim2) / 2) + dy

        if sx + x_lim2 > x_lim1 or sy + y_lim2 > y_lim1:
            return

        for y in range(0, y_lim2):
            for x in range(0, x_lim2):
                # if pixel2[x, y] < pixel1[sx + x, sy + y]:
                #     pixel1[sx + x, sy + y] = pixel2[x, y]
                if pixel1[sx + x, sy + y] > 240 and pixel2[x, y] < 254:
                    pixel1[sx + x, sy + y] = pixel2[x, y]

        fd = filename_doc.replace('PDF', '').replace('pdf', '')
        fl = filename_logo.replace('PNG', '').replace('png', '')
        img1.save(os.path.join(path_fusion, f'fusion_{size}_{fl}_{fd}'))
        # img1.show()

    @staticmethod
    def is_blank_page_from_img(img):
        histSum = 0
        threshold = 70000
        result = False
        try:
            # hist es una matriz de 256 * 1, cada valor corresponde al número de píxeles
            # con los valores de píxeles correspondientes en la imagen
            histogram = cv2.calcHist([img], [0], None, [256], [0, 128])

            histSum = sum(histogram)
            # print(f'Histogram accumulated value {histSum}')

            if histSum <= threshold:
                # logger.Information(f'Image {histSum} is empty')
                return True

        except Exception as ex:
            logger.Error(ex)

        # logger.Information(f'Image {histSum} has info')
        return result

    @staticmethod
    def is_blank_page_from_pil(pil_img):
        result = False
        try:
            img = np.array(pil_img)
            return ClassImage.is_blank_page_from_img(img)

        except Exception as ex:
            logger.Error(ex)

        return result

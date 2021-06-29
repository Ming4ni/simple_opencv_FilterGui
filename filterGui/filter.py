import sys
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QDialog, QFileDialog, QGridLayout, QLabel, QPushButton


class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.resize(700, 700)
        self.label = QLabel()
        self.btnOpen = QPushButton('Open Image', self)
        self.btnProcess1 = QPushButton('pencil', self)
        self.btnProcess2 = QPushButton('cooling', self)
        self.btnProcess3 = QPushButton('warming', self)
        self.btnProcess4 = QPushButton('cartoon', self)
        self.btnSave = QPushButton('Save Image', self)
        self.btnSave.setEnabled(False)
        

        layout = QGridLayout(self)
        layout.addWidget(self.label, 0, 0, 4, 4)
        layout.addWidget(self.btnOpen, 4, 0, 1, 1)
        layout.addWidget(self.btnProcess1, 4, 1, 1, 1)
        layout.addWidget(self.btnProcess2, 4, 2, 1, 1)
        layout.addWidget(self.btnProcess3, 4, 3, 1, 1)
        layout.addWidget(self.btnProcess4, 4, 4, 1, 1)
        layout.addWidget(self.btnSave, 4, 5, 1, 1)
        

        self.btnOpen.clicked.connect(self.openSlot)
        self.btnProcess1.clicked.connect(self.processSlot1)
        self.btnProcess2.clicked.connect(self.processSlot2)
        self.btnProcess3.clicked.connect(self.processSlot3)
        self.btnProcess4.clicked.connect(self.processSlot4)
        self.btnSave.clicked.connect(self.saveSlot)
        
    
    def openSlot(self):
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        self.img = cv2.imread(filename, -1)
        if self.img.size == 1:
            return
        
        height, width, channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        
        self.btnSave.setEnabled(True)
    
    def dodgeV2(self, image, mask): 
        return cv2.divide(image, 255-mask, scale=256)
        
    def processSlot1(self):
      
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_RGB2GRAY)
        
        img_gray_inv = 255 - img_gray
        
        img_blur = cv2.GaussianBlur(img_gray_inv, (21, 21), 0, 0)
        
        img_blend = self.dodgeV2(img_gray, img_blur)
    
        self.pencil_img = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2RGB)
        
        height, width, channel = self.pencil_img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.pencil_img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
        
    def create_LUT_8UC1(self, x, y):
        spl = UnivariateSpline(x, y)
        return spl(range(256))
    
    def processSlot2(self):
        
        
        self.incr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256],  #increase
                                                [0, 70, 140, 210, 256])
        self.decr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256],  #decrease
                                                [0, 30,  80, 120, 192])
        
        c_b, c_g, c_r = cv2.split(self.img)
        c_r = cv2.LUT(c_r, self.decr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.incr_ch_lut).astype(np.uint8)
        img_rg = cv2.merge((c_b, c_g, c_r))

        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_rg, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        self.img_cooling =  cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
        
        
        height, width, channel = self.img_cooling.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img_cooling.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    
    def processSlot3(self):
        
        
        self.incr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256],
                                                [0, 70, 140, 210, 256])
        self.decr_ch_lut = self.create_LUT_8UC1([0, 64, 128, 192, 256],
                                                [0, 30,  80, 120, 192])
        
        c_b, c_g, c_r = cv2.split(self.img)
        c_r = cv2.LUT(c_r, self.incr_ch_lut).astype(np.uint8)
        c_b = cv2.LUT(c_b, self.decr_ch_lut).astype(np.uint8)
        img_r = cv2.merge((c_b, c_g, c_r))

        c_h, c_s, c_v = cv2.split(cv2.cvtColor(img_r, cv2.COLOR_RGB2HSV))
        c_s = cv2.LUT(c_s, self.incr_ch_lut).astype(np.uint8)

        self.img_warming =  cv2.cvtColor(cv2.merge((c_h, c_s, c_v)), cv2.COLOR_HSV2RGB)
        
        
        height, width, channel = self.img_warming.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img_warming.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    
    def processSlot4(self):
        
        
        numDownSamples = 2       # number of downscaling steps
        numBilateralFilters = 7  # number of bilateral filtering steps
        
        img_ca =  cv2.resize(self.img, (600, 500))
        
        img_color = img_ca
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        # repeatedly apply small bilateral filter instead of applying
        ## one large filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        # upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # -- STEPS 2 and 3 --
        # convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_ca, cv2.COLOR_RGB2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)

        # -- STEP 4 --
        # detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 --
        # convert back to color so that it can be bit-ANDed with color image
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)
        self.cartoon_img = cv2.bitwise_and(img_color, img_edge)
        
        height, width, channel = self.cartoon_img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.cartoon_img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.label.setPixmap(QPixmap.fromImage(self.qImg))
    
    def saveSlot(self):
        filename, _ = QFileDialog.getSaveFileName(self, 'Save Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        cv2.imwrite(filename, self.img)
        

   

if __name__ == '__main__':
    a = QApplication(sys.argv)
    dialog = MyDialog()
    dialog.show()
    sys.exit(a.exec_())
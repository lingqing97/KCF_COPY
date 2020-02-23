import numpy as np
import cv2
import copy


def limit(rect, limit):
    '''
    @description:根据limit对rect进行裁剪
    @param rect:输入的矩阵,[左上角x，左上角y,w,h]
    @param limit:限制的范围,[左上角,左上角y,w,h]
    @return rect:裁剪后的范围
    '''
    if((rect[0]+rect[2]) > (limit[0]+limit[2])):
        rect[2] = limit[0]+limit[2]-rect[0]
    if((rect[1]+rect[3] > (limit[1]+limit[3]))):
        rect[3] = limit[1]+limit[3]-rect[1]
    if(rect[0] < 0):
        rect[2] -= (limit[0]-rect[0])
        rect[0] = limit[0]
    if(rect[1] < 0):
        rect[3] -= (limit[1]-rect[1])
        rect[1] = limit[1]
    if(rect[2] < 0):
        rect[2] = 0
    if(rect[3] < 0):
        rect[3] = 0
    return rect


def x2(win):
    return win[0]+win[2]


def y2(win):
    return win[1]+win[3]


def getBorder(win, limited):
    '''
    @description:获取填充区域
    @param win:未裁剪窗口
    @param limited:裁剪后的窗口
    @return res:[左边填充大小,上面填充大小，右边填充大小,下面填充大小]
    '''
    res = [0, 0, 0, 0]
    res[0] = limited[0]-win[0]
    res[1] = limited[1]-win[1]
    res[2] = x2(win)-x2(limited)
    res[3] = y2(win)-y2(limited)

    return res


class KCFtracker():
    def __init__(self, fixed_window=False):
        self.padding = 2.5  # 定义padding
        self.hann = None  # 汉明窗
        self.output_sigma_factor = 0.125  # 二维高斯参数sigma
        self.lmbda = 0.01  # alpha中的参数lambda
        self.sigma = 0.2  # RBF中的参数sigma
        self.eta = 0.075  # 学习权重
        if(fixed_window):
            self.template_size = 96  # 将特征图的大边调整到96
        else:
            self.template_size= 1

    def init(self, img, x1, y1, w, h):
        # 记录中点位置
        self.cx = int(x1+w/2)
        self.cy = int(y1+h/2)
        #记录左上角位置
        self.x1=x1
        self.y1=y1

        # 进行padding
        self.width = int(w*self.padding)
        self.height = int(h*self.padding)
        # 将width和height化为偶数便于处理
        self.width = self.width if self.width % 2 == 0 else self.width-1
        self.height = self.height if self.height % 2 == 0 else self.height-1

        # 获取跟踪框内图像的特征图,转灰度图,归一化并加汉明窗,获取训练集
        self.x = self.getFeature(img, 1)

        # 获取理想高斯相应
        self.y = self.target(self.width,self.height)

        # self.prev存储响应最大值的位置
        # 这里是理想响应，所以就是中心点是最大值
        self.prev = np.unravel_index(np.argmax(self.y), self.y.shape)

        # 训练
        self.alpha = self.train(self.x, self.y, self.sigma, self.lmbda)

    def train(self, x, y, sigma, lmbda):
        # 计算k_xx
        k = self.kernel_corralation(x, x, sigma)
        # 计算alpha
        alpha = np.fft.fft2(y)/(np.fft.fft2(k)+lmbda)
        return alpha

    def detect(self, alpha, x, z, sigma):
        # 计算k_xz
        k = self.kernel_corralation(x, z, sigma)
        # 计算响应
        response = np.real(np.fft.ifft2(np.fft.fft2(k)*alpha))
        return response

    def update(self, img):
        # 截取跟踪框中的图像,即获取预测集
        z = self.getFeature(img)
        # 计算响应
        response = self.detect(self.alpha, self.x, z, self.sigma)
        # 获取当前最大响应的位置
        curr = np.unravel_index(np.argmax(response), response.shape)
        # 计算偏移
        dx = self.prev[1]-curr[1]
        dy = self.prev[0]-curr[0]
        print(dx,dy)
        # if(abs(dx)>=4 or abs(dy)>=4):
        #     print('hello')
        # 更新当前位置
        self.cx=self.cx+dx
        self.cy=self.cy+dy
        self.x1 = self.x1+dx
        self.y1 = self.y1+dy
        # 更新训练集
        prevx = self.x
        currx = self.getFeature(img)
        self.alpha = self.eta * \
            self.train(currx, self.y, self.sigma, self.lmbda) + \
            (1-self.eta)*self.alpha
        self.x = self.eta*currx+(1-self.eta)*prevx
        # 返回左上角坐标
        return self.x1, self.y1

    def kernel_corralation(self, x1, x2, sigma):
        c = np.fft.fftshift(np.fft.ifft2(
            np.fft.fft2(x1)*np.conjugate(np.fft.fft2(x2))))
        mult = np.dot(np.conjugate(x1.flatten(1)), x1.flatten(1)) + \
            np.dot(np.conjugate(x2.flatten(1)), x2.flatten(1))-2*c
        k = np.exp(-1/sigma**2*np.abs(mult)/np.size(x1))

        return k

    def create_hann(self):

        i = np.arange(self.width)
        j = np.arange(self.height)
        I, J = np.meshgrid(i, j)
        self.hann = np.sin(np.pi*I/self.width)*np.sin(np.pi*J/self.height)

    def getFeature(self, img, inithann=0):

        window = [int(self.cx-self.width/2), int(self.cy-self.height/2), self.width, self.height]
        cutWin = copy.deepcopy(window)
        # 对越界的部分进行裁剪
        cutWin = limit(cutWin, [0, 0, img.shape[1], img.shape[0]])
        # 计算需要填充的区域
        border = getBorder(window, cutWin)

        res = img[cutWin[1]:cutWin[1]+cutWin[3], cutWin[0]:cutWin[0]+cutWin[2]]
        if border != [0, 0, 0, 0]:
            res = cv2.copyMakeBorder(
                res, border[1], border[3], border[0], border[2], cv2.BORDER_REPLICATE)

        # 若是彩色图，则转为灰度图
        if(res.ndim == 3 and res.shape[2] == 3):
            FeatureMap = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        else:
            FeatureMap = res
        # 归一化
        FeatureMap = FeatureMap.astype(np.float32)/255.0-0.5

        if(inithann):
            self.create_hann()

        # 加窗
        return FeatureMap*self.hann

    def target(self, width, height):
        i = np.arange(width)
        j = np.arange(height)
        # np.meshgrid(a,b)生成的形状为(len(b),len(a))
        I, J = np.meshgrid(i, j)
        # 生成以中点为中心的二维高斯相应
        xx = (I-width/2)**2+(J-height/2)**2

        # 二维高斯响应的凸出面积要与目标面积成比例
        output_sigma = np.sqrt(width*height) / \
            self.padding * self.output_sigma_factor
        mult = -0.5 / (output_sigma*output_sigma)

        return np.exp(mult*xx)

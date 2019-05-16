
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF

def network(x, y, test=False):
    # Input:x -> 3,64,64
    # AveragePooling -> 3,12,21
    h = F.average_pooling(x, (5,3), (5,3))
    # LeakyReLU_2
    h = F.leaky_relu(h, 0.1, True)
    # Convolution_2 -> 20,13,21
    h = PF.convolution(h, 20, (2,3), (1,1), name='Convolution_2')
    # BatchNormalization
    h = PF.batch_normalization(h, (1,), 0.9, 0.0001, not test, name='BatchNormalization')
    # ReLU
    h = F.relu(h, True)
    # DepthwiseConvolution
    h = PF.depthwise_convolution(h, (5,5), (2,2), name='DepthwiseConvolution')
    # MaxPooling_2 -> 20,6,7
    h = F.max_pooling(h, (2,3), (2,3))
    # LeakyReLU
    h = F.leaky_relu(h, 0.1, True)
    # Affine -> 2
    h = PF.affine(h, (2,), name='Affine')
    # Softmax
    h = F.softmax(h)
    return h

# nnlabla初期化
nn.load_parameters('aresults.nnp')
x=nn.Variable((1,3,64,64))
IMAGE_SIZE = 64

# webcamera初期化
cap = cv2.VideoCapture(0)

# 確認用のwindowを作成
root = tk.Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = tk.Label(root)
lmain.pack()
label = tk.Label(text="Apple",fg="red",font=("Helvetica", 30, "bold"))

def get_img():
	_, img = cap.read()
	frame = cv2.imencode('.jpg', img)[1].tobytes()
	yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def show_frame():
	_, frame = cap.read()
	frame = cv2.resize(frame, dsize=(560, 400))
	frame = cv2.flip(frame, 1)

	# cv2 image --> tkinter image
	cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
	img = Image.fromarray(cv2image)
	imgtk = ImageTk.PhotoImage(image=img)

	# 表示
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk,width=600,height=600)

	# 認識
	im = cv2.resize(frame, (IMAGE_SIZE,IMAGE_SIZE)).transpose(2,0,1)
	x = nn.Variable((1, ) + im.shape)
	x.d = im.reshape(x.shape)
	y = network(x, x,test=True)
	y.forward()
	if y.d[0][0] >= 0.49 :
		label.pack()
	lmain.after(20, show_frame)

show_frame()
root.mainloop()
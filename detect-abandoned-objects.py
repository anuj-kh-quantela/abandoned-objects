import cv2
import numpy as np
import time
import datetime

cap = cv2.VideoCapture('/home/anuj/git/personal/github/ObjLeft/pets2006_1.avi')
# cap = cv2.VideoCapture('/home/anuj/git/personal/github/Abandoned_Object/aban3.mp4')


interval = 0
count = 0
nPixel = 100
flag = 0

fgbg_mog2 = cv2.createBackgroundSubtractorMOG2(nPixel,cv2.THRESH_BINARY,2)
remove_shadow = True

aban = None
sub = None

algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

while cap.isOpened():
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	count += 1

	fgmask = fgbg_mog2.apply((gray))
	if remove_shadow:
		fgmask[fgmask == 127] = 0

	back = fgbg_mog2.getBackgroundImage()
	cv2.imshow('back', back)
	#________________ bgs over _________

	if flag == 0:
		aban = cv2.absdiff(back, back)
		cv2.imshow("first aban", aban)
		print("aban.shape: ", aban.shape)
		interval = 300
		flag = 10

	if ((flag == 10) and (count >= nPixel)):
		aban = back.copy()
		cv2.imshow("second aban", aban)
		flag = 20

	start_time = datetime.datetime.strptime(algo_start_time, '%H:%M:%S')
	check_time = datetime.datetime.now().time().strftime('%H:%M:%S')
	end_time = datetime.datetime.strptime(check_time, '%H:%M:%S')
	diff = (end_time - start_time)
	print("\nPrinting time: " +str((diff.seconds)))

	if(diff.seconds >= interval):
		aban = back.copy()
		cv2.imshow("third aban", aban)
		algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

	sub = cv2.absdiff(back, aban)
	cv2.imshow("sub", sub)

	count += 1
	k = cv2.waitKey()
	if k == 27:
		break
		cap.release()





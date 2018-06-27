import cv2
import numpy as np
import time
import datetime

# contour merging
# https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them/2889#2889

cap = cv2.VideoCapture('/home/anuj/git/personal/github/others/ObjLeft/pets2006_1.avi')
# cap = cv2.VideoCapture('/home/anuj/git/personal/github/others/Abandoned_Object/aban3.mp4')
# 2. video properties
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# out = cv2.VideoWriter("test.avi", cv2.VideoWriter_fourcc(*'MPEG'), fps, (w*2, h))

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
	cv2.imshow("original image", frame)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	print("count: ", count)
	count += 1

	fgmask = fgbg_mog2.apply((gray))
	if remove_shadow:
		fgmask[fgmask == 127] = 0

	back = fgbg_mog2.getBackgroundImage()
	cv2.imshow('Static Background', back)
	#________________ bgs over _________

	if flag == 0:
		# initialize aban
		aban = cv2.absdiff(back, back)
		interval = 300
		flag = 10

	if ((flag == 10) and (count >= nPixel)):
		print("above 100")
		# count >= nPixel - meaning that the background has been successfully captured
		aban = back.copy()
		flag = 20

	start_time = datetime.datetime.strptime(algo_start_time, '%H:%M:%S')
	check_time = datetime.datetime.now().time().strftime('%H:%M:%S')
	end_time = datetime.datetime.strptime(check_time, '%H:%M:%S')
	diff = (end_time - start_time)
	print("\nPrinting time: " +str((diff.seconds)))

	if(diff.seconds >= interval):
		aban = back.copy()
		algo_start_time = datetime.datetime.now().time().strftime('%H:%M:%S')

	
	
	# for video writing 
	# temp_aban = cv2.cvtColor(sub, cv2.COLOR_GRAY2BGR)
	# temp_concat = np.concatenate((frame, temp_aban), axis=1)
	# out.write(temp_concat)
	# cv2.imshow("output", temp_concat)

	# object localization
	# logic: put bounding box on those objects which are of near constant area 
	# and those object's coordinates are not changing overtime

	if count >= nPixel:
		sub = cv2.absdiff(back, aban)
		cv2.imshow("Abandoned Object", sub)
		ret,thresh1 = cv2.threshold(sub,127,255,cv2.THRESH_BINARY)
		kernel = np.ones((3,3),np.uint8)
		dilation = cv2.dilate(thresh1, kernel, iterations = 4)
		ret,thresh = cv2.threshold(dilation,127,255,0)
		im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(frame, contours, -1, (0,255,0), 3)
		cv2.imshow('dilation-cont', frame)
		test = frame.copy()
		for cnt in contours:
			x,y,w,h = cv2.boundingRect(cnt)
			cv2.rectangle(test, (x,y), (x+w,y+h), (255,255,255), 2)
		cv2.imshow("test", test)

	count += 1
	k = cv2.waitKey(1)
	if k == 27:
		break
		cap.release()





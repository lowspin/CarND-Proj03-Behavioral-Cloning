import matplotlib.pyplot as plt
import moviepy.editor as mpy

def show_driving_data(steering, throttle, brake, speed, X_center, X_left, X_right, startframe=0, endframe=100):

    if endframe==-1:
        endframe = len(X_center)-1;

    plt.figure(figsize=(15,15))

    # steering
    plt.subplot(2, 3, 1)
    plt.title("steering")
    plt.plot(steering[startframe:endframe]) #plt.hist(steering)

    # throttle / brake
    frameindex = range(startframe,endframe)
    ax = plt.subplot(2, 3, 2)
    plt.title("throttle / brake")
    plt.plot(frameindex,throttle[startframe:endframe],'g',label='throttle') #plt.hist(throttle)
    plt.plot(frameindex,brake[startframe:endframe],'r',label='brake') #plt.hist(brake)
    plt.legend(loc="best", ncol=2)

    # speed
    plt.subplot(2, 3, 3)
    plt.title("speed")
    plt.plot(speed[startframe:endframe]) #plt.hist(speed)

    # show camera images for last frame
    plt.subplot(2, 3, 4)
    plt.imshow(X_left[endframe])
    plt.title('left camera - last frame #%d'%(endframe-1))
    plt.subplot(2, 3, 5)
    plt.imshow(X_center[endframe])
    plt.title('center camera - last frame #%d'%(endframe-1))
    plt.subplot(2, 3, 6)
    plt.imshow(X_right[endframe])
    plt.title('right camera - last frame #%d'%(endframe-1))

    plt.show()

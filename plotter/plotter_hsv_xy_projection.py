import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import math


figure, axes = plt.subplots()
Drawing_colored_circle = plt.Circle(( 0.6 , 0.6 ), 0.2 )
axes.grid()
plt.xlim(-1.5,1.5)
plt.ylim(-1.5,1.5)
axes.scatter(0, 0, color="red")

a = plt.Circle(( 0 , 0 ), 1, fill=False, edgecolor="Green" )
angle = 2*math.pi/6
sample_x = math.cos(angle)
sample_y = math.sin(angle)
l = plt.Line2D([0,sample_x],[0,sample_y])
x_proj = plt.Line2D([0,sample_x],[0,0], color="Red")
y_proj = plt.Line2D([sample_x,sample_x],[0,sample_y], color="Red")
axes.set_aspect( 1 )
axes.add_artist( a)
axes.add_artist( l )
axes.add_artist( x_proj )
axes.add_artist( y_proj )
axes.annotate('Hue', xy=(sample_x,sample_y))
axes.text(0.25,-0.1,'x')
axes.text(sample_x+0.05,sample_y/2-0.1,'y')
plt.show()
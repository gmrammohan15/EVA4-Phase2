import os
import shutil
import random

#class_names = ['Flying Birds', 'Large QuadCopters', 'Small Quadcopters', 'Winged Drones']
class_names = ['Winged Drones']
for classes in class_names:
  print (classes)
  path1 = '/content/gdrive/My Drive/dronesdataset/' + classes + '/'
  path2 = '/content/gdrive/My Drive/eva/s2/' + classes + '/'
  train = path2 + 'train/'
  val = path2 + 'val/'

  try:
    os.makedirs (path2)
  except:
    pass
  try:
    os.makedirs (train)
  except:
    pass
  try:
    os.makedirs (val)
  except:
    pass

  filenames = os.listdir (path1)
  random.shuffle (filenames)
  l = len (filenames)
  print (l)

  for i, i_file in enumerate (filenames):
    if i % 1000 == 0:
      print (i)
    if i < 0.9 * l:
      shutil.copy (path1 + i_file, train + i_file)
    else:
      shutil.copy (path1 + i_file, val + i_file)
  print ('done')
  #break
from PIL import Image, ImageChops, ImageFilter
im_x = "img2.png"
im_y = "img3.png"
y = Image.open(im_y)
x = Image.open(im_x)
# x.show()
# y.show()
# print(y.size, y.mode)
merge = ImageChops.multiply(x, y) # japierdole zajebiste btw
# merge.show()
add = ImageChops.add(x,y) # to też
# add.show()
szarnuh = add.convert('L')
# szarnuh.show()
pixel = add.load()
for row in range(merge.size[0]):
    for column in range(merge.size[1]):
        if pixel[row, column] == (255,255,255):
            pixel[row, column] = (0,0,0)
add.show()
import random
import torch


class ImagePool():
    """
    This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    
    实现一个图像缓冲区，用于存储以前生成的图像。

    这个缓冲区使我们能够使用生成图像的历史记录来更新判别器
    """

    def __init__(self, pool_size):
        """
        Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
            pool_size (int) -- 图像缓冲区的大小，如果 pool_size=0，则不会创建缓冲区
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """
        Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # 如果缓冲区大小为 0，则不执行任何操作
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0) # 为了适应 torch.cat() 的格式，增加一个维度，这样shuchunk()后的维度为 b * c * h * w
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                # 如果缓冲区未满，则将当前图像插入缓冲区
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                # 如果缓冲区已满，则有 50% 的几率返回以前存储的图像，并将当前图像插入缓冲区
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    # 缓冲区有 50% 的几率返回以前存储的图像
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone() # 从缓冲区中复制图像
                    self.images[random_id] = image # 将当前图像替换缓冲区中被选中的图像
                    return_images.append(tmp) # 返回图像列表中加入被选中的图像
                else:       # by another 50% chance, the buffer will return the current image
                    # 另外 50% 的几率返回当前图像, 当前图像不加入缓冲区, 返回图像列表中依然加入当前图像
                    return_images.append(image) 
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images
